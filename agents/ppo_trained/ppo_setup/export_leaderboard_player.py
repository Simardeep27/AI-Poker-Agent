"""Export a trained MaskablePPO checkpoint as a one-file leaderboard player.

The generated file embeds the actor-network weights and reimplements only the
deterministic forward pass needed at evaluation time. It does not require
stable-baselines3 or sb3-contrib on the leaderboard runner; it only expects
numpy plus the PyPokerEngine package used by the competition.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from collections import OrderedDict
from pathlib import Path
import pickle
import re
import textwrap
import zipfile

import numpy as np

from ppo_setup.leaderboard import LEADERBOARD_EQUITY_SIMS, LEADERBOARD_INITIAL_STACK, LEADERBOARD_SMALL_BLIND


LEADERBOARD_TEMPLATE = r'''"""Single-file PPO PyPokerEngine player.

Generated from a MaskablePPO checkpoint. Submit this file as custom_player.py
when the leaderboard imports MyPlayer or setup_ai().
"""

import base64
import io
import math
import os
import sys

import numpy as np


def _ensure_engine_paths():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    parent = os.path.dirname(here)
    candidates = [
        here,
        os.path.join(here, "AI-Poker-Agent"),
        parent,
        os.path.join(parent, "AI-Poker-Agent"),
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "pypokerengine")) and candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_engine_paths()

from pypokerengine.players import BasePokerPlayer

try:
    from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
except Exception:
    estimate_hole_card_win_rate = None
    gen_cards = None


ACTION_TO_ID = {"fold": 0, "call": 1, "raise": 2}
ID_TO_ACTION = {0: "fold", 1: "call", 2: "raise"}
STREET_TO_ID = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 3}
RANKS = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
SUITS = {"C": 0, "D": 1, "H": 2, "S": 3}
OBSERVATION_SIZE = 71

_MODEL_META = @@MODEL_META@@
_MODEL_NPZ_B64 = """
@@MODEL_NPZ_B64@@
"""
_POLICY_ARRAYS = None


class EncoderConfig(object):
    def __init__(self, initial_stack=10000, small_blind=20, equity_sims=32):
        self.initial_stack = int(initial_stack)
        self.small_blind = int(small_blind)
        self.equity_sims = int(equity_sims)

    @property
    def big_blind(self):
        return self.small_blind * 2


class OpponentStats(object):
    def __init__(self):
        self.preflop_raises = 0
        self.preflop_actions = 0
        self.fold_to_raise = 0
        self.raises_we_made = 0
        self.street_raises = [0, 0, 0, 0]
        self.street_total = [0, 0, 0, 0]
        self.showdowns = 0
        self.hands_total = 0
        self.just_raised = False

    def start_round(self):
        self.hands_total += 1
        self.just_raised = False

    def observe_action(self, action, round_state, hero_uuid):
        actor = action.get("player_uuid", "")
        name = action.get("action", "").lower()
        if actor == hero_uuid:
            self.just_raised = name == "raise"
            return

        street = round_state.get("street", "preflop")
        street_num = STREET_TO_ID.get(street, 0)
        self.street_total[street_num] += 1
        if name == "raise":
            self.street_raises[street_num] += 1

        if street_num == 0 and name in ("raise", "call", "fold"):
            self.preflop_actions += 1
            if name == "raise":
                self.preflop_raises += 1

        if self.just_raised:
            self.raises_we_made += 1
            if name == "fold":
                self.fold_to_raise += 1
        self.just_raised = False

    def observe_round_result(self, hand_info):
        if hand_info:
            self.showdowns += 1
        self.just_raised = False

    def snapshot(self):
        total_actions = sum(self.street_total)
        total_raises = sum(self.street_raises)
        return {
            "preflop_raise_frequency": _smooth(self.preflop_raises, self.preflop_actions, 0.32),
            "aggression": _smooth(total_raises, total_actions, 0.42),
            "fold_to_raise": _smooth(self.fold_to_raise, self.raises_we_made, 0.40),
            "showdown_willingness": _smooth(self.showdowns, self.hands_total, 0.35),
            "hands_total": self.hands_total,
        }


class MyPlayer(BasePokerPlayer):
    def __init__(self, deterministic=True):
        super(MyPlayer, self).__init__()
        self.deterministic = bool(deterministic)
        self.config = EncoderConfig(
            initial_stack=_MODEL_META.get("initial_stack", 10000),
            small_blind=_MODEL_META.get("small_blind", 20),
            equity_sims=_MODEL_META.get("equity_sims", 32),
        )
        self.stats = OpponentStats()
        _policy_arrays()

    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            obs = encode_observation(
                valid_actions,
                hole_card,
                round_state,
                self.uuid,
                self.stats.snapshot(),
                self.config,
            )
            mask = action_mask(valid_actions)
            action_id = _predict_policy(obs, mask, self.deterministic)
            action = ID_TO_ACTION.get(int(action_id), "fold")
            return action if mask[ACTION_TO_ID[action]] else _fallback_action(valid_actions)
        except Exception:
            return _fallback_action(valid_actions)

    def receive_game_start_message(self, game_info):
        rule = game_info.get("rule", {}) if isinstance(game_info, dict) else {}
        self.config = EncoderConfig(
            initial_stack=rule.get("initial_stack", self.config.initial_stack),
            small_blind=rule.get("small_blind_amount", self.config.small_blind),
            equity_sims=self.config.equity_sims,
        )
        self.stats = OpponentStats()

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.stats.start_round()

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self.stats.observe_action(action, round_state, self.uuid)

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.stats.observe_round_result(hand_info)


def setup_ai():
    return MyPlayer()


def action_mask(valid_actions):
    legal = set(entry.get("action") for entry in valid_actions)
    return np.asarray(
        [
            "fold" in legal,
            "call" in legal,
            "raise" in legal,
        ],
        dtype=bool,
    )


def encode_observation(valid_actions, hole_card, round_state, hero_uuid, opponent_stats=None, config=None):
    config = config or EncoderConfig()
    opponent_stats = opponent_stats or {}

    street = round_state.get("street", "preflop")
    street_num = STREET_TO_ID.get(street, 0)
    community = round_state.get("community_card", [])
    seats = round_state.get("seats", [])
    hero_seat = next((seat for seat in seats if seat.get("uuid") == hero_uuid), {})
    opp_seat = next((seat for seat in seats if seat.get("uuid") != hero_uuid), {})

    hero_stack = float(hero_seat.get("stack", config.initial_stack))
    opp_stack = float(opp_seat.get("stack", config.initial_stack))
    effective_stack = min(hero_stack, opp_stack)
    pot = float(round_state.get("pot", {}).get("main", {}).get("amount", 0.0))
    to_call, to_raise = get_bet_amounts(round_state, hero_uuid, config.small_blind)
    legal_mask = action_mask(valid_actions)
    is_sb = _hero_is_small_blind(round_state, hero_uuid)

    equity = estimate_equity(hole_card, community, config.equity_sims)
    preflop = preflop_strength(hole_card)
    draw = draw_chance(hole_card, community)
    texture = board_texture(community)
    history = history_line(round_state.get("action_histories", {}), street)

    values = []
    values.extend(_one_hot(street_num, 4))
    for card in _pad_cards(hole_card, 2):
        values.extend(_encode_card(card))
    for card in _pad_cards(community, 5):
        values.extend(_encode_board_card(card))

    values.extend(
        [
            _scale_log(pot, config.big_blind, 8.0),
            _scale_log(to_call, config.big_blind, 6.0),
            _scale_log(to_raise, config.big_blind, 6.0),
            _clip01(hero_stack / config.initial_stack),
            _clip01(opp_stack / config.initial_stack),
            _clip01(effective_stack / config.initial_stack),
            _clip01(pot / max(pot + effective_stack, 1.0)),
            _clip01(pot / max(effective_stack, 1.0)),
            _clip01(to_call / max(pot + to_call, 1.0)),
            float(is_sb),
            float(not is_sb),
            float(legal_mask[2]),
            street_num / 3.0,
            equity,
            preflop,
            draw,
        ]
    )
    values.extend(_one_hot({"dry": 0, "semi": 1, "wet": 2}[texture], 3))
    values.extend(_one_hot({"passive": 0, "single_raise": 1, "reraise": 2}[history], 3))
    values.extend(
        [
            float(opponent_stats.get("preflop_raise_frequency", 0.32)),
            float(opponent_stats.get("aggression", 0.42)),
            float(opponent_stats.get("fold_to_raise", 0.40)),
            float(opponent_stats.get("showdown_willingness", 0.35)),
            _clip01(float(opponent_stats.get("hands_total", 0)) / 100.0),
        ]
    )

    obs = np.asarray(values, dtype=np.float32)
    if obs.shape != (OBSERVATION_SIZE,):
        raise ValueError("Observation shape mismatch: expected %d, got %d" % (OBSERVATION_SIZE, obs.shape[0]))
    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)


def estimate_equity(hole_card, community, equity_sims):
    if equity_sims <= 0 or estimate_hole_card_win_rate is None or gen_cards is None:
        return preflop_strength(hole_card)
    try:
        win_rate = estimate_hole_card_win_rate(equity_sims, 2, gen_cards(hole_card), gen_cards(community))
    except Exception:
        win_rate = preflop_strength(hole_card)
    if not community:
        return 0.70 * float(win_rate) + 0.30 * preflop_strength(hole_card)
    return float(win_rate)


def get_bet_amounts(round_state, hero_uuid, small_blind):
    street = round_state.get("street", "preflop")
    history = round_state.get("action_histories", {}).get(street, [])
    hero_bet = 0.0
    max_bet = 0.0
    for item in history:
        if item is None:
            continue
        action = item.get("action", "").upper()
        if action in ("RAISE", "CALL", "SMALLBLIND", "BIGBLIND"):
            amount = float(item.get("amount", 0.0))
            if item.get("uuid") == hero_uuid:
                hero_bet = max(hero_bet, amount)
            else:
                max_bet = max(max_bet, amount)
    to_call = max(0.0, max_bet - hero_bet)
    step = small_blind * 2 if STREET_TO_ID.get(street, 0) <= 1 else small_blind * 4
    return to_call, to_call + step


def preflop_strength(hole_card):
    if len(hole_card) < 2:
        return 0.0
    s1, r1_c = hole_card[0][0], hole_card[0][1]
    s2, r2_c = hole_card[1][0], hole_card[1][1]
    r1 = RANKS.get(r1_c, 2)
    r2 = RANKS.get(r2_c, 2)
    high, low = max(r1, r2), min(r1, r2)
    gap = high - low
    suited = s1 == s2
    pair = r1 == r2
    if pair:
        return min(0.95, 0.51 + high / 30.0)
    score = 0.30 + high / 40.0 + low / 90.0
    if suited:
        score += 0.035
    if gap == 1:
        score += 0.035
    elif gap == 2:
        score += 0.018
    elif gap >= 5:
        score -= 0.035
    if high == 14 and low >= 10:
        score += 0.075
    elif high >= 13 and low >= 10:
        score += 0.045
    return _clip01(score)


def draw_chance(hole_card, community):
    if not community:
        return 0.0
    all_cards = hole_card + community
    remaining = 52 - len(all_cards)
    if remaining <= 0:
        return 0.0

    outs = 0
    suit_counts = {}
    for card in all_cards:
        suit_counts[card[0]] = suit_counts.get(card[0], 0) + 1
    if any(count == 4 for count in suit_counts.values()):
        outs = max(outs, 9)

    ranks = sorted(set(RANKS.get(card[1], 2) for card in all_cards))
    for rank in ranks:
        window = [other for other in ranks if rank <= other <= rank + 4]
        if len(window) == 4:
            outs = max(outs, 8)
        elif len(window) == 3:
            outs = max(outs, 4)
    return _clip01(float(outs) / float(remaining))


def board_texture(community):
    if len(community) < 3:
        return "dry"
    suit_counts = {}
    rank_counts = {}
    for card in community:
        suit_counts[card[0]] = suit_counts.get(card[0], 0) + 1
        rank = RANKS.get(card[1], 2)
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    flush_draw = any(count >= 3 for count in suit_counts.values())
    ranks = sorted(rank_counts)
    straight_draw = any(len([other for other in ranks if rank <= other <= rank + 4]) >= 3 for rank in ranks)
    paired = any(count >= 2 for count in rank_counts.values())
    draws = int(flush_draw) + int(straight_draw)
    if draws >= 2 or (draws >= 1 and paired):
        return "wet"
    if draws == 1 or paired:
        return "semi"
    return "dry"


def history_line(action_histories, street):
    current = action_histories.get(street, [])
    raise_count = sum(
        1
        for item in current
        if item is not None and item.get("action", "").upper() == "RAISE"
    )
    if raise_count == 0:
        return "passive"
    if raise_count == 1:
        return "single_raise"
    return "reraise"


def _policy_arrays():
    global _POLICY_ARRAYS
    if _POLICY_ARRAYS is None:
        raw = base64.b64decode(_MODEL_NPZ_B64.encode("ascii"))
        with np.load(io.BytesIO(raw)) as data:
            _POLICY_ARRAYS = dict((key, data[key].astype(np.float32)) for key in data.files)
    return _POLICY_ARRAYS


def _predict_policy(obs, mask, deterministic=True):
    arrays = _policy_arrays()
    x = np.asarray(obs, dtype=np.float32)
    for op in _MODEL_META["policy_ops"]:
        if op[0] == "linear":
            name = "pi_%d" % int(op[1])
            x = np.dot(x, arrays[name + "_w"].T) + arrays[name + "_b"]
        elif op[0] == "activation":
            x = _activation(x, op[1])
    logits = np.dot(x, arrays["action_w"].T) + arrays["action_b"]
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != logits.shape or not mask.any():
        return ACTION_TO_ID["call"] if mask.shape == logits.shape and mask[ACTION_TO_ID["call"]] else ACTION_TO_ID["fold"]
    masked_logits = np.where(mask, logits, -1.0e30)
    if deterministic:
        return int(np.argmax(masked_logits))
    probs = _softmax(masked_logits)
    return int(np.random.choice(len(probs), p=probs))


def _activation(x, name):
    name = str(name).lower()
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "elu":
        return np.where(x > 0.0, x, np.exp(x) - 1.0)
    raise ValueError("Unsupported activation: %s" % name)


def _softmax(logits):
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(exp) / float(len(exp))
    return exp / total


def _fallback_action(valid_actions):
    legal = set(entry.get("action") for entry in valid_actions)
    if "call" in legal:
        return "call"
    if "fold" in legal:
        return "fold"
    if "raise" in legal:
        return "raise"
    return "fold"


def _hero_is_small_blind(round_state, hero_uuid):
    seats = round_state.get("seats", [])
    hero_pos = next((idx for idx, seat in enumerate(seats) if seat.get("uuid") == hero_uuid), 0)
    return hero_pos == round_state.get("small_blind_pos", 0)


def _pad_cards(cards, size):
    return list(cards[:size]) + [None] * max(0, size - len(cards))


def _encode_card(card):
    if not card:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [_rank_value(card)] + _one_hot(SUITS.get(card[0], 0), 4)


def _encode_board_card(card):
    if not card:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [1.0, _rank_value(card)] + _one_hot(SUITS.get(card[0], 0), 4)


def _rank_value(card):
    return (RANKS.get(card[1], 2) - 2) / 12.0


def _one_hot(index, size):
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _scale_log(value, unit, cap):
    if unit <= 0:
        return 0.0
    return _clip01(math.log1p(max(0.0, value) / unit) / cap)


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def _smooth(successes, opportunities, prior, strength=4):
    return float(successes + prior * strength) / float(opportunities + strength)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to final_model.zip or best_eval_model.zip")
    parser.add_argument("--output", default="cs683_group13/submission/custom_player.py", help="Generated single-file player path")
    parser.add_argument("--initial-stack", type=int, default=LEADERBOARD_INITIAL_STACK)
    parser.add_argument("--small-blind", type=int, default=LEADERBOARD_SMALL_BLIND)
    parser.add_argument("--equity-sims", type=int, default=LEADERBOARD_EQUITY_SIMS)
    parser.add_argument("--zip-output", default="", help="Optional ZIP to create for the CS683 submit form")
    parser.add_argument("--zip-member", default="submission/custom_player.py", help="Filename to store inside the ZIP")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it already exists")
    parser.add_argument("--skip-verify", action="store_true", help="Skip comparison against MaskablePPO.predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit("%s already exists; pass --force to overwrite it" % output)

    arrays, meta, model = extract_policy(
        args.model,
        initial_stack=args.initial_stack,
        small_blind=args.small_blind,
        equity_sims=args.equity_sims,
    )
    if not args.skip_verify:
        verify_export(model, arrays, meta)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_player(arrays, meta), encoding="utf-8")
    print("Wrote %s" % output)
    if args.zip_output:
        zip_output = Path(args.zip_output)
        if zip_output.exists() and not args.force:
            raise SystemExit("%s already exists; pass --force to overwrite it" % zip_output)
        if zip_output.parent != Path("."):
            zip_output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output, arcname=args.zip_member)
        print("Wrote %s containing %s" % (zip_output, args.zip_member))
    print("Submit the ZIP to CS683, or submit the generated file as custom_player.py if a single file is requested.")


def extract_policy(model_path: str, initial_stack: int, small_blind: int, equity_sims: int):
    path = Path(model_path)
    if path.is_dir() or path.suffix == ".pth":
        return extract_policy_without_sb3(
            path,
            initial_stack=initial_stack,
            small_blind=small_blind,
            equity_sims=equity_sims,
        )

    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        return extract_policy_without_sb3(
            path,
            initial_stack=initial_stack,
            small_blind=small_blind,
            equity_sims=equity_sims,
        )

    model = MaskablePPO.load(model_path, device="cpu")
    policy_net = list(model.policy.mlp_extractor.policy_net)
    arrays: dict[str, np.ndarray] = {}
    policy_ops: list[list[object]] = []
    linear_count = 0

    for module in policy_net:
        class_name = module.__class__.__name__.lower()
        if hasattr(module, "weight") and hasattr(module, "bias"):
            arrays["pi_%d_w" % linear_count] = module.weight.detach().cpu().numpy().astype(np.float32)
            arrays["pi_%d_b" % linear_count] = module.bias.detach().cpu().numpy().astype(np.float32)
            policy_ops.append(["linear", linear_count])
            linear_count += 1
        elif class_name in ("tanh", "relu", "elu"):
            policy_ops.append(["activation", class_name])
        elif class_name in ("identity", "flatten"):
            continue
        else:
            raise ValueError("Unsupported policy module in actor network: %s" % module.__class__.__name__)

    action_net = model.policy.action_net
    arrays["action_w"] = action_net.weight.detach().cpu().numpy().astype(np.float32)
    arrays["action_b"] = action_net.bias.detach().cpu().numpy().astype(np.float32)

    obs_shape = getattr(model.observation_space, "shape", None)
    observation_size = int(obs_shape[0]) if obs_shape else int(arrays["pi_0_w"].shape[1])
    action_size = int(arrays["action_b"].shape[0])
    if observation_size != 71:
        raise ValueError("Expected observation size 71, got %d" % observation_size)
    if action_size != 3:
        raise ValueError("Expected action size 3, got %d" % action_size)

    meta = {
        "policy_ops": policy_ops,
        "observation_size": observation_size,
        "action_size": action_size,
        "initial_stack": int(initial_stack),
        "small_blind": int(small_blind),
        "equity_sims": int(equity_sims),
    }
    return arrays, meta, model


def extract_policy_without_sb3(model_path: Path, initial_stack: int, small_blind: int, equity_sims: int):
    """Extract actor tensors from an SB3 checkpoint archive using only stdlib+NumPy."""
    policy_bytes, data = read_checkpoint_parts(model_path)
    state = load_torch_state_dict(policy_bytes)
    linear_layers = sorted(
        (
            int(match.group(1)),
            key,
        )
        for key in state
        for match in [re.match(r"mlp_extractor\.policy_net\.(\d+)\.weight$", key)]
        if match
    )
    if not linear_layers:
        raise ValueError("Could not find policy_net linear layers in %s" % model_path)

    arrays: dict[str, np.ndarray] = {}
    policy_ops: list[list[object]] = []
    activation = infer_activation(data)
    for export_idx, (layer_idx, weight_key) in enumerate(linear_layers):
        bias_key = "mlp_extractor.policy_net.%d.bias" % layer_idx
        arrays["pi_%d_w" % export_idx] = np.asarray(state[weight_key], dtype=np.float32)
        arrays["pi_%d_b" % export_idx] = np.asarray(state[bias_key], dtype=np.float32)
        policy_ops.append(["linear", export_idx])
        policy_ops.append(["activation", activation])

    arrays["action_w"] = np.asarray(state["action_net.weight"], dtype=np.float32)
    arrays["action_b"] = np.asarray(state["action_net.bias"], dtype=np.float32)

    observation_size = int(arrays["pi_0_w"].shape[1])
    action_size = int(arrays["action_b"].shape[0])
    if observation_size != 71:
        raise ValueError("Expected observation size 71, got %d" % observation_size)
    if action_size != 3:
        raise ValueError("Expected action size 3, got %d" % action_size)

    meta = {
        "policy_ops": policy_ops,
        "observation_size": observation_size,
        "action_size": action_size,
        "initial_stack": int(initial_stack),
        "small_blind": int(small_blind),
        "equity_sims": int(equity_sims),
    }
    return arrays, meta, None


def read_checkpoint_parts(model_path: Path) -> tuple[bytes, dict]:
    data: dict = {}
    if model_path.is_dir():
        data_path = model_path / "data"
        if data_path.exists():
            data = json.loads(data_path.read_text(encoding="utf-8"))
        return (model_path / "policy.pth").read_bytes(), data

    if model_path.suffix == ".pth":
        return model_path.read_bytes(), data

    with zipfile.ZipFile(model_path, "r") as archive:
        names = set(archive.namelist())
        if "data" in names:
            data = json.loads(archive.read("data").decode("utf-8"))
        if "policy.pth" in names:
            return archive.read("policy.pth"), data
    raise ValueError("Could not find policy.pth in %s" % model_path)


class _StorageType:
    pass


class _TorchStateUnpickler(pickle.Unpickler):
    def __init__(self, file, archive: zipfile.ZipFile):
        super().__init__(file)
        self.archive = archive

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self._rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return _StorageType
        return super().find_class(module, name)

    def persistent_load(self, pid):
        kind, _storage_type, key, _location, size = pid
        if kind != "storage":
            raise pickle.UnpicklingError("Unsupported persistent id: %r" % (pid,))
        raw = self.archive.read("archive/data/%s" % key)
        values = np.frombuffer(raw, dtype="<f4")
        if values.size != size:
            raise ValueError("Storage %s expected %d floats, got %d" % (key, size, values.size))
        return values

    @staticmethod
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        shape = tuple(size)
        byte_strides = tuple(int(item) * storage.itemsize for item in stride)
        base = storage[int(storage_offset):]
        return np.lib.stride_tricks.as_strided(base, shape=shape, strides=byte_strides).copy()


def load_torch_state_dict(policy_bytes: bytes) -> OrderedDict:
    with zipfile.ZipFile(io.BytesIO(policy_bytes), "r") as archive:
        payload = archive.read("archive/data.pkl")
        state = _TorchStateUnpickler(io.BytesIO(payload), archive).load()
    if not isinstance(state, OrderedDict):
        raise ValueError("Expected policy.pth to contain an OrderedDict state dict")
    return state


def infer_activation(data: dict) -> str:
    policy_kwargs = data.get("policy_kwargs", {}) if isinstance(data, dict) else {}
    activation_fn = str(policy_kwargs.get("activation_fn", "")).lower()
    if "relu" in activation_fn:
        return "relu"
    if "elu" in activation_fn:
        return "elu"
    return "tanh"


def verify_export(model, arrays: dict[str, np.ndarray], meta: dict) -> None:
    if model is None:
        verify_export_numpy_only(arrays, meta)
        return

    rng = np.random.default_rng(20240506)
    masks = [
        np.array([True, True, False], dtype=bool),
        np.array([True, True, True], dtype=bool),
        np.array([False, True, True], dtype=bool),
        np.array([True, False, True], dtype=bool),
    ]
    for _ in range(32):
        obs = rng.random(meta["observation_size"], dtype=np.float32)
        mask = masks[int(rng.integers(0, len(masks)))]
        sb3_action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        np_action = predict_numpy(obs, mask, arrays, meta)
        if int(sb3_action) != int(np_action):
            raise ValueError(
                "Export verification failed: SB3 chose %d but exported network chose %d"
                % (int(sb3_action), int(np_action))
            )


def verify_export_numpy_only(arrays: dict[str, np.ndarray], meta: dict) -> None:
    rng = np.random.default_rng(20240506)
    masks = [
        np.array([True, True, False], dtype=bool),
        np.array([True, True, True], dtype=bool),
    ]
    for _ in range(8):
        obs = rng.random(meta["observation_size"], dtype=np.float32)
        action = predict_numpy(obs, masks[int(rng.integers(0, len(masks)))], arrays, meta)
        if action not in (0, 1, 2):
            raise ValueError("Export verification failed: invalid action %r" % (action,))


def predict_numpy(obs: np.ndarray, mask: np.ndarray, arrays: dict[str, np.ndarray], meta: dict) -> int:
    x = np.asarray(obs, dtype=np.float32)
    for op in meta["policy_ops"]:
        if op[0] == "linear":
            name = "pi_%d" % int(op[1])
            x = np.dot(x, arrays[name + "_w"].T) + arrays[name + "_b"]
        elif op[0] == "activation":
            x = activate(x, str(op[1]))
    logits = np.dot(x, arrays["action_w"].T) + arrays["action_b"]
    masked_logits = np.where(mask, logits, -1.0e30)
    return int(np.argmax(masked_logits))


def activate(x: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "elu":
        return np.where(x > 0.0, x, np.exp(x) - 1.0)
    raise ValueError("Unsupported activation: %s" % name)


def render_player(arrays: dict[str, np.ndarray], meta: dict) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    model_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    wrapped_b64 = "\n".join(textwrap.wrap(model_b64, width=88))
    return (
        LEADERBOARD_TEMPLATE.replace("@@MODEL_META@@", json.dumps(meta, sort_keys=True))
        .replace("@@MODEL_NPZ_B64@@", wrapped_b64)
    )


if __name__ == "__main__":
    main()
