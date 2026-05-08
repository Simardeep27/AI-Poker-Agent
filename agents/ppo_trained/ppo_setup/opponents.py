"""Opponent factories used by PPO training and evaluation."""

from __future__ import annotations

import importlib
import importlib.util
import random
from pathlib import Path
from types import ModuleType

from ppo_setup.feature_encoder import (
    ACTION_TO_ID,
    EncoderConfig,
    action_mask,
    encode_observation,
    estimate_equity,
    get_bet_amounts,
    preflop_strength,
)
from ppo_setup.leaderboard import LEADERBOARD_EQUITY_SIMS, LEADERBOARD_INITIAL_STACK, LEADERBOARD_SMALL_BLIND
from ppo_setup.opponent_stats import OpponentStats
from ppo_setup.pathing import ensure_project_paths, find_project_file

ensure_project_paths()

from pypokerengine.players import BasePokerPlayer  # noqa: E402


PLAYER_SPECS = {
    "custom": ("custom_player", "MyPlayer", ("custom_player.py", "AI-Poker-Agent/custom_player.py")),
    "opponent": ("opponent_aware_player", "OpponentAwarePlayer", ("AI-Poker-Agent/opponent_aware_player.py",)),
    "submitted": ("submission_custom_player", "MyPlayer", ("cs683_group13/submission/custom_player.py",)),
    "random": ("randomplayer", "RandomPlayer", ("randomplayer.py", "AI-Poker-Agent/randomplayer.py")),
    "raise": ("raise_player", "RaisedPlayer", ("raise_player.py", "AI-Poker-Agent/raise_player.py")),
}

DIVERSE_POOL = ("custom", "opponent", "submitted", "random", "raise", "call", "fold", "tight", "loose", "maniac")
SNAPSHOT_SUFFIXES = (".zip",)
OPPONENT_FACTORIES = {}


def make_opponent(name: str):
    key = name.strip()
    normalized = key.lower()
    if normalized.startswith(("ppo:", "ckpt:")):
        return _checkpoint_factory(key.split(":", 1)[1])()
    if normalized in BUILTIN_FACTORIES:
        return BUILTIN_FACTORIES[normalized]()
    if Path(key).exists():
        return _checkpoint_factory(key)()
    if normalized not in PLAYER_SPECS:
        choices = sorted(set(PLAYER_SPECS) | set(BUILTIN_FACTORIES) | {"diverse", "ppo:<path>", "snapshots:<dir>"})
        raise ValueError("Unknown opponent %r. Choices include: %s" % (name, ", ".join(choices)))
    if normalized not in OPPONENT_FACTORIES:
        OPPONENT_FACTORIES[normalized] = _load_player_class(normalized)
    return OPPONENT_FACTORIES[normalized]()


def sample_opponent(opponent_pool: list[str] | tuple[str, ...] | str):
    names = resolve_opponent_pool(opponent_pool)
    if not names:
        names = ["custom"]
    return make_opponent(random.choice(names))


def resolve_opponent_pool(opponent_pool: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(opponent_pool, str):
        names = [name.strip() for name in opponent_pool.split(",") if name.strip()]
    else:
        names = [name.strip() for name in opponent_pool if name.strip()]

    expanded: list[str] = []
    for name in names:
        key = name.lower()
        if key == "diverse":
            expanded.extend(DIVERSE_POOL)
        elif key.startswith("snapshots:"):
            expanded.extend(_snapshot_opponents(name.split(":", 1)[1]))
        else:
            expanded.append(name)
    return expanded


def _load_player_class(key: str):
    module_name, class_name, relative_paths = PLAYER_SPECS[key]
    module = _import_module(module_name, key, relative_paths)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            "Found %s but it does not define %s. Check that your copied project files match this repo."
            % (module_name, class_name)
        ) from exc


def _import_module(module_name: str, key: str, relative_paths: tuple[str, ...]) -> ModuleType:
    ensure_project_paths()
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as original_exc:
        path = find_project_file(*relative_paths)
        if path is None:
            raise ModuleNotFoundError(_missing_player_message(module_name, relative_paths)) from original_exc
        return _load_module_from_path(module_name, key, path)


def _load_module_from_path(module_name: str, key: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("ppo_setup_loaded_%s_%s" % (key, module_name), path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load %s from %s" % (module_name, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _missing_player_message(module_name: str, relative_paths: tuple[str, ...]) -> str:
    expected = ", ".join(relative_paths)
    return (
        "Could not import %s. Expected to find one of: %s. "
        "In Colab, run from the full project root and make sure AI-Poker-Agent/ plus "
        "custom_player.py were copied beside ppo_setup/."
    ) % (module_name, expected)


class AlwaysCallPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        return _legal_or_fallback("call", valid_actions)

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class AlwaysFoldPlayer(AlwaysCallPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        return _legal_or_fallback("fold", valid_actions)


class TightPlayer(AlwaysCallPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        score = _hand_score(hole_card, round_state, equity_sims=8)
        to_call, _to_raise = get_bet_amounts(round_state, self.uuid, round_state.get("small_blind_amount", 10))
        if score >= 0.72:
            return _legal_or_fallback("raise", valid_actions)
        if score >= 0.55 or to_call <= round_state.get("small_blind_amount", 10) * 2:
            return _legal_or_fallback("call", valid_actions)
        return _legal_or_fallback("fold", valid_actions)


class LoosePlayer(AlwaysCallPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        score = _hand_score(hole_card, round_state, equity_sims=6)
        if score >= 0.66 or (score >= 0.48 and random.random() < 0.18):
            return _legal_or_fallback("raise", valid_actions)
        if score >= 0.28 or random.random() < 0.70:
            return _legal_or_fallback("call", valid_actions)
        return _legal_or_fallback("fold", valid_actions)


class ManiacPlayer(AlwaysCallPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        if random.random() < 0.82:
            return _legal_or_fallback("raise", valid_actions)
        return _legal_or_fallback("call", valid_actions)


class CheckpointPolicyPlayer(BasePokerPlayer):
    def __init__(self, model_path: str):
        from ppo_setup.export_leaderboard_player import extract_policy_without_sb3

        arrays, meta, _model = extract_policy_without_sb3(
            Path(model_path),
            initial_stack=LEADERBOARD_INITIAL_STACK,
            small_blind=LEADERBOARD_SMALL_BLIND,
            equity_sims=LEADERBOARD_EQUITY_SIMS,
        )
        self.arrays = arrays
        self.meta = meta
        self.config = EncoderConfig(
            initial_stack=meta.get("initial_stack", LEADERBOARD_INITIAL_STACK),
            small_blind=meta.get("small_blind", LEADERBOARD_SMALL_BLIND),
            equity_sims=meta.get("equity_sims", LEADERBOARD_EQUITY_SIMS),
        )
        self.stats = OpponentStats()

    def declare_action(self, valid_actions, hole_card, round_state):
        from ppo_setup.export_leaderboard_player import predict_numpy

        obs = encode_observation(
            valid_actions,
            hole_card,
            round_state,
            self.uuid,
            self.stats.snapshot(),
            self.config,
        )
        mask = action_mask(valid_actions)
        action_id = predict_numpy(obs, mask, self.arrays, self.meta)
        for name, idx in ACTION_TO_ID.items():
            if idx == action_id:
                return _legal_or_fallback(name, valid_actions)
        return _legal_or_fallback("call", valid_actions)

    def receive_game_start_message(self, game_info):
        rule = game_info.get("rule", {}) if isinstance(game_info, dict) else {}
        self.config = EncoderConfig(
            initial_stack=int(rule.get("initial_stack", self.config.initial_stack)),
            small_blind=int(rule.get("small_blind_amount", self.config.small_blind)),
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


BUILTIN_FACTORIES = {
    "call": AlwaysCallPlayer,
    "fold": AlwaysFoldPlayer,
    "tight": TightPlayer,
    "loose": LoosePlayer,
    "maniac": ManiacPlayer,
}


def _checkpoint_factory(path_text: str):
    path = _resolve_path(path_text)
    cache_key = "ppo:%s" % path
    if cache_key not in OPPONENT_FACTORIES:
        OPPONENT_FACTORIES[cache_key] = lambda resolved=str(path): CheckpointPolicyPlayer(resolved)
    return OPPONENT_FACTORIES[cache_key]


def _snapshot_opponents(path_text: str) -> list[str]:
    root = _resolve_path(path_text, must_exist=False)
    if not root.exists():
        return []
    candidates: list[Path] = []
    if root.is_file() and root.suffix in SNAPSHOT_SUFFIXES:
        candidates.append(root)
    else:
        candidates.extend(path for path in root.rglob("*") if _looks_like_checkpoint(path))
    return ["ppo:%s" % path for path in sorted(candidates)]


def _looks_like_checkpoint(path: Path) -> bool:
    return (path.is_file() and path.suffix in SNAPSHOT_SUFFIXES) or (path.is_dir() and (path / "policy.pth").exists())


def _resolve_path(path_text: str, must_exist: bool = True) -> Path:
    raw = Path(path_text).expanduser()
    if raw.exists() or raw.is_absolute():
        return raw
    found = find_project_file(path_text)
    if found is not None:
        return found
    if must_exist:
        raise FileNotFoundError("Could not resolve opponent checkpoint path: %s" % path_text)
    return raw


def _hand_score(hole_card, round_state, equity_sims: int) -> float:
    community = round_state.get("community_card", [])
    if not community:
        return preflop_strength(hole_card)
    return estimate_equity(hole_card, community, equity_sims)


def _legal_or_fallback(action: str, valid_actions) -> str:
    legal = {entry.get("action") for entry in valid_actions}
    if action in legal:
        return action
    if "call" in legal:
        return "call"
    if "fold" in legal:
        return "fold"
    return next(iter(legal), "fold")
