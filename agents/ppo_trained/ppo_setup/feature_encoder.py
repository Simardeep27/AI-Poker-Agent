"""Fixed-vector observations and action masks for PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ppo_setup.pathing import ensure_project_paths

ensure_project_paths()

from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards  # noqa: E402


ACTION_TO_ID = {"fold": 0, "call": 1, "raise": 2}
ID_TO_ACTION = {value: key for key, value in ACTION_TO_ID.items()}
STREET_TO_ID = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 3}
RANKS = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
SUITS = {"C": 0, "D": 1, "H": 2, "S": 3}
OBSERVATION_SIZE = 71


@dataclass(frozen=True)
class EncoderConfig:
    initial_stack: int = 10000
    small_blind: int = 20
    equity_sims: int = 32

    @property
    def big_blind(self) -> int:
        return self.small_blind * 2


def action_mask(valid_actions: list[dict]) -> np.ndarray:
    legal = {entry.get("action") for entry in valid_actions}
    return np.array(
        [
            "fold" in legal,
            "call" in legal,
            "raise" in legal,
        ],
        dtype=bool,
    )


def encode_observation(
    valid_actions: list[dict],
    hole_card: list[str],
    round_state: dict,
    hero_uuid: str,
    opponent_stats: dict | None = None,
    config: EncoderConfig | None = None,
) -> np.ndarray:
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

    values: list[float] = []
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


def estimate_equity(hole_card: list[str], community: list[str], equity_sims: int) -> float:
    if equity_sims <= 0:
        return preflop_strength(hole_card)
    try:
        win_rate = estimate_hole_card_win_rate(equity_sims, 2, gen_cards(hole_card), gen_cards(community))
    except Exception:
        win_rate = preflop_strength(hole_card)
    if not community:
        return 0.70 * float(win_rate) + 0.30 * preflop_strength(hole_card)
    return float(win_rate)


def get_bet_amounts(round_state: dict, hero_uuid: str, small_blind: int) -> tuple[float, float]:
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


def preflop_strength(hole_card: list[str]) -> float:
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


def draw_chance(hole_card: list[str], community: list[str]) -> float:
    if not community:
        return 0.0
    all_cards = hole_card + community
    remaining = 52 - len(all_cards)
    if remaining <= 0:
        return 0.0

    outs = 0
    suit_counts: dict[str, int] = {}
    for card in all_cards:
        suit_counts[card[0]] = suit_counts.get(card[0], 0) + 1
    if any(count == 4 for count in suit_counts.values()):
        outs = max(outs, 9)

    ranks = sorted({RANKS.get(card[1], 2) for card in all_cards})
    for rank in ranks:
        window = [other for other in ranks if rank <= other <= rank + 4]
        if len(window) == 4:
            outs = max(outs, 8)
        elif len(window) == 3:
            outs = max(outs, 4)
    return _clip01(float(outs) / float(remaining))


def board_texture(community: list[str]) -> str:
    if len(community) < 3:
        return "dry"
    suit_counts: dict[str, int] = {}
    rank_counts: dict[int, int] = {}
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


def history_line(action_histories: dict, street: str) -> str:
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


def _hero_is_small_blind(round_state: dict, hero_uuid: str) -> bool:
    seats = round_state.get("seats", [])
    hero_pos = next((idx for idx, seat in enumerate(seats) if seat.get("uuid") == hero_uuid), 0)
    return hero_pos == round_state.get("small_blind_pos", 0)


def _pad_cards(cards: list[str], size: int) -> list[str | None]:
    return list(cards[:size]) + [None] * max(0, size - len(cards))


def _encode_card(card: str | None) -> list[float]:
    if not card:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [_rank_value(card)] + _one_hot(SUITS.get(card[0], 0), 4)


def _encode_board_card(card: str | None) -> list[float]:
    if not card:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [1.0, _rank_value(card)] + _one_hot(SUITS.get(card[0], 0), 4)


def _rank_value(card: str) -> float:
    return (RANKS.get(card[1], 2) - 2) / 12.0


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _scale_log(value: float, unit: float, cap: float) -> float:
    if unit <= 0:
        return 0.0
    return _clip01(math.log1p(max(0.0, value) / unit) / cap)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
