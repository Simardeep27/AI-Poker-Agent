"""Action selection for the final agent.

The selector compares sampled EV estimates for call and raise, then applies a
small set of policy constraints that reduce known failure modes in limit poker.
"""

from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

from cards import fill_board, preflop_strength
from constants import BOARD_SAMPLES, NUM_SIMS, WEIGHTS


def choose_action(state, hand_info, response):
    """Choose a legal fold/call/raise action from state features and beliefs."""
    pot = state["pot"]
    to_call = state["to_call"]
    to_raise = state["to_raise"]
    raise_ok = state["raise_ok"]
    hole_card = state["hole_card"]
    community = state["community"]
    street = state["street"]
    wr = hand_info["win_rate"]
    made_score = hand_info.get("made_score", hand_info["strength"] / 4.0)
    pot_odds = to_call / float(max(pot + to_call, 1))
    cards_needed = max(0, 5 - len(community))

    # Start with raw chip EV. Later blocks add small strategic corrections.
    fold_ev = 0.0
    if cards_needed == 0:
        call_ev = call_ev_value(wr, pot, to_call)
        raise_ev = raise_ev_value(wr, pot, to_call, to_raise, response) if raise_ok else -1e9
    else:
        call_ev, raise_ev = sampled_action_values(
            state, response, cards_needed, raise_ok, hole_card, community, street
        )

    # Preserve drawing value before the river without letting draws dominate EV.
    future_draw = 0.0 if street == "river" else hand_info["draw_chance"]
    draw_bonus = 0.10 * future_draw * pot
    made_rate = 0.035 if street == "river" else 0.07
    made_bonus = made_rate * max(hand_info["strength"] / 4.0, made_score) * pot
    call_ev += draw_bonus + made_bonus
    if raise_ok and raise_ev > -1e8:
        raise_ev += draw_bonus + made_bonus

    if (
        hand_info["board_type"] == "wet"
        and hand_info["draw_chance"] < 0.35
        and hand_info["strength"] < 2
    ):
        call_ev -= 0.08 * pot
        if raise_ok and raise_ev > -1e8:
            raise_ev -= 0.08 * pot

    # Reduce thin bluffs against opponents who call too often.
    sticky = response.get("stickiness", 0.0)
    confidence = response.get("confidence", 0.0)
    if raise_ok and raise_ev > -1e8 and sticky > 0.40 and wr < 0.67:
        raise_ev -= confidence * 0.08 * pot

    maniac = response.get("natural_raise_rate", 0.0) >= 0.52 and confidence >= 0.45
    if maniac:
        if wr >= 0.61 and made_score >= 0.35:
            call_ev += 0.07 * pot
        if wr < pot_odds + 0.10 and made_score < 0.67:
            call_ev -= 0.12 * pot
        if raise_ok and raise_ev > -1e8 and wr < 0.70 and made_score < 0.78:
            raise_ev -= 0.18 * pot

    # River decisions use tighter pot-odds discipline because no future cards remain.
    if street == "river":
        call_ev, raise_ev = apply_river_discipline(
            call_ev, raise_ev, raise_ok, wr, made_score, pot, pot_odds, hand_info, response
        )

    # Re-raised pots require a larger margin before adding chips.
    history_line = hand_info["history_line"]
    call_slack_mult = 1.0
    raise_margin_mult = 1.0
    if history_line == "reraise":
        call_slack_mult = 0.45
        raise_margin_mult = 2.0
        if wr < 0.57 and raise_ok and raise_ev > -1e8:
            raise_ev -= 0.13 * pot
        if wr < 0.42:
            call_ev -= 0.07 * pot
        if street == "river" and wr < 0.62 and raise_ok:
            raise_ev -= 0.18 * pot
    elif history_line == "passive":
        if raise_ok and raise_ev > -1e8:
            raise_ev += 0.04 * pot
        call_slack_mult = 1.15

    # Leaf features break close EV ties but do not replace the EV calculation.
    bonus_weight = 0.08 + 0.14 * hand_info["street_progress"]
    bonus = score_state(hand_info, pot + to_call, response["fold"], response["opp_aggression"], state["is_sb"])
    call_ev += bonus_weight * bonus
    if raise_ok and raise_ev > -1e8:
        raise_ev += bonus_weight * response["call"] * bonus

    if state["spr"] < 2 and wr > 0.58 and raise_ok:
        raise_ev += 0.10 * pot
    elif state["spr"] > 15 and raise_ok:
        raise_ev -= 0.05 * pot

    if street == "preflop":
        call_ev, raise_ev = apply_preflop_adjustments(
            call_ev, raise_ev, raise_ok, hole_card, history_line, to_call, state, pot
        )

    if (
        raise_ok
        and hand_info["draw_chance"] > 0.40
        and hand_info["street_progress"] < 1.0
        and response["fold"] > 0.30
        and sticky < 0.58
        and not maniac
    ):
        step = to_raise - to_call
        semi_bluff_ev = response["fold"] * pot + response["call"] * (
            hand_info["draw_chance"] * (pot + to_raise + step) - to_raise
        )
        if semi_bluff_ev > call_ev:
            raise_ev = max(raise_ev, semi_bluff_ev)

    raise_allowed = raise_allowed_value(state, hand_info, response)
    if raise_ok and raise_allowed and wr >= value_raise_threshold(state, hand_info, response) and raise_ev > fold_ev:
        return "raise"

    raise_margin = (0.01 * pot if wr >= 0.60 else 0.07 * pot) * raise_margin_mult
    if raise_ok and raise_allowed and raise_ev >= call_ev + raise_margin and raise_ev > fold_ev:
        return "raise"

    call_slack = call_slack_value(hand_info, response["opp_aggression"], pot) * call_slack_mult
    if wr < 0.30:
        call_slack *= 0.35
    if call_ev >= -call_slack:
        return "call"
    return "fold"


def sampled_action_values(state, response, cards_needed, raise_ok, hole_card, community, street):
    sims_each = max(10, NUM_SIMS.get(street, 100) // BOARD_SAMPLES)
    call_acc = 0.0
    raise_acc = 0.0
    for _ in range(BOARD_SAMPLES):
        board = fill_board(hole_card, community, cards_needed)
        wr = estimate_hole_card_win_rate(sims_each, 2, gen_cards(hole_card), gen_cards(board))
        call_acc += call_ev_value(wr, state["pot"], state["to_call"])
        if raise_ok:
            raise_acc += raise_ev_value(wr, state["pot"], state["to_call"], state["to_raise"], response)
    call_ev = call_acc / BOARD_SAMPLES
    raise_ev = (raise_acc / BOARD_SAMPLES) if raise_ok else -1e9
    return call_ev, raise_ev


def raise_ev_value(win_rate, pot, to_call, to_raise, response):
    step = to_raise - to_call
    pot_opp_calls = pot + to_raise + step
    pot_opp_reraises = pot + to_raise + step * 2
    return (
        response["fold"] * pot
        + response["call"] * (win_rate * pot_opp_calls - to_raise)
        + response["raise"] * (win_rate * pot_opp_reraises - to_raise)
    )


def apply_river_discipline(call_ev, raise_ev, raise_ok, wr, made_score, pot, pot_odds, hand_info, response):
    river_need = pot_odds + 0.035
    if hand_info["history_line"] == "reraise":
        river_need += 0.08
    if hand_info["board_type"] == "wet":
        river_need += 0.035
    if wr < river_need and made_score < 0.55:
        call_ev -= 0.16 * pot
    if raise_ok and raise_ev > -1e8:
        if wr < 0.68 and made_score < 0.78 and response["fold"] < 0.52:
            raise_ev -= 0.22 * pot
        if wr < 0.54 and made_score < 0.67:
            raise_ev = min(raise_ev, call_ev - 0.05 * pot)
    return call_ev, raise_ev


def apply_preflop_adjustments(call_ev, raise_ev, raise_ok, hole_card, history_line, to_call, state, pot):
    pf_strength = preflop_strength(hole_card)
    if not state["is_sb"]:
        call_ev += 0.05 * pot
    else:
        call_ev -= 0.02 * pot
    if history_line == "reraise" and pf_strength < 0.58:
        call_ev -= 0.10 * pot
        if raise_ok and raise_ev > -1e8:
            raise_ev -= 0.18 * pot
    elif history_line == "single_raise" and pf_strength < 0.42 and to_call > state["blind"] * 2:
        call_ev -= 0.06 * pot
    return call_ev, raise_ev


def call_slack_value(hand_info, opp_aggression, pot):
    if hand_info["street_progress"] >= 1.0:
        return 0.0
    street_tightener = 1.0 - 0.65 * hand_info["street_progress"]
    return (
        0.02 + 0.05 * opp_aggression + 0.04 * hand_info["draw_chance"]
    ) * pot * street_tightener


def score_state(hand_info, pot, fold_prob, opp_aggression, is_sb):
    board_danger_map = {"dry": 0.0, "semi": 0.5, "wet": 1.0}
    position_bonus = 0.0 if is_sb else 0.10
    return (
        WEIGHTS["equity"] * hand_info["win_rate"]
        + WEIGHTS["draw_potential"] * hand_info["draw_chance"]
        + WEIGHTS["fold_equity"] * fold_prob
        + WEIGHTS["pot_pressure"] * hand_info["invested"]
        + WEIGHTS["board_danger"] * board_danger_map.get(hand_info["board_type"], 0.5)
        + WEIGHTS["showdown_value"] * (hand_info["strength"] / 4.0)
        + WEIGHTS["opp_aggression"] * opp_aggression
        + WEIGHTS["street_bonus"] * hand_info["street_progress"]
        + position_bonus
    )


def call_ev_value(win_rate, pot, to_call):
    if to_call <= 0:
        return win_rate * pot
    return win_rate * (pot + to_call) - to_call


def raise_allowed_value(state, hand_info, response):
    if not state["raise_ok"]:
        return False
    wr = hand_info["win_rate"]
    street = state["street"]
    history_line = hand_info["history_line"]
    sticky = response.get("stickiness", 0.0)
    if wr >= value_raise_threshold(state, hand_info, response):
        return True
    if street != "river":
        strong_draw = hand_info["draw_chance"] >= 0.43
        foldable = response["fold"] >= 0.32 and sticky < 0.58
        if strong_draw and foldable and history_line != "reraise":
            return True
    return (
        history_line == "passive"
        and hand_info["board_type"] == "dry"
        and response["fold"] >= 0.50
        and sticky < 0.38
        and wr >= 0.47
    )


def value_raise_threshold(state, hand_info, response):
    threshold = {
        "preflop": 0.63,
        "flop": 0.64,
        "turn": 0.66,
        "river": 0.69,
    }.get(state["street"], 0.66)
    if hand_info["history_line"] == "passive":
        threshold -= 0.02
    elif hand_info["history_line"] == "single_raise":
        threshold += 0.02
    else:
        threshold += 0.075
    if hand_info["board_type"] == "wet":
        threshold += 0.025
    elif hand_info["board_type"] == "dry":
        threshold -= 0.01
    if response["fold"] > 0.48 and response.get("stickiness", 0.0) < 0.42:
        threshold -= 0.045
    if state["street"] != "river" and hand_info["draw_chance"] > 0.44 and response["fold"] > 0.34:
        threshold -= 0.035
    if hand_info.get("made_score", 0.0) >= 0.78:
        threshold -= 0.05
    return max(0.54, min(0.78, threshold))
