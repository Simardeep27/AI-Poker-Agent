"""Round-state parsing and lightweight betting-line abstraction."""

from constants import STREETS


def read_state(valid_actions, hole_card, round_state, my_uuid):
    """Convert PyPokerEngine's nested round state into a compact decision state."""
    street = round_state["street"]
    street_num = STREETS.get(street, 0)
    community = round_state.get("community_card", [])
    pot = round_state["pot"]["main"]["amount"]
    blind = round_state["small_blind_amount"]
    seats = round_state["seats"]

    my_seat = next((s for s in seats if s["uuid"] == my_uuid), None)
    opp_seat = next((s for s in seats if s["uuid"] != my_uuid), None)
    my_chips = my_seat["stack"] if my_seat else 10000
    opp_chips = opp_seat["stack"] if opp_seat else 10000

    sb_pos = round_state.get("small_blind_pos", 0)
    my_pos = next((i for i, s in enumerate(seats) if s["uuid"] == my_uuid), 0)
    is_sb = my_pos == sb_pos

    names = [a["action"] for a in valid_actions]
    raise_ok = "raise" in names

    to_call, to_raise = _get_bet_amounts(round_state, street, blind, my_uuid)
    history_line = _get_history_line(round_state.get("action_histories", {}), street)

    return {
        "street": street,
        "street_num": street_num,
        "hole_card": hole_card,
        "community": community,
        "pot": pot,
        "blind": blind,
        "my_chips": my_chips,
        "opp_chips": opp_chips,
        "is_sb": is_sb,
        "raise_ok": raise_ok,
        "to_call": to_call,
        "to_raise": to_raise,
        "history_line": history_line,
        "my_uuid": my_uuid,
        "spr": pot / max(min(my_chips, opp_chips), 1),
    }


def _get_bet_amounts(round_state, street, blind, my_uuid):
    history = round_state.get("action_histories", {}).get(street, [])
    my_bet = 0
    max_bet = 0
    for h in history:
        if h is None:
            continue
        action = h.get("action", "").upper()
        if action in ("RAISE", "CALL", "SMALLBLIND", "BIGBLIND"):
            amount = h.get("amount", 0)
            if h.get("uuid") == my_uuid:
                my_bet = max(my_bet, amount)
            else:
                max_bet = max(max_bet, amount)
    to_call = max(0, max_bet - my_bet)
    street_num = STREETS.get(street, 0)
    step = blind * 2 if street_num <= 1 else blind * 4
    return to_call, to_call + step


def _get_history_line(action_histories, street):
    current = action_histories.get(street, [])
    raise_count = sum(
        1 for h in current if h is not None and h.get("action", "").upper() == "RAISE"
    )
    if raise_count == 0:
        return "passive"
    if raise_count == 1:
        return "single_raise"
    return "reraise"
