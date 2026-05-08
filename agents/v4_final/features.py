import random

from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

from cards import available_deck, preflop_strength
from constants import HAND_MADE_SCORE, NUM_SIMS, RANGE_SIMS


def analyze_state(state):
    hole_card = state["hole_card"]
    community = state["community"]
    pot = state["pot"]
    my_chips = state["my_chips"]
    street = state["street"]
    street_num = state["street_num"]

    hole_objs = gen_cards(hole_card)
    board_objs = gen_cards(community)
    win_rate = estimate_hole_card_win_rate(
        NUM_SIMS.get(street, 100), 2, hole_objs, board_objs
    )

    if street == "preflop":
        win_rate = 0.70 * win_rate + 0.30 * preflop_strength(hole_card)

    raw_win_rate = win_rate
    pressure = state["to_call"] / float(max(state["pot"] + state["to_call"], 1))
    range_rate = None
    hard_spot = street != "preflop" and (
        state["history_line"] != "passive" or pressure >= 0.16 or street == "river"
    )
    if hard_spot:
        range_rate = range_aware_win_rate(state, RANGE_SIMS.get(street, 0))
        if range_rate is not None:
            line_weight = {
                "passive": 0.14,
                "single_raise": 0.30,
                "reraise": 0.46,
            }.get(state["history_line"], 0.30)
            if street == "river":
                line_weight += 0.12
            if pressure >= 0.28:
                line_weight += 0.08
            line_weight = min(0.62, line_weight)
            win_rate = (1.0 - line_weight) * win_rate + line_weight * range_rate

    if win_rate >= 0.80:
        strength = 4
    elif win_rate >= 0.65:
        strength = 3
    elif win_rate >= 0.50:
        strength = 2
    elif win_rate >= 0.35:
        strength = 1
    else:
        strength = 0

    if pot < 100:
        pot_size = "small"
    elif pot < 400:
        pot_size = "medium"
    elif pot < 1000:
        pot_size = "large"
    else:
        pot_size = "huge"

    return {
        "win_rate": win_rate,
        "raw_win_rate": raw_win_rate,
        "range_win_rate": range_rate,
        "strength": strength,
        "made_score": made_hand_score(hole_objs, board_objs),
        "draw_chance": draw_outs(hole_objs, board_objs),
        "board_type": board_texture(board_objs),
        "pot_size": pot_size,
        "invested": min(pot / max(my_chips, 1), 1.0),
        "street_progress": street_num / 3.0,
        "history_line": state["history_line"],
    }


def range_aware_win_rate(state, n_sims):
    hole_card = state["hole_card"]
    community = state["community"]
    if len(hole_card) != 2 or n_sims <= 0:
        return None

    deck = available_deck(set(hole_card) | set(community))
    if len(deck) < 2:
        return None

    threshold = opponent_range_threshold(state)
    hero_objs = gen_cards(hole_card)
    wins = 0.0
    trials = 0
    cards_needed = max(0, 5 - len(community))

    for _ in range(n_sims):
        opp_cards = sample_range_hand(deck, threshold)
        if not opp_cards:
            continue
        runout_pool = [card for card in deck if card not in set(opp_cards)]
        if len(runout_pool) < cards_needed:
            continue
        board = list(community)
        if cards_needed:
            board += random.sample(runout_pool, cards_needed)
        opp_objs = gen_cards(opp_cards)
        board_objs = gen_cards(board)
        hero_score = HandEvaluator.eval_hand(hero_objs, board_objs)
        opp_score = HandEvaluator.eval_hand(opp_objs, board_objs)
        if hero_score > opp_score:
            wins += 1.0
        elif hero_score == opp_score:
            wins += 0.5
        trials += 1

    if trials < max(6, n_sims // 4):
        return None
    return wins / float(trials)


def opponent_range_threshold(state):
    line = state["history_line"]
    street = state["street"]
    to_call = state["to_call"]
    pot = max(state["pot"], 1)
    pressure = min(to_call / float(pot + to_call), 1.0)

    if line == "reraise":
        threshold = 0.56
    elif line == "single_raise":
        threshold = 0.42
    else:
        threshold = 0.20

    if street in ("turn", "river"):
        threshold += 0.035
    threshold += 0.10 * pressure
    return max(0.12, min(0.70, threshold))


def sample_range_hand(deck, threshold):
    best = None
    best_score = -1.0
    for _ in range(10):
        cards = random.sample(deck, 2)
        score = preflop_strength(cards)
        if score >= threshold:
            return cards
        if score > best_score:
            best = cards
            best_score = score
    return best


def made_hand_score(hole_objs, board_objs):
    if len(board_objs) < 3:
        return 0.0
    try:
        info = HandEvaluator.gen_hand_rank_info(hole_objs, board_objs)
        return HAND_MADE_SCORE.get(info["hand"]["strength"], 0.20)
    except Exception:
        return 0.0


def draw_outs(hole_objs, board_objs):
    if not board_objs:
        return 0.0
    all_cards = hole_objs + board_objs
    remaining = 52 - len(all_cards)
    if remaining <= 0:
        return 0.0

    outs = 0
    suit_counts = {}
    for card in all_cards:
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
    if any(count == 4 for count in suit_counts.values()):
        outs = max(outs, 9)

    ranks = sorted(set(card.rank for card in all_cards))
    for rank in ranks:
        window = [rr for rr in ranks if rank <= rr <= rank + 4]
        if len(window) == 4:
            outs = max(outs, 8)
        elif len(window) == 3:
            outs = max(outs, 4)

    return min(outs / remaining, 1.0)


def board_texture(board_objs):
    if len(board_objs) < 3:
        return "dry"

    suit_counts = {}
    for card in board_objs:
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
    flush_draw = any(count >= 3 for count in suit_counts.values())

    ranks = sorted(set(card.rank for card in board_objs))
    straight_draw = False
    for rank in ranks:
        window = [rr for rr in ranks if rank <= rr <= rank + 4]
        if len(window) >= 3:
            straight_draw = True
            break

    rank_counts = {}
    for card in board_objs:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
    paired = any(count >= 2 for count in rank_counts.values())

    draws = int(flush_draw) + int(straight_draw)
    if draws >= 2 or (draws >= 1 and paired):
        return "wet"
    if draws == 1 or paired:
        return "semi"
    return "dry"
