"""Card-level helper functions used by the final agent."""

import random


def available_deck(known):
    suits = ["C", "D", "H", "S"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    return [s + r for s in suits for r in ranks if s + r not in known]


def fill_board(hole_card, community, n_needed):
    known = set(hole_card) | set(community)
    return list(community) + random.sample(available_deck(known), n_needed)


def preflop_strength(hole_card):
    rank_map = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }
    s1, r1_c = hole_card[0][0], hole_card[0][1]
    s2, r2_c = hole_card[1][0], hole_card[1][1]
    r1 = rank_map.get(r1_c, 2)
    r2 = rank_map.get(r2_c, 2)
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
    return max(0.05, min(0.92, score))
