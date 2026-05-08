WEIGHTS = {
    "equity": 3.0,
    "draw_potential": 1.0,
    "fold_equity": 1.5,
    "pot_pressure": 0.6,
    "board_danger": -0.5,
    "showdown_value": 0.7,
    "opp_aggression": 0.4,
    "street_bonus": 0.3,
}

NUM_SIMS = {"preflop": 300, "flop": 180, "turn": 120, "river": 50}
RANGE_SIMS = {"preflop": 0, "flop": 20, "turn": 24, "river": 42}
BOARD_SAMPLES = 6
BASE_FOLD_RATE = 0.35
PRIOR_STRENGTH = 15
STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

HAND_MADE_SCORE = {
    "HIGHCARD": 0.15,
    "ONEPAIR": 0.35,
    "TWOPAIR": 0.55,
    "THREECARD": 0.67,
    "STRAIGHT": 0.78,
    "FLASH": 0.82,
    "FULLHOUSE": 0.90,
    "FOURCARD": 0.96,
    "STRAIGHTFLASH": 1.00,
}
