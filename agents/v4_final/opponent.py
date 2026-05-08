from constants import BASE_FOLD_RATE, PRIOR_STRENGTH, STREETS


def initial_history():
    return {
        "preflop_raises": 0,
        "preflop_actions": 0,
        "fold_to_raise": 0,
        "raises_we_made": 0,
        "street_raises": [0, 0, 0, 0],
        "street_calls": [0, 0, 0, 0],
        "street_folds": [0, 0, 0, 0],
        "street_total": [0, 0, 0, 0],
        "opponent_actions": 0,
        "opponent_calls": 0,
        "opponent_folds": 0,
        "opponent_raises": 0,
        "showdowns": 0,
        "hands_total": 0,
    }


def track_opponent(stats, action, round_state, we_just_raised):
    street = round_state.get("street", "preflop")
    street_num = STREETS.get(street, 0)

    stats["street_total"][street_num] += 1
    stats["opponent_actions"] += 1
    if action == "raise":
        stats["street_raises"][street_num] += 1
        stats["opponent_raises"] += 1
    elif action == "call":
        stats["street_calls"][street_num] += 1
        stats["opponent_calls"] += 1
    elif action == "fold":
        stats["street_folds"][street_num] += 1
        stats["opponent_folds"] += 1

    if street_num == 0 and action in ("raise", "call", "fold"):
        stats["preflop_actions"] += 1
        if action == "raise":
            stats["preflop_raises"] += 1

    if we_just_raised:
        stats["raises_we_made"] += 1
        if action == "fold":
            stats["fold_to_raise"] += 1


def predict_response(hand_info, stats):
    fold_prob = _fold_estimate(stats["fold_to_raise"], stats["raises_we_made"])

    total_opp_actions = max(stats.get("opponent_actions", 0), 1)
    natural_fold_rate = stats.get("opponent_folds", 0) / float(total_opp_actions)
    natural_call_rate = stats.get("opponent_calls", 0) / float(total_opp_actions)
    natural_raise_rate = stats.get("opponent_raises", 0) / float(total_opp_actions)

    texture_adj = {"dry": +0.05, "semi": 0.0, "wet": -0.08}.get(
        hand_info["board_type"], 0.0
    )
    sticky_adj = -0.10 * max(0.0, natural_call_rate - 0.42)
    foldy_adj = +0.08 * max(0.0, natural_fold_rate - 0.28)
    fold_prob = max(0.08, min(0.88, fold_prob + texture_adj + sticky_adj + foldy_adj))

    street_num = int(round(hand_info["street_progress"] * 3))
    street_raises = stats["street_raises"][street_num]
    street_total = stats["street_total"][street_num]
    raise_rate = (street_raises / street_total) if street_total > 0 else 0.15

    remaining = 1.0 - fold_prob
    raise_prob = remaining * raise_rate
    call_prob = remaining - raise_prob

    total_actions = sum(stats["street_total"])
    total_raises = sum(stats["street_raises"])
    opp_aggression = (total_raises / total_actions) if total_actions > 0 else 0.20
    opp_aggression = 0.65 * opp_aggression + 0.35 * natural_raise_rate

    pfr_total = max(stats.get("preflop_actions", 0), 1)
    pfr = stats.get("preflop_raises", 0) / float(pfr_total)
    showdown_rate = stats.get("showdowns", 0) / float(max(stats.get("hands_total", 0), 1))
    stickiness = max(0.0, min(1.0, 0.65 * natural_call_rate + 0.35 * showdown_rate))
    confidence = min(1.0, total_opp_actions / 24.0)

    return {
        "fold": fold_prob,
        "call": call_prob,
        "raise": raise_prob,
        "opp_aggression": opp_aggression,
        "stickiness": stickiness,
        "pfr": pfr,
        "confidence": confidence,
        "natural_fold_rate": natural_fold_rate,
        "natural_call_rate": natural_call_rate,
        "natural_raise_rate": natural_raise_rate,
    }


def _fold_estimate(n_fold, n_total):
    pseudo = BASE_FOLD_RATE * PRIOR_STRENGTH
    est = (n_fold + pseudo) / (n_total + PRIOR_STRENGTH)
    return max(0.08, min(0.88, est))
