"""Version 6: mixed-strategy experiment.

This snapshot tested extra deception and randomized action mixing. It is useful
as a comparison point, but it was not selected as the final policy.
"""

import time
import random

from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


WEIGHTS = {
    'equity':         3.0,
    'draw_potential': 1.0,
    'fold_equity':    1.5,
    'pot_pressure':   0.6,
    'board_danger':  -0.5,
    'showdown_value': 0.7,
    'opp_aggression': 0.4,
    'street_bonus':   0.3,
}

NUM_SIMS     = {'preflop': 300, 'flop': 180, 'turn': 120, 'river': 50}
RANGE_SIMS   = {'preflop': 0, 'flop': 20, 'turn': 24, 'river': 42}
BOARD_SAMPLES = 6
BASE_FOLD_RATE  = 0.35
PRIOR_STRENGTH  = 15
STREETS = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
HAND_MADE_SCORE = {
    'HIGHCARD': 0.15,
    'ONEPAIR': 0.35,
    'TWOPAIR': 0.55,
    'THREECARD': 0.67,
    'STRAIGHT': 0.78,
    'FLASH': 0.82,
    'FULLHOUSE': 0.90,
    'FOURCARD': 0.96,
    'STRAIGHTFLASH': 1.00,
}


class MyPlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()
        self.history = {
            'preflop_raises':  0,
            'preflop_actions': 0,
            'fold_to_raise':   0,
            'raises_we_made':  0,
            'street_raises':   [0, 0, 0, 0],
            'street_calls':    [0, 0, 0, 0],
            'street_folds':    [0, 0, 0, 0],
            'street_total':    [0, 0, 0, 0],
            'opponent_actions': 0,
            'opponent_calls':   0,
            'opponent_folds':   0,
            'opponent_raises':  0,
            'showdowns':       0,
            'hands_total':     0,
        }
        self.just_raised = False

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.history['hands_total'] += 1

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        who  = action.get('player_uuid', '')
        what = action.get('action', '').lower()
        if who == self.uuid:
            self.just_raised = (what == 'raise')
        else:
            track_opponent(self.history, what, round_state, self.just_raised)
            self.just_raised = False

    def receive_round_result_message(self, winners, hand_info, round_state):
        if hand_info:
            self.history['showdowns'] += 1

    def declare_action(self, valid_actions, hole_card, round_state):
        start = time.time()

        state    = read_state(valid_actions, hole_card, round_state, self.uuid)
        hand_info = analyze_state(state)
        response  = predict_response(hand_info, self.history)
        action    = choose_action(state, hand_info, response)

        if time.time() - start > 0.35:
            names  = [a['action'] for a in valid_actions]
            action = 'call' if 'call' in names else 'fold'

        return action


# State Parser

def read_state(valid_actions, hole_card, round_state, my_uuid):
    street      = round_state['street']
    street_num  = STREETS.get(street, 0)
    community   = round_state.get('community_card', [])
    pot         = round_state['pot']['main']['amount']
    blind       = round_state['small_blind_amount']
    seats       = round_state['seats']

    my_seat   = next((s for s in seats if s['uuid'] == my_uuid),  None)
    opp_seat  = next((s for s in seats if s['uuid'] != my_uuid),  None)
    my_chips  = my_seat['stack']  if my_seat  else 10000
    opp_chips = opp_seat['stack'] if opp_seat else 10000

    sb_pos    = round_state.get('small_blind_pos', 0)
    my_pos    = next((i for i, s in enumerate(seats) if s['uuid'] == my_uuid), 0)
    is_sb     = (my_pos == sb_pos)

    names     = [a['action'] for a in valid_actions]
    raise_ok  = 'raise' in names

    to_call, to_raise = _get_bet_amounts(round_state, street, blind, my_uuid)
    history_line      = _get_history_line(round_state.get('action_histories', {}), street)

    return {
        'street':       street,
        'street_num':   street_num,
        'hole_card':    hole_card,
        'community':    community,
        'pot':          pot,
        'blind':        blind,
        'my_chips':     my_chips,
        'opp_chips':    opp_chips,
        'is_sb':        is_sb,
        'raise_ok':     raise_ok,
        'to_call':      to_call,
        'to_raise':     to_raise,
        'history_line': history_line,
        'my_uuid':      my_uuid,
        'round_count':  round_state.get('round_count', 0),
        'spr':          pot / max(min(my_chips, opp_chips), 1),
    }


# Opponent Profiler

def track_opponent(stats, action, round_state, we_just_raised):
    street     = round_state.get('street', 'preflop')
    street_num = STREETS.get(street, 0)

    stats['street_total'][street_num] += 1
    stats['opponent_actions'] += 1
    if action == 'raise':
        stats['street_raises'][street_num] += 1
        stats['opponent_raises'] += 1
    elif action == 'call':
        stats['street_calls'][street_num] += 1
        stats['opponent_calls'] += 1
    elif action == 'fold':
        stats['street_folds'][street_num] += 1
        stats['opponent_folds'] += 1

    if street_num == 0 and action in ('raise', 'call', 'fold'):
        stats['preflop_actions'] += 1
        if action == 'raise':
            stats['preflop_raises'] += 1

    if we_just_raised:
        stats['raises_we_made'] += 1
        if action == 'fold':
            stats['fold_to_raise'] += 1


# Abstraction Layer

def analyze_state(state):
    hole_card   = state['hole_card']
    community   = state['community']
    pot         = state['pot']
    my_chips    = state['my_chips']
    street      = state['street']
    street_num  = state['street_num']

    n_sims      = NUM_SIMS.get(street, 100)
    hole_objs   = gen_cards(hole_card)
    board_objs  = gen_cards(community)
    win_rate    = estimate_hole_card_win_rate(n_sims, 2, hole_objs, board_objs)

    if street == 'preflop':
        approx   = _preflop_strength(hole_card)
        win_rate = 0.70 * win_rate + 0.30 * approx

    raw_win_rate = win_rate
    pressure = state['to_call'] / float(max(state['pot'] + state['to_call'], 1))
    range_rate = None
    hard_spot = (
        street != 'preflop' and (
            state['history_line'] != 'passive' or
            pressure >= 0.16 or
            street == 'river'
        )
    )
    if hard_spot:
        range_rate = _range_aware_win_rate(state, n_sims=RANGE_SIMS.get(street, 0))
        if range_rate is not None:
            line_weight = {
                'passive':      0.14,
                'single_raise': 0.30,
                'reraise':      0.46,
            }.get(state['history_line'], 0.30)
            if street == 'river':
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

    draw_chance = _draw_outs(hole_objs, board_objs)
    board_type  = _board_texture(board_objs)
    made_score  = _made_hand_score(hole_objs, board_objs)

    if pot < 100:
        pot_size = 'small'
    elif pot < 400:
        pot_size = 'medium'
    elif pot < 1000:
        pot_size = 'large'
    else:
        pot_size = 'huge'

    invested        = min(pot / max(my_chips, 1), 1.0)
    street_progress = street_num / 3.0

    return {
        'win_rate':       win_rate,
        'raw_win_rate':   raw_win_rate,
        'range_win_rate': range_rate,
        'strength':       strength,
        'made_score':     made_score,
        'draw_chance':    draw_chance,
        'board_type':     board_type,
        'pot_size':       pot_size,
        'invested':       invested,
        'street_progress': street_progress,
        'history_line':   state['history_line'],
    }


# Belief Estimator

def predict_response(hand_info, stats):
    n_fold   = stats['fold_to_raise']
    n_raises = stats['raises_we_made']
    fold_prob = _fold_estimate(n_fold, n_raises)

    total_opp_actions = max(stats.get('opponent_actions', 0), 1)
    natural_fold_rate = stats.get('opponent_folds', 0) / float(total_opp_actions)
    natural_call_rate = stats.get('opponent_calls', 0) / float(total_opp_actions)
    natural_raise_rate = stats.get('opponent_raises', 0) / float(total_opp_actions)

    texture_adj = {'dry': +0.05, 'semi': 0.0, 'wet': -0.08}.get(hand_info['board_type'], 0.0)
    sticky_adj  = -0.10 * max(0.0, natural_call_rate - 0.42)
    foldy_adj   = +0.08 * max(0.0, natural_fold_rate - 0.28)
    fold_prob   = max(0.08, min(0.88, fold_prob + texture_adj + sticky_adj + foldy_adj))

    street_num  = int(round(hand_info['street_progress'] * 3))
    s_raises    = stats['street_raises'][street_num]
    s_total     = stats['street_total'][street_num]
    raise_rate  = (s_raises / s_total) if s_total > 0 else 0.15

    remaining   = 1.0 - fold_prob
    raise_prob  = remaining * raise_rate
    call_prob   = remaining - raise_prob

    total_actions  = sum(stats['street_total'])
    total_raises   = sum(stats['street_raises'])
    opp_aggression = (total_raises / total_actions) if total_actions > 0 else 0.20
    opp_aggression = 0.65 * opp_aggression + 0.35 * natural_raise_rate

    pfr_total = max(stats.get('preflop_actions', 0), 1)
    pfr = stats.get('preflop_raises', 0) / float(pfr_total)
    showdown_rate = stats.get('showdowns', 0) / float(max(stats.get('hands_total', 0), 1))
    stickiness = max(0.0, min(1.0, 0.65 * natural_call_rate + 0.35 * showdown_rate))
    confidence = min(1.0, total_opp_actions / 24.0)

    return {
        'fold': fold_prob,
        'call': call_prob,
        'raise': raise_prob,
        'opp_aggression': opp_aggression,
        'stickiness': stickiness,
        'pfr': pfr,
        'confidence': confidence,
        'natural_fold_rate': natural_fold_rate,
        'natural_call_rate': natural_call_rate,
        'natural_raise_rate': natural_raise_rate,
    }


# Search Engine

def choose_action(state, hand_info, response):
    pot        = state['pot']
    to_call    = state['to_call']
    to_raise   = state['to_raise']
    raise_ok   = state['raise_ok']
    hole_card  = state['hole_card']
    community  = state['community']
    street     = state['street']
    wr         = hand_info['win_rate']
    made_score = hand_info.get('made_score', hand_info['strength'] / 4.0)
    pot_odds   = to_call / float(max(pot + to_call, 1))

    cards_needed = max(0, 5 - len(community))

    fold_ev = 0.0

    if cards_needed == 0:
        call_ev  = _call_ev(wr, pot, to_call)
        if raise_ok:
            step            = to_raise - to_call
            pot_opp_calls   = pot + to_raise + step
            pot_opp_reraises= pot + to_raise + step * 2
            raise_ev = (
                response['fold']  * pot +
                response['call']  * (wr * pot_opp_calls   - to_raise) +
                response['raise'] * (wr * pot_opp_reraises - to_raise)
            )
        else:
            raise_ev = -1e9
    else:
        sims_each  = max(10, NUM_SIMS.get(street, 100) // BOARD_SAMPLES)
        call_acc   = 0.0
        raise_acc  = 0.0

        for _ in range(BOARD_SAMPLES):
            board      = _fill_board(hole_card, community, cards_needed)
            hole_objs  = gen_cards(hole_card)
            board_objs = gen_cards(board)
            wr         = estimate_hole_card_win_rate(sims_each, 2, hole_objs, board_objs)

            call_acc += _call_ev(wr, pot, to_call)

            if raise_ok:
                step            = to_raise - to_call
                pot_opp_calls   = pot + to_raise + step
                pot_opp_reraises= pot + to_raise + step * 2
                raise_acc += (
                    response['fold']  * pot +
                    response['call']  * (wr * pot_opp_calls   - to_raise) +
                    response['raise'] * (wr * pot_opp_reraises - to_raise)
                )

        call_ev  = call_acc  / BOARD_SAMPLES
        raise_ev = (raise_acc / BOARD_SAMPLES) if raise_ok else -1e9

    future_draw = 0.0 if street == 'river' else hand_info['draw_chance']
    draw_bonus = 0.10 * future_draw * pot
    made_rate = 0.035 if street == 'river' else 0.07
    made_bonus = made_rate * max(hand_info['strength'] / 4.0, made_score) * pot
    call_ev  += draw_bonus + made_bonus
    if raise_ok and raise_ev > -1e8:
        raise_ev += draw_bonus + made_bonus

    if (hand_info['board_type'] == 'wet'
            and hand_info['draw_chance'] < 0.35
            and hand_info['strength'] < 2):
        call_ev  -= 0.08 * pot
        if raise_ok and raise_ev > -1e8:
            raise_ev -= 0.08 * pot

    sticky = response.get('stickiness', 0.0)
    confidence = response.get('confidence', 0.0)
    if raise_ok and raise_ev > -1e8 and sticky > 0.40 and wr < 0.67:
        raise_ev -= confidence * 0.08 * pot

    maniac = response.get('natural_raise_rate', 0.0) >= 0.52 and confidence >= 0.45
    if maniac:
        if wr >= 0.61 and made_score >= 0.35:
            call_ev += 0.07 * pot
        if wr < pot_odds + 0.10 and made_score < 0.67:
            call_ev -= 0.12 * pot
        if raise_ok and raise_ev > -1e8 and wr < 0.70 and made_score < 0.78:
            raise_ev -= 0.18 * pot

    if street == 'river':
        river_need = pot_odds + 0.035
        if hand_info['history_line'] == 'reraise':
            river_need += 0.08
        if hand_info['board_type'] == 'wet':
            river_need += 0.035
        if wr < river_need and made_score < 0.55:
            call_ev -= 0.16 * pot
        if raise_ok and raise_ev > -1e8:
            if wr < 0.68 and made_score < 0.78 and response['fold'] < 0.52:
                raise_ev -= 0.22 * pot
            if wr < 0.54 and made_score < 0.67:
                raise_ev = min(raise_ev, call_ev - 0.05 * pot)

    spr            = state['spr']
    is_sb          = state['is_sb']
    opp_aggression = response['opp_aggression']
    history_line   = hand_info['history_line']
    bonus_weight   = 0.08 + 0.14 * hand_info['street_progress']

    if history_line == 'reraise':
        call_slack_mult   = 0.45
        raise_margin_mult = 2.0
        if wr < 0.57 and raise_ok and raise_ev > -1e8:
            raise_ev -= 0.13 * pot
        if wr < 0.42:
            call_ev -= 0.07 * pot
        if street == 'river' and wr < 0.62 and raise_ok:
            raise_ev -= 0.18 * pot
    elif history_line == 'passive':
        if raise_ok and raise_ev > -1e8:
            raise_ev += 0.04 * pot
        call_slack_mult   = 1.15
        raise_margin_mult = 1.0
    else:
        call_slack_mult   = 1.0
        raise_margin_mult = 1.0

    bonus    = score_state(hand_info, pot + to_call, response['fold'], opp_aggression, is_sb)
    call_ev  += bonus_weight * bonus
    if raise_ok and raise_ev > -1e8:
        raise_ev += bonus_weight * response['call'] * bonus

    if spr < 2 and hand_info['win_rate'] > 0.58 and raise_ok:
        raise_ev += 0.10 * pot
    elif spr > 15 and raise_ok:
        raise_ev -= 0.05 * pot

    if street == 'preflop':
        pf_strength = _preflop_strength(hole_card)
        if not is_sb:
            call_ev += 0.05 * pot
        else:
            call_ev -= 0.02 * pot
        if history_line == 'reraise' and pf_strength < 0.58:
            call_ev -= 0.10 * pot
            if raise_ok and raise_ev > -1e8:
                raise_ev -= 0.18 * pot
        elif history_line == 'single_raise' and pf_strength < 0.42 and to_call > state['blind'] * 2:
            call_ev -= 0.06 * pot

    if (raise_ok and hand_info['draw_chance'] > 0.40 and
            hand_info['street_progress'] < 1.0 and response['fold'] > 0.30 and
            sticky < 0.58 and not maniac):
        step = to_raise - to_call
        semi_bluff_ev = (
            response['fold'] * pot +
            response['call'] * (hand_info['draw_chance'] * (pot + to_raise + step) - to_raise)
        )
        if semi_bluff_ev > call_ev:
            raise_ev = max(raise_ev, semi_bluff_ev)

    raise_allowed = _raise_allowed(state, hand_info, response)
    if (raise_ok and raise_allowed and wr >= _value_raise_threshold(state, hand_info, response)
            and raise_ev > fold_ev):
        if _slowplay_mix(state, hand_info, response, raise_ev, call_ev):
            return 'call'
        return 'raise'

    raise_margin = (0.01 * pot if wr >= 0.60 else 0.07 * pot) * raise_margin_mult
    if raise_ok and raise_allowed and raise_ev >= call_ev + raise_margin and raise_ev > fold_ev:
        if _slowplay_mix(state, hand_info, response, raise_ev, call_ev):
            return 'call'
        return 'raise'

    street_tightener = 1.0 - 0.65 * hand_info['street_progress']
    if hand_info['street_progress'] >= 1.0:
        call_slack = 0.0
    else:
        call_slack = (0.02 + 0.05 * opp_aggression + 0.04 * hand_info['draw_chance']) * pot * street_tightener
    call_slack *= call_slack_mult
    if hand_info['win_rate'] < 0.30:
        call_slack *= 0.35
    if _probe_raise_mix(state, hand_info, response, raise_ev, call_ev, call_slack):
        return 'raise'
    if call_ev >= -call_slack:
        return 'call'
    return 'fold'


# Leaf Evaluator

def score_state(hand_info, pot, fold_prob, opp_aggression, is_sb):
    w = WEIGHTS

    board_danger_map = {'dry': 0.0, 'semi': 0.5, 'wet': 1.0}
    position_bonus   = 0.0 if is_sb else 0.10

    value = (
        w['equity']         * hand_info['win_rate']                              +
        w['draw_potential'] * hand_info['draw_chance']                           +
        w['fold_equity']    * fold_prob                                          +
        w['pot_pressure']   * hand_info['invested']                              +
        w['board_danger']   * board_danger_map.get(hand_info['board_type'], 0.5) +
        w['showdown_value'] * (hand_info['strength'] / 4.0)                     +
        w['opp_aggression'] * opp_aggression                                     +
        w['street_bonus']   * hand_info['street_progress']                       +
        position_bonus
    )
    return value


def _get_bet_amounts(round_state, street, blind, my_uuid):
    history = round_state.get('action_histories', {}).get(street, [])
    my_bet  = 0
    max_bet = 0
    for h in history:
        if h is None:
            continue
        act = h.get('action', '').upper()
        if act in ('RAISE', 'CALL', 'SMALLBLIND', 'BIGBLIND'):
            amt = h.get('amount', 0)
            if h.get('uuid') == my_uuid:
                my_bet  = max(my_bet,  amt)
            else:
                max_bet = max(max_bet, amt)
    to_call    = max(0, max_bet - my_bet)
    street_num = STREETS.get(street, 0)
    step       = blind * 2 if street_num <= 1 else blind * 4
    to_raise   = to_call + step
    return to_call, to_raise


def _get_history_line(action_histories, street):
    current     = action_histories.get(street, [])
    raise_count = sum(
        1 for h in current
        if h is not None and h.get('action', '').upper() == 'RAISE'
    )
    if raise_count == 0:
        return 'passive'
    if raise_count == 1:
        return 'single_raise'
    return 'reraise'


def _fold_estimate(n_fold, n_total):
    pseudo = BASE_FOLD_RATE * PRIOR_STRENGTH
    est    = (n_fold + pseudo) / (n_total + PRIOR_STRENGTH)
    return max(0.08, min(0.88, est))


def _call_ev(win_rate, pot, to_call):
    if to_call <= 0:
        return win_rate * pot
    return win_rate * (pot + to_call) - to_call


def _slowplay_mix(state, hand_info, response, raise_ev, call_ev):
    if state['street'] == 'preflop':
        return False
    if hand_info['history_line'] != 'passive':
        return False
    if hand_info['board_type'] == 'wet':
        return False
    if hand_info['win_rate'] < 0.72 or hand_info.get('made_score', 0.0) < 0.55:
        return False
    if response.get('stickiness', 0.0) > 0.62:
        return False
    if response.get('opp_aggression', 0.0) < 0.18 and response.get('natural_raise_rate', 0.0) < 0.18:
        return False
    if raise_ev > call_ev + max(12.0, 0.09 * state['pot']):
        return False
    return _mix_bucket(state, 'slow') < 5


def _probe_raise_mix(state, hand_info, response, raise_ev, call_ev, call_slack):
    if not state['raise_ok'] or state['street'] == 'river':
        return False
    if hand_info['history_line'] == 'reraise':
        return False
    if response.get('stickiness', 0.0) > 0.50:
        return False
    if response.get('fold', 0.0) < 0.44:
        return False
    if hand_info['win_rate'] < 0.44 and hand_info['draw_chance'] < 0.32:
        return False
    if raise_ev < call_ev - max(call_slack, 0.035 * state['pot']):
        return False
    return _mix_bucket(state, 'probe') < 4


def _mix_bucket(state, salt):
    text = "%s:%s:%s:%s:%s:%s" % (
        salt,
        state.get('round_count', 0),
        ''.join(state.get('hole_card', [])),
        ''.join(state.get('community', [])),
        state.get('pot', 0),
        state.get('history_line', ''),
    )
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1000003
    return value % 100


def _raise_allowed(state, hand_info, response):
    if not state['raise_ok']:
        return False

    wr = hand_info['win_rate']
    street = state['street']
    history_line = hand_info['history_line']
    sticky = response.get('stickiness', 0.0)

    if wr >= _value_raise_threshold(state, hand_info, response):
        return True

    if street != 'river':
        strong_draw = hand_info['draw_chance'] >= 0.43
        foldable = response['fold'] >= 0.32 and sticky < 0.58
        if strong_draw and foldable and history_line != 'reraise':
            return True

    if (history_line == 'passive' and hand_info['board_type'] == 'dry' and
            response['fold'] >= 0.50 and sticky < 0.38 and wr >= 0.47):
        return True

    return False


def _value_raise_threshold(state, hand_info, response):
    street = state['street']
    threshold = {
        'preflop': 0.63,
        'flop':    0.64,
        'turn':    0.66,
        'river':   0.69,
    }.get(street, 0.66)

    history_line = hand_info['history_line']
    if history_line == 'passive':
        threshold -= 0.02
    elif history_line == 'single_raise':
        threshold += 0.02
    else:
        threshold += 0.075

    if hand_info['board_type'] == 'wet':
        threshold += 0.025
    elif hand_info['board_type'] == 'dry':
        threshold -= 0.01

    if response['fold'] > 0.48 and response.get('stickiness', 0.0) < 0.42:
        threshold -= 0.045
    if street != 'river' and hand_info['draw_chance'] > 0.44 and response['fold'] > 0.34:
        threshold -= 0.035
    if hand_info.get('made_score', 0.0) >= 0.78:
        threshold -= 0.05

    return max(0.54, min(0.78, threshold))


def _made_hand_score(hole_objs, board_objs):
    if len(board_objs) < 3:
        return 0.0
    try:
        info = HandEvaluator.gen_hand_rank_info(hole_objs, board_objs)
        strength = info['hand']['strength']
        return HAND_MADE_SCORE.get(strength, 0.20)
    except Exception:
        return 0.0


def _range_aware_win_rate(state, n_sims):
    hole_card = state['hole_card']
    community = state['community']
    if len(hole_card) != 2 or n_sims <= 0:
        return None

    known = set(hole_card) | set(community)
    deck = _available_deck(known)
    if len(deck) < 2:
        return None

    threshold = _opponent_range_threshold(state)
    hero_objs = gen_cards(hole_card)
    wins = 0.0
    trials = 0
    cards_needed = max(0, 5 - len(community))

    for _ in range(n_sims):
        opp_cards = _sample_range_hand(deck, threshold)
        if not opp_cards:
            continue
        blocked = set(opp_cards)
        runout_pool = [card for card in deck if card not in blocked]
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


def _opponent_range_threshold(state):
    line = state['history_line']
    street = state['street']
    to_call = state['to_call']
    pot = max(state['pot'], 1)
    pressure = min(to_call / float(pot + to_call), 1.0)

    if line == 'reraise':
        threshold = 0.56
    elif line == 'single_raise':
        threshold = 0.42
    else:
        threshold = 0.20

    if street in ('turn', 'river'):
        threshold += 0.035
    threshold += 0.10 * pressure
    return max(0.12, min(0.70, threshold))


def _sample_range_hand(deck, threshold):
    if len(deck) < 2:
        return None
    best = None
    best_score = -1.0
    for _ in range(10):
        cards = random.sample(deck, 2)
        score = _preflop_strength(cards)
        if score >= threshold:
            return cards
        if score > best_score:
            best = cards
            best_score = score
    return best


def _available_deck(known):
    suits = ['C', 'D', 'H', 'S']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return [s + r for s in suits for r in ranks if s + r not in known]


def _fill_board(hole_card, community, n_needed):
    known     = set(hole_card) | set(community)
    available = _available_deck(known)
    drawn     = random.sample(available, n_needed)
    return list(community) + drawn


def _preflop_strength(hole_card):
    RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
                'T':10,'J':11,'Q':12,'K':13,'A':14}
    s1, r1_c = hole_card[0][0], hole_card[0][1]
    s2, r2_c = hole_card[1][0], hole_card[1][1]
    r1 = RANK_MAP.get(r1_c, 2)
    r2 = RANK_MAP.get(r2_c, 2)
    high, low = max(r1, r2), min(r1, r2)
    gap    = high - low
    suited = s1 == s2
    pair   = r1 == r2

    if pair:
        return min(0.95, 0.51 + high / 30.0)

    score = 0.30 + high / 40.0 + low / 90.0
    if suited:        score += 0.035
    if gap == 1:      score += 0.035
    elif gap == 2:    score += 0.018
    elif gap >= 5:    score -= 0.035
    if high == 14 and low >= 10:   score += 0.075
    elif high >= 13 and low >= 10: score += 0.045
    return max(0.05, min(0.92, score))


def _draw_outs(hole_objs, board_objs):
    if not board_objs:
        return 0.0
    all_cards = hole_objs + board_objs
    remaining = 52 - len(all_cards)
    if remaining <= 0:
        return 0.0

    outs = 0

    suit_counts = {}
    for c in all_cards:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
    if any(v == 4 for v in suit_counts.values()):
        outs = max(outs, 9)

    ranks = sorted(set(c.rank for c in all_cards))
    for r in ranks:
        window = [rr for rr in ranks if r <= rr <= r + 4]
        if len(window) == 4:
            outs = max(outs, 8)
        elif len(window) == 3:
            outs = max(outs, 4)

    return min(outs / remaining, 1.0)


def _board_texture(board_objs):
    if len(board_objs) < 3:
        return 'dry'

    suit_counts = {}
    for c in board_objs:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
    flush_draw = any(v >= 3 for v in suit_counts.values())

    ranks = sorted(set(c.rank for c in board_objs))
    straight_draw = False
    for r in ranks:
        window = [rr for rr in ranks if r <= rr <= r + 4]
        if len(window) >= 3:
            straight_draw = True
            break

    rank_counts = {}
    for c in board_objs:
        rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1
    paired = any(v >= 2 for v in rank_counts.values())

    draws = int(flush_draw) + int(straight_draw)
    if draws >= 2 or (draws >= 1 and paired):
        return 'wet'
    if draws == 1 or paired:
        return 'semi'
    return 'dry'


def setup_ai():
    return MyPlayer()
    
