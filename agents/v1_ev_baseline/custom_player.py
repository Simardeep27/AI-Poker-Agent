import time
import random

from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.card import Card
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
BOARD_SAMPLES = 6
BASE_FOLD_RATE  = 0.35
PRIOR_STRENGTH  = 15
STREETS = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}


class MyPlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()
        self.history = {
            'preflop_raises':  0,
            'preflop_actions': 0,
            'fold_to_raise':   0,
            'raises_we_made':  0,
            'street_raises':   [0, 0, 0, 0],
            'street_total':    [0, 0, 0, 0],
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
        'spr':          pot / max(min(my_chips, opp_chips), 1),
    }


# Opponent Profiler

def track_opponent(stats, action, round_state, we_just_raised):
    street     = round_state.get('street', 'preflop')
    street_num = STREETS.get(street, 0)

    stats['street_total'][street_num] += 1
    if action == 'raise':
        stats['street_raises'][street_num] += 1

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
        'strength':       strength,
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

    texture_adj = {'dry': +0.05, 'semi': 0.0, 'wet': -0.08}.get(hand_info['board_type'], 0.0)
    fold_prob   = max(0.08, min(0.88, fold_prob + texture_adj))

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

    return {'fold': fold_prob, 'call': call_prob, 'raise': raise_prob, 'opp_aggression': opp_aggression}


# Search Engine

def choose_action(state, hand_info, response):
    pot        = state['pot']
    to_call    = state['to_call']
    to_raise   = state['to_raise']
    raise_ok   = state['raise_ok']
    hole_card  = state['hole_card']
    community  = state['community']
    street     = state['street']

    cards_needed = max(0, 5 - len(community))

    fold_ev = 0.0

    if cards_needed == 0:
        wr       = hand_info['win_rate']
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

    draw_bonus = 0.10 * hand_info['draw_chance'] * pot
    made_bonus = 0.08 * (hand_info['strength'] / 4.0) * pot
    call_ev  += draw_bonus + made_bonus
    if raise_ok and raise_ev > -1e8:
        raise_ev += draw_bonus + made_bonus

    if (hand_info['board_type'] == 'wet'
            and hand_info['draw_chance'] < 0.35
            and hand_info['strength'] < 2):
        call_ev  -= 0.08 * pot
        if raise_ok and raise_ev > -1e8:
            raise_ev -= 0.08 * pot

    spr            = state['spr']
    is_sb          = state['is_sb']
    opp_aggression = response['opp_aggression']
    history_line   = hand_info['history_line']
    bonus_weight   = 0.08 + 0.14 * hand_info['street_progress']

    if history_line == 'reraise':
        call_slack_mult   = 0.45
        raise_margin_mult = 2.0
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
        if not is_sb:
            call_ev += 0.05 * pot
        else:
            call_ev -= 0.02 * pot

    if (raise_ok and hand_info['draw_chance'] > 0.40 and
            hand_info['street_progress'] < 1.0 and response['fold'] > 0.30):
        step = to_raise - to_call
        semi_bluff_ev = (
            response['fold'] * pot +
            response['call'] * (hand_info['draw_chance'] * (pot + to_raise + step) - to_raise)
        )
        if semi_bluff_ev > call_ev:
            raise_ev = max(raise_ev, semi_bluff_ev)

    if raise_ok and hand_info['win_rate'] >= 0.62 and raise_ev > fold_ev:
        return 'raise'

    raise_margin = (0.01 * pot if hand_info['win_rate'] >= 0.60 else 0.07 * pot) * raise_margin_mult
    if raise_ok and raise_ev >= call_ev + raise_margin and raise_ev > fold_ev:
        return 'raise'

    street_tightener = 1.0 - 0.65 * hand_info['street_progress']
    if hand_info['street_progress'] >= 1.0:
        call_slack = 0.0
    else:
        call_slack = (0.02 + 0.05 * opp_aggression + 0.04 * hand_info['draw_chance']) * pot * street_tightener
    call_slack *= call_slack_mult
    if hand_info['win_rate'] < 0.30:
        call_slack *= 0.35
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


def _fill_board(hole_card, community, n_needed):
    suits     = ['C', 'D', 'H', 'S']
    ranks     = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    full_deck = [s + r for s in suits for r in ranks]
    known     = set(hole_card) | set(community)
    available = [c for c in full_deck if c not in known]
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
    