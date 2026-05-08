"""PyPokerEngine player wrapper for the final modular agent."""

import time

from pypokerengine.players import BasePokerPlayer

from decision import choose_action
from features import analyze_state
from opponent import initial_history, predict_response, track_opponent
from state import read_state


class MyPlayer(BasePokerPlayer):
    """Thin callback layer around state parsing, opponent modeling, and decision logic."""

    def __init__(self):
        super().__init__()
        self.history = initial_history()
        self.just_raised = False

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.history["hands_total"] += 1

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        player_uuid = action.get("player_uuid", "")
        action_name = action.get("action", "").lower()
        if player_uuid == self.uuid:
            self.just_raised = action_name == "raise"
            return

        track_opponent(self.history, action_name, round_state, self.just_raised)
        self.just_raised = False

    def receive_round_result_message(self, winners, hand_info, round_state):
        if hand_info:
            self.history["showdowns"] += 1

    def declare_action(self, valid_actions, hole_card, round_state):
        start = time.time()
        state = read_state(valid_actions, hole_card, round_state, self.uuid)
        hand_info = analyze_state(state)
        response = predict_response(hand_info, self.history)
        action = choose_action(state, hand_info, response)

        if time.time() - start > 0.35:
            names = [item["action"] for item in valid_actions]
            action = "call" if "call" in names else "fold"

        return action


def setup_ai():
    return MyPlayer()
