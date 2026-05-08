"""A BasePokerPlayer wrapper for using a trained PPO model in PyPokerEngine."""

from __future__ import annotations

from ppo_setup.feature_encoder import ID_TO_ACTION, EncoderConfig, action_mask, encode_observation
from ppo_setup.leaderboard import LEADERBOARD_EQUITY_SIMS, LEADERBOARD_INITIAL_STACK, LEADERBOARD_SMALL_BLIND
from ppo_setup.opponent_stats import OpponentStats
from ppo_setup.pathing import ensure_project_paths

ensure_project_paths()

from pypokerengine.players import BasePokerPlayer  # noqa: E402


class PPOPolicyPlayer(BasePokerPlayer):
    def __init__(
        self,
        model_path: str,
        deterministic: bool = True,
        initial_stack: int = LEADERBOARD_INITIAL_STACK,
        small_blind: int = LEADERBOARD_SMALL_BLIND,
        equity_sims: int = LEADERBOARD_EQUITY_SIMS,
    ):
        super().__init__()
        from sb3_contrib import MaskablePPO

        self.model = MaskablePPO.load(model_path, device="auto")
        self.deterministic = deterministic
        self.encoder_config = EncoderConfig(
            initial_stack=initial_stack,
            small_blind=small_blind,
            equity_sims=equity_sims,
        )
        self.stats = OpponentStats()

    def declare_action(self, valid_actions, hole_card, round_state):
        obs = encode_observation(
            valid_actions,
            hole_card,
            round_state,
            self.uuid,
            self.stats.snapshot(),
            self.encoder_config,
        )
        mask = action_mask(valid_actions)
        action, _ = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
        return ID_TO_ACTION[int(action)]

    def receive_game_start_message(self, game_info):
        self.stats = OpponentStats()

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.stats.start_round()

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self.stats.observe_action(action, round_state, self.uuid)

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.stats.observe_round_result(hand_info)
