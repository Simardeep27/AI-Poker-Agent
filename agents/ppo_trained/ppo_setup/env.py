"""Gymnasium wrapper around the local heads-up PyPokerEngine game."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ppo_setup.feature_encoder import (
    ACTION_TO_ID,
    ID_TO_ACTION,
    OBSERVATION_SIZE,
    EncoderConfig,
    action_mask,
    encode_observation,
)
from ppo_setup.opponent_stats import OpponentStats
from ppo_setup.opponents import sample_opponent
from ppo_setup.pathing import ensure_project_paths

ensure_project_paths()

from pypokerengine.api.emulator import exclude_short_of_money_players, update_blind_level  # noqa: E402
from pypokerengine.engine.message_builder import MessageBuilder  # noqa: E402
from pypokerengine.engine.player import Player  # noqa: E402
from pypokerengine.engine.poker_constants import PokerConstants as Const  # noqa: E402
from pypokerengine.engine.round_manager import RoundManager  # noqa: E402
from pypokerengine.engine.table import Table  # noqa: E402


class HeadsUpPokerEnv(gym.Env):
    """Decision-level PPO environment.

    Episodes are short heads-up games. A step applies one hero action, then the
    environment advances opponent actions and streets until the hero is asked
    again or the game ends. Rewards are end-of-hand chip deltas in big blinds.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_pool: list[str] | tuple[str, ...] | str = ("custom",),
        max_rounds: int = 100,
        initial_stack: int = 10000,
        small_blind: int = 20,
        ante: int = 0,
        equity_sims: int = 32,
        randomize_position: bool = True,
        blind_structure: dict | None = None,
        hero_observer=None,
    ):
        super().__init__()
        self.opponent_pool = opponent_pool
        self.max_rounds = int(max_rounds)
        self.initial_stack = int(initial_stack)
        self.small_blind = int(small_blind)
        self.ante = int(ante)
        self.equity_sims = int(equity_sims)
        self.randomize_position = bool(randomize_position)
        self.blind_structure = blind_structure or {}
        self._hero_observer = hero_observer

        self.hero_uuid = "ppo-hero"
        self.opponent_uuid = "ppo-villain"
        self.hero_name = "ppo"
        self.opponent_name = "opponent"
        self.encoder_config = EncoderConfig(
            initial_stack=self.initial_stack,
            small_blind=self.small_blind,
            equity_sims=self.equity_sims,
        )
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._rng = random.Random()
        self._zero_obs = np.zeros((OBSERVATION_SIZE,), dtype=np.float32)
        self._pending_ask: dict[str, Any] | None = None
        self._last_mask = np.array([True, True, False], dtype=bool)
        self._opponent = None
        self._opponent_stats = OpponentStats()
        self._table = None
        self._state = None
        self._round_count = 0
        self._hand_start_stack = float(self.initial_stack)
        self._current_sb = self.small_blind
        self._current_ante = self.ante
        self._terminated = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self._build_game()
        obs, reward, terminated, truncated, info = self._start_next_round_and_run()
        if terminated or truncated:
            obs = self._zero_obs.copy()
        return obs, info

    def step(self, action: int):
        if self._terminated:
            return self._zero_obs.copy(), 0.0, True, False, self._info()

        legal = self.action_masks()
        action_id = int(action)
        if action_id < 0 or action_id >= len(legal) or not legal[action_id]:
            action_id = self._fallback_legal_action(legal)
        action_name = ID_TO_ACTION[action_id]

        self._state, messages = RoundManager.apply_action(self._state, action_name)
        obs, reward, terminated, truncated, info = self._run_until_hero_decision(messages)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._last_mask.copy()

    @property
    def current_ask(self) -> dict[str, Any] | None:
        """Latest hero ask message, useful for expert-label collection."""
        return self._pending_ask

    def render(self):
        return None

    def close(self):
        return None

    def _build_game(self) -> None:
        self._pending_ask = None
        self._last_mask = np.array([True, True, False], dtype=bool)
        self._opponent_stats = OpponentStats()
        self._opponent = sample_opponent(self.opponent_pool)
        self._opponent.set_uuid(self.opponent_uuid)
        if self._hero_observer is not None:
            self._hero_observer.set_uuid(self.hero_uuid)
        self._round_count = 0
        self._current_sb = self.small_blind
        self._current_ante = self.ante
        self._terminated = False

        self._table = Table()
        hero = Player(self.hero_uuid, self.initial_stack, self.hero_name)
        villain = Player(self.opponent_uuid, self.initial_stack, self.opponent_name)
        if self.randomize_position and self._rng.random() < 0.5:
            self._table.seats.sitdown(villain)
            self._table.seats.sitdown(hero)
        else:
            self._table.seats.sitdown(hero)
            self._table.seats.sitdown(villain)
        self._table.dealer_btn = 0

        config = {
            "initial_stack": self.initial_stack,
            "max_round": self.max_rounds,
            "small_blind_amount": self.small_blind,
            "ante": self.ante,
            "blind_structure": self.blind_structure,
        }
        self._notify(-1, MessageBuilder.build_game_start_message(config, self._table.seats))

    def _start_next_round_and_run(self):
        while True:
            if self._round_count >= self.max_rounds or self._winner_decided(self._table):
                self._terminated = True
                return self._zero_obs.copy(), 0.0, True, False, self._info()

            self._round_count += 1
            self._current_ante, self._current_sb = update_blind_level(
                self._current_ante,
                self._current_sb,
                self._round_count,
                self.blind_structure,
            )
            self._table = exclude_short_of_money_players(self._table, self._current_ante, self._current_sb)
            if self._winner_decided(self._table):
                self._terminated = True
                return self._zero_obs.copy(), 0.0, True, False, self._info()

            self._hand_start_stack = self._hero_stack_from_table(self._table)
            self._opponent_stats.start_round()
            self._state, messages = RoundManager.start_new_round(
                self._round_count,
                self._current_sb,
                self._current_ante,
                self._table,
            )
            return self._run_until_hero_decision(messages)

    def _run_until_hero_decision(self, messages):
        queue = deque(messages)
        reward = 0.0

        while True:
            round_finished = False
            while queue:
                address, message = queue.popleft()
                if message["type"] == "notification":
                    message_type = message["message"].get("message_type")
                    self._notify(address, message)
                    if message_type == MessageBuilder.GAME_UPDATE_MESSAGE:
                        action = message["message"]["action"]
                        round_state = message["message"]["round_state"]
                        self._opponent_stats.observe_action(action, round_state, self.hero_uuid)
                    elif message_type == MessageBuilder.ROUND_RESULT_MESSAGE:
                        self._opponent_stats.observe_round_result(message["message"].get("hand_info"))
                        round_finished = True
                    continue

                if message["type"] != "ask":
                    continue

                ask = message["message"]
                if address == self.hero_uuid:
                    self._pending_ask = ask
                    self._last_mask = action_mask(ask["valid_actions"])
                    obs = encode_observation(
                        ask["valid_actions"],
                        ask["hole_card"],
                        ask["round_state"],
                        self.hero_uuid,
                        self._opponent_stats.snapshot(),
                        self.encoder_config,
                    )
                    return obs, reward, False, False, self._info()

                action = self._opponent_action(ask)
                self._state, next_messages = RoundManager.apply_action(self._state, action)
                queue.extend(next_messages)

            if round_finished or (self._state and self._state.get("street") == Const.Street.FINISHED):
                reward += self._finish_round_reward()
                if self._round_count >= self.max_rounds or self._winner_decided(self._table):
                    self._terminated = True
                    return self._zero_obs.copy(), reward, True, False, self._info()
                obs, next_reward, terminated, truncated, info = self._start_next_round_and_run()
                return obs, reward + next_reward, terminated, truncated, info

            self._terminated = True
            return self._zero_obs.copy(), reward, True, False, self._info(error="message_queue_exhausted")

    def _finish_round_reward(self) -> float:
        self._table = self._state["table"]
        final_stack = self._hero_stack_from_table(self._table)
        delta = final_stack - self._hand_start_stack
        if self._table.seats.count_active_players() > 0:
            self._table.shift_dealer_btn()
        return float(delta) / float(max(self._current_sb * 2, 1))

    def _opponent_action(self, ask: dict) -> str:
        response = self._opponent.respond_to_ask(ask)
        if isinstance(response, tuple):
            response = response[0]
        if response not in ACTION_TO_ID:
            response = "fold"
        legal = {entry.get("action") for entry in ask["valid_actions"]}
        if response not in legal:
            response = "call" if "call" in legal else "fold"
        return response

    def _notify(self, address, message: dict) -> None:
        targets = []
        if address in (-1, self.opponent_uuid):
            targets.append(self._opponent)
        if self._hero_observer is not None and address in (-1, self.hero_uuid):
            targets.append(self._hero_observer)
        for target in targets:
            try:
                target.receive_notification(message["message"])
            except Exception:
                pass

    def _info(self, **extra) -> dict:
        info = {
            "round_count": self._round_count,
            "hero_stack": self.hero_stack,
            "opponent_stack": self.opponent_stack,
            "opponent_stats": self._opponent_stats.snapshot(),
        }
        info.update(extra)
        return info

    @property
    def hero_stack(self) -> float:
        table = self._state["table"] if self._state else self._table
        return self._hero_stack_from_table(table) if table else float(self.initial_stack)

    @property
    def opponent_stack(self) -> float:
        table = self._state["table"] if self._state else self._table
        return self._stack_from_table(table, self.opponent_uuid) if table else float(self.initial_stack)

    def _hero_stack_from_table(self, table) -> float:
        return self._stack_from_table(table, self.hero_uuid)

    @staticmethod
    def _stack_from_table(table, uuid: str) -> float:
        player = next((seat for seat in table.seats.players if seat.uuid == uuid), None)
        return float(player.stack if player else 0.0)

    @staticmethod
    def _winner_decided(table) -> bool:
        return len([player for player in table.seats.players if player.stack > 0]) <= 1

    @staticmethod
    def _fallback_legal_action(mask: np.ndarray) -> int:
        if mask[ACTION_TO_ID["call"]]:
            return ACTION_TO_ID["call"]
        if mask[ACTION_TO_ID["fold"]]:
            return ACTION_TO_ID["fold"]
        legal = np.flatnonzero(mask)
        return int(legal[0]) if len(legal) else ACTION_TO_ID["fold"]
