"""Small online opponent profile used as PPO observation context."""

from __future__ import annotations

from dataclasses import dataclass, field


STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 3}


@dataclass
class OpponentStats:
    preflop_raises: int = 0
    preflop_actions: int = 0
    fold_to_raise: int = 0
    raises_we_made: int = 0
    street_raises: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    street_total: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    showdowns: int = 0
    hands_total: int = 0
    just_raised: bool = False

    def start_round(self) -> None:
        self.hands_total += 1
        self.just_raised = False

    def observe_action(self, action: dict, round_state: dict, hero_uuid: str) -> None:
        actor = action.get("player_uuid", "")
        name = action.get("action", "").lower()
        if actor == hero_uuid:
            self.just_raised = name == "raise"
            return

        street = round_state.get("street", "preflop")
        street_num = STREETS.get(street, 0)
        self.street_total[street_num] += 1
        if name == "raise":
            self.street_raises[street_num] += 1

        if street_num == 0 and name in ("raise", "call", "fold"):
            self.preflop_actions += 1
            if name == "raise":
                self.preflop_raises += 1

        if self.just_raised:
            self.raises_we_made += 1
            if name == "fold":
                self.fold_to_raise += 1
        self.just_raised = False

    def observe_round_result(self, hand_info: list | None) -> None:
        if hand_info:
            self.showdowns += 1
        self.just_raised = False

    def snapshot(self) -> dict:
        total_actions = sum(self.street_total)
        total_raises = sum(self.street_raises)
        pfr = _smooth(self.preflop_raises, self.preflop_actions, 0.32)
        aggression = _smooth(total_raises, total_actions, 0.42)
        fold_to_raise = _smooth(self.fold_to_raise, self.raises_we_made, 0.40)
        showdown = _smooth(self.showdowns, self.hands_total, 0.35)
        return {
            "preflop_raise_frequency": pfr,
            "aggression": aggression,
            "fold_to_raise": fold_to_raise,
            "showdown_willingness": showdown,
            "hands_total": self.hands_total,
        }


def _smooth(successes: int, opportunities: int, prior: float, strength: int = 4) -> float:
    return float(successes + prior * strength) / float(opportunities + strength)

