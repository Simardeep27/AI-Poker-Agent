"""Tournament-facing defaults inferred from the CS683 leaderboard."""

from __future__ import annotations


# The public leaderboard reports best/worst gains at +/-1000 chips and
# Avg Gain / Run == Avg Gain / Game * 32, so these are the useful local
# defaults for tournament-style training and evaluation.
LEADERBOARD_INITIAL_STACK = 1000
LEADERBOARD_SMALL_BLIND = 10
LEADERBOARD_ANTE = 0
LEADERBOARD_ROUNDS = 100
LEADERBOARD_GAMES = 32
LEADERBOARD_EQUITY_SIMS = 32

LEADERBOARD_OPPONENTS = "diverse,snapshots:ppo_setup/runs/leaderboard/checkpoints"
