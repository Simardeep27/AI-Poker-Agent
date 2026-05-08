"""Evaluate a trained MaskablePPO policy against a fixed opponent."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from ppo_setup.env import HeadsUpPokerEnv
from ppo_setup.leaderboard import (
    LEADERBOARD_ANTE,
    LEADERBOARD_EQUITY_SIMS,
    LEADERBOARD_GAMES,
    LEADERBOARD_INITIAL_STACK,
    LEADERBOARD_ROUNDS,
    LEADERBOARD_SMALL_BLIND,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--opponent", default="diverse")
    parser.add_argument("--games", type=int, default=LEADERBOARD_GAMES)
    parser.add_argument("--rounds", type=int, default=LEADERBOARD_ROUNDS)
    parser.add_argument("--initial-stack", type=int, default=LEADERBOARD_INITIAL_STACK)
    parser.add_argument("--small-blind", type=int, default=LEADERBOARD_SMALL_BLIND)
    parser.add_argument("--ante", type=int, default=LEADERBOARD_ANTE)
    parser.add_argument("--equity-sims", type=int, default=LEADERBOARD_EQUITY_SIMS)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(args.model, device="auto")
    results = evaluate_model(
        model,
        opponent=args.opponent,
        games=args.games,
        rounds=args.rounds,
        initial_stack=args.initial_stack,
        small_blind=args.small_blind,
        ante=args.ante,
        equity_sims=args.equity_sims,
        seed=args.seed,
        stochastic=args.stochastic,
        progress_every=args.progress_every,
    )
    print_summary(results)


def evaluate_model(
    model,
    opponent: str = "diverse",
    games: int = 100,
    rounds: int = 100,
    initial_stack: int = 10000,
    small_blind: int = 20,
    ante: int = 0,
    equity_sims: int = 32,
    seed: int = 101,
    stochastic: bool = False,
    progress_every: int = 0,
    progress_prefix: str = "",
) -> dict:
    game_gains: list[float] = []
    game_rewards: list[float] = []
    wins = 0
    start_time = time.perf_counter()

    for idx in range(games):
        env = HeadsUpPokerEnv(
            opponent_pool=opponent,
            max_rounds=rounds,
            initial_stack=initial_stack,
            small_blind=small_blind,
            ante=ante,
            equity_sims=equity_sims,
            randomize_position=bool(idx % 2),
        )
        obs, info = env.reset(seed=seed + idx)
        terminated = False
        truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action_masks = env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=not stochastic,
                action_masks=action_masks,
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
        hero_stack = info["hero_stack"]
        villain_stack = info["opponent_stack"]
        game_gain = hero_stack - initial_stack
        game_gains.append(game_gain)
        game_rewards.append(total_reward)
        wins += int(hero_stack > villain_stack)
        env.close()
        if progress_every > 0 and ((idx + 1) % progress_every == 0 or idx + 1 == games):
            print_progress(
                completed=idx + 1,
                games=games,
                rounds=rounds,
                small_blind=small_blind,
                game_gains=game_gains,
                wins=wins,
                start_time=start_time,
                prefix=progress_prefix,
            )

    total_hands = games * rounds
    big_blind = small_blind * 2
    total_delta = sum(game_gains)
    mean_gain = _mean(game_gains)
    std = _sample_std(game_gains)
    ci = 1.96 * std / math.sqrt(max(len(game_gains), 1)) if len(game_gains) > 1 else 0.0
    return {
        "opponent": opponent,
        "games": games,
        "rounds": rounds,
        "total_chip_delta": total_delta,
        "avg_gain_game": mean_gain,
        "win_rate": 100.0 * wins / max(games, 1),
        "bb_per_100": (total_delta / big_blind) * 100.0 / max(total_hands, 1),
        "reward_mean_game": _mean(game_rewards),
        "ci95_low_game_gain": mean_gain - ci,
        "ci95_high_game_gain": mean_gain + ci,
        "ci95_margin_game_gain": ci,
        "elapsed_sec": time.perf_counter() - start_time,
    }


def print_summary(results: dict, prefix: str = "") -> None:
    print("%sPPO vs %s" % (prefix, results["opponent"]))
    print("%sGames: %d, rounds/game: %d" % (prefix, results["games"], results["rounds"]))
    print("%sTotal chip delta: %.2f" % (prefix, results["total_chip_delta"]))
    print("%sAvg gain/game: %.2f" % (prefix, results["avg_gain_game"]))
    print("%sWin rate: %.2f%%" % (prefix, results["win_rate"]))
    print("%sBB/100: %.4f" % (prefix, results["bb_per_100"]))
    print("%sReward mean/game: %.4f" % (prefix, results["reward_mean_game"]))
    print("%s95%% CI gain/game: %.2f to %.2f" % (prefix, results["ci95_low_game_gain"], results["ci95_high_game_gain"]))


def print_progress(
    completed: int,
    games: int,
    rounds: int,
    small_blind: int,
    game_gains: list[float],
    wins: int,
    start_time: float,
    prefix: str = "",
) -> None:
    elapsed = time.perf_counter() - start_time
    per_game = elapsed / max(completed, 1)
    remaining = per_game * max(games - completed, 0)
    total_delta = sum(game_gains)
    total_hands = completed * rounds
    big_blind = small_blind * 2
    bb_per_100 = (total_delta / big_blind) * 100.0 / max(total_hands, 1)
    print(
        "%s[%d/%d] elapsed=%s eta=%s avg_gain=%.2f win_rate=%.2f%% bb/100=%.4f"
        % (
            prefix,
            completed,
            games,
            _format_duration(elapsed),
            _format_duration(remaining),
            _mean(game_gains),
            100.0 * wins / max(completed, 1),
            bb_per_100,
        ),
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "%dh%02dm%02ds" % (hours, minutes, sec)
    return "%dm%02ds" % (minutes, sec)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


if __name__ == "__main__":
    main()
