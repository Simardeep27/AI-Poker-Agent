"""Train MaskablePPO on the local heads-up poker environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from ppo_setup.evaluate import evaluate_model, print_summary
from ppo_setup.env import HeadsUpPokerEnv
from ppo_setup.leaderboard import (
    LEADERBOARD_ANTE,
    LEADERBOARD_EQUITY_SIMS,
    LEADERBOARD_GAMES,
    LEADERBOARD_INITIAL_STACK,
    LEADERBOARD_OPPONENTS,
    LEADERBOARD_ROUNDS,
    LEADERBOARD_SMALL_BLIND,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--opponents", default=LEADERBOARD_OPPONENTS)
    parser.add_argument("--rounds", type=int, default=LEADERBOARD_ROUNDS)
    parser.add_argument("--initial-stack", type=int, default=LEADERBOARD_INITIAL_STACK)
    parser.add_argument("--small-blind", type=int, default=LEADERBOARD_SMALL_BLIND)
    parser.add_argument("--ante", type=int, default=LEADERBOARD_ANTE)
    parser.add_argument("--equity-sims", type=int, default=LEADERBOARD_EQUITY_SIMS)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="ppo_setup/runs/leaderboard")
    parser.add_argument("--resume", default="")
    parser.add_argument("--reset-num-timesteps", action="store_true")
    parser.add_argument("--tb-log-name", default="maskable_ppo_poker")
    parser.add_argument("--bc-examples", type=int, default=0)
    parser.add_argument("--bc-experts", default="opponent,custom")
    parser.add_argument("--bc-opponents", default="")
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-progress-every", type=int, default=1000)
    parser.add_argument("--eval-every-timesteps", type=int, default=0)
    parser.add_argument("--eval-opponent", default="diverse")
    parser.add_argument("--eval-games", type=int, default=LEADERBOARD_GAMES)
    parser.add_argument("--eval-rounds", type=int, default=LEADERBOARD_ROUNDS)
    parser.add_argument("--eval-equity-sims", type=int, default=-1)
    parser.add_argument("--eval-seed", type=int, default=1001)
    parser.add_argument("--eval-progress-every", type=int, default=0)
    parser.add_argument("--eval-stochastic", action="store_true")
    parser.add_argument("--no-save-best-eval", action="store_true")
    return parser.parse_args()


def make_env(args: argparse.Namespace, rank: int):
    opponent_pool = [name.strip() for name in args.opponents.split(",") if name.strip()]

    def _init():
        env = HeadsUpPokerEnv(
            opponent_pool=opponent_pool,
            max_rounds=args.rounds,
            initial_stack=args.initial_stack,
            small_blind=args.small_blind,
            ante=args.ante,
            equity_sims=args.equity_sims,
        )
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main() -> None:
    args = parse_args()

    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from sb3_contrib import MaskablePPO

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    env = vec_cls([make_env(args, idx) for idx in range(args.n_envs)])

    policy_kwargs = {
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256],
        }
    }
    if args.resume:
        model = MaskablePPO.load(
            args.resume,
            env=env,
            device=args.device,
            tensorboard_log=str(out_dir / "tensorboard"),
        )
        print("Resuming from %s at num_timesteps=%d" % (args.resume, model.num_timesteps))
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(out_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    if args.bc_examples > 0:
        run_behavior_cloning_warm_start(model, args, out_dir)

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.n_steps * args.n_envs * 10, 10_000),
            save_path=str(checkpoint_dir),
            name_prefix="maskable_ppo_poker",
        )
    ]
    if args.eval_every_timesteps > 0:
        callbacks.append(make_periodic_eval_callback(args, out_dir))
    callback = CallbackList(callbacks)
    reset_num_timesteps = args.reset_num_timesteps or not bool(args.resume)
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=args.tb_log_name,
    )
    model.save(str(out_dir / "final_model"))
    env.close()
    print("Saved model to %s" % (out_dir / "final_model.zip"))


def run_behavior_cloning_warm_start(model, args: argparse.Namespace, out_dir: Path) -> None:
    from ppo_setup.behavior_cloning import behavior_clone_model, collect_expert_examples

    bc_opponents = args.bc_opponents or args.opponents
    print(
        "[bc] collecting %d expert examples from experts=%s against opponents=%s"
        % (args.bc_examples, args.bc_experts, bc_opponents),
        flush=True,
    )
    dataset = collect_expert_examples(
        expert_pool=args.bc_experts,
        opponent_pool=bc_opponents,
        examples=args.bc_examples,
        rounds=args.rounds,
        initial_stack=args.initial_stack,
        small_blind=args.small_blind,
        ante=args.ante,
        equity_sims=args.equity_sims,
        seed=args.seed + 50_000,
        progress_every=args.bc_progress_every,
    )
    behavior_clone_model(
        model,
        dataset=dataset,
        epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        learning_rate=args.bc_learning_rate,
    )
    warm_path = out_dir / "bc_warm_start"
    model.save(str(warm_path))
    print("[bc] saved warm-start policy to %s.zip" % warm_path, flush=True)


def make_periodic_eval_callback(args: argparse.Namespace, out_dir: Path):
    """Build an SB3 callback without importing SB3 at module import time."""
    from stable_baselines3.common.callbacks import BaseCallback

    class PeriodicPokerEvalCallback(BaseCallback):
        """Evaluate the live policy against a fixed poker opponent during training."""

        def __init__(self):
            super().__init__(verbose=0)
            self.best_bb_per_100 = float("-inf")
            self.next_eval_at = args.eval_every_timesteps

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_eval_at:
                return True
            while self.next_eval_at <= self.num_timesteps:
                self.next_eval_at += args.eval_every_timesteps

            print("")
            print("Periodic eval at %d training timesteps" % self.num_timesteps, flush=True)
            eval_equity_sims = args.equity_sims if args.eval_equity_sims < 0 else args.eval_equity_sims
            results = evaluate_model(
                self.model,
                opponent=args.eval_opponent,
                games=args.eval_games,
                rounds=args.eval_rounds,
                initial_stack=args.initial_stack,
                small_blind=args.small_blind,
                ante=args.ante,
                equity_sims=eval_equity_sims,
                seed=args.eval_seed + self.num_timesteps,
                stochastic=args.eval_stochastic,
                progress_every=args.eval_progress_every,
                progress_prefix="[eval] ",
            )
            print_summary(results, prefix="[eval] ")
            metric_prefix = "eval/%s" % args.eval_opponent
            self.logger.record("%s_bb_per_100" % metric_prefix, results["bb_per_100"])
            self.logger.record("%s_avg_gain_game" % metric_prefix, results["avg_gain_game"])
            self.logger.record("%s_win_rate" % metric_prefix, results["win_rate"])
            self.logger.record("%s_ci95_margin_game_gain" % metric_prefix, results["ci95_margin_game_gain"])

            if not args.no_save_best_eval and results["bb_per_100"] > self.best_bb_per_100:
                self.best_bb_per_100 = results["bb_per_100"]
                best_path = out_dir / "best_eval_model"
                self.model.save(str(best_path))
                print("[eval] New best BB/100 %.4f; saved %s.zip" % (self.best_bb_per_100, best_path), flush=True)
            return True

    return PeriodicPokerEvalCallback()


if __name__ == "__main__":
    main()
