# PPO Setup

This directory contains a Colab-ready MaskablePPO setup for the local heads-up
PyPokerEngine bot. Defaults now target the CS683 leaderboard scale:

- `initial_stack=1000`
- `small_blind=10`
- `rounds=100`
- `eval_games=32`

The PPO setup uses:

- observation vectors from poker features already used by `custom_player.py`
- three discrete actions: `fold`, `call`, `raise`
- legal-action masking through `sb3-contrib` `MaskablePPO`
- end-of-hand reward: hero chip delta measured in big blinds
- diverse opponents: `custom`, `opponent`, submitted checkpoint export,
  `random`, `raise`, `call`, `fold`, `tight`, `loose`, and `maniac`
- checkpoint opponents through `ppo:<path>` and dynamic snapshot self-play through
  `snapshots:<directory>`

## Colab Install

From the repo root in Colab:

```bash
pip install -r ppo_setup/requirements-colab.txt
```

Run the training command from the folder that contains both `ppo_setup/` and
`AI-Poker-Agent/`. The loader expects to find:

```text
ppo_setup/
AI-Poker-Agent/pypokerengine/
AI-Poker-Agent/custom_player.py
AI-Poker-Agent/randomplayer.py
AI-Poker-Agent/raise_player.py
```

If your best bot lives as `custom_player.py` at the repo root instead, that is
also supported.

Colab A100 runtimes usually already include CUDA PyTorch. If your runtime does
not, install the matching PyTorch build from pytorch.org before installing the
requirements above.

## Smoke Test

```bash
python -m ppo_setup.train \
  --timesteps 5000 \
  --n-envs 1 \
  --rounds 20 \
  --equity-sims 0 \
  --out-dir ppo_setup/runs/smoke
```

This only checks that the environment, masks, and model loop work.

## First Real Run

```bash
python -m ppo_setup.train \
  --timesteps 1000000 \
  --n-envs 4 \
  --opponents diverse,snapshots:ppo_setup/runs/leaderboard/checkpoints \
  --device cuda \
  --out-dir ppo_setup/runs/leaderboard
```

The `snapshots:` entry is re-expanded whenever a new opponent is sampled, so
new checkpoint files from the same run become self-play opponents automatically.

## Behavior-Cloning Warm Start

Warm-start the policy by imitating the strongest search/heuristic bots, then
continue PPO:

```bash
python -m ppo_setup.train \
  --bc-examples 50000 \
  --bc-experts opponent,custom \
  --bc-opponents diverse,snapshots:AI-Poker-Agent \
  --bc-epochs 4 \
  --timesteps 1500000 \
  --opponents diverse,snapshots:ppo_setup/runs/leaderboard/checkpoints,snapshots:AI-Poker-Agent \
  --eval-every-timesteps 100000 \
  --eval-opponent diverse \
  --eval-progress-every 8 \
  --device cuda \
  --out-dir ppo_setup/runs/leaderboard
```

To evaluate the live checkpoint during training, add:

```bash
python -m ppo_setup.train \
  --timesteps 1000000 \
  --n-envs 4 \
  --opponents diverse,snapshots:ppo_setup/runs/leaderboard/checkpoints \
  --device cuda \
  --out-dir ppo_setup/runs/leaderboard \
  --eval-every-timesteps 100000 \
  --eval-opponent diverse \
  --eval-progress-every 8
```

`--eval-every-timesteps` is measured in PPO training timesteps. Each eval runs
the current policy against `--eval-opponent` for `--eval-games` games of
`--eval-rounds` poker hands. The best eval model is saved as
`best_eval_model.zip` unless `--no-save-best-eval` is set.

The engine simulation is CPU-bound, so the A100 helps the network updates more
than the card game rollout. If rollout speed is poor, lower `--equity-sims` or
start with `--equity-sims 0`.

## Evaluate

```bash
python -m ppo_setup.evaluate \
  --model ppo_setup/runs/leaderboard/final_model.zip \
  --opponent diverse
```

Also evaluate against the easy baselines:

```bash
python -m ppo_setup.evaluate --model ppo_setup/runs/leaderboard/final_model.zip --opponent random
python -m ppo_setup.evaluate --model ppo_setup/runs/leaderboard/final_model.zip --opponent raise
python -m ppo_setup.evaluate --model ppo_setup/runs/leaderboard/final_model.zip --opponent opponent
python -m ppo_setup.evaluate --model ppo_setup/runs/leaderboard/final_model.zip --opponent submitted
```

## Export a Leaderboard Submission

For normal local eval, the checkpoint is loaded by `ppo_setup.evaluate` and the
PPO policy chooses actions from the encoded 71-feature observation plus the legal
action mask.

For a leaderboard that expects one `custom_player.py` file, export the actor
network into a standalone player:

```bash
python -m ppo_setup.export_leaderboard_player \
  --model ppo_setup/runs/leaderboard/best_eval_model.zip \
  --output cs683_group13/submission/custom_player.py \
  --zip-output cs683_group13.zip \
  --zip-member submission/custom_player.py \
  --force
```

The generated file embeds the trained policy weights and includes the same
feature encoder and opponent-stat tracker used during training. It does not need
`stable-baselines3` or `sb3-contrib` at eval time; it only needs `numpy` and the
competition's `pypokerengine`. The CS683 site asks for a ZIP, so upload
`cs683_group13.zip` and choose `Group 13` in the form.

## Recommended Progression

1. Run the smoke test.
2. Run `--bc-examples 5000 --timesteps 20000` to verify BC and PPO glue.
3. Scale BC to `50k+` examples from `opponent,custom`.
4. Train PPO against `diverse,snapshots:<run>/checkpoints`.
5. Evaluate against `diverse`, then each fixed opponent one by one.
6. Export the best eval checkpoint into `cs683_group13.zip`.

## Files

- `env.py`: Gymnasium environment around PyPokerEngine
- `feature_encoder.py`: fixed observation vector and action masks
- `train.py`: MaskablePPO training entry point
- `evaluate.py`: deterministic policy evaluation
- `behavior_cloning.py`: supervised warm-start from expert player decisions
- `leaderboard.py`: tournament-scale defaults
- `export_leaderboard_player.py`: one-file NumPy player and CS683 ZIP export
- `ppo_player.py`: optional `BasePokerPlayer` wrapper for a trained model
- `opponents.py`: opponent pool factories
