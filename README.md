# Poker Agent Project

This repository contains the final poker agent, earlier agent versions, PPO training code,
and benchmark/report artifacts for the COMPSCI 683 final project.

## Main Submission

The course-compatible submission file is:

- `cs683_group13/submission/custom_player.py`

The final development version is also kept at:

- `custom_player_4.py`
- `agents/v4_final/`

`custom_player_4.py` is the safest file to submit when the evaluator expects a single
`custom_player.py`. The modular version in `agents/v4_final/` contains the same final
pipeline split into smaller files for readability.

## Agent Versions

- `agents/v1_ev_baseline/`: initial EV/heuristic baseline.
- `agents/ppo_trained/`: exported PPO-trained policy.
- `agents/v3_range_aware/`: intermediate range-aware heuristic agent.
- `agents/v4_final/`: final sampled-EV, opponent-aware agent.
- `agents/v5_experiment/`: later experimental variant.
- `agents/v6_mixed_strategy/`: mixed-strategy/deception experiment.

Root-level `custom_player*.py` files are retained as compatibility snapshots.

## Experiments

Benchmark and plotting scripts live in `scripts/`. Generated report assets are in
`report_assets/`, including comparison plots, tables, and PPO training diagnostics.

To regenerate the agent comparison benchmark:

```bash
python3 scripts/benchmark_agent_versions.py --runs 3 --games 10 --rounds 100
```

This writes:

- `report_assets/agent_benchmarks/agent_version_benchmark.csv`
- `report_assets/agent_benchmarks/agent_version_table.tex`
- `report_assets/agent_benchmarks/agent_version_compact_table.tex`
- `report_assets/agent_benchmarks/agent_version_comparison.png`

To regenerate PPO training plots:

```bash
python3 scripts/plot_training_metrics.py
```
