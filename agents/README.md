# Agent Versions

Each directory stores one playable agent version with a `custom_player.py` entrypoint.
The directories are organized chronologically so experiments can be reproduced without
renaming root-level files.

| Directory | Role |
| --- | --- |
| `v1_ev_baseline/` | First EV-oriented heuristic baseline. |
| `ppo_trained/` | Exported PPO policy trained with behavior cloning warm start and self-play-style opponents. |
| `v3_range_aware/` | Intermediate heuristic version with stronger range and board features. |
| `v4_final/` | Final selected agent: sampled EV search, opponent model, range-aware correction, and conservative river logic. |
| `v5_experiment/` | Experimental post-v4 variant. |
| `v6_mixed_strategy/` | Mixed-strategy/deception experiment that was not selected for final submission. |

The final official submission should still be packaged as a single `custom_player.py`
because that is the format expected by the course evaluator.

The optimal one is v4_final/