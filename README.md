# Poker Agent

Opponent-aware poker bot built on top of PyPokerEngine. The custom player combines
lightweight online opponent modeling, state abstraction, sampled lookahead search,
and a weighted leaf evaluator.

The main implementation lives in `AI-Poker-Agent/opponent_aware/`, with
`AI-Poker-Agent/opponent_aware_player.py` exposing the `BasePokerPlayer` entry
point used by the demo and benchmark scripts.

## Quick Start

Run a one-hand debug demo:

```bash
cd AI-Poker-Agent
python example.py
```

Run a reproducible benchmark against a baseline player:

```bash
cd AI-Poker-Agent
python benchmark.py --hero opponent --villain random --runs 3 --games 10 --rounds 100
```

Available benchmark players are:

- `custom`: the `custom_player.py` agent
- `opponent`: the custom opponent-aware search bot
- `random`: random baseline player
- `raise`: simple raise-heavy baseline player

## Custom Model Stats and Metrics

### Online Opponent Profile

`OpponentProfileTracker` updates after every observed action and keeps a smoothed
profile for the opponent. Ratios use a four-observation prior:

```text
smoothed_ratio = (observed_count + prior * 4) / (opportunities + 4)
```

| Metric | Definition | Default prior | Used for |
| --- | --- | ---: | --- |
| `preflop_raise_frequency` | Hands where opponent raised preflop / hands observed | `0.32` | Loose/tight style label and raise response prior |
| `aggression` | Raises / (raises + calls) | `0.42` | Opponent raise probability and exploitability adjustment |
| `fold_to_raise` | Folds to our raises / opportunities after our raise | `0.40` | Fold equity and opponent fold response prior |
| `showdown_willingness` | Showdowns reached / hands observed | `0.35` | Call/fold response prior |

The profile also emits a style label:

- `tight` when PFR `< 0.30`, otherwise `loose`
- `aggressive` when aggression `>= 0.45`, otherwise `passive`

### Search Configuration

The default action search is intentionally small enough for live play:

| Setting | Value | Meaning |
| --- | ---: | --- |
| `time_budget_sec` | `0.35` | Per-decision search budget |
| `root_samples` | `6` | Max sampled rollouts per root action |
| `max_depth` | `2` | Lookahead depth after the root action |
| `equity_samples` | `12` | Monte Carlo board completions for leaf equity |

Search returns per-action metrics in debug mode:

- `score`: average simulated value for the action
- `samples`: number of sampled rollouts used
- `source`: `search`, `fallback`, or `forced`
- `chosen_action`: best-scoring legal action
- `fallback_action`: rule-based action used when search cannot sample enough

### Leaf Evaluation Features

Leaf states are scored as:

```text
realized_stack_delta + pot_amount * weighted_feature_score - call_liability
```

`call_liability` increases when the bot must call with equity below `0.55`.

| Feature | Weight | Description |
| --- | ---: | --- |
| `equity` | `0.62` | Estimated win/tie equity against sampled opponent hole cards |
| `fold_equity` | `0.18` | Current opponent `fold_to_raise` profile value |
| `pot_pressure` | `0.12` | Pot / (pot + effective stack) |
| `showdown_value` | `0.16` | Better of made-hand score and `0.90 * equity` |
| `opponent_exploitability` | `0.08` | `fold_to_raise - 0.5 * aggression`, floored at zero |
| `position` | `0.04` | Button/in-position bonus |
| `commitment` | `0.05` | Low/medium/high commitment score |
| `board_danger` | `-0.14` | Penalty for wet or coordinated boards |

### State Abstraction Metrics

The model compresses each decision into poker-specific buckets:

| Abstraction | Metric |
| --- | --- |
| Hole-card strength | `trash` starts at `0.30`; premium pairs score up to `0.95` |
| Made hand | High card through straight flush, scored from `0.15` to `1.00` |
| Board texture | `dry`, `semi_coordinated`, or `wet` from flush, straight, and paired-board danger |
| Pot size | `small` when pot/effective stack `< 0.25`, `medium` when `< 0.75`, otherwise `large` |
| Commitment | `low` when pot share `< 0.18`, `medium` when `< 0.35`, otherwise `high` |
| Betting history | `passive_line`, `single_raise_pressure`, `reraise_line`, or `mixed_line` |

## Benchmark Metrics

`benchmark.py` prints the comparison metrics directly and can append them to a
CSV file for tracking changes over time:

```bash
python benchmark.py --hero opponent --villain random --runs 3 --games 20 --rounds 100 --csv ../benchmark_metrics.csv --label baseline
```

Compare the current opponent-aware agent against `custom_player.py`:

```bash
python benchmark.py --hero opponent --villain custom --runs 3 --games 20 --rounds 100 --csv ../benchmark_metrics.csv --label opponent_vs_custom
```

Use chip delta as the core result. A single game gain is the bot's profit
relative to its starting stack:

```text
game_gain = hero_final_stack - initial_stack
```

Report custom model runs with these summary columns:

| Metric | Definition |
| --- | --- |
| `Avg Gain / Game` | Mean `game_gain` across all games in the benchmark |
| `Avg Gain / Run` | Mean total chip gain across repeated benchmark runs |
| `Best Game Gain` | Highest single-game `game_gain` |
| `Worst Game Gain` | Lowest single-game `game_gain` |

Also track the following evaluation metrics:

| Metric | What it measures | Use for |
| --- | --- | --- |
| **Average chip delta / hand** | Expected profit per hand | Main performance metric |
| **Average chip delta / game** | Total profit across many rounds | Easy README metric |
| **Win rate** | Percent of games where final stack is greater than opponent stack | Simple comparison |
| **BB/100** | Big blinds won per 100 hands | Poker-standard metric |
| **Std dev of returns** | Variance/risk | Stability |
| **95% confidence interval** | Reliability of result | Whether gains are meaningful |

Useful formulas:

```text
average_chip_delta_per_hand = total_chip_delta / total_hands
average_chip_delta_per_game = total_chip_delta / total_games
win_rate = winning_games / total_games * 100
BB_per_100 = (total_chip_delta / big_blind) / total_hands * 100
std_dev_of_returns = sample_standard_deviation(game_gain)
95_percent_CI = mean(game_gain) +/- 1.96 * std_dev_of_returns / sqrt(total_games)
```

The script treats the big blind as `2 * --blind`, because PyPokerEngine's
`small_blind_amount` is passed through the `--blind` argument.

The benchmark alternates seat position each game to reduce button-order bias.

For stronger evidence, increase `--games` and compare against both `random` and
`raise`.

## Debug Trace

Enable debug mode with `OpponentAwarePlayer(debug=True)` to print:

- round start, stacks, hole cards, street, board, and pot
- current abstraction buckets
- opponent profile metrics
- search budget and sample counts
- root action scores
- selected action and showdown details
