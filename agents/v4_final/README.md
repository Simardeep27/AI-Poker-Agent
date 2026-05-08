# Final Agent

This is the readable modular version of the final submitted poker agent.

## Files

- `custom_player.py`: evaluator-facing entrypoint.
- `player.py`: `MyPlayer` class and PyPokerEngine callbacks.
- `state.py`: raw round-state parsing and betting-line abstraction.
- `features.py`: hand equity, range-aware correction, draw potential, board texture, and made-hand features.
- `opponent.py`: online opponent action tracking and response-prior estimation.
- `decision.py`: sampled EV search, leaf correction, semi-bluff logic, and river/reraise discipline.
- `constants.py`: tuned weights and simulation budgets.
- `cards.py`: card helpers and preflop strength approximation.

The root-level `custom_player_4.py` is retained as the single-file compatibility version.
