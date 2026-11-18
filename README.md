# CFR Poker Solver (6-max MTT)

A game-theory-optimal (GTO) preflop solver for 6-max tournament poker, built from scratch with
**Counterfactual Regret Minimization (CFR+)** over the full 1,326-combo hand space.

## Highlights
- **CFR+** with regret clamping and linearly weighted average strategy (`infoset.py`).
- **Vectorized traversal**: all 169 canonical hands are updated in a single tree pass
  (matrix form) instead of one traversal per hand: about **11× faster** than the scalar version.
- **Exact card removal**: 1,326 × 1,326 conflict matrix and per-combo ranges (`cards.py`, `combo_ranges.py`).
- **Precomputed equity tables** (preflop + 52 reachable postflop board clusters) built with `phevaluator`.
- **ICM** payout model for tournament spots (`icm.py`).
- Persistent, resumable training (`cli.py train`) and range queries (`cli.py query`).
- **Streamlit UI** with range grids, EV and convergence charts (`ui/`).

## Performance
| Game | Tree size | Time / iteration |
|---|---|---|
| 3-player (HJ, BU, BB) | 1,137 nodes | ~0.25 s |

## Quick start
```bash
pip install -r requirements.txt
python main.py train --help
python main.py query --help
streamlit run cfr_mtt_solver/ui/app.py
pytest tests
```

## Structure
```
cfr_mtt_solver/
  cards.py, combo_ranges.py   card encoding, 1326-combo ranges with card removal
  equity_engine.py            precomputed equity tables
  tree_builder.py, tree.py    multi-way preflop game tree
  infoset.py, cfr_solver.py   CFR+ and vectorized traversal
  icm.py                      tournament equity
  ui/                         Streamlit app
tests/                        pytest suite
```
