"""Precompute equity lookup tables.

Usage:
    python -m cfr_mtt_solver.precompute_tables [--preflop] [--postflop] [--samples N]

Both tables are written to tables/ (gitignored).
  preflop_equity.npy   ~110 KB  (169x169 float32)
  postflop_equity.npy  ~17 MB   (52x169x169 float32)
  cluster_index.npy    ~144 B   (72 int16)
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from .equity_engine import precompute_preflop, precompute_postflop

_TABLES_DIR = Path(__file__).parent.parent / "tables"


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute equity tables for CFR solver")
    parser.add_argument("--preflop", action="store_true", default=False)
    parser.add_argument("--postflop", action="store_true", default=False)
    parser.add_argument("--all", dest="all_tables", action="store_true", default=False)
    parser.add_argument("--samples", type=int, default=2000,
                        help="MC samples per hand pair (preflop). Postflop uses samples/10.")
    args = parser.parse_args()

    if args.all_tables or not (args.preflop or args.postflop):
        args.preflop = True
        args.postflop = True

    _TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.preflop:
        print(f"Computing preflop equity table ({args.samples} samples/pair)...")
        t0 = time.time()
        precompute_preflop(n_samples=args.samples)
        print(f"  Done in {time.time()-t0:.1f}s")

    if args.postflop:
        n_runouts = max(50, args.samples // 10)
        print(f"Computing postflop equity table ({n_runouts} runouts/cluster)...")
        t0 = time.time()
        precompute_postflop(n_runouts=n_runouts)
        print(f"  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
