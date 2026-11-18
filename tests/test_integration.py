from __future__ import annotations
import json
import os
import tempfile
import numpy as np
import pytest
from cfr_mtt_solver.equity import EquityModel
from cfr_mtt_solver.icm import ICMModel
from cfr_mtt_solver.ranges import get_opening_range
from cfr_mtt_solver.tree_builder import TreeBuilder
from cfr_mtt_solver.cfr_solver import CFRSolver
from cfr_mtt_solver.exporter import export_json


def _run_solver(positions, stack_bb, iterations=20, use_icm=False, payouts=None):
    builder = TreeBuilder(
        positions=positions,
        stack_bb=stack_bb,
        max_postflop_depth=0,  # preflop-only for speed in tests
        use_icm=use_icm,
    )
    root = builder.build_preflop_tree()
    equity_model = EquityModel()
    range_dists = {p: get_opening_range(p, stack_bb) for p in positions}
    icm_model = ICMModel(payouts) if (use_icm and payouts) else None
    solver = CFRSolver(
        root=root,
        equity_model=equity_model,
        range_dists=range_dists,
        icm_model=icm_model,
    )
    result = solver.solve(iterations, progress=False)
    return solver, result


def test_full_bu_bb_50bb():
    solver, result = _run_solver(["BU", "BB"], 50, iterations=20)
    assert result.iterations == 20
    assert len(result.infosets) > 0
    assert result.total_positive_regret() >= 0.0


def test_full_6max_tree_builds():
    """Verify 6-max tree builds correctly without running full CFR (too slow for CI)."""
    from cfr_mtt_solver.tree_builder import TreeBuilder
    positions = ["LJ", "HJ", "CO", "BU", "SB", "BB"]
    builder = TreeBuilder(positions=positions, stack_bb=50, max_postflop_depth=0)
    root = builder.build_preflop_tree()
    assert root.node_type == "decision"
    assert len(root.actions) > 0
    # Spot-check a child exists
    assert len(root.children) > 0


def test_10bb_short_stack():
    solver, result = _run_solver(["SB", "BB"], 10, iterations=20)
    assert result.iterations == 20


def test_icm_enabled():
    solver, result = _run_solver(
        ["BU", "BB"], 20, iterations=20,
        use_icm=True, payouts=[0.5, 0.3, 0.2]
    )
    assert len(result.infosets) > 0


def test_export_json():
    solver, _ = _run_solver(["BU", "BB"], 50, iterations=20)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        export_json(solver, path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) > 0
        for key, strat in data.items():
            assert isinstance(strat, dict)
            total = sum(strat.values())
            assert abs(total - 1.0) < 1e-4, f"Strategy at {key} sums to {total}"
    finally:
        os.unlink(path)


def test_convergence_decreases():
    """Total positive regret should generally decrease (not guaranteed every iter, but overall)."""
    solver, result = _run_solver(["BU", "BB"], 50, iterations=40)
    hist = result.convergence_history
    if len(hist) < 10:
        pytest.skip("Not enough iterations to test convergence trend")
    first_half = sum(hist[:len(hist)//2]) / (len(hist)//2)
    second_half = sum(hist[len(hist)//2:]) / (len(hist) - len(hist)//2)
    # Second half average should be <= first half average (convergence)
    assert second_half <= first_half * 1.1, (
        f"Regret not converging: first_half_avg={first_half:.4f}, second_half_avg={second_half:.4f}"
    )
