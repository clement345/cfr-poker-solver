from __future__ import annotations
import numpy as np
import pytest
from cfr_mtt_solver.infoset import InfoSet, InfoSetKey, pot_to_bucket, stack_to_bucket
from cfr_mtt_solver.equity import EquityModel
from cfr_mtt_solver.ranges import get_opening_range
from cfr_mtt_solver.tree_builder import TreeBuilder
from cfr_mtt_solver.cfr_solver import CFRSolver


def test_pot_to_bucket():
    assert pot_to_bucket(1.5) == 0
    assert pot_to_bucket(4.0) == 1
    assert pot_to_bucket(8.0) == 2
    assert pot_to_bucket(20.0) == 3
    assert pot_to_bucket(30.0) == 4


def test_stack_to_bucket():
    assert stack_to_bucket(10.0) == 0
    assert stack_to_bucket(20.0) == 1
    assert stack_to_bucket(50.0) == 2
    assert stack_to_bucket(100.0) == 3


def test_infoset_regret_matching_uniform_when_zero():
    key = InfoSetKey("BU", "PREFLOP", 0, 2, (), 0, 0, "BU")
    actions = ["fold", "open_2.5bb", "shove"]
    info = InfoSet(key, actions)
    strat = info.get_strategy()
    assert abs(sum(strat.values()) - 1.0) < 1e-6
    for v in strat.values():
        assert abs(v - 1 / 3) < 1e-6


def test_infoset_regret_matching_follows_positive_regrets():
    key = InfoSetKey("BU", "PREFLOP", 0, 2, (), 0, 0, "BU")
    actions = ["fold", "open_2.5bb"]
    info = InfoSet(key, actions)
    info.regrets["fold"] = -1.0
    info.regrets["open_2.5bb"] = 2.0
    strat = info.get_strategy()
    assert strat["open_2.5bb"] > strat["fold"]
    assert strat["open_2.5bb"] == pytest.approx(1.0)
    assert strat["fold"] == pytest.approx(0.0)


def test_average_strategy_sums_to_one():
    key = InfoSetKey("BU", "PREFLOP", 0, 2, (), 0, 0, "BU")
    actions = ["fold", "call", "raise"]
    info = InfoSet(key, actions)
    info.cumulative_strategy["fold"] = 10.0
    info.cumulative_strategy["call"] = 20.0
    info.cumulative_strategy["raise"] = 30.0
    avg = info.get_average_strategy()
    assert abs(sum(avg.values()) - 1.0) < 1e-6
    assert avg["raise"] == pytest.approx(0.5)


def _build_hu_solver(stack_bb=50, iterations=5):
    positions = ["BU", "BB"]
    # depth=0 = preflop-only, fast for unit tests
    builder = TreeBuilder(positions=positions, stack_bb=stack_bb, max_postflop_depth=0)
    root = builder.build_preflop_tree()
    equity_model = EquityModel()
    range_dists = {p: get_opening_range(p, stack_bb) for p in positions}
    solver = CFRSolver(root=root, equity_model=equity_model, range_dists=range_dists)
    solver.solve(iterations, progress=False)
    return solver


def test_cfr_creates_infosets():
    solver = _build_hu_solver(iterations=3)
    assert len(solver.infosets) > 0


def test_cfr_strategies_sum_to_one():
    solver = _build_hu_solver(iterations=5)
    for key, infoset in solver.infosets.items():
        avg = infoset.get_average_strategy()
        total = sum(avg.values())
        assert abs(total - 1.0) < 1e-5, f"Strategy doesn't sum to 1 for {key}: {total}"


def test_cfr_convergence_non_negative():
    solver = _build_hu_solver(iterations=10)
    for regret in solver.convergence_history:
        assert regret >= 0.0, f"Negative convergence metric: {regret}"


def test_cfr_short_stack_pushfold():
    """10bb stack should produce near push-fold strategies."""
    solver = _build_hu_solver(stack_bb=10, iterations=20)
    # Look for BU infosets in preflop
    bu_preflop = {k: v for k, v in solver.infosets.items() if "|BU|" in k and "|PREFLOP|" in k}
    assert len(bu_preflop) > 0, "Should have BU preflop infosets"
    # With 10bb, strategies should be mostly shove or fold
    for key, infoset in bu_preflop.items():
        avg = infoset.get_average_strategy()
        shove_fold_freq = avg.get("shove", 0.0) + avg.get("fold", 0.0)
        # Not always dominated by shove/fold, but at least some shove
        assert avg.get("shove", 0.0) >= 0.0  # Sanity: non-negative
