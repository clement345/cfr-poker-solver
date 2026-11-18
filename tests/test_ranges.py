from __future__ import annotations
import numpy as np
import pytest
from cfr_mtt_solver.ranges import get_opening_range, get_calling_range, get_3bet_range
from cfr_mtt_solver.hands import HAND_INDEX, N_HANDS
from cfr_mtt_solver.constants import POSITION_ORDER


def test_opening_range_sums_to_one():
    for pos in [p.value for p in POSITION_ORDER]:
        for stack in [10, 20, 50, 100]:
            r = get_opening_range(pos, stack)
            assert abs(r.sum() - 1.0) < 1e-6, f"{pos} {stack}bb sums to {r.sum()}"


def test_opening_range_shape():
    r = get_opening_range("BU", 50)
    assert r.shape == (169,)


def test_bu_opens_wider_than_lj():
    bu = get_opening_range("BU", 50)
    lj = get_opening_range("LJ", 50)
    # LJ is tighter: concentrates more probability on AA (strongest hand)
    aa_idx = HAND_INDEX['AA']
    assert lj[aa_idx] > bu[aa_idx], "LJ should give AA more relative weight than BU"


def test_premium_pairs_always_open():
    for pos in ['LJ', 'HJ', 'CO', 'BU', 'SB', 'BB']:
        for stack in [10, 20, 50, 100]:
            r = get_opening_range(pos, stack)
            aa_idx = HAND_INDEX['AA']
            assert r[aa_idx] > 0.0, f"AA should always open from {pos} {stack}bb"


def test_trash_has_zero_from_lj():
    # 72o should not be in LJ opening range at 100bb
    r = get_opening_range("LJ", 100)
    idx = HAND_INDEX['72o']
    assert r[idx] == pytest.approx(0.0, abs=1e-6), f"72o from LJ 100bb: {r[idx]}"


def test_calling_range_sums_to_one():
    for pos in ['BU', 'BB', 'CO']:
        r = get_calling_range(pos, 50)
        assert abs(r.sum() - 1.0) < 1e-6


def test_3bet_range_sums_to_one():
    for pos in ['BU', 'SB', 'CO']:
        r = get_3bet_range(pos, 100)
        assert abs(r.sum() - 1.0) < 1e-6


def test_short_stack_tight_range():
    r10 = get_opening_range("BU", 10)
    r100 = get_opening_range("BU", 100)
    aa_idx = HAND_INDEX['AA']
    # 10bb range gives AA relatively more weight
    assert r10[aa_idx] >= r100[aa_idx], "10bb should give AA at least as much weight as 100bb"
