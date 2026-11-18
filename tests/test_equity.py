from __future__ import annotations
import numpy as np
import pytest
from cfr_mtt_solver.equity import EquityModel
from cfr_mtt_solver.hands import HAND_INDEX, N_HANDS


def test_equity_table_shape():
    m = EquityModel()
    assert m._table.shape == (169, 169, 8)


def test_same_hand_equity():
    m = EquityModel()
    for hand in ['AA', 'KK', 'AKs', 'AKo', '72o']:
        idx = HAND_INDEX[hand]
        for tex in range(8):
            eq = m.get_equity(idx, idx, tex)
            assert abs(eq - 0.5) < 0.02, f"{hand} vs itself tex={tex}: {eq}"


def test_stronger_hand_wins_more():
    m = EquityModel()
    aa, kk, two_two = HAND_INDEX['AA'], HAND_INDEX['KK'], HAND_INDEX['22']
    # AA beats 22 more than AA beats KK
    eq_aa_22 = m.get_equity(aa, two_two, 0)
    eq_aa_kk = m.get_equity(aa, kk, 0)
    assert eq_aa_22 > 0.60, f"AA vs 22 equity too low: {eq_aa_22}"
    assert eq_aa_kk > 0.60, f"AA vs KK equity too low: {eq_aa_kk}"
    assert eq_aa_22 > eq_aa_kk, "AA should have more equity vs 22 than vs KK"


def test_pair_beats_nonpair():
    m = EquityModel()
    qq, ako = HAND_INDEX['QQ'], HAND_INDEX['AKo']
    # QQ is slight favourite vs AKo (coin-flip ~55-56%)
    eq = m.get_equity(qq, ako, 0)
    assert 0.48 < eq < 0.70, f"QQ vs AKo equity out of expected range: {eq}"


def test_suited_bonus():
    m = EquityModel()
    aks, ako = HAND_INDEX['AKs'], HAND_INDEX['AKo']
    kqs = HAND_INDEX['KQs']
    # AKs should beat AKo
    eq_aks_vs_ako = m.get_equity(aks, ako, 0)
    assert eq_aks_vs_ako > 0.50, f"AKs should beat AKo: {eq_aks_vs_ako}"


def test_equity_in_range():
    m = EquityModel()
    for hero in range(N_HANDS):
        for villain in range(N_HANDS):
            for tex in range(8):
                eq = m.get_equity(hero, villain, tex)
                assert 0.0 < eq < 1.0, f"Equity out of (0,1) at ({hero},{villain},{tex}): {eq}"


def test_range_equity_uniform():
    m = EquityModel()
    uniform = np.ones(N_HANDS) / N_HANDS
    eq = m.get_range_equity(HAND_INDEX['AA'], uniform, 0)
    assert eq > 0.60, f"AA vs uniform range equity too low: {eq}"


def test_range_equity_sums_correctly():
    m = EquityModel()
    aa_idx, kk_idx = HAND_INDEX['AA'], HAND_INDEX['KK']
    villain_range = np.zeros(N_HANDS)
    villain_range[kk_idx] = 1.0
    eq_range = m.get_range_equity(aa_idx, villain_range, 0)
    eq_direct = m.get_equity(aa_idx, kk_idx, 0)
    assert abs(eq_range - eq_direct) < 1e-6
