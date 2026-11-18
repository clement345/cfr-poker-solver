"""Tests for cards.py, combo_ranges.py, board_clusters.py."""
from __future__ import annotations
import numpy as np
import pytest

from cfr_mtt_solver.cards import (
    card_from_str, card_to_str, card_rank, card_suit,
    COMBOS_1326, N_COMBOS, CONFLICT_MATRIX, COMBO_TO_HAND_IDX,
    combo_to_idx,
)
from cfr_mtt_solver.combo_ranges import ComboRange
from cfr_mtt_solver.board_clusters import cluster_flop, cluster_features, REACHABLE_CLUSTERS


# ── cards.py ────────────────────────────────────────────────────────────────

def test_card_from_str_roundtrip():
    for rank in "23456789TJQKA":
        for suit in "cdhs":
            s = rank + suit
            assert card_to_str(card_from_str(s)) == s


def test_n_combos():
    assert N_COMBOS == 1326


def test_conflict_matrix_shape():
    assert CONFLICT_MATRIX.shape == (1326, 1326)


def test_conflict_matrix_diagonal():
    # Every combo conflicts with itself
    assert CONFLICT_MATRIX[0, 0]
    assert CONFLICT_MATRIX[1325, 1325]


def test_conflict_matrix_shared_card():
    # As (51) and Ah (50): combos containing As or Ah share a card
    As = card_from_str("As")
    Ah = card_from_str("Ah")
    idx_AsAh = combo_to_idx[(min(As, Ah), max(As, Ah))]
    idx_AsKs = combo_to_idx[(min(As, card_from_str("Ks")), max(As, card_from_str("Ks")))]
    # AsAh and AsKs share As
    assert CONFLICT_MATRIX[idx_AsAh, idx_AsKs]


def test_no_false_conflicts():
    # 2c (0) and 3d (5): no shared card with Ah/Kh
    c1 = card_from_str("2c")
    c2 = card_from_str("3d")
    c3 = card_from_str("Ah")
    c4 = card_from_str("Kh")
    # combo_to_idx requires c1 < c2
    idx_low = combo_to_idx[(min(c1, c2), max(c1, c2))]
    idx_high = combo_to_idx[(min(c3, c4), max(c3, c4))]
    assert not CONFLICT_MATRIX[idx_low, idx_high]


def test_combo_to_hand_idx_aa():
    from cfr_mtt_solver.hands import HAND_TYPES
    As = card_from_str("As")
    Ah = card_from_str("Ah")
    idx = combo_to_idx[(min(As, Ah), max(As, Ah))]
    assert HAND_TYPES[COMBO_TO_HAND_IDX[idx]] == "AA"


def test_combo_to_hand_idx_aks():
    from cfr_mtt_solver.hands import HAND_TYPES
    As = card_from_str("As")
    Ks = card_from_str("Ks")
    idx = combo_to_idx[(min(As, Ks), max(As, Ks))]
    assert HAND_TYPES[COMBO_TO_HAND_IDX[idx]] == "AKs"


def test_combo_to_hand_idx_ako():
    from cfr_mtt_solver.hands import HAND_TYPES
    Ah = card_from_str("Ah")
    Kd = card_from_str("Kd")
    idx = combo_to_idx[(min(Ah, Kd), max(Ah, Kd))]
    assert HAND_TYPES[COMBO_TO_HAND_IDX[idx]] == "AKo"


# ── combo_ranges.py ─────────────────────────────────────────────────────────

def test_uniform_range():
    cr = ComboRange.uniform()
    assert cr.weights.shape == (1326,)
    assert np.all(cr.weights == 1.0)


def test_remove_blockers_reduces_count():
    cr = ComboRange.uniform()
    As = card_from_str("As")
    Ah = card_from_str("Ah")
    idx = combo_to_idx[(min(As, Ah), max(As, Ah))]
    cr2 = cr.remove_blockers(idx)
    # AsAh uses As and Ah; each blocks 50 combos sharing that card, minus AsAh itself = 101 total
    assert int((cr2.weights == 0).sum()) == 101
    assert int((cr2.weights > 0).sum()) == 1225


def test_normalized_sums_to_one():
    cr = ComboRange.uniform().remove_blockers(0)
    norm = cr.normalized()
    assert abs(norm.sum() - 1.0) < 1e-5


def test_from_hand_type_weights():
    from cfr_mtt_solver.hands import N_HANDS
    weights = np.zeros(N_HANDS)
    weights[0] = 1.0  # Only AA
    cr = ComboRange.from_hand_type_weights(weights)
    # AA has 6 combos, each should have weight 1/6
    nonzero = (cr.weights > 0).sum()
    assert nonzero == 6


# ── board_clusters.py ────────────────────────────────────────────────────────

def test_reachable_clusters_count():
    # 72 - 20 unreachable (conn_tier=2 + is_paired=1) = 52
    assert len(REACHABLE_CLUSTERS) == 52


def test_cluster_flop_known_boards():
    from cfr_mtt_solver.cards import card_from_str
    # A-high rainbow disconnected: hr=3, flush=0, conn=0, pair=0 → cluster 54
    c1, c2, c3 = card_from_str("As"), card_from_str("Kd"), card_from_str("2h")
    cid = cluster_flop(c1, c2, c3)
    f = cluster_features(cid)
    assert f["high_rank_tier"] == 3
    assert f["flush_tier"] == 0
    assert f["is_paired"] == 0


def test_cluster_monotone():
    from cfr_mtt_solver.cards import card_from_str
    # All spades → flush_tier=2
    c1, c2, c3 = card_from_str("As"), card_from_str("Ks"), card_from_str("Qs")
    cid = cluster_flop(c1, c2, c3)
    f = cluster_features(cid)
    assert f["flush_tier"] == 2


def test_cluster_paired():
    from cfr_mtt_solver.cards import card_from_str
    # Paired board
    c1, c2, c3 = card_from_str("As"), card_from_str("Ad"), card_from_str("2h")
    cid = cluster_flop(c1, c2, c3)
    f = cluster_features(cid)
    assert f["is_paired"] == 1


def test_cluster_flop_in_reachable():
    from cfr_mtt_solver.cards import card_from_str
    # Any actual flop should produce a reachable cluster
    boards = [
        ("As", "Kd", "Qh"),
        ("9s", "8h", "7d"),
        ("As", "Ks", "Qs"),
        ("2c", "2d", "3h"),
    ]
    for cs in boards:
        c1, c2, c3 = [card_from_str(c) for c in cs]
        cid = cluster_flop(c1, c2, c3)
        assert cid in REACHABLE_CLUSTERS, f"Board {cs} gave unreachable cluster {cid}"
