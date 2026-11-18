"""Equity engine backed by phevaluator and precomputed lookup tables.

Tables stored as numpy arrays in tables/ directory (gitignored):
  preflop_equity.npy   — (169, 169) float32 hero win equity vs villain hand type
  postflop_equity.npy  — (52, 169, 169) float32, indexed by REACHABLE_CLUSTERS position
  cluster_index.npy    — (72,) int16, maps raw cluster_id → row in postflop table (-1=unreachable)
"""
from __future__ import annotations
import os
import random
import numpy as np
from pathlib import Path

_TABLES_DIR = Path(__file__).parent.parent / "tables"

# Lazy-loaded singletons
_preflop_table: np.ndarray | None = None       # (169, 169) float32
_postflop_table: np.ndarray | None = None      # (N_reach, 169, 169) float32
_cluster_index: np.ndarray | None = None       # (72,) int16


def _card_str(c: int) -> str:
    """Convert card int to phevaluator string."""
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return ranks[c >> 2] + suits[c & 3]


def _eval_hand(hole1: int, hole2: int, board: list[int]) -> int:
    from phevaluator import evaluate_cards
    cards = [_card_str(c) for c in [hole1, hole2] + board]
    return evaluate_cards(*cards)


def _preflop_mc_equity(h_idx: int, v_idx: int, n_samples: int = 2000) -> float:
    """Monte Carlo preflop equity for hand type h_idx vs v_idx."""
    from .hands import HAND_TYPES
    from .cards import COMBOS_1326, COMBO_TO_HAND_IDX, N_CARDS, CONFLICT_MATRIX

    # Collect combos for each hand type
    h_combos = [i for i, hi in enumerate(COMBO_TO_HAND_IDX) if hi == h_idx]
    v_combos = [i for i, hi in enumerate(COMBO_TO_HAND_IDX) if hi == v_idx]
    if not h_combos or not v_combos:
        return 0.5

    wins = 0.0
    total = 0
    deck = list(range(N_CARDS))

    for _ in range(n_samples):
        hci = random.choice(h_combos)
        vci = random.choice(v_combos)
        # Resample if they conflict
        attempts = 0
        while CONFLICT_MATRIX[hci, vci] and attempts < 20:
            vci = random.choice(v_combos)
            attempts += 1
        if CONFLICT_MATRIX[hci, vci]:
            continue

        hc1, hc2 = COMBOS_1326[hci]
        vc1, vc2 = COMBOS_1326[vci]
        used = {hc1, hc2, vc1, vc2}
        remaining = [c for c in deck if c not in used]
        board = random.sample(remaining, 5)

        h_score = _eval_hand(hc1, hc2, board)
        v_score = _eval_hand(vc1, vc2, board)
        if h_score < v_score:
            wins += 1.0
        elif h_score == v_score:
            wins += 0.5
        total += 1

    return wins / total if total > 0 else 0.5


def _postflop_mc_equity(h_idx: int, v_idx: int, board3: list[int], n_samples: int = 500) -> float:
    """Monte Carlo postflop equity: enumerate turn+river given 3-card board."""
    from .cards import COMBOS_1326, COMBO_TO_HAND_IDX, N_CARDS, CONFLICT_MATRIX

    h_combos = [i for i, hi in enumerate(COMBO_TO_HAND_IDX) if hi == h_idx]
    v_combos = [i for i, hi in enumerate(COMBO_TO_HAND_IDX) if hi == v_idx]
    if not h_combos or not v_combos:
        return 0.5

    board_set = set(board3)
    wins = 0.0
    total = 0
    deck = list(range(N_CARDS))

    for _ in range(n_samples):
        hci = random.choice(h_combos)
        vci = random.choice(v_combos)
        attempts = 0
        while (CONFLICT_MATRIX[hci, vci] or
               COMBOS_1326[hci][0] in board_set or
               COMBOS_1326[hci][1] in board_set or
               COMBOS_1326[vci][0] in board_set or
               COMBOS_1326[vci][1] in board_set) and attempts < 30:
            hci = random.choice(h_combos)
            vci = random.choice(v_combos)
            attempts += 1
        hc1, hc2 = COMBOS_1326[hci]
        vc1, vc2 = COMBOS_1326[vci]
        if CONFLICT_MATRIX[hci, vci] or hc1 in board_set or hc2 in board_set or vc1 in board_set or vc2 in board_set:
            continue

        used = {hc1, hc2, vc1, vc2} | board_set
        remaining = [c for c in deck if c not in used]
        if len(remaining) < 2:
            continue
        turn, river = random.sample(remaining, 2)
        board5 = board3 + [turn, river]

        h_score = _eval_hand(hc1, hc2, board5)
        v_score = _eval_hand(vc1, vc2, board5)
        if h_score < v_score:
            wins += 1.0
        elif h_score == v_score:
            wins += 0.5
        total += 1

    return wins / total if total > 0 else 0.5


def precompute_preflop(out_path: Path | None = None, n_samples: int = 2000) -> np.ndarray:
    """Compute (169, 169) preflop equity table. Symmetric: table[i,j] + table[j,i] = 1."""
    from .hands import N_HANDS
    if out_path is None:
        out_path = _TABLES_DIR / "preflop_equity.npy"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = np.full((N_HANDS, N_HANDS), 0.5, dtype=np.float32)
    total_pairs = N_HANDS * (N_HANDS + 1) // 2
    done = 0
    for i in range(N_HANDS):
        for j in range(i, N_HANDS):
            eq = _preflop_mc_equity(i, j, n_samples)
            table[i, j] = eq
            table[j, i] = 1.0 - eq if i != j else 0.5
            done += 1
            if done % 500 == 0:
                print(f"  preflop equity: {done}/{total_pairs} pairs")
    np.save(out_path, table)
    print(f"Saved preflop equity table → {out_path} ({table.nbytes // 1024} KB)")
    return table


def _cluster_equity_fast(board3: list[int], n_runouts: int = 150) -> np.ndarray:
    """Compute (169, 169) float32 equity table for a board cluster.

    Uses vectorized numpy operations: sample runouts once, evaluate all valid
    combos per runout, accumulate wins matrix in bulk.
    """
    from phevaluator import evaluate_cards
    from .cards import COMBOS_1326, COMBO_TO_HAND_IDX, N_CARDS, N_COMBOS
    from .hands import N_HANDS

    board_set = set(board3)
    available = [c for c in range(N_CARDS) if c not in board_set]

    # Which combos don't conflict with the board?
    valid_mask = np.array([
        c1 not in board_set and c2 not in board_set
        for c1, c2 in COMBOS_1326
    ], dtype=bool)
    valid_idx = np.where(valid_mask)[0]  # ~1250 combos

    hand_type = COMBO_TO_HAND_IDX[valid_idx]  # (N_valid,) hand type per valid combo

    # Accumulate per-hand-type equity over runouts
    wins_ht = np.zeros((N_HANDS, N_HANDS), dtype=np.float64)
    count_ht = np.zeros((N_HANDS, N_HANDS), dtype=np.float64)

    for _ in range(n_runouts):
        turn, river = random.sample(available, 2)
        board5 = board3 + [turn, river]
        board5_strs = [_card_str(c) for c in board5]
        extra = {turn, river}

        # Evaluate scores for all valid combos on this runout
        scores = np.full(N_COMBOS, -1, dtype=np.int32)
        for i in valid_idx:
            c1, c2 = COMBOS_1326[i]
            if c1 in extra or c2 in extra:
                continue
            scores[i] = evaluate_cards(_card_str(c1), _card_str(c2), *board5_strs)

        # Mask to combos valid on this runout
        runout_valid = (scores[valid_idx] >= 0)
        rv_idx = valid_idx[runout_valid]   # subset
        rv_scores = scores[rv_idx]         # (N_rv,)
        rv_ht = COMBO_TO_HAND_IDX[rv_idx]  # (N_rv,)

        # Build wins matrix: wins[a,b] += 1 if combo a beats combo b
        # Use broadcasting: (N_rv, N_rv) — only ~1200×1200 = 1.44M elements
        s_i = rv_scores[:, None]            # (N_rv, 1)
        s_j = rv_scores[None, :]            # (1, N_rv)
        # lower = better in phevaluator
        win_matrix = (s_i < s_j).astype(np.float32) + 0.5 * (s_i == s_j).astype(np.float32)

        # Aggregate by hand type using np.add.at
        # For each pair (ht_i, ht_j), accumulate win_matrix[i,j]
        ht_i = rv_ht[:, None].repeat(len(rv_idx), axis=1)  # (N_rv, N_rv)
        ht_j = rv_ht[None, :].repeat(len(rv_idx), axis=0)  # (N_rv, N_rv)
        np.add.at(wins_ht, (ht_i, ht_j), win_matrix)
        np.add.at(count_ht, (ht_i, ht_j), 1.0)

    # Normalize
    with np.errstate(invalid='ignore', divide='ignore'):
        equity_ht = np.where(count_ht > 0, wins_ht / count_ht, 0.5)

    return equity_ht.astype(np.float32)


def precompute_postflop(out_path: Path | None = None, n_runouts: int = 150) -> np.ndarray:
    """Compute (N_reach, 169, 169) postflop equity table using vectorized runout approach."""
    from .hands import N_HANDS
    from .board_clusters import REACHABLE_CLUSTERS, CLUSTER_REPR

    if out_path is None:
        out_path = _TABLES_DIR / "postflop_equity.npy"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_reach = len(REACHABLE_CLUSTERS)
    table = np.full((n_reach, N_HANDS, N_HANDS), 0.5, dtype=np.float32)

    for row, cid in enumerate(REACHABLE_CLUSTERS):
        board3 = list(CLUSTER_REPR[cid])
        table[row] = _cluster_equity_fast(board3, n_runouts)
        if (row + 1) % 5 == 0 or row == 0:
            print(f"  postflop equity: cluster {row+1}/{n_reach}", flush=True)

    np.save(out_path, table)

    # Save cluster index mapping
    idx_table = np.full(72, -1, dtype=np.int16)
    for row, cid in enumerate(REACHABLE_CLUSTERS):
        idx_table[cid] = row
    np.save(_TABLES_DIR / "cluster_index.npy", idx_table)

    print(f"Saved postflop equity table → {out_path} ({table.nbytes // 1024 // 1024} MB)", flush=True)
    return table


def _load_tables() -> None:
    global _preflop_table, _postflop_table, _cluster_index
    pf_path = _TABLES_DIR / "preflop_equity.npy"
    if _preflop_table is None and pf_path.exists():
        _preflop_table = np.load(pf_path).astype(np.float64)
    po_path = _TABLES_DIR / "postflop_equity.npy"
    ci_path = _TABLES_DIR / "cluster_index.npy"
    if _postflop_table is None and po_path.exists() and ci_path.exists():
        _postflop_table = np.load(po_path).astype(np.float64)
        _cluster_index = np.load(ci_path)


class EquityEngine:
    """Equity lookup with precomputed tables + Monte Carlo fallback."""

    def __init__(self) -> None:
        _load_tables()

    def get_preflop_equity(self, hero_hand_idx: int, villain_hand_idx: int) -> float:
        """Return hero equity [0,1] vs villain hand type."""
        if _preflop_table is not None:
            return float(_preflop_table[hero_hand_idx, villain_hand_idx])
        return _preflop_mc_equity(hero_hand_idx, villain_hand_idx, n_samples=500)

    def get_postflop_equity(
        self, hero_hand_idx: int, villain_hand_idx: int, cluster_id: int
    ) -> float:
        """Return hero equity [0,1] on a given board cluster."""
        if _postflop_table is not None and _cluster_index is not None:
            row = int(_cluster_index[cluster_id])
            if row >= 0:
                return float(_postflop_table[row, hero_hand_idx, villain_hand_idx])
        return _preflop_mc_equity(hero_hand_idx, villain_hand_idx, n_samples=200)

    def get_all_equities(self, normalized_villain: np.ndarray, board_texture: int = 0) -> np.ndarray:
        """Return equity for all 169 hero hands vs normalized villain range. Shape: (169,)."""
        from .hands import N_HANDS
        if board_texture == 0:
            if _preflop_table is not None:
                return (_preflop_table @ normalized_villain).astype(np.float64)
        else:
            if _postflop_table is not None and _cluster_index is not None:
                row_idx = int(_cluster_index[board_texture])
                if row_idx >= 0:
                    return (_postflop_table[row_idx] @ normalized_villain).astype(np.float64)
            if _preflop_table is not None:
                return (_preflop_table @ normalized_villain).astype(np.float64)
        return np.full(N_HANDS, 0.5, dtype=np.float64)

    def get_range_equity(
        self,
        hero_hand_idx: int,
        villain_range: np.ndarray,
        board_texture: int = 0,
    ) -> float:
        """Range-weighted equity of hero hand type vs villain range distribution.

        Compatible with the legacy EquityModel interface used in cfr_solver.py.
        villain_range: (169,) float array of weights (need not sum to 1).
        board_texture: 0 = preflop/use preflop table, else cluster_id.
        """
        from .hands import N_HANDS
        total_w = float(villain_range.sum())
        if total_w < 1e-12:
            return 0.5

        if board_texture == 0:
            if _preflop_table is not None:
                row = _preflop_table[hero_hand_idx]        # (169,) float32
                return float(np.dot(row, villain_range) / total_w)
            # Fallback: per-hand MC (slow, only if no table)
            eq = sum(
                float(villain_range[j]) * _preflop_mc_equity(hero_hand_idx, j, 200)
                for j in range(N_HANDS) if villain_range[j] > 1e-10
            )
            return eq / total_w
        else:
            if _postflop_table is not None and _cluster_index is not None:
                row_idx = int(_cluster_index[board_texture])
                if row_idx >= 0:
                    row = _postflop_table[row_idx, hero_hand_idx]  # (169,)
                    return float(np.dot(row, villain_range) / total_w)
            # Fallback: use preflop table (much faster than MC, good approximation)
            if _preflop_table is not None:
                row = _preflop_table[hero_hand_idx]
                return float(np.dot(row, villain_range) / total_w)
            # Last resort: MC
            eq = sum(
                float(villain_range[j]) * self.get_postflop_equity(hero_hand_idx, j, board_texture)
                for j in range(N_HANDS) if villain_range[j] > 1e-10
            )
            return eq / total_w
