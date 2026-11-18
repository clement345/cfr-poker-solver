"""52-card representation, 1326 two-card combos, and conflict matrix."""
from __future__ import annotations
import numpy as np

# card = rank_idx * 4 + suit_idx
# rank_idx: 0=2, 1=3, ..., 12=A  (ascending)
# suit_idx: 0=c, 1=d, 2=h, 3=s

RANK_CHARS = "23456789TJQKA"  # index 0–12
SUIT_CHARS = "cdhs"           # index 0–3
N_CARDS = 52

RANK_IDX: dict[str, int] = {r: i for i, r in enumerate(RANK_CHARS)}
SUIT_IDX: dict[str, int] = {s: i for i, s in enumerate(SUIT_CHARS)}


def card_from_str(s: str) -> int:
    """'As' → 51, '2c' → 0."""
    return RANK_IDX[s[0]] * 4 + SUIT_IDX[s[1]]


def card_to_str(c: int) -> str:
    return RANK_CHARS[c >> 2] + SUIT_CHARS[c & 3]


def card_rank(c: int) -> int:
    return c >> 2   # 0=2 … 12=A


def card_suit(c: int) -> int:
    return c & 3


# ── 1326 two-card combos (c1 < c2) ─────────────────────────────────────────

COMBOS_1326: list[tuple[int, int]] = [
    (c1, c2) for c1 in range(N_CARDS) for c2 in range(c1 + 1, N_CARDS)
]

combo_to_idx: dict[tuple[int, int], int] = {
    combo: i for i, combo in enumerate(COMBOS_1326)
}

N_COMBOS = len(COMBOS_1326)  # 1326


# ── Conflict matrix ─────────────────────────────────────────────────────────

def _build_conflict_matrix() -> np.ndarray:
    mat = np.zeros((N_COMBOS, N_COMBOS), dtype=bool)
    for i, (a, b) in enumerate(COMBOS_1326):
        for j, (c, d) in enumerate(COMBOS_1326):
            if a == c or a == d or b == c or b == d:
                mat[i, j] = True
    return mat


CONFLICT_MATRIX: np.ndarray = _build_conflict_matrix()  # (1326, 1326) bool


# ── Canonical hand type for a combo ─────────────────────────────────────────
# Maps each combo index → index in hands.HAND_TYPES (0-168)

def _build_combo_to_hand_idx() -> np.ndarray:
    """Return array of shape (1326,) mapping combo idx → hand type idx (0-168)."""
    from .hands import HAND_INDEX, RANKS

    rank_to_hand_rank: dict[int, str] = {RANK_IDX[r]: r for r in RANK_CHARS}

    out = np.empty(N_COMBOS, dtype=np.int16)
    for i, (c1, c2) in enumerate(COMBOS_1326):
        r1, r2 = card_rank(c1), card_rank(c2)
        s1, s2 = card_suit(c1), card_suit(c2)
        hr1, hr2 = rank_to_hand_rank[r1], rank_to_hand_rank[r2]
        # Canonical order: higher rank first (A=12 highest)
        if r1 < r2:
            hr1, hr2 = hr2, hr1
        if hr1 == hr2:
            key = hr1 + hr2               # pair
        elif s1 == s2:
            key = hr1 + hr2 + "s"        # suited
        else:
            key = hr1 + hr2 + "o"        # offsuit
        out[i] = HAND_INDEX[key]
    return out


COMBO_TO_HAND_IDX: np.ndarray = _build_combo_to_hand_idx()  # (1326,) int16
