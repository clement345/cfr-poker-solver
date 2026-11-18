from __future__ import annotations
import numpy as np

RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
RANK_VAL: dict[str, int] = {r: 14 - i for i, r in enumerate(RANKS)}  # A=14, K=13, ..., 2=2

# 169 canonical hand types: pairs (AA..22), suited (AKs..32s), offsuit (AKo..32o)
HAND_TYPES: list[str] = []

for _r in RANKS:
    HAND_TYPES.append(_r + _r)

for _i, _r1 in enumerate(RANKS):
    for _r2 in RANKS[_i + 1:]:
        HAND_TYPES.append(_r1 + _r2 + 's')

for _i, _r1 in enumerate(RANKS):
    for _r2 in RANKS[_i + 1:]:
        HAND_TYPES.append(_r1 + _r2 + 'o')

N_HANDS: int = len(HAND_TYPES)  # 169

HAND_INDEX: dict[str, int] = {h: i for i, h in enumerate(HAND_TYPES)}

# Number of specific combos: pairs=6 (C(4,2)), suited=4, offsuit=12
HAND_COMBOS: dict[str, int] = {}
for _h in HAND_TYPES:
    if len(_h) == 2:
        HAND_COMBOS[_h] = 6
    elif _h[-1] == 's':
        HAND_COMBOS[_h] = 4
    else:
        HAND_COMBOS[_h] = 12

HAND_COMBOS_ARRAY: np.ndarray = np.array([HAND_COMBOS[h] for h in HAND_TYPES], dtype=float)


def parse_hand(hand: str) -> tuple[int, int, bool, bool]:
    """Return (high_rank_val, low_rank_val, is_pair, is_suited)."""
    if len(hand) == 2:
        rv = RANK_VAL[hand[0]]
        return rv, rv, True, False
    return RANK_VAL[hand[0]], RANK_VAL[hand[1]], False, hand[-1] == 's'


def hand_strength(hand: str) -> float:
    """Preflop playability score. Pairs: 54–78. Non-pairs: ~7–27. Higher = stronger."""
    rh, rl, pair, suited = parse_hand(hand)
    if pair:
        return 50.0 + rh * 2.0
    gap = rh - rl - 1
    conn_bonus = max(0.0, 3.0 - gap) * (0.5 if suited else 0.15)
    suit_bonus = 3.0 if suited else 0.0
    hcw = 1.1 if suited else 0.85
    lcw = 0.55 if suited else 0.35
    score = rh * hcw + rl * lcw + suit_bonus + conn_bonus
    # Penalty for big-gap offsuit non-ace hands (K2o, Q2o, etc.)
    if not suited and rh != 14 and gap >= 5:
        score -= (gap - 4) * 0.5
    return score
