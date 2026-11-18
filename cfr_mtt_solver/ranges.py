from __future__ import annotations
import numpy as np
from .hands import HAND_TYPES, N_HANDS, HAND_COMBOS_ARRAY, hand_strength, parse_hand

# Minimum hand strength to open from each position at each stack size.
# Hand strength: pairs 54–78, non-pairs ~7–27 (higher = stronger).
_OPEN_THRESH: dict[str, dict[int, float]] = {
    'LJ': {10: 20.0, 20: 16.0, 50: 14.5, 100: 14.0},
    'HJ': {10: 18.0, 20: 14.5, 50: 13.0, 100: 12.5},
    'CO': {10: 16.0, 20: 13.0, 50: 12.0, 100: 11.5},
    'BU': {10: 14.0, 20: 11.0, 50: 10.5, 100: 10.0},
    'SB': {10: 13.0, 20: 10.0, 50:  9.8, 100:  9.5},
    'BB': {10: 15.0, 20: 12.0, 50: 11.0, 100: 10.5},
}
_ZONE = 1.5  # Width of mixed-frequency zone below threshold


def _opening_freq(hand: str, position: str, stack_bb: int) -> float:
    thresh = _OPEN_THRESH[position][stack_bb]
    s = hand_strength(hand)
    if s >= thresh + _ZONE:
        return 1.0
    if s >= thresh:
        return (s - thresh) / _ZONE * 0.8 + 0.2  # 0.2–1.0 in fuzzy zone
    if s >= thresh - 0.5:
        return 0.05
    return 0.0


def _normalize(freqs: np.ndarray) -> np.ndarray:
    """Weight by combo count and normalize to a probability distribution."""
    weighted = freqs * HAND_COMBOS_ARRAY
    total = weighted.sum()
    if total == 0:
        return np.ones(N_HANDS) / N_HANDS
    return weighted / total


def _snap_stack(stack_bb: int) -> int:
    return min([10, 20, 50, 100], key=lambda s: abs(s - stack_bb))


def get_opening_range(position: str, stack_bb: int) -> np.ndarray:
    """Normalized probability distribution over 169 hand types for opening."""
    stack_bb = _snap_stack(stack_bb)
    freqs = np.array([_opening_freq(h, position, stack_bb) for h in HAND_TYPES])
    return _normalize(freqs)


def get_calling_range(position: str, stack_bb: int, vs_raise_size: float = 2.5) -> np.ndarray:
    """Calling range: middle tier of opening range (not 3bet-worthy, not fold)."""
    stack_bb = _snap_stack(stack_bb)
    thresh = _OPEN_THRESH[position][stack_bb]
    lo = thresh - 2.0    # Bottom of call range
    hi = thresh + 8.0    # Top of call range (above this → 3bet)
    freqs = np.zeros(N_HANDS)
    for i, hand in enumerate(HAND_TYPES):
        s = hand_strength(hand)
        if lo <= s < hi:
            freqs[i] = 1.0 if s >= lo + 1.5 else (s - lo) / 1.5 * 0.8 + 0.2
    return _normalize(freqs)


# 3bet thresholds: minimum hand strength to 3bet for value
_3BET_THRESH: dict[str, float] = {
    'BU': 20.0, 'SB': 20.0, 'CO': 21.0,
    'HJ': 22.0, 'LJ': 23.0, 'BB': 20.0,
}


def get_3bet_range(position: str, stack_bb: int) -> np.ndarray:
    """3bet range: top value hands + suited-ace bluffs (A5s–A2s) and suited connectors."""
    stack_bb = _snap_stack(stack_bb)
    thresh = _3BET_THRESH.get(position, 21.0)
    freqs = np.zeros(N_HANDS)
    for i, hand in enumerate(HAND_TYPES):
        s = hand_strength(hand)
        rh, rl, pair, suited = parse_hand(hand)
        if s >= thresh:
            freqs[i] = 1.0
        elif s >= thresh - 1.5:
            freqs[i] = 0.5
        elif suited and rh == 14 and rl <= 5 and stack_bb >= 50:
            # Suited ace bluffs: A5s–A2s block opponent's AA/AK combos
            freqs[i] = 0.6
        elif suited and not pair and rh <= 8 and rh - rl == 1 and stack_bb >= 50:
            # Suited connector bluffs from position
            if position in ('BU', 'SB'):
                freqs[i] = 0.3
    return _normalize(freqs)
