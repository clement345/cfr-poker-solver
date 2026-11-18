from __future__ import annotations
import numpy as np
from .hands import HAND_TYPES, N_HANDS, parse_hand
from .constants import NUM_BOARD_TEXTURES


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return np.log(p / (1.0 - p))


def _hu_equity(h1: str, h2: str) -> float:
    """Approximate HU showdown equity for h1 vs h2 (pre-board, heads-up)."""
    r1h, r1l, pair1, suit1 = parse_hand(h1)
    r2h, r2l, pair2, suit2 = parse_hand(h2)

    if h1 == h2:
        return 0.50

    if pair1 and pair2:
        if r1h > r2h:
            return min(0.84, 0.80 + 0.003 * (r1h - r2h))
        return 1.0 - _hu_equity(h2, h1)

    if pair1 and not pair2:
        p, vh, vl = r1h, r2h, r2l
        sp = 0.02 if suit2 else 0.0
        if p == vh or p == vl:  # Domination: pair rank in villain's hand
            return max(0.60, 0.69 - sp)
        if p > vh:  # Overpair vs two undercards
            base = 0.50 + 0.42 * float(np.tanh(0.12 * (p - vl)))
            return min(0.92, base - sp)
        overcards = (1 if vh > p else 0) + (1 if vl > p else 0)
        if overcards == 2:  # Classic pair vs two overcards
            base = 0.51 + p * 0.004
            return max(0.48, min(0.58, base - sp))
        oc = max(vh, vl)  # One overcard
        base = 0.64 - (oc - p) * 0.022
        return max(0.52, min(0.72, base - sp))

    if not pair1 and pair2:
        return 1.0 - _hu_equity(h2, h1)

    # Both non-pairs
    h1s, h2s = {r1h, r1l}, {r2h, r2l}

    if h1s == h2s:  # Same ranks, different suitedness
        if suit1 and not suit2: return 0.60
        if not suit1 and suit2: return 0.40
        return 0.50

    shared = h1s & h2s
    if len(shared) == 1:  # One shared rank: domination
        sh = next(iter(shared))
        k1 = r1l if r1h == sh else r1h
        k2 = r2l if r2h == sh else r2h
        if k1 > k2:
            base = 0.67 + (k1 - k2) * 0.005
            return min(0.78, base + (0.04 if suit1 else 0.0) - (0.03 if suit2 else 0.0))
        if k1 < k2:
            return 1.0 - _hu_equity(h2, h1)
        return 0.50

    # No shared ranks
    if r1h < r2h:
        return 1.0 - _hu_equity(h2, h1)
    if r1l > r2h:  # Complete domination: AK vs 54, etc.
        base = 0.63 + (r1l - r2h) * 0.012
    elif r1l > r2l:
        base = 0.61
    else:
        base = 0.57
    return min(0.88, max(0.12, base + (0.03 if suit1 else 0.0) - (0.03 if suit2 else 0.0)))


def _hand_category(hand: str) -> str:
    rh, rl, pair, suited = parse_hand(hand)
    if pair:
        if rh >= 11: return 'pp_high'
        if rh >= 6:  return 'pp_mid'
        return 'pp_low'
    if suited:
        if rh == 14:         return 'ax_suited'
        if rh >= 12:         return 'broadway_s'
        if rh - rl <= 4:     return 'sc_suited'
        return 'broadway_s'
    if rh >= 11 and rl >= 10: return 'broadway_o'
    return 'weak_o'


# Per-category logit adjustments per board texture
# Textures: 0=dry A-high, 1=dry K/Q-high, 2=dry low, 3=monotone,
#            4=wet two-tone, 5=semi-wet, 6=paired, 7=two-paired
_CAT_ADJ: dict[str, list[float]] = {
    'pp_high':    [+0.25, +0.20, +0.15, -0.10, -0.20, -0.10, +0.00, +0.00],
    'pp_mid':     [+0.10, +0.10, +0.05, -0.05, -0.10, -0.05, +0.05, +0.00],
    'pp_low':     [+0.05, +0.00, +0.00, +0.05, +0.05, +0.05, +0.05, +0.00],
    'ax_suited':  [+0.15, +0.05, +0.00, +0.20, +0.20, +0.10, -0.05, -0.05],
    'broadway_s': [+0.10, +0.10, -0.05, +0.08, +0.05, +0.08, -0.05, -0.05],
    'sc_suited':  [-0.05, -0.05, +0.05, +0.20, +0.25, +0.15, -0.05, -0.10],
    'broadway_o': [+0.05, +0.05, -0.05, -0.05, -0.10, -0.05, +0.00, +0.00],
    'weak_o':     [-0.10, -0.05, +0.00, +0.00, +0.05, +0.00, +0.00, -0.05],
}

_CACHED_TABLE: np.ndarray | None = None


def _build_equity_table() -> np.ndarray:
    cats = [_hand_category(h) for h in HAND_TYPES]

    # Base HU equity (pre-board) for all 169×169 pairs
    base = np.empty((N_HANDS, N_HANDS))
    for i in range(N_HANDS):
        for j in range(N_HANDS):
            base[i, j] = _hu_equity(HAND_TYPES[i], HAND_TYPES[j])

    # Logit-space base
    logit_base = np.log(np.clip(base, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(base, 1e-6, 1.0 - 1e-6)))

    # Board-texture adjustments (vectorized)
    adj = np.array([[_CAT_ADJ[cats[i]][t] for t in range(NUM_BOARD_TEXTURES)]
                    for i in range(N_HANDS)])  # (169, 8)

    table = np.empty((N_HANDS, N_HANDS, NUM_BOARD_TEXTURES))
    for tex in range(NUM_BOARD_TEXTURES):
        hero_adj = adj[:, tex].reshape(-1, 1)    # (169, 1)
        vil_adj  = adj[:, tex].reshape(1, -1)    # (1, 169)
        logit = logit_base + hero_adj - vil_adj   # (169, 169)
        table[:, :, tex] = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))

    return np.clip(table, 0.01, 0.99)


def _get_table() -> np.ndarray:
    global _CACHED_TABLE
    if _CACHED_TABLE is None:
        _CACHED_TABLE = _build_equity_table()
    return _CACHED_TABLE


class EquityModel:
    def __init__(self) -> None:
        self._table = _get_table()

    def get_equity(self, hero_idx: int, villain_idx: int, board_texture: int) -> float:
        return float(self._table[hero_idx, villain_idx, board_texture])

    def get_range_equity(
        self,
        hero_idx: int,
        villain_range: np.ndarray,
        board_texture: int,
    ) -> float:
        """Equity of hero_idx vs a weighted range distribution over 169 hands."""
        total = villain_range.sum()
        if total == 0:
            return 0.5
        weights = villain_range / total
        return float(np.dot(weights, self._table[hero_idx, :, board_texture]))

    def get_all_equities(self, normalized_villain: np.ndarray, board_texture: int) -> np.ndarray:
        """Return equity for all 169 hero hands vs normalized villain range. Shape: (169,)."""
        return self._table[:, :, board_texture] @ normalized_villain

    def get_ev(
        self,
        hero_idx: int,
        villain_range: np.ndarray,
        board_texture: int,
        pot: float,
    ) -> float:
        equity = self.get_range_equity(hero_idx, villain_range, board_texture)
        return equity * pot - (1.0 - equity) * pot
