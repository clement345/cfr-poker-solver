"""ComboRange: a probability distribution over 1326 two-card combos."""
from __future__ import annotations
import numpy as np
from .cards import N_COMBOS, CONFLICT_MATRIX, COMBO_TO_HAND_IDX


class ComboRange:
    """Probability distribution over the 1326 canonical two-card combos.

    `weights[i]` is the un-normalised probability of holding combo i.
    Call `.normalized()` to get a distribution summing to 1.
    """

    __slots__ = ("weights",)

    def __init__(self, weights: np.ndarray) -> None:
        assert weights.shape == (N_COMBOS,), f"Expected ({N_COMBOS},), got {weights.shape}"
        self.weights: np.ndarray = weights.astype(np.float32)

    # ── Constructors ────────────────────────────────────────────────────────

    @classmethod
    def uniform(cls) -> "ComboRange":
        return cls(np.ones(N_COMBOS, dtype=np.float32))

    @classmethod
    def from_hand_type_weights(cls, weights_169: np.ndarray) -> "ComboRange":
        """Map 169-hand-type weights to 1326 combos.

        Each combo inherits the weight of its canonical hand type divided by
        the number of combos in that hand type (pairs=6, suited=4, offsuit=12)
        so that the total weight is preserved.
        """
        from .hands import HAND_COMBOS_ARRAY
        per_combo_weight = weights_169 / HAND_COMBOS_ARRAY  # (169,)
        out = per_combo_weight[COMBO_TO_HAND_IDX]           # (1326,) broadcast
        return cls(out)

    # ── Card removal ────────────────────────────────────────────────────────

    def remove_blockers(self, hero_combo_idx: int) -> "ComboRange":
        """Return copy with all combos that share a card with hero's hand zeroed."""
        mask = CONFLICT_MATRIX[hero_combo_idx]          # (1326,) bool
        new_weights = self.weights.copy()
        new_weights[mask] = 0.0
        return ComboRange(new_weights)

    # ── Normalisation ───────────────────────────────────────────────────────

    def normalized(self) -> np.ndarray:
        """Return (1326,) float32 array summing to 1 (or all-zeros if empty)."""
        total = self.weights.sum()
        if total < 1e-12:
            return np.zeros(N_COMBOS, dtype=np.float32)
        return self.weights / total

    # ── Utilities ───────────────────────────────────────────────────────────

    def total_weight(self) -> float:
        return float(self.weights.sum())

    def __repr__(self) -> str:
        nz = int((self.weights > 0).sum())
        return f"ComboRange(nonzero={nz}/{N_COMBOS}, total={self.total_weight():.4f})"
