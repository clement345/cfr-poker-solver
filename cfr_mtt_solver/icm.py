from __future__ import annotations
import math
from functools import lru_cache


class ICMModel:
    def __init__(self, payouts: list[float]) -> None:
        # Normalize payouts to sum to 1
        total = sum(payouts)
        self.payouts = [p / total for p in payouts] if total > 0 else payouts

    def chip_ev_to_dollar_ev(self, chip_stacks: list[float]) -> list[float]:
        """Malmuth-Harville ICM: dollar equity per player given chip stacks."""
        n = len(chip_stacks)
        total_chips = sum(chip_stacks)
        if total_chips == 0:
            return [0.0] * n

        dollar_evs = [0.0] * n
        n_payouts = min(len(self.payouts), n)

        for player_idx in range(n):
            dollar_evs[player_idx] = self._player_dollar_ev(
                player_idx, chip_stacks, total_chips, n_payouts
            )

        return dollar_evs

    def _player_dollar_ev(
        self,
        player_idx: int,
        chip_stacks: list[float],
        total_chips: float,
        n_payouts: int,
    ) -> float:
        """Recursive Malmuth-Harville for one player."""
        ev = 0.0
        n = len(chip_stacks)

        # Enumerate all orderings by induction (Harville approximation)
        # P(finish 1st) = chips_i / total
        # P(finish kth | others above) = conditional via Harville

        def finish_probs(stacks: list[float], remaining_places: list[int]) -> list[float]:
            """Get finish probability for each remaining place using Harville."""
            total = sum(stacks)
            if total == 0:
                return [1.0 / len(stacks)] * len(stacks)
            return [s / total for s in stacks]

        # Use recursive computation with memoization via tuple
        indices = list(range(n))
        ev = self._recursive_icm(player_idx, tuple(chip_stacks), total_chips, n_payouts, 0)
        return ev

    def _recursive_icm(
        self,
        player_idx: int,
        stacks: tuple[float, ...],
        total_chips: float,
        n_payouts: int,
        place: int,
        excluded: frozenset[int] | None = None,
    ) -> float:
        if excluded is None:
            excluded = frozenset()
        if place >= n_payouts:
            return 0.0
        if place >= len(self.payouts):
            return 0.0

        active_indices = [i for i in range(len(stacks)) if i not in excluded]
        if not active_indices:
            return 0.0

        total = sum(stacks[i] for i in active_indices)
        if total == 0:
            return 0.0

        ev = 0.0
        for winner_idx in active_indices:
            p_win = stacks[winner_idx] / total
            if winner_idx == player_idx:
                # Player wins this place
                ev += p_win * self.payouts[place]
            # Continue to next place
            new_excluded = excluded | {winner_idx}
            ev += p_win * self._recursive_icm(
                player_idx, stacks, total_chips, n_payouts, place + 1, new_excluded
            )

        return ev

    def adjust_ev(self, chip_delta: float, player_idx: int, stacks: list[float]) -> float:
        """Convert chip EV gain to dollar EV by comparing before/after ICM."""
        before = self.chip_ev_to_dollar_ev(stacks)
        new_stacks = list(stacks)
        new_stacks[player_idx] = max(0.0, stacks[player_idx] + chip_delta)
        # Opponent loses chips (simplification: take from player 1-player_idx)
        opponent = 1 - player_idx if len(stacks) == 2 else (player_idx + 1) % len(stacks)
        new_stacks[opponent] = max(0.0, stacks[opponent] - chip_delta)
        after = self.chip_ev_to_dollar_ev(new_stacks)
        return after[player_idx] - before[player_idx]
