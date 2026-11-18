from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class InfoSetKey:
    position: str
    street: str
    pot_bucket: int        # 5 bins: <3/3-6/6-12/12-25/25+
    stack_bucket: int      # 4 bins: 10/20/50/100
    action_history: tuple  # compressed action sequence
    board_texture: int     # 0-7 (0 preflop)
    hand_idx: int          # 0-168 (index into hands.HAND_TYPES)
    player_to_act: str

    def to_str(self) -> str:
        # Serialize each (player, action) as "PLAYER:action_value" for readability
        hist = "|".join(
            f"{p}:{a.value if hasattr(a, 'value') else a}"
            for p, a in self.action_history
        )
        return (
            f"{self.position}|{self.street}|pot{self.pot_bucket}"
            f"|stk{self.stack_bucket}|tex{self.board_texture}"
            f"|hnd{self.hand_idx}|{self.player_to_act}|{hist}"
        )


def pot_to_bucket(pot: float) -> int:
    if pot < 3:
        return 0
    if pot < 6:
        return 1
    if pot < 12:
        return 2
    if pot < 25:
        return 3
    return 4


def stack_to_bucket(stack: float) -> int:
    if stack <= 12:
        return 0   # ~10bb
    if stack <= 30:
        return 1   # ~20bb
    if stack <= 75:
        return 2   # ~50bb
    return 3       # ~100bb


class InfoSet:
    def __init__(self, key: InfoSetKey, actions: list[str]) -> None:
        self.key = key
        self.actions = actions
        self.regrets: dict[str, float] = {a: 0.0 for a in actions}
        self.cumulative_strategy: dict[str, float] = {a: 0.0 for a in actions}

    def get_strategy(self) -> dict[str, float]:
        """Regret matching: σ(a) = max(R(a),0) / Σmax(R,0). Uniform if all ≤ 0."""
        positive = {a: max(self.regrets[a], 0.0) for a in self.actions}
        total = sum(positive.values())
        if total > 0:
            return {a: positive[a] / total for a in self.actions}
        n = len(self.actions)
        return {a: 1.0 / n for a in self.actions}

    def get_average_strategy(self) -> dict[str, float]:
        """Nash-converging average strategy."""
        total = sum(self.cumulative_strategy.values())
        if total > 0:
            return {a: self.cumulative_strategy[a] / total for a in self.actions}
        n = len(self.actions)
        return {a: 1.0 / n for a in self.actions}

    def update_regrets(self, action_evs: dict[str, float], node_ev: float, cf_reach: float) -> None:
        # CFR+: clamp regrets to 0 after each update
        for a in self.actions:
            ev = action_evs.get(a, node_ev)
            self.regrets[a] = max(0.0, self.regrets[a] + cf_reach * (ev - node_ev))

    def update_cumulative(self, strategy: dict[str, float], own_reach: float, iteration: int = 1) -> None:
        # Linear weighted averaging: weight by iteration index → faster convergence
        for a in self.actions:
            self.cumulative_strategy[a] += iteration * own_reach * strategy.get(a, 0.0)

    def total_positive_regret(self) -> float:
        return sum(max(r, 0.0) for r in self.regrets.values())
