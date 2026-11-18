from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .constants import Street


@dataclass(frozen=True)
class GameState:
    positions: tuple[str, ...]
    stacks: tuple[tuple[str, float], ...]   # frozen as tuple of (pos, stack) pairs
    pot: float
    street: Street
    action_history: tuple[tuple[str, str], ...]  # ((position, action_str), ...)
    current_player: str
    board_texture: int                           # 0–7
    effective_stack: float
    use_icm: bool = False
    tournament_chip_stacks: tuple[float, ...] = ()

    @staticmethod
    def make(
        positions: list[str],
        stacks: dict[str, float],
        pot: float,
        street: Street,
        action_history: list[tuple[str, str]],
        current_player: str,
        board_texture: int = 0,
        use_icm: bool = False,
        tournament_chip_stacks: Optional[list[float]] = None,
    ) -> "GameState":
        eff = min(stacks[p] for p in positions) if len(positions) > 1 else list(stacks.values())[0]
        return GameState(
            positions=tuple(positions),
            stacks=tuple(sorted(stacks.items())),
            pot=pot,
            street=street,
            action_history=tuple(action_history),
            current_player=current_player,
            board_texture=board_texture,
            effective_stack=eff,
            use_icm=use_icm,
            tournament_chip_stacks=tuple(tournament_chip_stacks or []),
        )

    def get_stacks(self) -> dict[str, float]:
        return dict(self.stacks)

    def get_stack(self, position: str) -> float:
        return dict(self.stacks).get(position, 0.0)

    def with_updates(self, **kwargs) -> "GameState":
        stacks = dict(self.stacks)
        if "stacks" in kwargs and isinstance(kwargs["stacks"], dict):
            kwargs["stacks"] = tuple(sorted(kwargs["stacks"].items()))
        if "action_history" in kwargs and isinstance(kwargs["action_history"], list):
            kwargs["action_history"] = tuple(kwargs["action_history"])
        if "positions" in kwargs and isinstance(kwargs["positions"], list):
            kwargs["positions"] = tuple(kwargs["positions"])
        current = {
            "positions": self.positions,
            "stacks": self.stacks,
            "pot": self.pot,
            "street": self.street,
            "action_history": self.action_history,
            "current_player": self.current_player,
            "board_texture": self.board_texture,
            "effective_stack": self.effective_stack,
            "use_icm": self.use_icm,
            "tournament_chip_stacks": self.tournament_chip_stacks,
        }
        current.update(kwargs)
        new_stacks = dict(current["stacks"])
        eff = min(new_stacks[p] for p in current["positions"]) if len(current["positions"]) > 1 else list(new_stacks.values())[0]
        current["effective_stack"] = eff
        return GameState(**current)
