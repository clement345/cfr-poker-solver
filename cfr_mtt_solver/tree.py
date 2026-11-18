from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .game_state import GameState


@dataclass
class GameNode:
    state: GameState
    node_type: str              # 'decision' | 'chance' | 'terminal'
    player: Optional[str]       # None for chance/terminal
    actions: list[str]
    children: dict[str, "GameNode"] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_type: str = ""     # 'fold' | 'showdown' | ''
    # Filled after solving
    ev: dict[str, float] = field(default_factory=dict)
    # For terminal nodes: who won/lost
    winner: Optional[str] = None
    loser: Optional[str] = None
