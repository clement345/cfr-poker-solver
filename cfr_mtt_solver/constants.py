from __future__ import annotations
from enum import Enum, auto


class Position(str, Enum):
    LJ = "LJ"
    HJ = "HJ"
    CO = "CO"
    BU = "BU"
    SB = "SB"
    BB = "BB"


POSITION_ORDER = [Position.LJ, Position.HJ, Position.CO, Position.BU, Position.SB, Position.BB]


class Street(str, Enum):
    PREFLOP = "PREFLOP"
    FLOP = "FLOP"
    TURN = "TURN"
    RIVER = "RIVER"


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    LIMP = "limp"
    LIMP_BEHIND = "limp_behind"
    OPEN_2_5 = "open_2.5bb"
    OPEN_3_5 = "open_3.5bb"
    THREE_BET = "3bet"
    FOUR_BET = "4bet"
    SHOVE = "shove"
    ISO_RAISE = "iso_raise"
    LIMP_RAISE = "limp_raise"
    BET_QUARTER = "bet_quarter"
    BET_THIRD = "bet_third"
    BET_HALF = "bet_half"
    BET_TWOTHIRDS = "bet_twothirds"
    BET_THREEQUARTERS = "bet_threequarters"
    BET_POT = "bet_pot"
    RAISE_2X = "raise_2x"
    RAISE_3X = "raise_3x"


STACK_SIZES = [10, 20, 50, 100]
DEFAULT_STACK = 50

# Postflop bet sizes as fraction of pot
BET_FRACTIONS = {
    ActionType.BET_QUARTER: 0.25,
    ActionType.BET_THIRD: 0.33,
    ActionType.BET_HALF: 0.50,
    ActionType.BET_TWOTHIRDS: 0.67,
    ActionType.BET_THREEQUARTERS: 0.75,
    ActionType.BET_POT: 1.00,
}

NUM_BOARD_TEXTURES = 8

# Board texture probabilities (unnormalized weights, will be normalized)
BOARD_TEXTURE_WEIGHTS = {
    0: 6,   # Dry rainbow A-high
    1: 8,   # Dry rainbow K/Q-high
    2: 10,  # Dry rainbow low
    3: 4,   # Monotone
    4: 8,   # Two-tone wet
    5: 10,  # Two-tone semi-wet
    6: 6,   # Paired board
    7: 3,   # Two-paired / trips
}
_total = sum(BOARD_TEXTURE_WEIGHTS.values())
BOARD_TEXTURE_PROBS = {k: v / _total for k, v in BOARD_TEXTURE_WEIGHTS.items()}
