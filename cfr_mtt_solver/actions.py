from __future__ import annotations
from .constants import Street, ActionType, BET_FRACTIONS
from .game_state import GameState


def get_legal_actions(state: GameState) -> list[str]:
    if state.street == Street.PREFLOP:
        return _preflop_actions(state)
    return _postflop_actions(state)


def _preflop_actions(state: GameState) -> list[str]:
    pos = state.current_player
    stack = state.get_stack(pos)
    history = state.action_history

    if stack <= 0:
        return []  # All-in player cannot act

    non_fold = [a for (p, a) in history if a != ActionType.FOLD]
    aggro = {
        ActionType.OPEN_2_5, ActionType.OPEN_3_5, ActionType.THREE_BET,
        ActionType.FOUR_BET, ActionType.SHOVE, ActionType.ISO_RAISE,
        ActionType.LIMP_RAISE,
    }

    # Find last aggressive action
    last_aggro = None
    for p, a in reversed(history):
        if a in aggro:
            last_aggro = a
            break

    limps = [a for a in non_fold if a in (ActionType.LIMP, ActionType.LIMP_BEHIND)]

    facing_raise = last_aggro in aggro if last_aggro else False
    facing_shove = last_aggro == ActionType.SHOVE
    facing_3bet = any(a == ActionType.THREE_BET for a in non_fold)
    facing_4bet = any(a == ActionType.FOUR_BET for a in non_fold)
    facing_limp = len(limps) > 0 and not facing_raise

    is_sb = pos == "SB"
    is_bb = pos == "BB"

    actions: list[str] = []

    # Facing an all-in: only fold or call
    if facing_shove:
        actions.append(ActionType.FOLD)
        actions.append(ActionType.CALL)
        return actions

    if not facing_raise and not facing_limp:
        # First in
        if is_sb or is_bb:
            if stack > 2:
                actions.append(ActionType.LIMP)
        actions.append(ActionType.FOLD)
        if stack <= 15:
            actions.append(ActionType.SHOVE)
        else:
            open_action = ActionType.OPEN_3_5 if is_sb else ActionType.OPEN_2_5
            actions.append(open_action)
        return actions

    if facing_limp and not facing_raise:
        if is_bb:
            actions.append(ActionType.CHECK)
            if stack > 5:
                actions.append(ActionType.ISO_RAISE)
            actions.append(ActionType.SHOVE)
        else:
            actions.append(ActionType.FOLD)
            actions.append(ActionType.LIMP_BEHIND)
            if stack > 5:
                actions.append(ActionType.ISO_RAISE)
            actions.append(ActionType.SHOVE)
        return actions

    # Facing raise, 3bet, or 4bet
    actions.append(ActionType.FOLD)
    actions.append(ActionType.CALL)

    if facing_4bet:
        actions.append(ActionType.SHOVE)
        return actions

    if facing_3bet:
        if stack > 20:
            actions.append(ActionType.FOUR_BET)
        actions.append(ActionType.SHOVE)
        return actions

    # Facing open or iso
    if stack > 25:
        actions.append(ActionType.THREE_BET)
        actions.append(ActionType.FOUR_BET)
    elif stack > 15:
        actions.append(ActionType.THREE_BET)
    actions.append(ActionType.SHOVE)
    return actions


def _postflop_actions(state: GameState) -> list[str]:
    pos = state.current_player
    stack = state.get_stack(pos)
    pot = state.pot

    if stack <= 0:
        return []

    history_actions = [a for (p, a) in state.action_history]
    postflop_aggro = set(BET_FRACTIONS.keys()) | {ActionType.SHOVE, ActionType.RAISE_2X, ActionType.RAISE_3X}

    # Find last bet in current street history
    last_bet = None
    for p, a in reversed(state.action_history):
        if a in postflop_aggro:
            last_bet = a
            break

    facing_bet = last_bet is not None
    facing_shove = last_bet == ActionType.SHOVE

    actions: list[str] = []

    if facing_shove:
        actions.append(ActionType.FOLD)
        actions.append(ActionType.CALL)
        return actions

    if not facing_bet:
        actions.append(ActionType.CHECK)
        if stack <= pot:
            actions.append(ActionType.SHOVE)
        else:
            # Reduced bet sizes to keep tree tractable
            actions.append(ActionType.BET_HALF)
            actions.append(ActionType.BET_POT)
            actions.append(ActionType.SHOVE)
    else:
        actions.append(ActionType.FOLD)
        actions.append(ActionType.CALL)
        last_bet_size = _estimate_last_bet_size(last_bet, pot)
        if stack > last_bet_size * 2:
            actions.append(ActionType.RAISE_2X)
        actions.append(ActionType.SHOVE)

    return actions


def _estimate_last_bet_size(action: str, pot: float) -> float:
    if action == ActionType.SHOVE:
        return pot * 2
    fraction = BET_FRACTIONS.get(action, 0.5)
    return pot * fraction


def get_bet_size_bb(action: str, pot: float, stack: float) -> float:
    if action == ActionType.SHOVE:
        return stack
    if action in BET_FRACTIONS:
        return min(BET_FRACTIONS[action] * pot, stack)
    if action == ActionType.RAISE_2X:
        return min(pot * 1.0, stack)
    if action == ActionType.RAISE_3X:
        return min(pot * 1.5, stack)
    return 0.0
