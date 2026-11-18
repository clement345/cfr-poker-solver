from __future__ import annotations
from .constants import (
    Street, ActionType, POSITION_ORDER, BOARD_TEXTURE_PROBS,
    BET_FRACTIONS,
)
from .game_state import GameState
from .tree import GameNode
from .actions import get_legal_actions, get_bet_size_bb


# Actions that "open" or "reopen" the betting
_AGGRO_ACTIONS = {
    ActionType.OPEN_2_5, ActionType.OPEN_3_5, ActionType.THREE_BET,
    ActionType.FOUR_BET, ActionType.SHOVE, ActionType.ISO_RAISE,
    ActionType.LIMP_RAISE,
}


class TreeBuilder:
    def __init__(
        self,
        positions: list[str],
        stack_bb: int = 50,
        max_postflop_depth: int = 3,
        use_icm: bool = False,
        tournament_chip_stacks: list[float] | None = None,
    ) -> None:
        self.positions = positions
        self.stack_bb = float(stack_bb)
        self.max_postflop_depth = max_postflop_depth
        self.use_icm = use_icm
        self.tournament_chip_stacks = tournament_chip_stacks or []

    def build_preflop_tree(self) -> GameNode:
        stacks: dict[str, float] = {p: self.stack_bb for p in self.positions}
        # Post blinds
        if "SB" in stacks:
            stacks["SB"] = self.stack_bb - 0.5
        if "BB" in stacks:
            stacks["BB"] = self.stack_bb - 1.0
        pot = 0.0
        if "SB" in stacks:
            pot += 0.5
        if "BB" in stacks:
            pot += 1.0
        elif pot == 0.0:
            pot = 1.5  # default heads-up
        # Store post-blind stacks as the reference for computing call amounts
        self._initial_stacks: dict[str, float] = dict(stacks)

        ordered = [p.value for p in POSITION_ORDER if p.value in self.positions]
        first_player = ordered[0] if ordered else self.positions[0]

        state = GameState.make(
            positions=list(stacks.keys()),
            stacks=stacks,
            pot=pot,
            street=Street.PREFLOP,
            action_history=[],
            current_player=first_player,
            board_texture=0,
            use_icm=self.use_icm,
            tournament_chip_stacks=self.tournament_chip_stacks or None,
        )
        # Every player must act at least once (BB gets the option last)
        initial_respondents = frozenset(ordered)
        return self._build_preflop_node(state, ordered[:], initial_respondents, depth=0)

    def _build_preflop_node(
        self,
        state: GameState,
        ordered_positions: list[str],
        respondents: frozenset[str],  # players who still owe an action this round
        depth: int = 0,
    ) -> GameNode:
        if depth > 40:
            return self._make_terminal(state, "showdown")

        active = list(state.positions)
        if len(active) == 1:
            return self._make_terminal(state, "fold", winner=active[0])

        actions = get_legal_actions(state)
        if not actions:
            return self._make_terminal(state, "showdown")

        node = GameNode(
            state=state,
            node_type="decision",
            player=state.current_player,
            actions=actions,
        )

        for action in actions:
            next_state, went_to_flop, is_fold_terminal, next_respondents = \
                self._apply_preflop_action(state, action, ordered_positions, respondents)

            if went_to_flop:
                node.children[action] = self._build_chance_node(next_state)
            elif is_fold_terminal:
                winner = next_state.positions[0] if next_state.positions else None
                node.children[action] = self._make_terminal(next_state, "fold", winner=winner)
            else:
                active_ordered = [p for p in ordered_positions if p in next_state.positions]
                node.children[action] = self._build_preflop_node(
                    next_state, active_ordered, next_respondents, depth + 1
                )

        return node

    # ── Preflop action amounts ──────────────────────────────────────────────

    def _open_size(self, pos: str) -> float:
        """Total chips a player puts into the pot when opening."""
        return 3.5 if pos == "SB" else 2.5

    def _3bet_size(self, pos: str) -> float:
        return 9.0

    def _4bet_size(self, pos: str, stack: float) -> float:
        return min(max(stack * 0.4, 20.0), stack)

    def _invested_so_far(self, pos: str) -> float:
        """Chips already in pot (blinds only — raise investments tracked via stacks)."""
        if pos == "SB":
            return 0.5
        if pos == "BB":
            return 1.0
        return 0.0

    def _apply_preflop_action(
        self,
        state: GameState,
        action: str,
        ordered_positions: list[str],
        respondents: frozenset[str],
    ) -> tuple[GameState, bool, bool, frozenset[str]]:
        """Returns (new_state, went_to_flop, is_fold_terminal, new_respondents)."""
        pos = state.current_player
        stacks = state.get_stacks()
        pot = state.pot
        history = list(state.action_history) + [(pos, action)]
        active = list(state.positions)

        def next_player(skip: str | None = None) -> str:
            remaining = [p for p in ordered_positions if p in active and p != skip]
            if not remaining:
                return pos
            try:
                pos_idx = ordered_positions.index(pos)
            except ValueError:
                return remaining[0]
            n = len(ordered_positions)
            for offset in range(1, n + 1):
                candidate = ordered_positions[(pos_idx + offset) % n]
                if candidate in remaining:
                    return candidate
            return remaining[0]

        # ── Respondents logic ───────────────────────────────────────────────
        # After fold/call/check: this player no longer owes an action
        # After any raise: everyone else must respond again

        if action == ActionType.FOLD:
            new_positions = [p for p in active if p != pos]
            new_resp = respondents - {pos}
            if len(new_positions) == 1:
                new_state = state.with_updates(
                    positions=new_positions,
                    action_history=history,
                    current_player=new_positions[0],
                )
                return new_state, False, True, frozenset()
            nxt = next_player(skip=pos)
            new_state = state.with_updates(
                positions=new_positions,
                action_history=history,
                current_player=nxt,
            )
            new_active = [p for p in ordered_positions if p in new_positions]
            new_resp = new_resp & frozenset(new_positions)  # remove folded
            closes = len(new_resp) == 0
            return new_state, closes, False, new_resp

        elif action in (ActionType.CALL, ActionType.CHECK):
            call_amt = self._call_amount(state) if action == ActionType.CALL else 0.0
            new_stacks = dict(stacks)
            new_stacks[pos] = max(0.0, stacks[pos] - call_amt)
            new_resp = respondents - {pos}
            closes = len(new_resp) == 0
            nxt = next_player()  # advance clockwise; irrelevant when closes=True
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + call_amt,
                action_history=history,
                current_player=nxt,
            )
            return new_state, closes, False, new_resp

        elif action in (ActionType.LIMP, ActionType.LIMP_BEHIND):
            cost = max(1.0 - (0.5 if pos == "SB" else 0.0), 0.0)
            new_stacks = dict(stacks)
            new_stacks[pos] = max(0.0, stacks[pos] - cost)
            nxt = next_player()
            new_resp = respondents - {pos}
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + cost,
                action_history=history,
                current_player=nxt,
            )
            closes = len(new_resp) == 0
            return new_state, closes, False, new_resp

        # ── Aggressive actions: everyone else must respond again ─────────────

        elif action in (ActionType.OPEN_2_5, ActionType.OPEN_3_5):
            size = self._open_size(pos)
            already_in = self._invested_so_far(pos)
            invested = max(size - already_in, 0.0)
            new_stacks = dict(stacks)
            new_stacks[pos] = max(0.0, stacks[pos] - invested)
            nxt = next_player()
            new_resp = frozenset(p for p in active if p != pos)
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + invested,
                action_history=history,
                current_player=nxt,
            )
            return new_state, False, False, new_resp

        elif action in (ActionType.THREE_BET, ActionType.ISO_RAISE, ActionType.LIMP_RAISE):
            size = self._3bet_size(pos)
            already_in = self._initial_stacks.get(pos, self.stack_bb) - stacks.get(pos, 0.0)
            invested = max(size - already_in, 0.0)
            new_stacks = dict(stacks)
            new_stacks[pos] = max(0.0, stacks[pos] - invested)
            nxt = next_player()
            new_resp = frozenset(p for p in active if p != pos)
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + invested,
                action_history=history,
                current_player=nxt,
            )
            return new_state, False, False, new_resp

        elif action == ActionType.FOUR_BET:
            invested = self._4bet_size(pos, stacks[pos])
            new_stacks = dict(stacks)
            new_stacks[pos] = max(0.0, stacks[pos] - invested)
            nxt = next_player()
            new_resp = frozenset(p for p in active if p != pos)
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + invested,
                action_history=history,
                current_player=nxt,
            )
            return new_state, False, False, new_resp

        elif action == ActionType.SHOVE:
            invested = stacks[pos]
            new_stacks = dict(stacks)
            new_stacks[pos] = 0.0
            nxt = next_player()
            new_resp = frozenset(p for p in active if p != pos)
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + invested,
                action_history=history,
                current_player=nxt,
            )
            return new_state, False, False, new_resp

        # Fallback
        new_resp = respondents - {pos}
        nxt = next_player()
        new_state = state.with_updates(action_history=history, current_player=nxt)
        return new_state, len(new_resp) == 0, False, new_resp

    def _call_amount(self, state: GameState) -> float:
        """Amount caller needs to add to match the largest outstanding bet.

        Computed as: max opponent contribution so far − caller's own contribution so far,
        capped by the caller's remaining stack (to handle all-in situations).
        Contribution = initial_stack − current_stack, which correctly accounts for
        every prior raise/open made during the hand.
        """
        stacks = state.get_stacks()
        pos = state.current_player
        initial = self._initial_stacks
        # How much each player has committed to the pot
        contributions = {
            p: max(0.0, initial.get(p, self.stack_bb) - stacks.get(p, 0.0))
            for p in state.positions
        }
        max_opp = max(
            (v for p, v in contributions.items() if p != pos),
            default=0.0,
        )
        caller_in = contributions.get(pos, 0.0)
        return min(max(max_opp - caller_in, 0.0), stacks.get(pos, 0.0))

    def _build_chance_node(self, state: GameState) -> GameNode:
        flop_state = state.with_updates(
            street=Street.FLOP,
            action_history=[],
        )
        node = GameNode(
            state=flop_state,
            node_type="chance",
            player=None,
            actions=[str(t) for t in range(8)],
        )
        ordered = [p.value for p in POSITION_ORDER if p.value in flop_state.positions]
        first_postflop = ordered[0] if ordered else list(flop_state.positions)[0]

        for texture, prob in BOARD_TEXTURE_PROBS.items():
            textured_state = flop_state.with_updates(
                board_texture=texture,
                current_player=first_postflop,
            )
            node.children[str(texture)] = self._build_postflop_node(
                textured_state, depth=0, raise_count=0
            )
        return node

    def _build_postflop_node(
        self, state: GameState, depth: int, raise_count: int
    ) -> GameNode:
        # max_postflop_depth=0 → immediate showdown, =1 → 2 levels, =2 → 4 levels
        if depth >= self.max_postflop_depth * 2 or raise_count > self.max_postflop_depth:
            return self._make_terminal(state, "showdown")

        actions = get_legal_actions(state)
        if not actions:
            return self._make_terminal(state, "showdown")

        node = GameNode(
            state=state,
            node_type="decision",
            player=state.current_player,
            actions=actions,
        )

        for action in actions:
            next_state, is_terminal, terminal_type, next_raise_count = self._apply_postflop_action(
                state, action, raise_count
            )
            if is_terminal:
                node.children[action] = self._make_terminal(next_state, terminal_type)
            elif self._street_over(action, next_state):
                node.children[action] = self._advance_street(next_state)
            else:
                node.children[action] = self._build_postflop_node(
                    next_state, depth + 1, next_raise_count
                )

        return node

    def _apply_postflop_action(
        self, state: GameState, action: str, raise_count: int
    ) -> tuple[GameState, bool, str, int]:
        pos = state.current_player
        stacks = state.get_stacks()
        pot = state.pot
        history = list(state.action_history) + [(pos, action)]
        new_raise_count = raise_count

        ordered = [p.value for p in POSITION_ORDER if p.value in state.positions]
        idx = ordered.index(pos) if pos in ordered else 0
        next_player = ordered[(idx + 1) % len(ordered)] if ordered else pos

        if action == ActionType.FOLD:
            new_positions = [p for p in state.positions if p != pos]
            winner = new_positions[0] if new_positions else next_player
            new_state = state.with_updates(
                positions=new_positions,
                action_history=history,
                current_player=winner,
            )
            return new_state, True, "fold", new_raise_count

        elif action == ActionType.CHECK:
            new_state = state.with_updates(action_history=history, current_player=next_player)
            return new_state, False, "", new_raise_count

        elif action == ActionType.CALL:
            call_amt = self._postflop_call_amount(state)
            new_stacks = dict(stacks)
            new_stacks[pos] = max(0.0, stacks[pos] - call_amt)
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + call_amt,
                action_history=history,
                current_player=next_player,
            )
            return new_state, False, "", new_raise_count

        elif action == ActionType.SHOVE:
            invested = stacks[pos]
            new_stacks = dict(stacks)
            new_stacks[pos] = 0.0
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + invested,
                action_history=history,
                current_player=next_player,
            )
            return new_state, False, "", new_raise_count + 1

        elif action in BET_FRACTIONS:
            fraction = BET_FRACTIONS[action]
            bet_size = min(fraction * pot, stacks[pos])
            new_stacks = dict(stacks)
            new_stacks[pos] = stacks[pos] - bet_size
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + bet_size,
                action_history=history,
                current_player=next_player,
            )
            return new_state, False, "", new_raise_count + 1

        elif action in (ActionType.RAISE_2X, ActionType.RAISE_3X):
            multiplier = 2.0 if action == ActionType.RAISE_2X else 3.0
            last_bet = self._postflop_call_amount(state)
            raise_size = min(last_bet * multiplier, stacks[pos])
            new_stacks = dict(stacks)
            new_stacks[pos] = stacks[pos] - raise_size
            new_state = state.with_updates(
                stacks=new_stacks,
                pot=pot + raise_size,
                action_history=history,
                current_player=next_player,
            )
            return new_state, False, "", new_raise_count + 1

        new_state = state.with_updates(action_history=history, current_player=next_player)
        return new_state, False, "", new_raise_count

    def _postflop_call_amount(self, state: GameState) -> float:
        stacks = state.get_stacks()
        pos = state.current_player
        for p, a in reversed(state.action_history):
            if a in BET_FRACTIONS:
                return min(BET_FRACTIONS[a] * state.pot, stacks.get(pos, 0.0))
            if a == ActionType.SHOVE:
                return stacks.get(pos, 0.0)
            if a in (ActionType.RAISE_2X, ActionType.RAISE_3X):
                return stacks.get(pos, 0.0) * 0.5
        return 0.0

    def _street_over(self, action: str, next_state: GameState) -> bool:
        history = next_state.action_history
        if not history:
            return False
        last_actions = [a for (p, a) in history]
        # Check-check
        n = len(last_actions)
        if n >= 2 and last_actions[-1] == ActionType.CHECK and last_actions[-2] == ActionType.CHECK:
            return True
        # Bet/raise followed by call
        if n >= 2 and last_actions[-1] == ActionType.CALL:
            prev = last_actions[-2]
            if prev in list(BET_FRACTIONS.keys()) + [ActionType.SHOVE, ActionType.RAISE_2X, ActionType.RAISE_3X]:
                return True
        return False

    def _advance_street(self, state: GameState) -> GameNode:
        street_order = [Street.FLOP, Street.TURN, Street.RIVER]
        if state.street not in street_order:
            return self._make_terminal(state, "showdown")
        idx = street_order.index(state.street)
        if idx >= len(street_order) - 1:
            return self._make_terminal(state, "showdown")
        next_street = street_order[idx + 1]
        ordered = [p.value for p in POSITION_ORDER if p.value in state.positions]
        first = ordered[0] if ordered else list(state.positions)[0]
        new_state = state.with_updates(
            street=next_street,
            action_history=[],
            current_player=first,
        )
        return self._build_postflop_node(new_state, depth=0, raise_count=0)

    def _make_terminal(
        self, state: GameState, terminal_type: str, winner: str | None = None
    ) -> GameNode:
        return GameNode(
            state=state,
            node_type="terminal",
            player=None,
            actions=[],
            is_terminal=True,
            terminal_type=terminal_type,
            winner=winner,
        )
