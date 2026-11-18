from __future__ import annotations
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

from .constants import (
    BOARD_TEXTURE_PROBS, Street, ActionType, BET_FRACTIONS,
)
from .hands import N_HANDS, HAND_TYPES
from .equity import EquityModel
from .game_state import GameState
from .icm import ICMModel
from .infoset import InfoSet, InfoSetKey, pot_to_bucket, stack_to_bucket
from .tree import GameNode


@dataclass
class SolverResult:
    infosets: dict[str, InfoSet]
    iterations: int
    convergence_history: list[float]
    root_ev: float

    def get_strategy(self, infoset_key: str) -> dict[str, float] | None:
        if infoset_key in self.infosets:
            return self.infosets[infoset_key].get_average_strategy()
        return None

    def total_positive_regret(self) -> float:
        return sum(s.total_positive_regret() for s in self.infosets.values())


class CFRSolver:
    def __init__(
        self,
        root: GameNode,
        equity_model: EquityModel,
        range_dists: dict[str, np.ndarray],  # position -> 169-hand probability dist
        icm_model: Optional[ICMModel] = None,
        debug: bool = False,
    ) -> None:
        self.root = root
        self.equity_model = equity_model
        self.range_dists = range_dists
        self.icm_model = icm_model
        self.debug = debug

        self.infosets: dict[str, InfoSet] = {}
        self.iteration = 0
        self.convergence_history: list[float] = []
        self._root_ev = 0.0
        self._initial_stacks: dict[str, float] = dict(root.state.stacks)
        self._opp_strat_cache: dict[tuple, dict] = {}

        # Precompute normalized ranges for fast vectorized equity computation
        self._default_range_norm = np.ones(N_HANDS, dtype=np.float64) / N_HANDS
        self._normalized_ranges: dict[str, np.ndarray] = {}
        self._refresh_normalized_ranges()

    def _refresh_normalized_ranges(self) -> None:
        for p, r in self.range_dists.items():
            total = float(r.sum())
            self._normalized_ranges[p] = (r / total).astype(np.float64) if total > 0 else self._default_range_norm

    def solve(self, num_iterations: int, progress: bool = True) -> SolverResult:
        players = list(self.root.state.positions)
        disable_bar = not progress
        iter_ev = 0.0

        for _ in tqdm(range(num_iterations), disable=disable_bar, desc="CFR iterations"):
            iter_ev = 0.0
            for update_player in players:
                self._opp_strat_cache = {}

                # Counterfactual reach base: product of all opponents' hand-prob distributions
                cf_reach_base = np.ones(N_HANDS, dtype=np.float64)
                for p in players:
                    if p != update_player:
                        opp_range = self.range_dists.get(p)
                        if opp_range is not None:
                            cf_reach_base *= opp_range
                        else:
                            cf_reach_base /= N_HANDS

                hero_reach = self.range_dists.get(
                    update_player, np.ones(N_HANDS) / N_HANDS
                ).astype(np.float64).copy()

                ev_vec = self._cfr_traverse_vec(
                    self.root, hero_reach, cf_reach_base, 1.0, update_player
                )
                iter_ev += float(np.sum(ev_vec))

            self.iteration += 1
            regret = self.get_convergence_metric()
            self.convergence_history.append(regret)

            if self.debug and self.iteration % 100 == 0:
                print(f"  Iteration {self.iteration} | Total positive regret: {regret:.4f}")

        self._root_ev = iter_ev / max(1, len(players))
        return self._build_result()

    def _cfr_traverse_vec(
        self,
        node: GameNode,
        hero_reach: np.ndarray,    # (N_HANDS,) hero reach probabilities
        cf_reach_base: np.ndarray, # (N_HANDS,) product of opponents' range distributions
        cf_scalar: float,          # scalar: opp strategy probabilities accumulated along path
        update_player: str,
    ) -> np.ndarray:               # (N_HANDS,) EV for each hero hand
        if node.is_terminal:
            return self._terminal_utility_vec(node, update_player)

        if node.node_type == "chance":
            ev = np.zeros(N_HANDS, dtype=np.float64)
            for texture_str, prob in BOARD_TEXTURE_PROBS.items():
                key = str(int(texture_str))
                if key in node.children:
                    ev += prob * self._cfr_traverse_vec(
                        node.children[key], hero_reach, cf_reach_base, cf_scalar, update_player
                    )
            return ev

        current_player = node.player
        if current_player is None:
            return np.zeros(N_HANDS, dtype=np.float64)

        actions = node.actions
        n_actions = len(actions)

        if current_player == update_player:
            infosets, strat_matrix = self._get_strategy_matrix(node, update_player)
            # strat_matrix: (N_HANDS, n_actions)

            action_ev_matrix = np.zeros((N_HANDS, n_actions), dtype=np.float64)
            for i, action in enumerate(actions):
                if action not in node.children:
                    continue
                new_hero_reach = hero_reach * strat_matrix[:, i]
                action_ev_matrix[:, i] = self._cfr_traverse_vec(
                    node.children[action], new_hero_reach, cf_reach_base, cf_scalar, update_player
                )

            node_ev = (strat_matrix * action_ev_matrix).sum(axis=1)  # (N_HANDS,)

            cf_reach_full = cf_reach_base * cf_scalar  # (N_HANDS,)
            regret_deltas = cf_reach_full[:, None] * (action_ev_matrix - node_ev[:, None])
            cum_deltas = (self.iteration + 1) * hero_reach[:, None] * strat_matrix

            for h, infoset in enumerate(infosets):
                if infoset is None:
                    continue
                for i, a in enumerate(actions):
                    infoset.regrets[a] = max(0.0, infoset.regrets[a] + float(regret_deltas[h, i]))
                    infoset.cumulative_strategy[a] += float(cum_deltas[h, i])

            return node_ev

        else:
            cache_key = (id(node), current_player)
            if cache_key not in self._opp_strat_cache:
                self._opp_strat_cache[cache_key] = self._opponent_current_strategy(node, current_player)
            strategy = self._opp_strat_cache[cache_key]

            node_ev = np.zeros(N_HANDS, dtype=np.float64)
            for action in actions:
                if action not in node.children:
                    continue
                prob = strategy.get(action, 0.0)
                if prob < 1e-12:
                    continue
                action_ev = self._cfr_traverse_vec(
                    node.children[action], hero_reach, cf_reach_base, cf_scalar * prob, update_player
                )
                node_ev += prob * action_ev
            return node_ev

    def _terminal_utility_vec(self, node: GameNode, update_player: str) -> np.ndarray:
        """Vectorized terminal utility: returns (N_HANDS,) chip EV for each hero hand."""
        state = node.state
        pot = state.pot
        positions = list(state.positions)
        invested = self._initial_stacks.get(update_player, 0.0) - state.get_stack(update_player)

        if node.terminal_type == "fold" and node.winner is not None:
            utility = (pot - invested) if node.winner == update_player else -invested
            return np.full(N_HANDS, utility, dtype=np.float64)

        opponents = [p for p in positions if p != update_player]
        if not opponents:
            return np.full(N_HANDS, pot - invested, dtype=np.float64)

        opponent = opponents[0]
        norm_villain = self._normalized_ranges.get(opponent, self._default_range_norm)
        board_texture = state.board_texture

        equity_vec = self.equity_model.get_all_equities(norm_villain, board_texture)  # (N_HANDS,)
        chip_ev_vec = equity_vec * pot - invested

        if self.icm_model and state.use_icm and len(state.tournament_chip_stacks) > 0:
            stacks_list = list(state.tournament_chip_stacks)
            player_idx = _find_player_idx(update_player, positions)
            if 0 <= player_idx < len(stacks_list):
                chip_ev_vec = np.array([
                    self.icm_model.adjust_ev(float(ev), player_idx, stacks_list)
                    for ev in chip_ev_vec
                ])
        return chip_ev_vec

    def _get_strategy_matrix(
        self, node: GameNode, player: str
    ) -> tuple[list, np.ndarray]:
        """Return (infosets list len N_HANDS, strat_matrix shape (N_HANDS, n_actions)).

        Precomputes key prefix/suffix once to avoid rebuilding action history 169 times.
        """
        state = node.state
        pos = node.player or player
        actions = node.actions
        n_actions = len(actions)

        pot_b = pot_to_bucket(state.pot)
        stk_b = stack_to_bucket(state.effective_stack)
        hist = "|".join(
            f"{p}:{a.value if hasattr(a, 'value') else a}"
            for p, a in state.action_history
        )
        key_prefix = f"{pos}|{state.street.value}|pot{pot_b}|stk{stk_b}|tex{state.board_texture}"
        key_suffix = f"|{pos}|{hist}"

        strat_matrix = np.zeros((N_HANDS, n_actions), dtype=np.float64)
        infosets = []

        for h in range(N_HANDS):
            key = f"{key_prefix}|hnd{h}{key_suffix}"
            infoset = self._get_or_create_infoset(key, actions)
            infosets.append(infoset)
            strat = infoset.get_strategy()
            for i, a in enumerate(actions):
                strat_matrix[h, i] = strat.get(a, 0.0)

        return infosets, strat_matrix

    def _make_infoset_key(
        self, node: GameNode, hero_bucket: int, update_player: str
    ) -> str:
        state = node.state
        pos = node.player or update_player
        isk = InfoSetKey(
            position=pos,
            street=state.street.value,
            pot_bucket=pot_to_bucket(state.pot),
            stack_bucket=stack_to_bucket(state.effective_stack),
            action_history=state.action_history,
            board_texture=state.board_texture,
            hand_idx=hero_bucket,
            player_to_act=pos,
        )
        return isk.to_str()

    def _get_or_create_infoset(self, key: str, actions: list[str]) -> InfoSet:
        if key not in self.infosets:
            # Create dummy InfoSetKey (key is already stringified)
            dummy_key = InfoSetKey(
                position="", street="", pot_bucket=0, stack_bucket=0,
                action_history=(), board_texture=0, hand_idx=0, player_to_act=""
            )
            self.infosets[key] = InfoSet(dummy_key, actions)
        existing = self.infosets[key]
        # Add any new actions that weren't there before
        for a in actions:
            if a not in existing.actions:
                existing.actions.append(a)
                existing.regrets[a] = 0.0
                existing.cumulative_strategy[a] = 0.0
        return existing

    def compute_hand_evs(self, player: str) -> dict[str, float]:
        """Net EV in chips for each of the 169 hand types, from `player`'s perspective.

        Net EV = expected profit (chips won minus chips invested). Uses the
        converged average (Nash) strategy. Call after solve().
        """
        from .hands import HAND_TYPES
        players = list(self.root.state.positions)
        # Stacks at the root (after blinds are posted) — used to compute contribution
        initial_stacks = dict(self.root.state.stacks)
        evs: dict[str, float] = {}
        for hand_idx, hand in enumerate(HAND_TYPES):
            initial_reach = {
                p: float(self.range_dists.get(p, np.ones(N_HANDS) / N_HANDS)[hand_idx])
                for p in players
            }
            evs[hand] = self._ev_traverse(
                self.root, initial_reach, player, hand_idx, initial_stacks
            )
        return evs

    def _ev_traverse(
        self,
        node: GameNode,
        reach_probs: dict[str, float],
        update_player: str,
        hand_idx: int,
        initial_stacks: dict[str, float],
    ) -> float:
        """Traversal using average strategies — no regret updates, returns net EV."""
        if node.is_terminal:
            return self._net_ev_terminal(node, update_player, hand_idx, initial_stacks)

        if node.node_type == "chance":
            ev = 0.0
            for texture_str, prob in BOARD_TEXTURE_PROBS.items():
                key = str(int(texture_str))
                if key in node.children:
                    ev += prob * self._ev_traverse(
                        node.children[key], reach_probs, update_player, hand_idx, initial_stacks
                    )
            return ev

        current_player = node.player
        if current_player is None:
            return 0.0

        if current_player == update_player:
            infoset_key = self._make_infoset_key(node, hand_idx, update_player)
            infoset = self.infosets.get(infoset_key)
            n = len(node.actions)
            strategy = infoset.get_average_strategy() if infoset else {a: 1.0 / n for a in node.actions}
        else:
            strategy = self._opponent_avg_strategy(node, current_player)

        ev = 0.0
        for action in node.actions:
            if action not in node.children:
                continue
            new_reach = dict(reach_probs)
            if current_player in new_reach:
                new_reach[current_player] = reach_probs[current_player] * strategy.get(action, 0.0)
            ev += strategy.get(action, 0.0) * self._ev_traverse(
                node.children[action], new_reach, update_player, hand_idx, initial_stacks
            )
        return ev

    def _opponent_range_strategy(self, node: GameNode, opponent: str, use_current: bool) -> dict[str, float]:
        """Range-weighted strategy for opponent: weighted over their hand distribution."""
        opp_range = self.range_dists.get(opponent, np.ones(N_HANDS) / N_HANDS)
        n = len(node.actions)
        weighted: dict[str, float] = {a: 0.0 for a in node.actions}
        total_w = 0.0
        for opp_idx in range(N_HANDS):
            w = float(opp_range[opp_idx])
            if w <= 1e-10:
                continue
            key = self._make_infoset_key(node, opp_idx, opponent)
            infoset = self.infosets.get(key)
            if infoset:
                strat = infoset.get_strategy() if use_current else infoset.get_average_strategy()
            else:
                strat = {a: 1.0 / n for a in node.actions}
            for a in node.actions:
                weighted[a] += w * strat.get(a, 0.0)
            total_w += w
        if total_w > 1e-10:
            for a in weighted:
                weighted[a] /= total_w
        else:
            inv_n = 1.0 / n if n > 0 else 0.0
            for a in weighted:
                weighted[a] = inv_n
        return weighted

    def _opponent_current_strategy(self, node: GameNode, opponent: str) -> dict[str, float]:
        return self._opponent_range_strategy(node, opponent, use_current=True)

    def _opponent_avg_strategy(self, node: GameNode, opponent: str) -> dict[str, float]:
        return self._opponent_range_strategy(node, opponent, use_current=False)

    def _net_ev_terminal(
        self,
        node: GameNode,
        update_player: str,
        hand_idx: int,
        initial_stacks: dict[str, float],
    ) -> float:
        """Net EV at a terminal node = chips won from pot − chips invested."""
        state = node.state
        pot = state.pot
        positions = list(state.positions)

        # Chips this player put into the pot during the hand
        invested = initial_stacks.get(update_player, 0.0) - state.get_stack(update_player)

        if node.terminal_type == "fold" and node.winner is not None:
            net = (pot - invested) if node.winner == update_player else -invested
            if self.icm_model and state.use_icm and len(state.tournament_chip_stacks) > 0:
                stacks_list = list(state.tournament_chip_stacks)
                player_idx = _find_player_idx(update_player, positions)
                if 0 <= player_idx < len(stacks_list):
                    net = self.icm_model.adjust_ev(net, player_idx, stacks_list)
            return net

        # Showdown
        opponents = [p for p in positions if p != update_player]
        if not opponents:
            return pot - invested

        norm_villain = self._normalized_ranges.get(opponents[0], self._default_range_norm)
        equity = float(self.equity_model.get_all_equities(norm_villain, state.board_texture)[hand_idx])
        net = equity * pot - invested

        if self.icm_model and state.use_icm and len(state.tournament_chip_stacks) > 0:
            stacks_list = list(state.tournament_chip_stacks)
            player_idx = _find_player_idx(update_player, positions)
            if 0 <= player_idx < len(stacks_list):
                net = self.icm_model.adjust_ev(net, player_idx, stacks_list)
        return net

    def get_convergence_metric(self) -> float:
        return sum(s.total_positive_regret() for s in self.infosets.values())

    def _build_result(self) -> SolverResult:
        return SolverResult(
            infosets=self.infosets,
            iterations=self.iteration,
            convergence_history=self.convergence_history,
            root_ev=self._root_ev,
        )


    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist trained infosets and training state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "iteration": self.iteration,
            "convergence_history": self.convergence_history,
            "infosets": {
                k: {
                    "actions": is_.actions,
                    "regrets": dict(is_.regrets),
                    "cumulative": dict(is_.cumulative_strategy),
                }
                for k, is_ in self.infosets.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=4)

    def load(self, path: str | Path) -> None:
        """Load infosets from a checkpoint and resume training from that state."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.iteration = data.get("iteration", 0)
        self.convergence_history = data.get("convergence_history", [])
        dummy = InfoSetKey("", "", 0, 0, (), 0, 0, "")
        for k, d in data["infosets"].items():
            is_ = InfoSet(dummy, d["actions"])
            is_.regrets = d["regrets"]
            is_.cumulative_strategy = d["cumulative"]
            self.infosets[k] = is_
        self._refresh_normalized_ranges()

    # ── Query ────────────────────────────────────────────────────────────────

    def query_range(
        self,
        position: str,
        action_history: list[tuple[str, str]],
        street: str = "PREFLOP",
    ) -> dict[str, dict[str, float]]:
        """Return GTO strategy for each of the 169 hands at a given spot.

        Finds infosets where all specified (player, action) pairs appear in order
        in the key's action history, with only fold actions in between and after.
        This lets you query "BB vs HJ open" without listing the intermediate folds
        from CO/BU/SB.

        Returns {hand: {action: frequency}} for all 169 hand types.
        """
        # Normalize action values (handle ActionType enums or raw strings)
        specified: list[tuple[str, str]] = [
            (p, a.value if hasattr(a, "value") else str(a))
            for p, a in action_history
        ]

        prefix = f"{position}|{street}|"
        mid = f"|{position}|"  # player_to_act == position

        # Pre-filter candidate keys (cheap string checks before expensive parse)
        candidates: list[tuple[str, InfoSet]] = [
            (k, is_)
            for k, is_ in self.infosets.items()
            if k.startswith(prefix) and mid in k
        ]

        # Parse action-history portion of each candidate key once
        parsed: list[tuple[list[tuple[str, str]], InfoSet]] = []
        for k, is_ in candidates:
            parts = k.split("|")
            # Format: pos|street|potN|stkN|texN|hndN|player|P:action|...
            key_actions: list[tuple[str, str]] = []
            for part in parts[7:]:
                if ":" in part:
                    p, a = part.split(":", 1)
                    key_actions.append((p, a))
            parsed.append((key_actions, is_))

        results: dict[str, dict[str, float]] = {}
        for hand_idx, hand in enumerate(HAND_TYPES):
            hnd_tag = f"|hnd{hand_idx}|"
            matching: list[InfoSet] = []
            for (key_actions, is_), (k, _) in zip(parsed, candidates):
                if hnd_tag not in k:
                    continue
                if _history_matches(specified, key_actions):
                    matching.append(is_)

            if matching:
                agg: dict[str, float] = {}
                for is_ in matching:
                    for a, freq in is_.get_average_strategy().items():
                        agg[a] = agg.get(a, 0.0) + freq
                n = len(matching)
                results[hand] = {a: v / n for a, v in agg.items()}
            else:
                results[hand] = {}

        return results

    def list_trained_spots(self, street: str = "PREFLOP") -> list[str]:
        """Return unique action-history strings found in the trained infosets.

        Each string is in "PLAYER:action" format, "|"-separated for multi-action sequences.
        """
        seen: set[str] = set()
        for k in self.infosets:
            parts = k.split("|")
            # key: pos|street|potN|stkN|texN|hndN|player|P:action|P:action...
            if len(parts) >= 8 and parts[1] == street:
                hist = "|".join(parts[7:])
                if hist:
                    seen.add(hist)
        return sorted(seen)


def _find_player_idx(player: str, positions: list[str]) -> int:
    try:
        return positions.index(player)
    except ValueError:
        return -1


def _history_matches(
    specified: list[tuple[str, str]],
    key_actions: list[tuple[str, str]],
) -> bool:
    """True if key_actions represents the specified spot.

    All specified (player, action) pairs must appear in key_actions in order.
    Any key action that is NOT a specified action must be a fold.
    Empty specified = first-to-act only (key_actions must also be empty).
    """
    if not specified:
        return len(key_actions) == 0

    spec_idx = 0
    for kp, ka in key_actions:
        if spec_idx < len(specified):
            sp, sa = specified[spec_idx]
            if (kp, ka) == (sp, sa):
                spec_idx += 1
                continue
        # This action is not the next expected specified action
        # It must be a fold (intermediate or trailing) to be acceptable
        if ka != "fold":
            return False

    return spec_idx == len(specified)
