from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cfr_solver import CFRSolver


def export_json(solver: "CFRSolver", path: str) -> None:
    """Write average strategies for all infosets to JSON."""
    data: dict[str, dict[str, float]] = {}
    for key, infoset in solver.infosets.items():
        data[key] = infoset.get_average_strategy()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported {len(data)} infosets to {path}")


def print_strategy_table(
    solver: "CFRSolver",
    position: str,
    street: str,
    top_n: int = 20,
) -> None:
    """Print strategy table filtered by position and street."""
    try:
        from rich.table import Table
        from rich.console import Console
        _print_rich_table(solver, position, street, top_n)
    except ImportError:
        _print_plain_table(solver, position, street, top_n)


def _print_rich_table(solver: "CFRSolver", position: str, street: str, top_n: int) -> None:
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title=f"Strategy: {position} | {street}")

    # Find matching infosets
    matching = {
        k: v for k, v in solver.infosets.items()
        if f"|{position}|" in k and f"|{street}|" in k
    }

    if not matching:
        console.print(f"[yellow]No infosets found for {position} {street}[/yellow]")
        return

    # Collect all actions
    all_actions: set[str] = set()
    for infoset in matching.values():
        all_actions.update(infoset.actions)
    action_list = sorted(all_actions)

    table.add_column("InfoSet Key", style="cyan", no_wrap=False, max_width=50)
    for action in action_list:
        table.add_column(action, justify="right")

    shown = 0
    for key, infoset in list(matching.items())[:top_n]:
        avg = infoset.get_average_strategy()
        row = [key[:50]]
        for action in action_list:
            freq = avg.get(action, 0.0)
            row.append(f"{freq:.2f}")
        table.add_row(*row)
        shown += 1

    console.print(table)
    console.print(f"Showing {shown} of {len(matching)} infosets for {position} {street}")


def _print_plain_table(solver: "CFRSolver", position: str, street: str, top_n: int) -> None:
    matching = {
        k: v for k, v in solver.infosets.items()
        if f"|{position}|" in k and f"|{street}|" in k
    }
    if not matching:
        print(f"No infosets found for {position} {street}")
        return

    all_actions: set[str] = set()
    for infoset in matching.values():
        all_actions.update(infoset.actions)
    action_list = sorted(all_actions)

    header = f"{'Key':<60}" + "".join(f"{a:>15}" for a in action_list)
    print(header)
    print("-" * len(header))

    for key, infoset in list(matching.items())[:top_n]:
        avg = infoset.get_average_strategy()
        row = f"{key[:60]:<60}"
        for action in action_list:
            row += f"{avg.get(action, 0.0):>15.3f}"
        print(row)

    print(f"\n{min(top_n, len(matching))} of {len(matching)} infosets shown")


def print_hand_evs(evs: dict[str, float], player: str, top_n: int = 50) -> None:
    """Print EV per hand sorted best → worst."""
    sorted_evs = sorted(evs.items(), key=lambda x: x[1], reverse=True)
    try:
        from rich.table import Table
        from rich.console import Console
        console = Console()
        table = Table(title=f"Hand EV — {player}")
        table.add_column("Hand", style="cyan", width=6)
        table.add_column("EV (bb)", justify="right", width=10)
        table.add_column("Bar", width=30)
        max_ev = sorted_evs[0][1] if sorted_evs else 1.0
        min_ev = sorted_evs[-1][1] if sorted_evs else 0.0
        span = max(abs(max_ev), abs(min_ev), 0.01)
        for hand, ev in sorted_evs[:top_n]:
            bar_len = int(abs(ev) / span * 20)
            bar = ("+" if ev >= 0 else "-") * bar_len
            color = "green" if ev >= 0 else "red"
            table.add_row(hand, f"{ev:+.3f}", f"[{color}]{bar}[/{color}]")
        console.print(table)
    except ImportError:
        print(f"\nHand EV — {player}:")
        print(f"{'Hand':<6} {'EV (bb)':>10}")
        print("-" * 18)
        for hand, ev in sorted_evs[:top_n]:
            print(f"{hand:<6} {ev:>+10.3f}")


def print_convergence(convergence_history: list[float]) -> None:
    """Print convergence chart to terminal."""
    if not convergence_history:
        print("No convergence data.")
        return
    print(f"\nConvergence over {len(convergence_history)} iterations:")
    print(f"  Start regret: {convergence_history[0]:.4f}")
    print(f"  Final regret: {convergence_history[-1]:.4f}")
    if len(convergence_history) > 1:
        reduction = (1 - convergence_history[-1] / max(convergence_history[0], 1e-9)) * 100
        print(f"  Reduction: {reduction:.1f}%")
