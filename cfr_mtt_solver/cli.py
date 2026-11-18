from __future__ import annotations
import json
import sys
from pathlib import Path
import click
import numpy as np

from .constants import POSITION_ORDER, Street
from .equity import EquityModel
from .equity_engine import EquityEngine
from .icm import ICMModel
from .ranges import get_opening_range
from .tree_builder import TreeBuilder
from .cfr_solver import CFRSolver
from .exporter import export_json, print_strategy_table, print_convergence, print_hand_evs
from .hands import HAND_TYPES


VALID_POSITIONS = [p.value for p in POSITION_ORDER]
VALID_STACKS = [10, 20, 50, 100]


@click.group()
def cli() -> None:
    """CFR MTT Solver — 6-max poker GTO approximation."""
    pass


@cli.command()
@click.option("--position", "-p", type=click.Choice(VALID_POSITIONS), required=True)
@click.option("--stack", "-s", type=click.Choice(["10", "20", "50", "100"]), default="50", show_default=True)
@click.option("--iterations", "-n", type=int, default=500, show_default=True)
@click.option("--depth", type=int, default=1, show_default=True, help="Postflop tree depth (0=preflop-only, 1=fast, 2=full)")
@click.option("--icm/--no-icm", default=False, show_default=True)
@click.option("--payouts", type=str, default="0.5,0.3,0.2", show_default=True, help="Comma-separated payout fractions")
@click.option("--output", "-o", type=str, default=None)
@click.option("--debug/--no-debug", default=False)
@click.option("--show-evs/--no-show-evs", default=True, show_default=True, help="Display EV per hand after solving")
@click.option("--positions-in-hand", type=str, default="BU,BB", show_default=True, help="Comma-separated positions (e.g. BU,BB or CO,BU,SB,BB)")
def solve(
    position: str,
    stack: str,
    iterations: int,
    depth: int,
    icm: bool,
    payouts: str,
    output: str | None,
    debug: bool,
    show_evs: bool,
    positions_in_hand: str,
) -> None:
    """Solve a spot and output GTO strategies."""
    stack_bb = int(stack)
    # Use real phevaluator-backed engine when tables are available, else heuristic fallback
    try:
        equity_model = EquityEngine()
    except Exception:
        equity_model = EquityModel()

    hand_positions = [p.strip().upper() for p in positions_in_hand.split(",")]

    # Validate positions
    for pos in hand_positions:
        if pos not in VALID_POSITIONS:
            click.echo(f"Invalid position: {pos}", err=True)
            sys.exit(1)

    click.echo(f"Positions: {hand_positions} | Stack: {stack_bb}bb | Iterations: {iterations}")

    icm_model: ICMModel | None = None
    if icm:
        payout_list = [float(x) for x in payouts.split(",")]
        icm_model = ICMModel(payout_list)
        click.echo(f"ICM enabled | Payouts: {payout_list}")

    # Build range distributions
    range_dists = {
        pos: get_opening_range(pos, stack_bb) for pos in hand_positions
    }

    # Build tree
    click.echo(f"Building game tree (postflop depth={depth})...")
    builder = TreeBuilder(
        positions=hand_positions,
        stack_bb=stack_bb,
        max_postflop_depth=depth,
        use_icm=icm,
    )
    root = builder.build_preflop_tree()
    click.echo("Tree built.")

    # Run CFR
    solver = CFRSolver(
        root=root,
        equity_model=equity_model,
        range_dists=range_dists,
        icm_model=icm_model,
        debug=debug,
    )

    click.echo(f"Running CFR for {iterations} iterations...")
    result = solver.solve(iterations)

    click.echo(f"\nDone. Infosets created: {len(result.infosets)}")
    click.echo(f"Final total positive regret: {result.total_positive_regret():.4f}")
    print_convergence(result.convergence_history)

    # Show sample strategy for the requested position
    click.echo(f"\nSample strategies for {position} PREFLOP:")
    print_strategy_table(solver, position, Street.PREFLOP.value, top_n=10)

    if show_evs:
        click.echo(f"\nComputing EV per hand for {position}...")
        evs = solver.compute_hand_evs(position)
        print_hand_evs(evs, position, top_n=169)

    if output:
        export_json(solver, output)


@cli.command()
@click.option("--stack", "-s", type=click.Choice(["10", "20", "50", "100"]), default="50", show_default=True)
@click.option("--model", "-m", type=str, default="models/6max_{stack}bb.pkl",
              help="Path to checkpoint file (loaded if exists, saved after each round)")
@click.option("--iterations", "-n", type=int, default=500, show_default=True,
              help="Iterations per training round")
@click.option("--rounds", "-r", type=int, default=1, show_default=True,
              help="Number of training rounds (checkpoint saved after each)")
@click.option("--positions", type=str, default="LJ,HJ,CO,BU,SB,BB", show_default=True,
              help="Comma-separated positions in the hand")
@click.option("--icm/--no-icm", default=False, show_default=True)
@click.option("--debug/--no-debug", default=False)
def train(
    stack: str,
    model: str,
    iterations: int,
    rounds: int,
    positions: str,
    icm: bool,
    debug: bool,
) -> None:
    """Train CFR+ on the full preflop tree and save to disk.

    The bot remembers everything across sessions — run multiple times to keep
    improving. Once trained, use `query` to get ranges for any situation.

    \b
    Example — initial training:
      python -m cfr_mtt_solver.cli train --stack 50 --iterations 1000 --rounds 5

    \b
    Example — continue training:
      python -m cfr_mtt_solver.cli train --stack 50 --model models/6max_50bb.pkl --rounds 10
    """
    stack_bb = int(stack)
    model_path = model.replace("{stack}", stack)
    position_list = [p.strip().upper() for p in positions.split(",")]

    for pos in position_list:
        if pos not in VALID_POSITIONS:
            click.echo(f"Invalid position: {pos}", err=True)
            sys.exit(1)

    click.echo(f"Positions: {position_list} | Stack: {stack_bb}bb | "
               f"{rounds} rounds × {iterations} iter/round")
    click.echo(f"Model: {model_path}")

    try:
        equity_model = EquityEngine()
        click.echo("Equity: phevaluator (real)")
    except Exception:
        equity_model = EquityModel()
        click.echo("Equity: heuristic (run precompute_tables for real equity)")

    # Uniform range for all players — CFR+ discovers optimal ranges
    range_dists = {p: np.ones(169) / 169 for p in position_list}

    click.echo("Building game tree...")
    builder = TreeBuilder(
        positions=position_list,
        stack_bb=stack_bb,
        max_postflop_depth=0,
        use_icm=icm,
    )
    root = builder.build_preflop_tree()
    click.echo("Tree built.")

    solver = CFRSolver(
        root=root,
        equity_model=equity_model,
        range_dists=range_dists,
        debug=debug,
    )

    # Load existing checkpoint if present
    p = Path(model_path)
    if p.exists():
        solver.load(model_path)
        click.echo(f"Loaded checkpoint: iteration {solver.iteration}, "
                   f"{len(solver.infosets):,} infosets")
    else:
        click.echo("Starting fresh training.")

    for rnd in range(1, rounds + 1):
        click.echo(f"\nRound {rnd}/{rounds} — running {iterations} iterations...")
        solver.solve(iterations, progress=True)
        solver.save(model_path)
        regret = solver.get_convergence_metric()
        click.echo(f"  Saved | iteration {solver.iteration} | "
                   f"{len(solver.infosets):,} infosets | regret {regret:.2f}")

    click.echo(f"\nTraining complete. Model saved to {model_path}")
    click.echo(f"Infosets: {len(solver.infosets):,} | "
               f"Use `query` to explore any situation.")


@cli.command()
@click.option("--model", "-m", type=str, required=True,
              help="Path to trained model (.pkl)")
@click.option("--position", "-p", type=click.Choice(VALID_POSITIONS), required=True,
              help="Player whose range to query")
@click.option("--spot", type=str, default="",
              help='Action history as comma-sep "PLAYER:ACTION" pairs. '
                   'E.g. "HJ:open_2.5bb" or "BU:open_2.5bb,BB:3bet"')
@click.option("--top", type=int, default=30, show_default=True,
              help="Show top N hands by EV/frequency")
def query(model: str, position: str, spot: str, top: int) -> None:
    """Query GTO strategy for any situation from a trained model.

    \b
    Examples:
      # BB defense vs HJ open
      python -m cfr_mtt_solver.cli query -m models/6max_50bb.pkl -p BB --spot "HJ:open_2.5bb"

      # BTN vs BB 3bet (BTN opened, BB 3bet)
      python -m cfr_mtt_solver.cli query -m models/6max_50bb.pkl -p BU --spot "BU:open_2.5bb,BB:3bet"
    """
    p = Path(model)
    if not p.exists():
        click.echo(f"Model not found: {model}. Run `train` first.", err=True)
        sys.exit(1)

    click.echo(f"Loading model from {model}...")

    # Minimal dummy solver to load infosets
    from .equity import EquityModel
    from .tree_builder import TreeBuilder
    builder = TreeBuilder(positions=["BU", "BB"], stack_bb=50, max_postflop_depth=0)
    root = builder.build_preflop_tree()
    solver = CFRSolver(root=root, equity_model=EquityModel(), range_dists={})
    solver.load(model)

    click.echo(f"Loaded {len(solver.infosets):,} infosets (iteration {solver.iteration})")

    # Parse spot
    action_history: list[tuple[str, str]] = []
    if spot.strip():
        for part in spot.split(","):
            part = part.strip()
            if ":" in part:
                pl, ac = part.split(":", 1)
                action_history.append((pl.strip().upper(), ac.strip()))
            else:
                click.echo(f"Bad format in --spot: '{part}' (expect PLAYER:ACTION)", err=True)
                sys.exit(1)

    spot_str = " → ".join(f"{p}:{a}" for p, a in action_history) if action_history else "(first to act)"
    click.echo(f"\nSpot: {spot_str}")
    click.echo(f"Querying {position}'s strategy...\n")

    ranges = solver.query_range(position, action_history)

    found = {h: s for h, s in ranges.items() if s}
    if not found:
        click.echo("No infosets found for this spot. Try more training iterations or check the spot.")
        click.echo("\nTrained spots (sample):")
        for s in solver.list_trained_spots()[:20]:
            click.echo(f"  {s}")
        return

    # Display range table
    all_actions: set[str] = set()
    for strat in found.values():
        all_actions.update(strat.keys())
    action_list = sorted(all_actions)

    header = f"{'Hand':<8}" + "".join(f"{a:>10}" for a in action_list)
    click.echo(header)
    click.echo("─" * len(header))

    # Sort by primary aggressive action (raise/open/3bet)
    aggro_actions = [a for a in action_list if a not in ("fold", "call", "check", "limp")]
    def sort_key(item):
        h, strat = item
        return -max((strat.get(a, 0) for a in aggro_actions), default=0)

    shown = 0
    for hand, strat in sorted(found.items(), key=sort_key):
        if shown >= top:
            break
        row = f"{hand:<8}" + "".join(f"{strat.get(a, 0):>10.1%}" for a in action_list)
        click.echo(row)
        shown += 1

    # Summary: overall range frequency
    if aggro_actions:
        avg_aggro = np.mean([s.get(aggro_actions[0], 0) for s in found.values() if s])
        click.echo(f"\n{position} primary action ({aggro_actions[0]}): {avg_aggro:.1%} of range")


@cli.command()
@click.option("--infoset-key", "-k", type=str, required=True)
@click.option("--input", "-i", "input_path", type=str, required=True)
def inspect(infoset_key: str, input_path: str) -> None:
    """Inspect a specific infoset from a saved JSON."""
    with open(input_path) as f:
        data = json.load(f)
    if infoset_key in data:
        click.echo(f"Strategy for: {infoset_key}")
        for action, freq in data[infoset_key].items():
            click.echo(f"  {action}: {freq:.4f}")
    else:
        click.echo(f"Key not found. Available keys (first 10):")
        for k in list(data.keys())[:10]:
            click.echo(f"  {k}")


@cli.command()
@click.option("--input", "-i", "input_path", type=str, required=True)
@click.option("--format", "-f", type=click.Choice(["json", "table"]), default="table")
@click.option("--position", "-p", type=click.Choice(VALID_POSITIONS), default=None)
@click.option("--street", "-s", type=click.Choice(["PREFLOP", "FLOP", "TURN", "RIVER"]), default="PREFLOP")
def export(input_path: str, format: str, position: str | None, street: str) -> None:
    """Export or display saved strategies."""
    with open(input_path) as f:
        data = json.load(f)

    if format == "json":
        click.echo(json.dumps(data, indent=2))
    else:
        if not position:
            click.echo("Specify --position for table format")
            return
        # Filter and display
        matching = {k: v for k, v in data.items() if f"|{position}|" in k and f"|{street}|" in k}
        if not matching:
            click.echo(f"No entries for {position} {street}")
            return
        all_actions: set[str] = set()
        for strat in matching.values():
            all_actions.update(strat.keys())
        action_list = sorted(all_actions)
        header = f"{'Key':<60}" + "".join(f"{a:>15}" for a in action_list)
        click.echo(header)
        click.echo("-" * len(header))
        for key, strat in list(matching.items())[:20]:
            row = f"{key[:60]:<60}"
            for a in action_list:
                row += f"{strat.get(a, 0.0):>15.3f}"
            click.echo(row)
