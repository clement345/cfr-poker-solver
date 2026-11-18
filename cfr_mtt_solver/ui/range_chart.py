"""13×13 Plotly heatmap showing raise/call/fold frequencies per hand."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


RANKS = list("AKQJT98765432")


def build_range_matrix(
    solver,
    position: str,
    street: str = "PREFLOP",
    action_order: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return dict of action → (13, 13) frequency matrices (rows=high rank, cols=low rank).

    Diagonal = pairs, upper-right = suited, lower-left = offsuit.
    """
    from cfr_mtt_solver.infoset import pot_to_bucket, stack_to_bucket
    from cfr_mtt_solver.hands import HAND_TYPES, N_HANDS

    all_actions: set[str] = set()
    hand_strats: dict[int, dict[str, float]] = {}

    for hand_idx, hand in enumerate(HAND_TYPES):
        # Find infosets for this hand/position/street and average their strategies
        matching = {
            k: v for k, v in solver.infosets.items()
            if f"|{position}|" in k and f"|{street}|" in k and f"|hnd{hand_idx}|" in k
        }
        if not matching:
            continue
        # Average across pot/stack buckets
        agg: dict[str, float] = {}
        for infoset in matching.values():
            strat = infoset.get_average_strategy()
            for a, freq in strat.items():
                agg[a] = agg.get(a, 0.0) + freq
        n = len(matching)
        agg = {a: v / n for a, v in agg.items()}
        hand_strats[hand_idx] = agg
        all_actions.update(agg.keys())

    if not hand_strats:
        return {}

    if action_order is None:
        action_order = sorted(all_actions)

    matrices: dict[str, np.ndarray] = {a: np.zeros((13, 13)) for a in action_order}

    for hand_idx, hand in enumerate(HAND_TYPES):
        strat = hand_strats.get(hand_idx, {})
        if not strat:
            continue

        r0, r1 = hand[0], hand[1] if len(hand) > 1 else hand[0]
        row = RANKS.index(r0)   # high rank → row
        col = RANKS.index(r1)   # low rank → col

        if hand in [r + r for r in RANKS]:
            # Pair: on the diagonal
            row = col = RANKS.index(r0)
        elif hand.endswith("s"):
            # Suited: upper triangle (row < col, i.e. high rank row, low rank col)
            row = RANKS.index(r0)
            col = RANKS.index(r1)
        else:
            # Offsuit: lower triangle
            row = RANKS.index(r1)
            col = RANKS.index(r0)

        for a in action_order:
            matrices[a][row, col] = strat.get(a, 0.0)

    return matrices


def render_range_heatmap(matrices: dict[str, np.ndarray], title: str = "Range Matrix") -> go.Figure:
    """Render a stacked-color heatmap: green=raise, yellow=call, grey=fold."""
    if not matrices:
        fig = go.Figure()
        fig.add_annotation(text="No strategy data available", showarrow=False)
        return fig

    action_colors = {
        "RAISE": "#2ecc71",
        "BET": "#2ecc71",
        "SHOVE": "#27ae60",
        "CALL": "#f39c12",
        "CHECK": "#3498db",
        "FOLD": "#e74c3c",
    }

    # Determine dominant action per cell for coloring
    all_actions = list(matrices.keys())
    n = 13
    z = np.zeros((n, n))
    text = [["" for _ in range(n)] for _ in range(n)]

    for r in range(n):
        for c in range(n):
            best_a = max(all_actions, key=lambda a: matrices[a][r, c])
            z[r, c] = matrices[best_a][r, c] * 100  # percentage
            parts = [f"{a[:1]}:{matrices[a][r,c]*100:.0f}%" for a in all_actions if matrices[a][r, c] > 0.02]
            text[r][c] = "<br>".join(parts)

    # Use custom colorscale per action
    colorscale = [
        [0.0, "#e74c3c"],   # fold (red)
        [0.5, "#f39c12"],   # call (orange)
        [1.0, "#2ecc71"],   # raise (green)
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=RANKS,
        y=RANKS,
        colorscale=colorscale,
        zmin=0, zmax=100,
        text=text,
        hovertemplate="%{y}%{x}<br>%{text}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Action %", tickvals=[0, 50, 100], ticktext=["Fold", "Call", "Raise"]),
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Low rank", side="top"),
        yaxis=dict(title="High rank", autorange="reversed"),
        width=620,
        height=600,
        margin=dict(l=40, r=40, t=60, b=20),
        font=dict(size=11),
    )
    return fig
