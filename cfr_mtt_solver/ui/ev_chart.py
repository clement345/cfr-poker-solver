"""EV table with inline bar visualization."""
from __future__ import annotations
import plotly.graph_objects as go


def render_ev_table(evs: dict[str, float], position: str = "") -> go.Figure:
    """Render a bar chart of hand EVs (sorted descending)."""
    if not evs:
        fig = go.Figure()
        fig.add_annotation(text="No EV data", showarrow=False)
        return fig

    sorted_hands = sorted(evs.items(), key=lambda x: x[1], reverse=True)
    hands = [h for h, _ in sorted_hands]
    values = [v for _, v in sorted_hands]

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig = go.Figure(go.Bar(
        x=hands,
        y=values,
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:+.3f} bb<extra></extra>",
    ))

    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.update_layout(
        title=f"Hand EV — {position}" if position else "Hand EV",
        xaxis_title="Hand",
        yaxis_title="Net EV (bb)",
        height=500,
        margin=dict(l=40, r=20, t=50, b=60),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="white"),
        xaxis=dict(tickangle=45),
    )
    return fig
