"""Convergence plot — total positive regret over iterations."""
from __future__ import annotations
import plotly.graph_objects as go


def render_convergence(history: list[float]) -> go.Figure:
    if not history:
        fig = go.Figure()
        fig.add_annotation(text="No convergence data", showarrow=False)
        return fig

    fig = go.Figure(go.Scatter(
        x=list(range(1, len(history) + 1)),
        y=history,
        mode="lines",
        line=dict(color="#3498db", width=2),
        fill="tozeroy",
        fillcolor="rgba(52,152,219,0.15)",
        hovertemplate="Iteration %{x}: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="CFR+ Convergence — Total Positive Regret",
        xaxis_title="Iteration",
        yaxis_title="Total Positive Regret",
        height=400,
        margin=dict(l=60, r=20, t=50, b=50),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="white"),
        yaxis=dict(type="log") if (history and max(history) / max(min(history), 1e-6) > 100) else {},
    )
    return fig
