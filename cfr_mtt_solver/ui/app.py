"""Streamlit GTO Solver UI — run with: streamlit run cfr_mtt_solver/ui/app.py"""
from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import numpy as np

from cfr_mtt_solver.constants import POSITION_ORDER, Street
from cfr_mtt_solver.equity_engine import EquityEngine
from cfr_mtt_solver.equity import EquityModel
from cfr_mtt_solver.icm import ICMModel
from cfr_mtt_solver.ranges import get_opening_range
from cfr_mtt_solver.tree_builder import TreeBuilder
from cfr_mtt_solver.cfr_solver import CFRSolver
from cfr_mtt_solver.ui.range_chart import build_range_matrix, render_range_heatmap
from cfr_mtt_solver.ui.ev_chart import render_ev_table
from cfr_mtt_solver.ui.convergence_chart import render_convergence

VALID_POSITIONS = [p.value for p in POSITION_ORDER]
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# ── Common spots for query ───────────────────────────────────────────────────
COMMON_SPOTS = {
    "RFI — LJ open": [("LJ", "open_2.5bb")],
    "RFI — HJ open": [("HJ", "open_2.5bb")],
    "RFI — CO open": [("CO", "open_2.5bb")],
    "RFI — BU open": [("BU", "open_2.5bb")],
    "SB vs BB (RFI)": [("SB", "open_3.5bb")],
    "BB vs LJ open": [("LJ", "open_2.5bb")],
    "BB vs HJ open": [("HJ", "open_2.5bb")],
    "BB vs CO open": [("CO", "open_2.5bb")],
    "BB vs BU open": [("BU", "open_2.5bb")],
    "BB vs SB open": [("SB", "open_3.5bb")],
    "BU vs BB 3bet (BU opened)": [("BU", "open_2.5bb"), ("BB", "3bet")],
    "CO vs BB 3bet (CO opened)": [("CO", "open_2.5bb"), ("BB", "3bet")],
    "HJ vs BB 3bet (HJ opened)": [("HJ", "open_2.5bb"), ("BB", "3bet")],
    "BU vs SB 3bet (BU opened)": [("BU", "open_2.5bb"), ("SB", "open_3.5bb")],
    "BB vs BU — 3bet pot (BB called BU open)": [("BU", "open_2.5bb"), ("BB", "3bet")],
    "BU 4bet vs BB 3bet": [("BU", "open_2.5bb"), ("BB", "3bet"), ("BU", "4bet")],
    "Custom": [],
}

st.set_page_config(
    page_title="CFR GTO Solver",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stButton > button { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }
.stButton > button:hover { background-color: #45475a; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("♠ CFR GTO Solver")
    st.markdown("---")

    mode = st.radio("Mode", ["Query trained model", "Train new spot"], index=0)

    stack_bb = st.select_slider("Stack (bb)", options=[10, 20, 30, 50, 75, 100], value=50)

    if mode == "Query trained model":
        # Model selector
        available_models = sorted(MODELS_DIR.glob("*.pkl")) if MODELS_DIR.exists() else []
        model_names = [m.name for m in available_models]
        if model_names:
            selected_model = st.selectbox("Trained model", model_names)
            model_path = MODELS_DIR / selected_model
        else:
            st.warning("No trained models found. Run `train` or switch to training mode.")
            model_path = None

        st.markdown("---")
        # Spot selector
        spot_name = st.selectbox("Spot", list(COMMON_SPOTS.keys()))
        action_history = COMMON_SPOTS[spot_name]

        if spot_name == "Custom":
            custom_spot = st.text_input(
                "Custom spot (e.g. HJ:open_2.5bb,BB:3bet)",
                help="PLAYER:ACTION pairs separated by commas",
            )
            if custom_spot.strip():
                action_history = []
                for part in custom_spot.split(","):
                    if ":" in part:
                        pl, ac = part.split(":", 1)
                        action_history.append((pl.strip().upper(), ac.strip()))

        # Hero position
        positions_in_spot = list({p for p, a in action_history})
        hero_position = st.selectbox("Query position", VALID_POSITIONS)

        query_btn = st.button("Query Range", type="primary", use_container_width=True)

    else:  # Train
        positions_str = st.text_input("Positions", value="LJ,HJ,CO,BU,SB,BB")
        positions_in_hand = [p.strip().upper() for p in positions_str.split(",")]
        iterations = st.slider("Iterations per run", 50, 2000, 500, 50)
        model_name = st.text_input("Model name", value=f"6max_{stack_bb}bb.pkl")
        train_btn = st.button("▶ Train", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("CFR+ · phevaluator · persistent memory")


# ── Main panel ───────────────────────────────────────────────────────────────
st.title("GTO Strategy Solver")

if mode == "Query trained model":
    if not query_btn and "query_result" not in st.session_state:
        st.info("Select a trained model and a spot, then click **Query Range**.")
        if not (MODELS_DIR.exists() and list(MODELS_DIR.glob("*.pkl"))):
            st.markdown("""
### Getting started

No trained model found. To train one:

```bash
python -m cfr_mtt_solver.cli train --stack 50 --iterations 1000 --rounds 5
```

This creates `models/6max_50bb.pkl`. Once trained, reload this page and query any spot.
""")
        st.stop()

    if query_btn and model_path and model_path.exists():
        with st.spinner(f"Loading {model_path.name}..."):
            builder = TreeBuilder(positions=["BU", "BB"], stack_bb=stack_bb, max_postflop_depth=0)
            root = builder.build_preflop_tree()
            solver = CFRSolver(root=root, equity_model=EquityModel(), range_dists={})
            solver.load(str(model_path))

        with st.spinner("Querying..."):
            ranges = solver.query_range(hero_position, action_history)

        st.session_state["query_result"] = ranges
        st.session_state["query_position"] = hero_position
        st.session_state["query_spot"] = spot_name
        st.session_state["query_history"] = action_history
        st.session_state["solver"] = solver

    if "query_result" in st.session_state:
        ranges = st.session_state["query_result"]
        hero_position = st.session_state["query_position"]
        spot_name = st.session_state["query_spot"]
        action_history = st.session_state["query_history"]
        solver = st.session_state.get("solver")

        found = {h: s for h, s in ranges.items() if s}

        spot_display = " → ".join(f"{p}: {a}" for p, a in action_history)
        st.subheader(f"**{hero_position}** strategy | Spot: {spot_display or 'first to act'}")

        if not found:
            st.warning("No infosets found for this spot in the model. Try training more iterations.")
            if solver:
                with st.expander("Trained spots available"):
                    for s in solver.list_trained_spots()[:50]:
                        st.code(s)
            st.stop()

        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.markdown("#### Range Matrix")
            # Build 13×13 matrices from query result
            from cfr_mtt_solver.ui.range_chart import RANKS
            import plotly.graph_objects as go

            all_actions = sorted({a for s in found.values() for a in s})
            matrices = {a: np.zeros((13, 13)) for a in all_actions}
            from cfr_mtt_solver.hands import HAND_TYPES
            for hand_idx, hand in enumerate(HAND_TYPES):
                strat = found.get(hand, {})
                if not strat:
                    continue
                r0 = hand[0]
                r1 = hand[1] if len(hand) > 1 else hand[0]
                if hand in [r + r for r in RANKS]:
                    row = col = RANKS.index(r0)
                elif hand.endswith("s"):
                    row, col = RANKS.index(r0), RANKS.index(r1)
                else:
                    row, col = RANKS.index(r1), RANKS.index(r0)
                for a in all_actions:
                    matrices[a][row, col] = strat.get(a, 0.0)

            from cfr_mtt_solver.ui.range_chart import render_range_heatmap
            fig = render_range_heatmap(matrices, title=f"{hero_position} — {spot_name}")
            st.plotly_chart(fig, use_container_width=False)

        with col2:
            st.markdown("#### Strategy Table")
            rows = []
            for hand, strat in sorted(found.items()):
                row = {"Hand": hand}
                for a in all_actions:
                    row[a] = f"{strat.get(a, 0):.1%}"
                rows.append(row)
            st.dataframe(rows, hide_index=True, height=600)

else:  # Train mode
    if not train_btn:
        st.info("Configure training settings in the sidebar and click **Train**.")
        st.markdown("""
### How training works

1. CFR+ runs on the full preflop tree (all positions and action sequences)
2. The bot learns optimal strategies for every situation simultaneously
3. Results are saved to disk — training accumulates across sessions
4. More iterations = better approximation of Nash equilibrium

**After training**, switch to "Query trained model" to explore any spot instantly.
""")
        st.stop()

    if train_btn:
        try:
            equity_model = EquityEngine()
            equity_label = "phevaluator (real)"
        except Exception:
            equity_model = EquityModel()
            equity_label = "heuristic"

        range_dists = {p: np.ones(169) / 169 for p in positions_in_hand}

        with st.spinner("Building game tree..."):
            builder = TreeBuilder(
                positions=positions_in_hand,
                stack_bb=stack_bb,
                max_postflop_depth=0,
                use_icm=False,
            )
            root = builder.build_preflop_tree()

        model_path = MODELS_DIR / model_name
        solver = CFRSolver(root=root, equity_model=equity_model, range_dists=range_dists)

        if model_path.exists():
            solver.load(str(model_path))
            st.info(f"Resuming from iteration {solver.iteration} ({len(solver.infosets):,} infosets)")

        progress_bar = st.progress(0, text="Running CFR+...")

        CHUNK = max(1, iterations // 20)
        done = 0
        while done < iterations:
            chunk = min(CHUNK, iterations - done)
            solver.solve(chunk, progress=False)
            done += chunk
            progress_bar.progress(done / iterations,
                                  text=f"CFR+ — iteration {solver.iteration} | "
                                       f"infosets: {len(solver.infosets):,}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        solver.save(str(model_path))
        progress_bar.empty()

        st.success(f"Training done! Saved to `{model_path}`")
        col1, col2, col3 = st.columns(3)
        col1.metric("Iterations", solver.iteration)
        col2.metric("Infosets", f"{len(solver.infosets):,}")
        col3.metric("Equity", equity_label)

        st.markdown("**Switch to 'Query trained model' to explore any spot.**")

        fig_conv = render_convergence(solver.convergence_history)
        st.plotly_chart(fig_conv, use_container_width=True)
