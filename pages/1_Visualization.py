import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🔬 Atomic Structure Viewer (MD Animation)")

# -----------------------------
# Load data
# -----------------------------
df = st.session_state.df.copy()

# Ensure frame is clean for animation
df["frame"] = df["frame"].astype(str)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

atom_list = sorted(df["atom"].unique())

selected_atom = st.sidebar.selectbox(
    "Filter by Atom Type",
    options=atom_list,
    index=0,
    help="Select which atomic species to display"
)

# ✅ FILTER DATA (safe with animation)
df_filtered = df[df["atom"] == selected_atom].copy()

# -----------------------------
# Highlight adsorbates
# -----------------------------
adsorbates = ["C", "O", "H"]

df_filtered["size"] = df_filtered["atom"].apply(
    lambda a: 10 if a in adsorbates else 5
)

# -----------------------------
# Animated plot
# -----------------------------
fig = px.scatter_3d(
    df_filtered,
    x="x",
    y="y",
    z="z",
    color="atom",
    size="size",
    animation_frame="frame",
    title="Atomic Motion Over Simulation Time"
)

fig.update_layout(
    scene=dict(
        xaxis_title="X (Å)",
        yaxis_title="Y (Å)",
        zaxis_title="Z (Å)"
    )
)

# -----------------------------
# Slow down animation
# -----------------------------
if fig.layout.updatemenus:
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0

# -----------------------------
# Display
# -----------------------------
st.plotly_chart(fig, width="stretch")

# -----------------------------
# Explanation
# -----------------------------
st.markdown(
    """
**Explanation**  
This animation shows how a selected atomic species moves over time during a molecular dynamics (MD) simulation.  
Each frame corresponds to one simulation time step.  
Atoms such as carbon, oxygen, and hydrogen are emphasized because they are part of CO₂-derived intermediates interacting with the catalyst surface.
"""
)
