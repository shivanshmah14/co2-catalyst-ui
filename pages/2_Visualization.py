import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🔬 Atomic Structure Viewer (Frame Slider)")

# -----------------------------
# Load data (already in session)
# -----------------------------
df = st.session_state.df.copy()

# Make sure frame is numeric
df["frame"] = df["frame"].astype(int)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

# Atom selector
atoms = sorted(df["atom"].unique())
selected_atom = st.sidebar.selectbox(
    "Select atom type",
    atoms
)

# Filter by atom FIRST
df_atom = df[df["atom"] == selected_atom]

# Get available frames for this atom
frames = sorted(df_atom["frame"].unique())

# Frame slider (INDEX based – this is critical)
frame_idx = st.sidebar.slider(
    "Simulation frame",
    min_value=0,
    max_value=len(frames) - 1,
    value=0,
    help="Drag to move through simulation time"
)

# Actual frame value
current_frame = frames[frame_idx]

# Filter data for this frame
df_frame = df_atom[df_atom["frame"] == current_frame]

# -----------------------------
# Marker size (highlight adsorbates)
# -----------------------------
adsorbates = {"C", "O", "H"}
marker_size = 10 if selected_atom in adsorbates else 5

# -----------------------------
# Plot
# -----------------------------
fig = go.Figure(
    go.Scatter3d(
        x=df_frame["x"],
        y=df_frame["y"],
        z=df_frame["z"],
        mode="markers",
        marker=dict(
            size=marker_size,
            opacity=0.9
        )
    )
)

fig.update_layout(
    title=f"Atom: {selected_atom} | Frame: {current_frame}",
    scene=dict(
        xaxis_title="X (Å)",
        yaxis_title="Y (Å)",
        zaxis_title="Z (Å)"
    )
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Explanation
# -----------------------------
st.markdown(
    """
**What you are seeing**

- Each **frame** is one snapshot from a molecular dynamics (MD) simulation  
- Dragging the slider moves forward/backward in simulation time  
- You are visualizing **only one atom type at a time**, which keeps things fast  
- Different frames → different atomic positions → different image
"""
)
