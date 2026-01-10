import streamlit as st
import pandas as pd

st.title("📊 Structure Analytics")

# ---- Explanation ----
st.markdown(
    """
### What is this data?

This table shows the **raw atomic structure data** extracted from a  
**Molecular Dynamics (MD) simulation**.

Each row represents **one atom at a specific simulation time step (frame)**.

The data answers three fundamental questions:
- **Which atoms are present?** (chemical composition)
- **Where are they located in space?** (x, y, z coordinates)
- **How does the structure change over time?** (different frames)

This is the *lowest-level representation* of the system and is the
foundation for visualization, analytics, and machine-learning models.
"""
)

df = st.session_state.df

# ---- Frame selection ----
frame = st.selectbox(
    "Select Simulation Time Step",
    sorted(df["frame"].unique()),
    help=(
        "Each frame is a snapshot of the atomic positions at a specific "
        "time during the simulation. Selecting different frames lets you "
        "inspect how the structure evolves."
    )
)

frame_df = df[df["frame"] == frame]

# ---- Column explanation ----
st.markdown(
    """
### How to read the table

- **frame** → Simulation time step (snapshot in time)
- **atom** → Chemical element of the atom (e.g., Ni, O, C)
- **x, y, z** → 3D spatial coordinates of the atom (in Ångström)

Together, these columns fully describe the atomic configuration at this moment in time.
"""
)

# ---- Raw table ----
st.subheader("Raw Atomic Data")
st.dataframe(frame_df, height=400)

# ---- Why this matters ----
st.markdown(
    """
### Why is this important?

This raw data is used to:
- Build **3D visualizations** of atomic structures
- Compute **structural statistics** (atom counts, distances, coordination)
- Generate **features for machine learning**
- Validate that the simulation ran correctly

All higher-level insights come from this table.
"""
)

# ---- Export ----
csv = frame_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download frame as CSV",
    csv,
    file_name=f"frame_{frame}.csv",
    mime="text/csv"
)
