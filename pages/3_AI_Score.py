import streamlit as st
import pandas as pd
import numpy as np

st.title("🤖 AI Catalyst Activity Score")

df = st.session_state.df

frame = st.selectbox("Select frame", sorted(df["frame"].unique()))
frame_df = df[df["frame"] == frame]

# ---- Fake AI logic ----
num_adsorbates = frame_df["atom"].isin(["C", "O", "H"]).sum()
num_metal = frame_df["atom"].isin(["Pt", "Cu", "Ni"]).sum()

score = min(100, num_adsorbates * 2 + np.random.randint(5, 20))

st.metric("Predicted CO₂ Activity Score", f"{score}/100")

st.markdown("""
🧠 **How this works (demo logic):**
- More adsorbates → higher activity
- Random noise simulates ML uncertainty
- Replace this later with a real ML model
""")
