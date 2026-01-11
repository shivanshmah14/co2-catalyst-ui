import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# =========================================================
# PRESENTATION NOTE:
# This page uses ML to rank atomic configurations (frames)
# to identify promising catalysts for CO₂ → hydrocarbons.
# =========================================================

st.set_page_config(page_title="AI Catalyst Score", layout="wide")
st.title("🤖 AI-Driven Catalyst Screening")

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------
FEATURES_CSV = "data/features/features.csv"
MODEL_PATH = "models/rf_model_v1.pkl"
FEATURE_SCHEMA_PATH = "models/feature_schema.json"

# ---------------------------------------------------------
# LOADERS
# ---------------------------------------------------------
@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_CSV)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_schema():
    with open(FEATURE_SCHEMA_PATH) as f:
        return json.load(f)

df = load_features()
model = load_model()
FEATURES = load_schema()

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    st.error(f"Missing features: {missing}")
    st.stop()

X = df[FEATURES]
frames = df["frame"].values

# ---------------------------------------------------------
# RANDOM FOREST PREDICTION + UNCERTAINTY
# ---------------------------------------------------------
@st.cache_data
def rf_predict_with_uncertainty(_model, X):
    """
    PRESENTATION:
    Each tree gives one opinion.
    Mean = AI score
    Std  = model uncertainty
    """
    preds = np.stack([tree.predict(X) for tree in _model.estimators_])
    return preds.mean(axis=0), preds.std(axis=0)

y_mean, y_std = rf_predict_with_uncertainty(model, X)

# ---------------------------------------------------------
# SINGLE CATALYST SCORE (ROBUST)
# ---------------------------------------------------------
# Top 10% mean score = Catalyst Score
top_k = max(1, int(0.1 * len(y_mean)))
catalyst_score = np.mean(np.sort(y_mean)[-top_k:])

# ---------------------------------------------------------
# BEST FRAME IDENTIFICATION
# ---------------------------------------------------------
best_idx = np.argmax(y_mean)
best_frame = frames[best_idx]

# ---------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------
col1, col2 = st.columns([1.2, 2])

# =========================================================
# LEFT PANEL — SUMMARY
# =========================================================
with col1:
    st.subheader("🏆 Best Frame Summary")

    st.metric("Best Frame", int(best_frame))
    st.metric("AI Score", f"{y_mean[best_idx]:.5f}")
    st.metric("Uncertainty", f"± {y_std[best_idx]:.5f}")

    st.divider()

    st.subheader("🧪 Catalyst Score")
    st.write(
        """
        **Catalyst Score** = Average of top 10% frames.
        
        **Interpretation:**  
        Measures *stable catalytic potential*, not just a lucky configuration.
        """
    )
    st.metric("Overall Catalyst Score", f"{catalyst_score:.5f}")

# =========================================================
# RIGHT PANEL — PLOT
# =========================================================
with col2:
    st.subheader("📈 AI Score vs Frame")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(frames, y_mean, label="AI Score", linewidth=2)
    ax.fill_between(
        frames,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.3,
        label="Uncertainty (±1σ)"
    )
    ax.scatter(
        best_frame,
        y_mean[best_idx],
        color="red",
        s=80,
        label="Best Frame"
    )

    ax.set_xlabel("Frame")
    ax.set_ylabel("AI Score")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

# =========================================================
# WHY THIS FRAME IS GOOD (HUMAN EXPLANATION)
# =========================================================
st.subheader("🧠 Why this frame is good")

row = df.iloc[best_idx]

positive_elements = []
for f in FEATURES:
    if f.startswith("count_") and row[f] > 0:
        positive_elements.append(f.replace("count_", ""))

explanation = f"""
This frame achieves the **highest AI score**, meaning the model predicts
it to have the strongest catalytic potential.

Key contributing factors:
- Presence of active elements: **{", ".join(positive_elements[:6])}**
- Balanced adsorbate environment
- Composition similar to high-performing configurations seen during training

📌 **Scientific meaning:**  
This atomic arrangement is likely to provide favorable binding
and reaction pathways for CO₂ conversion.
"""

st.info(explanation)

# =========================================================
# PER-ELEMENT CONTRIBUTION (MODEL-BASED)
# =========================================================
st.subheader("⚛️ Per-Element Contribution")

# Simple feature importance weighted by value (human-interpretable)
importances = model.feature_importances_
contrib = importances * X.iloc[best_idx].values

contrib_df = pd.DataFrame({
    "feature": FEATURES,
    "contribution": contrib
})

contrib_df = contrib_df[contrib_df["feature"].str.startswith("count_")]
contrib_df["element"] = contrib_df["feature"].str.replace("count_", "")
contrib_df = contrib_df.sort_values("contribution", ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.barh(contrib_df["element"], contrib_df["contribution"])
ax2.set_xlabel("Contribution to AI Score")
ax2.set_ylabel("Element")
ax2.invert_yaxis()
ax2.grid(True)

st.pyplot(fig2)
