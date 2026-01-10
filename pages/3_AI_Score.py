import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# Optional SHAP
SHAP_AVAILABLE = True
try:
    import shap
except Exception:
    SHAP_AVAILABLE = False


st.set_page_config(page_title="AI Score", layout="wide")
st.title("🤖 AI Catalyst Score")


# ---------------------------
# Paths (IMPORTANT)
# ---------------------------
FEATURES_CSV = "data/features/features.csv"
MODEL_PATH = "models/rf_model_v1.pkl"
FEATURE_SCHEMA_PATH = "models/feature_schema.json"


# ---------------------------
# Loaders
# ---------------------------
@st.cache_data
def load_features_df():
    return pd.read_csv(FEATURES_CSV)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_schema():
    with open(FEATURE_SCHEMA_PATH) as f:
        return json.load(f)   # LIST


df = load_features_df()
model = load_model()
FEATURES = load_schema()


# ---------------------------
# Validation
# ---------------------------
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    st.error(f"❌ Missing features in features.csv: {missing}")
    st.stop()


X = df[FEATURES]
frames = df["frame"].values


# ---------------------------
# RF prediction + uncertainty
# ---------------------------
@st.cache_data
def predict_with_uncertainty(_model, X):
    preds = np.stack(
        [tree.predict(X) for tree in _model.estimators_]
    )
    return preds.mean(axis=0), preds.std(axis=0)



y_mean, y_std = predict_with_uncertainty(model, X)


# ---------------------------
# Frame selector
# ---------------------------
st.subheader("🎞 Frame Control")

idx = st.slider(
    "Frame",
    0,
    len(frames) - 1,
    0
)

st.metric(
    "AI Score",
    f"{y_mean[idx]:.5f}",
    f"± {y_std[idx]:.5f}"
)


# ---------------------------
# Plot
# ---------------------------
st.subheader("📈 AI Score vs Frame")

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(frames, y_mean, label="AI Score", linewidth=2)
ax.fill_between(
    frames,
    y_mean - y_std,
    y_mean + y_std,
    alpha=0.3,
    label="Confidence band (±1σ)"
)

ax.scatter(
    frames[idx],
    y_mean[idx],
    color="red",
    s=60,
    label="Selected frame"
)

ax.set_xlabel("Frame")
ax.set_ylabel("AI Score")
ax.grid(True)
ax.legend()

st.pyplot(fig)


# ---------------------------
# SHAP (optional)
# ---------------------------
st.subheader("🧠 SHAP Explanation")

if SHAP_AVAILABLE and st.checkbox("Show SHAP for selected frame"):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X.iloc[[idx]])

    fig_shap, ax_shap = plt.subplots(figsize=(9, 4))
    shap.bar_plot(
        shap_vals[0],
        feature_names=FEATURES,
        max_display=10,
        show=False
    )
    st.pyplot(fig_shap)
