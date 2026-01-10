# utils/data.py
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_structure_data():
    df = pd.read_csv(
        "data/testingdata/all_frames.csv",
        names=["file", "frame", "atom", "x", "y", "z"],
        dtype={"frame": int},
        skiprows=1
    )
    return df
