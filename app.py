# app.py
import streamlit as st
from utils.data import load_structure_data

st.set_page_config(page_title="CO₂ Catalyst Explorer", layout="wide")

# Background load
if "df" not in st.session_state:
    with st.spinner("Initializing application..."):
        st.session_state.df = load_structure_data()

st.title("🧪 CO₂ Catalyst Explorer")

st.markdown("""
Welcome!  
Use the sidebar to navigate between:
- Structure visualization
- Analytics
- AI scoring
""")
