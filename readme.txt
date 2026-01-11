CO₂ Catalyst Discovery – AI Visualization Platform
=================================================

This application is an AI-assisted catalyst screening and visualization tool
for analyzing molecular dynamics (MD) simulation frames and predicting
relative catalyst performance for CO₂ conversion reactions.

A pre-trained machine learning model is used strictly for inference.
No model training is performed inside this application.


-------------------------------------------------
Project Structure
-------------------------------------------------

co2-catalyst-ui/
├── app.py                      Main Streamlit entry point
├── pages/
│   ├── 1_Analytics.py           Dataset statistics & exploration
│   ├── 2_Visualization.py       Atomic structure visualization
│   └── 3_AI_Score.py            AI-based catalyst scoring & explanation
├── models/
│   ├── rf_model_v1.pkl          Pre-trained ML model (inference only)
│   └── feature_schema.json      Feature definitions used by the model
├── data/
│   ├── testingData/
│   │   └── all_frames.csv       Atomic positions (MD output, large file)
│   └── features/
│       └── features.csv         Precomputed frame-level features
├── utils/
│   └── data.py                  Data processing utilities
└── readme.txt


-------------------------------------------------
System Requirements
-------------------------------------------------

- macOS (Intel or Apple Silicon)
- Python 3.9 – 3.11
- Minimum 8 GB RAM (16 GB recommended)


-------------------------------------------------
Installation (macOS)
-------------------------------------------------

1. Clone the repository:

   git clone <repository-url>
   cd co2-catalyst-ui


2. Prepare the data directory:

   mkdir -p data/testingData

   Place the file `all_frames.csv` (≈800 MB) inside:

   data/testingData/all_frames.csv


3. (Recommended) Create a virtual environment:

   python -m venv venv
   source venv/bin/activate


4. Install required dependencies (inference only):

   pip install streamlit pandas numpy matplotlib plotly joblib scikit-learn


5. Optional: Install SHAP for model explainability:

   pip install shap

   SHAP is used to explain *why* the model assigns a high or low score to a
   specific frame by highlighting the contribution of individual elements.
   The application runs fully without SHAP installed.


-------------------------------------------------
Running the Application
-------------------------------------------------

Start the Streamlit app:

   streamlit run app.py

Open in browser:

   http://localhost:8501


-------------------------------------------------
What the Application Does
-------------------------------------------------

✔ Visualizes atomic structures from MD simulations  
✔ Predicts AI-based catalyst scores per frame  
✔ Identifies the best-performing atomic configuration  
✔ Explains model decisions (optional SHAP)  
✔ Aggregates frame scores into a single catalyst score  

This tool is designed for **screening and prioritization**, not final validation.


-------------------------------------------------
Scientific Interpretation
-------------------------------------------------

The AI score represents a **relative catalyst suitability metric** learned
from historical simulation data.

Higher scores indicate atomic environments that statistically resemble
high-performing CO₂ conversion catalysts in the training dataset.

The predictions should be used to:
- Rank simulation frames
- Select candidates for DFT refinement
- Guide experimental catalyst design


-------------------------------------------------
Disclaimer
-------------------------------------------------

AI predictions are probabilistic and model-dependent.
Final catalyst validation must be performed using
DFT calculations and/or laboratory experiments.


-------------------------------------------------
Intended Audience
-------------------------------------------------

- Computational chemistry researchers
- Materials science & catalysis researchers
- CO₂ utilization and sustainability projects
- AI-assisted scientific discovery studies
