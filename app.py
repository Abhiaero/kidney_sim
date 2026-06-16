import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier

# Setup page
st.set_page_config(page_title="Renal CFD & ML Dashboard", layout="wide", page_icon="🫘")

st.title("🫘 Renal Fluid Flow & CKD Prediction Dashboard")
st.markdown("Interactive dashboard integrating **Computational Fluid Dynamics (Non-Newtonian Rheology & Porous Media Filtration)** over real geometry and **Machine Learning** predictions for Chronic Kidney Disease (CKD).")

# Sidebar for ML inputs
st.sidebar.header("Patient Hemodynamics")
st.sidebar.markdown("Adjust parameters to predict CKD stage.")
wss_avg = st.sidebar.slider("Wall Shear Stress Avg (Pa)", 1.0, 30.0, 15.0)
wss_max = st.sidebar.slider("Wall Shear Stress Max (Pa)", 10.0, 80.0, 30.0)
p_drop = st.sidebar.slider("Pressure Drop (mmHg)", 5.0, 50.0, 10.0)
gfr = st.sidebar.slider("GFR (mL/min)", 5.0, 150.0, 120.0)

# Load or Train simple ML model in memory
@st.cache_resource
def get_ml_model():
    # Train dummy RF model matching our script
    n_samples = 1000
    np.random.seed(42)
    X = np.zeros((n_samples, 4))
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        stage = np.random.choice([0, 1, 2])
        y[i] = stage
        if stage == 0:
            X[i] = [np.random.normal(15, 2), np.random.normal(30, 5), np.random.normal(10, 1), np.random.normal(120, 10)]
        elif stage == 1:
            X[i] = [np.random.normal(10, 3), np.random.normal(45, 8), np.random.normal(15, 2), np.random.normal(75, 15)]
        else:
            X[i] = [np.random.normal(5, 2), np.random.normal(60, 10), np.random.normal(25, 4), np.random.normal(20, 10)]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

model = get_ml_model()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Machine Learning Prediction")
    prediction = model.predict([[wss_avg, wss_max, p_drop, gfr]])[0]
    
    if prediction == 0:
        st.success("### Prediction: Normal / Stage 1")
        st.write("Parameters indicate healthy glomerular filtration.")
    elif prediction == 1:
        st.warning("### Prediction: CKD Stage 2/3")
        st.write("Parameters indicate mild to moderate renal stress.")
    else:
        st.error("### Prediction: CKD Stage 4/5")
        st.write("Parameters indicate severe renal hemodynamic distress.")
        
    st.markdown("---")
    st.write("### Classifier Confusion Matrix")
    if os.path.exists("results/ckd_classifier_cm.png"):
        st.image("results/ckd_classifier_cm.png", use_container_width=True)
    else:
        st.info("Run `train_ckd_classifier.py` to generate the matrix.")

with col2:
    st.subheader("CFD Simulation Output")
    st.markdown("Velocity vectors, pressure fields, and **Non-Newtonian Dynamic Viscosity** mapped to the glomerulus cross-section.")
    if os.path.exists("results/masked_cfd_simulation_results.png"):
        st.image("results/masked_cfd_simulation_results.png", use_container_width=True)
    else:
        st.info("Run `simulate_renal_flow.py` and `visualize_results.py` to generate CFD visuals.")
