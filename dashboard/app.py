import streamlit as st
import numpy as np
import os
import plotly.graph_objects as go
import xgboost as xgb
import plotly.express as px

import yaml

# Setup page
st.set_page_config(page_title="Renal CFD & ML Dashboard", layout="wide", page_icon="🫘")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
results_dir = os.path.join(PROJECT_ROOT, 'results')
assets_dir = os.path.join(PROJECT_ROOT, 'assets')

config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

fluid_cfg = config['physics']['fluid']
st.title("🫘 Renal Fluid Flow & CKD Prediction Dashboard")
st.markdown("Interactive dashboard integrating **Computational Fluid Dynamics**, **Nephron ODE Models**, **AI Surrogates**, and **Machine Learning** predictions for Chronic Kidney Disease (CKD).")
st.markdown(f"*(Fluid Config Active: $\\rho={fluid_cfg['rho']}$, $\\nu_{{baseline}}={fluid_cfg['nu']}$)*")

# Sidebar for ML inputs
st.sidebar.header("Patient Hemodynamics")
st.sidebar.markdown("Adjust parameters to predict CKD stage.")
wss_avg = st.sidebar.slider("Wall Shear Stress Avg (Pa)", 2.0, 30.0, 15.0)
wss_max = st.sidebar.slider("Wall Shear Stress Max (Pa)", 10.0, 80.0, 30.0)
p_drop = st.sidebar.slider("Pressure Drop (mmHg)", 5.0, 50.0, 10.0)
gfr = st.sidebar.slider("GFR (mL/min)", 10.0, 150.0, 120.0)

# Load XGBoost Model
@st.cache_resource
def get_ml_model():
    model_path = os.path.join(assets_dir, 'xgboost_ckd_model.json')
    if os.path.exists(model_path):
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
        return clf
    return None

model = get_ml_model()

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Clinical Prediction (ML)", "CFD Visualization", "Nephron ODE Model", "Feature Importance (SHAP)", "PINN Validation"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("XGBoost Machine Learning Prediction")
        if model is not None:
            # XGBoost expects 2D array
            input_data = np.array([[wss_avg, wss_max, p_drop, gfr]])
            prediction = model.predict(input_data)[0]
            
            if prediction == 0:
                st.success("### Prediction: Normal / Stage 1")
                st.write("Parameters indicate healthy glomerular filtration.")
            elif prediction == 1:
                st.warning("### Prediction: CKD Stage 2/3")
                st.write("Parameters indicate mild to moderate renal stress.")
            else:
                st.error("### Prediction: CKD Stage 4/5")
                st.write("Parameters indicate severe renal hemodynamic distress.")
        else:
            st.error("XGBoost model not found. Run the pipeline first.")
            
        # Radar chart for the patient vs normal
        categories = ['WSS Avg', 'WSS Max', 'Pressure Drop', 'GFR (normalized)']
        patient_vals = [wss_avg, wss_max, p_drop, gfr * (30/120)] # scale GFR for radar
        normal_vals = [15, 30, 10, 30]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=normal_vals, theta=categories, fill='toself', name='Healthy Baseline'))
        fig.add_trace(go.Scatterpolar(r=patient_vals, theta=categories, fill='toself', name='Current Patient'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 80])), showlegend=True, title="Hemodynamic Profile")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("### Classifier Confusion Matrix")
        cm_path = os.path.join(results_dir, "ckd_classifier_cm.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Run the ML classifier pipeline to generate the matrix.")

with tab2:
    st.subheader("CFD Simulation Output")
    st.markdown("Velocity vectors, pressure fields, and **Non-Newtonian Dynamic Viscosity** mapped to the glomerulus cross-section.")
    cfd_path = os.path.join(results_dir, "masked_cfd_simulation_results.png")
    if os.path.exists(cfd_path):
        st.image(cfd_path, use_container_width=True)
    else:
        st.info("Run the CFD pipeline to generate CFD visuals.")

with tab3:
    st.subheader("Nephron-Scale Tubular Transport Model")
    st.markdown("Weinstein-Stephenson ODE model showing fluid reabsorption and NaCl concentration along the proximal tubule.")
    ode_path = os.path.join(results_dir, "nephron_profile.png")
    if os.path.exists(ode_path):
        st.image(ode_path, use_container_width=True)
    else:
        st.info("Run the Nephron ODE script to generate this profile.")

with tab4:
    st.subheader("SHAP Feature Importance")
    st.markdown("Actual SHAP (SHapley Additive exPlanations) values from the XGBoost model evaluated on the 10,000-patient Latin Hypercube cohort.")
    
    shap_path = os.path.join(results_dir, "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, use_container_width=True)
    else:
        st.info("Run the ML Classifier pipeline to generate SHAP values.")

with tab5:
    st.subheader("Fourier Feature PINN vs CFD Validation")
    st.markdown("Validation of the AI Surrogate (Physics-Informed Neural Network) against the classical Navier-Stokes FDM solver.")
    val_path = os.path.join(results_dir, "pinn_vs_cfd_validation.png")
    if os.path.exists(val_path):
        st.image(val_path, use_container_width=True)
    else:
        st.info("Run the PINN pipeline to generate validation metrics.")
