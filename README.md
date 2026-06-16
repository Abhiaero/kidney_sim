# Computational Fluid Dynamics of Human Renal Fluid Flow for Early Disease Detection

This project provides an industrial-grade foundational model to simulate human renal fluid flow using both classical CFD techniques and Machine Learning models. The core objective is to calculate and visualize complex hemodynamics and tubular flow patterns—most notably the Wall Shear Stress (WSS)—which serve as critical indicators for early-stage renal diseases such as Chronic Kidney Disease (CKD) and Diabetic Nephropathy.

## Project Structure

*   **`methodology.md`**: A detailed document outlining the governing mathematical equations (Navier-Stokes), key terminologies (WSS, Reynolds Number, Porous Media), and advanced future plans. Read this first to understand the science behind the code.
*   **`assets/`**: Contains the input image geometries used for masking real biological structures.
    *   `glomerulus_cross_section.png`: High-contrast microscopic cross-section of a renal glomerulus capillary network.
*   **`extract_geometry.py`**: Image processing script. Reads the picture of the kidney component (`assets/glomerulus_cross_section.png`), thresholds it, and creates a computational binary mask (`assets/mask.npy`) for the CFD solver to use.
*   **`simulate_renal_flow.py`**: The core CFD engine. It uses a custom 2D Finite Difference Method (FDM) with a time-marching projection algorithm to solve the incompressible Navier-Stokes equations. It incorporates the **Darcy-Brinkman Porous Media** equations to allow fluid filtration through biological tissue and dynamically calculates Non-Newtonian blood viscosity using the **Carreau-Yasuda Rheological model**.
*   **`visualize_results.py`**: The post-processing script. It loads the simulation data matrices and uses Matplotlib to generate high-quality vector fields (with log-scale highlighting porous permeation), pressure contours, and dynamic viscosity maps overlaid on the real geometry.
*   **`pinn_surrogate.py`**: Machine Learning script implementing a Physics-Informed Neural Network (PINN) using PyTorch. This acts as a proof-of-concept fast surrogate model learning the physics of the Navier-Stokes equations.
*   **`train_ckd_classifier.py`**: Machine Learning script using Scikit-Learn. It generates synthetic patient hemodynamic data (WSS, pressure gradients) and trains a Random Forest Classifier to predict CKD stages. Outputs a confusion matrix.
*   **`app.py`**: Interactive Streamlit Web Dashboard. Acts as the frontend UI where users can adjust patient parameters on sliders, view real-time CKD stage predictions from the ML model, and visualize the CFD fluid flow overlaid on the kidney geometry.
*   **`requirements.txt`**: Standard Python dependencies required to run the simulation, data processing, and ML algorithms.
*   **`results/`**: A generated folder containing raw `.npy` arrays output by the solver, the final CFD plotted image (`masked_cfd_simulation_results.png`), and the ML classification confusion matrix (`ckd_classifier_cm.png`).

## Getting Started (Short Run Guide)

Follow these steps to run the complete simulation pipeline on your local machine:

### 1. Setup the Environment
It is recommended to run this project inside an isolated virtual environment.
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Pipeline in Order
Execute the scripts in the following order to go from image extraction to full simulation and machine learning predictions:

**A. Geometry Extraction** (Processes the kidney picture into a mask):
```powershell
python extract_geometry.py
```

**B. Run CFD Simulation** (Calculates flow velocity, pressure, and Wall Shear Stress over the mask):
```powershell
python simulate_renal_flow.py
```

**C. Visualize Fluid Flow** (Saves `results/masked_cfd_simulation_results.png`):
```powershell
python visualize_results.py
```

**D. Train Physics-Informed Neural Network Surrogate** (Saves PyTorch model):
```powershell
python pinn_surrogate.py
```

**E. Train CKD Machine Learning Classifier** (Saves `results/ckd_classifier_cm.png`):
```powershell
python train_ckd_classifier.py
```

**F. Launch Interactive Dashboard** (Opens UI in default browser):
```powershell
streamlit run app.py
```

**G. Run the Full Pipeline at Once** (Executes steps A-E sequentially):
```powershell
python extract_geometry.py ; python simulate_renal_flow.py ; python visualize_results.py ; python pinn_surrogate.py ; python train_ckd_classifier.py
```
*(After the pipeline finishes, you can run step F to open the dashboard!)*

## Future Advanced Usage
For advanced 3D patient-specific modeling and Fluid-Structure Interactions, refer to the **Future Plan** section listed in `methodology.md`.
