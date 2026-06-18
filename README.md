# Computational Fluid Dynamics of Human Renal Fluid Flow for Early Disease Detection

This project provides an industrial-grade foundational model to simulate human renal fluid flow using both classical CFD techniques and Machine Learning models. The core objective is to calculate and visualize complex hemodynamics and tubular flow patterns—most notably the Wall Shear Stress (WSS)—which serve as critical indicators for early-stage renal diseases such as Chronic Kidney Disease (CKD) and Diabetic Nephropathy.

### 📁 Project Structure

Following the 18-month research plan architecture, the repository is modularized for scalability:

```text
kidney_sim/
├── data/
│   ├── raw/            # Raw DICOM/NIfTI scans (gitignored)
│   ├── meshes/         # Generated .vtu, .msh files
│   └── biobank/        # Clinical feature datasets (NHANES / UK Biobank)
├── src/
│   ├── cfd/            # Fluid dynamics solvers (Navier-Stokes/Stokes)
│   │   └── simulate_renal_flow.py
│   ├── viz/            # PyVista and Matplotlib rendering scripts
│   │   └── visualize_results.py
│   ├── classifier/     # Machine Learning CKD predictors (XGBoost/RF)
│   │   └── train_ckd_classifier.py
│   ├── pinn/           # Physics-Informed Neural Network surrogates
│   │   └── pinn_surrogate.py
│   ├── meshing/        # Geometry extraction and meshing pipelines
│   │   └── extract_geometry.py
│   ├── segmentation/   # Medical image segmentation tools
│   └── nephron/        # ODE models for tubular transport
│       └── tubular_transport_ode.py
├── dashboard/          # Clinical Dashboard (Streamlit/Dash)
│   └── app.py
├── notebooks/          # Jupyter notebooks for data exploration
├── tests/              # Pytest suite
├── docs/               # Sphinx/Jupyter Book documentation
├── environment.yml     # Conda environment definition
├── dvc.yaml            # Data Version Control pipeline
└── README.md
```

### 📁 Core Files

1.  **`src/meshing/extract_geometry.py`**: A preprocessing script that takes a medical image cross-section of a glomerulus (`assets/glomerulus_cross_section.png`), applies computer vision techniques (Otsu's thresholding), and generates a binary numerical mask (`assets/mask.npy`). This mask directly informs the CFD solver where the porous solid boundaries of the capillaries exist.
2.  **`src/cfd/simulate_renal_flow.py`**: The core CFD engine. It solves the incompressible Navier-Stokes equations over the domain using a Finite Difference Method. It incorporates a Non-Newtonian Carreau-Yasuda model to simulate blood's shear-thinning behavior and enforces no-slip boundaries over the complex geometry mask.
3.  **`src/viz/visualize_results.py`**: Post-processing module. It loads the simulation tensors and generates publication-ready colormaps for Velocity, Pressure, and Dynamic Viscosity using Matplotlib.
4.  **`src/nephron/tubular_transport_ode.py`**: Implementation of Phase 3. A rigid mathematical model using a system of ODEs (solved via SciPy's stiff `Radau` integrator) simulating the **Weinstein-Stephenson epithelial transport** along the proximal tubule. Computes fluid flow and NaCl concentration drops mimicking reabsorption.
5.  **`src/classifier/train_ckd_classifier.py`**: The ML component (Phase 5). Generates a rigorous **10,000-patient virtual cohort** via Latin Hypercube Sampling. Trains an **XGBoost Classifier** on hemodynamic features to predict CKD stages and uses **SHAP** to compute actionable feature importances for clinicians.
6.  **`src/pinn/pinn_surrogate.py`**: Implementation of Phase 4. A journal-grade **Fourier Feature Physics-Informed Neural Network (PINN)** built in PyTorch. It samples collocation points from the biological fluid mask and computes spatial derivatives ($\partial u/\partial x$, $\partial^2 u/\partial x^2$) via PyTorch `autograd` to explicitly minimize the 2D Stokes Flow PDE residuals. It uses a robust two-stage **Adam $\rightarrow$ L-BFGS** optimization strategy and cross-validates its predictions by computing the Relative $L_2$ error against the CFD solver.
7.  **`dashboard/app.py`**: An interactive web dashboard built with Streamlit and Plotly. It unifies the CFD results, the Nephron ODE profiles, the XGBoost/SHAP ML predictions, and the PINN validation metrics into a single, cohesive clinical interface.

### ⚙️ Centralized Configuration
For strict reproducibility (a requirement for journal publication), all hardcoded hyperparameters have been stripped from the python scripts. Every script dynamically loads its physical constants (viscosity, lengths) and machine learning hyperparams (epochs, learning rates) from **`config.yaml`**. 

## 🚀 Run the Pipeline

You can run the entire pipeline (Extract -> CFD -> Visualize -> ODE -> ML -> PINN) automatically with a single command using the included batch script:

```bash
.\run_pipeline.bat
```

*(Alternatively, because this project includes a `dvc.yaml` file, you can also run the pipeline using Data Version Control: `dvc repro`)*

### Run Individually
If you prefer to run the steps individually, execute the following commands from the root directory (using the virtual environment):

**A. Generate Geometry Mask**
```bash
.\.venv\Scripts\python src/meshing/extract_geometry.py
```

**B. Run 2D CFD Solver**
*(Note: Requires `extract_geometry.py` to be run first)*
```bash
.\.venv\Scripts\python src/cfd/simulate_renal_flow.py
```

**C. Visualize Fluid Dynamics**
*(Note: Requires `simulate_renal_flow.py` to be run first)*
```bash
.\.venv\Scripts\python src/viz/visualize_results.py
```

**D. Run Nephron ODE Model**
```bash
.\.venv\Scripts\python src/nephron/tubular_transport_ode.py
```

**E. Train ML CKD Classifier (XGBoost + SHAP)**
```bash
.\.venv\Scripts\python src/classifier/train_ckd_classifier.py
```

**F. Train PINN Surrogate**
```bash
.\.venv\Scripts\python src/pinn/pinn_surrogate.py
```

**G. Launch the Dashboard**
```bash
.\.venv\Scripts\streamlit run dashboard/app.py
```

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
