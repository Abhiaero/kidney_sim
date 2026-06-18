@echo off
echo Starting Full RenalFlow-Sim Pipeline...
echo ========================================

echo.
echo [1/6] Extracting Geometry...
.\.venv\Scripts\python src\meshing\extract_geometry.py
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [2/6] Running CFD Simulation (Stokes Flow)...
.\.venv\Scripts\python src\cfd\simulate_renal_flow.py
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [3/6] Visualizing Results...
.\.venv\Scripts\python src\viz\visualize_results.py
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [4/6] Running Nephron ODE Transport Model...
.\.venv\Scripts\python src\nephron\tubular_transport_ode.py
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [5/6] Training ML Classifier...
.\.venv\Scripts\python src\classifier\train_ckd_classifier.py
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo [6/6] Training PINN Surrogate...
.\.venv\Scripts\python src\pinn\pinn_surrogate.py
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo ========================================
echo Pipeline Completed Successfully! 
echo You can now run the dashboard using: streamlit run dashboard\app.py
goto eof

:error
echo.
echo ========================================
echo Pipeline Failed! Check the errors above.
exit /b %ERRORLEVEL%

:eof
