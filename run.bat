@echo off
echo ============================================================
echo   PotholeSense - Predictive Pothole Formation System
echo   Pune Smart City Initiative
echo ============================================================
echo.

cd /d %~dp0

SET PYTHON=C:\Users\Rishi\AppData\Local\Programs\Python\Python314\python.exe
SET STREAMLIT=C:\Users\Rishi\AppData\Local\Programs\Python\Python314\Scripts\streamlit.exe

echo [1/3] Generating synthetic dataset...
%PYTHON% data_generator.py
if errorlevel 1 (echo ERROR: data_generator.py failed & pause & exit /b 1)

echo.
echo [2/3] Training ML model...
%PYTHON% train_model.py
if errorlevel 1 (echo ERROR: train_model.py failed & pause & exit /b 1)

echo.
echo [3/3] Launching Streamlit Dashboard...
echo       Visit: http://localhost:8501
echo.
%STREAMLIT% run dashboard.py --server.port 8501

pause