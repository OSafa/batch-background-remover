@echo off
title AI Portrait Remover Setup
echo ====================================================
echo   Setting up Environment with CUDA 13 PyTorch
echo ====================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA 13 (cu130)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo Installing supporting packages...
pip install -r requirements.txt

echo.
echo ====================================================
echo   Setup Complete! You can now launch run_gui.bat
echo ====================================================
pause
