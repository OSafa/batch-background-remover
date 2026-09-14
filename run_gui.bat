@echo off
title AI Portrait Matting & BC7 DDS Exporter
echo ====================================================
echo   Starting AI Portrait Background Remover (CUDA 13)
echo ====================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found! Please run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python app.py
pause
