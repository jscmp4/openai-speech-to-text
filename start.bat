@echo off
title Speech to Text
cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r requirements.txt

echo.
echo Starting Speech to Text server...
echo Opening browser at http://localhost:8080
echo.
start http://localhost:8080
python app.py
pause
