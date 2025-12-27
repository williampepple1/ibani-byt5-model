@echo off
REM Quick start script for Ibani Translation System

echo ================================================
echo Ibani-English Translation System Setup
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [4/4] Setup complete!
echo.
echo ================================================
echo Next Steps:
echo ================================================
echo.
echo 1. Train the model:
echo    python train.py
echo.
echo 2. Start the API server:
echo    python app.py
echo.
echo 3. Test the API:
echo    python test_api.py
echo.
echo 4. Or use Docker:
echo    docker-compose up -d
echo.
echo ================================================
echo.
pause
