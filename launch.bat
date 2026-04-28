@echo off
:: =======================================================
:: LAUNCHER FOR SPE DETECTOR
:: =======================================================

echo 1. Navigating to project folder...
cd /d "C:\Users\Public\Shared Confocal Files\SPE_Automation"

echo 2. Checking for virtual environment...
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo    - Virtual environment activated.
) else (
    echo    - ERROR: Could not find .venv\Scripts\activate.bat
    echo    - Check if your venv folder is named ".venv" or just "venv"
    pause
    exit /b
)

echo 3. Launching Python...
:: Running "main.py" with python (console visible for errors)
python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================
    echo CRITICAL ERROR OCCURRED!
    echo Read the error message above to see what went wrong.
    echo =======================================================
    pause
)