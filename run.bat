@echo off

set VENV_DIR=venv

rem Check if virtual environment exists
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

rem Activate the virtual environment
call %VENV_DIR%\Scripts\activate

rem Upgrade pip if needed
python -m pip --version | find "python 3." > nul
if %errorlevel% neq 0 (
    echo Upgrading pip...
    python -m pip install --upgrade pip
)

rem Install dependencies if not already installed
python -m pip check -r requirements.txt > nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    python -m pip install -r requirements.txt
) else (
    echo Requirements already satisfied.
)

rem Run the main Python script
python main.py

rem Deactivate the virtual environment
call %VENV_DIR%\Scripts\deactivate
