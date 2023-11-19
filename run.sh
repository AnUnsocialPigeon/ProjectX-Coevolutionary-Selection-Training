#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip if needed
if [ "$(pip --version | grep -o 'python 3.')" != "python 3." ]; then
    echo "Upgrading pip..."
    pip install --upgrade pip
fi

# Install dependencies if not already installed
if ! pip check -r requirements.txt > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Requirements already satisfied."
fi

# Run the main Python script
python main.py

# Deactivate the virtual environment
deactivate
