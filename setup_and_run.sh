#!/bin/bash
# One-click Setup and Run for Adversarial Arena

echo "Step 1: Checking Python..."
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python3."
    exit 1
fi

echo "Step 2: Creating Virtual Environment (if missing)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
fi

echo "Step 3: Installing Dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit matplotlib opencv-python scipy torch torchvision tqdm requests

echo "Step 4: Launching Arena..."
streamlit run src/app.py
