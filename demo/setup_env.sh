#!/bin/bash
# setup_env.sh - Create .venv and install dependencies for Mac demonstration

set -e

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEMO_DIR"

echo "============================================="
echo "  RabiGuard Mac Demo: Environment Setup"
echo "============================================="

# 1. Create a virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv' in $DEMO_DIR..."
    python3 -m venv .venv
    echo "✅ Virtual environment created."
else
    echo "⚠️  Virtual environment '.venv' already exists. Skipping creation."
fi

# 2. Activate virtual environment and upgrade pip
echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

# 3. Install core dependencies
echo "Installing required Python packages..."
# PyTorch, Torchvision, OpenCV, Ultralytics, NCNN, Flask, Firebase-admin, Timm
pip install numpy opencv-python flask ultralytics ncnn torch torchvision firebase-admin timm

echo "============================================="
echo "✅ Environment setup successfully completed!"
echo "To activate this virtual environment, run:"
echo "  source demo/.venv/bin/activate"
echo "============================================="
