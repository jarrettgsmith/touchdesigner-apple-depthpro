#!/bin/bash
# Setup script for TouchDesigner Depth Pro Server
# Creates virtual environment and installs all dependencies

set -e  # Exit on error

echo "========================================"
echo "Depth Pro Server Setup"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="$SCRIPT_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing virtual environment"
        exit 0
    fi
fi

echo ""
echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Installing Depth Pro package..."
DEPTH_PRO_DIR="$SCRIPT_DIR/external/ml-depth-pro"

if [ ! -d "$DEPTH_PRO_DIR" ]; then
    echo "ERROR: Depth Pro directory not found: $DEPTH_PRO_DIR"
    echo "Please ensure ml-depth-pro is in the external/ directory"
    exit 1
fi

# Install in editable mode
pip install -e "$DEPTH_PRO_DIR"

echo ""
echo "Downloading Depth Pro checkpoint..."
CHECKPOINT_DIR="$DEPTH_PRO_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# Check if checkpoint exists
if [ -f "$CHECKPOINT_DIR/depth_pro.pt" ]; then
    echo "Checkpoint already exists"
else
    echo "Running checkpoint download script..."
    cd "$DEPTH_PRO_DIR"

    if [ -f "get_pretrained_models.sh" ]; then
        bash get_pretrained_models.sh
    else
        echo "WARNING: get_pretrained_models.sh not found"
        echo "You may need to manually download the checkpoint"
        echo "See: https://github.com/apple/ml-depth-pro"
    fi

    cd "$SCRIPT_DIR"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Verify checkpoint exists at: $CHECKPOINT_DIR/depth_pro.pt"
echo "2. Run the server with: ./run_server.sh"
echo "3. Or use from TouchDesigner with scriptTOP_depthpro.py"
echo ""
