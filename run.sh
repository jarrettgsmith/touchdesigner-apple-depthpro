#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run setup.sh first:"
    echo "  chmod +x setup.sh"
    echo "  ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run server
echo "Starting Apple Depth Pro server..."
echo "Server will listen on localhost:9995"
echo "Press Ctrl+C to stop"
echo ""

python3 depth_pro_server.py
