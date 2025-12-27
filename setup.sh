#!/bin/bash
# Quick start script for Ibani Translation System

echo "================================================"
echo "Ibani-English Translation System Setup"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11 or higher"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo ""
echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[4/4] Setup complete!"
echo ""
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo ""
echo "1. Train the model:"
echo "   python train.py"
echo ""
echo "2. Start the API server:"
echo "   python app.py"
echo ""
echo "3. Test the API:"
echo "   python test_api.py"
echo ""
echo "4. Or use Docker:"
echo "   docker-compose up -d"
echo ""
echo "================================================"
