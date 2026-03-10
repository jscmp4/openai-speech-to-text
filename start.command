#!/bin/bash
cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.10+"
    echo "  macOS: brew install python3"
    exit 1
fi

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found. Please install ffmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    exit 1
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Starting Speech to Text server..."
echo "Opening browser at http://localhost:8080"
echo ""

# Open browser (macOS / Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8080
else
    xdg-open http://localhost:8080 2>/dev/null || true
fi

python app.py
