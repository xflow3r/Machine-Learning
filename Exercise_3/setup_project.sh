#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating project directory structure..."

mkdir -p src/models results/tables results/figures report fashion_mnist_data
touch src/__init__.py src/models/__init__.py

if [ ! -d "venv" ]; then
  echo "Virtual environment not found. Creating one..."
  python3 -m venv venv
  echo "Virtual environment created"
fi

echo ""
echo "Installing dependencies..."
./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
echo "Dependencies installed"

echo ""
echo "Project structure created successfully!"
