#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"

echo "=========================================="
echo "Fashion-MNIST & CIFAR-10 Classification"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please run setup_project.sh first to create the environment."
    exit 1
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo ""
echo "Creating directories..."
mkdir -p src/models results/tables results/figures report fashion_mnist_data cifar10_data
touch src/__init__.py src/models/__init__.py
echo "Directories ready"

echo ""
echo "=========================================="
echo "Starting Experiments"
echo "=========================================="
echo ""
echo "This will run:"
echo "  - 2 datasets (Fashion-MNIST, CIFAR-10)"
echo "  - 6 models per dataset (4 traditional + 2 deep learning)"
echo "  - Total: 12 experiments"
echo ""
echo "Estimated time:"
echo "  - GPU: ~1-2 hours (20 Minutes on a 4090)"
echo "  - CPU: ~4-8 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Running all models on both datasets..."
echo ""

# Run experiments
python src/main.py --model all --dataset both --seed 42 --epochs 10 --device cuda

echo ""
echo "=========================================="
echo "Experiments Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/tables/results.csv"
echo "  - results/figures/cm_*.png"
echo ""

echo "Running augmented data experiments for CNN models..."
echo ""
# Run augmented data experiments for CNN models
python src/main.py --model cnn_small --dataset both --seed 42 --epochs 10 --device cuda --augment
python src/main.py --model cnn_medium --dataset both --seed 42 --epochs 10 --device cuda --augment
