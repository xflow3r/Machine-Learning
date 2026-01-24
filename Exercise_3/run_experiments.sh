#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

requirements_ok() {
  "$VENV_PY" - <<'PY'
import sys
from importlib import metadata
from packaging.requirements import Requirement

req_path = "requirements.txt"

def iter_reqs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r ", "--requirement ", "-e ", "--editable ", "--find-links", "-f ", "--extra-index-url", "--index-url")):
                continue
            yield line

bad = []
for line in iter_reqs(req_path):
    try:
        req = Requirement(line)
    except Exception:
        continue

    try:
        installed = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        bad.append(f"{req.name} (not installed)")
        continue

    if req.specifier and installed not in req.specifier:
        bad.append(f"{req.name} ({installed} does not satisfy {req.specifier})")

if bad:
    print("Requirements not satisfied:")
    for b in bad:
        print(" -", b)
    sys.exit(1)
sys.exit(0)
PY
}

run_setup_in_new_terminal_and_wait() {
  local setup_cmd
  setup_cmd="cd \"$SCRIPT_DIR\" && bash \"$SCRIPT_DIR/setup_project.sh\""

  if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    bash "$SCRIPT_DIR/setup_project.sh"
    return 0
  fi

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --wait -- bash -lc "$setup_cmd" &
    wait $!
  elif command -v konsole >/dev/null 2>&1; then
    konsole -e bash -lc "$setup_cmd" &
    wait $!
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -e bash -lc "$setup_cmd" &
    wait $!
  elif command -v xterm >/dev/null 2>&1; then
    xterm -e bash -lc "$setup_cmd" &
    wait $!
  else
    bash "$SCRIPT_DIR/setup_project.sh"
  fi
}

echo "=========================================="
echo "Fashion-MNIST & CIFAR-10 Classification"
echo "=========================================="
echo ""

need_setup=0

if [ ! -d "$VENV_DIR" ]; then
  need_setup=1
else
  if [ -f "$REQ_FILE" ]; then
    if ! requirements_ok; then
      need_setup=1
    fi
  fi
fi

if [ "$need_setup" -eq 1 ]; then
  echo "Setup required (missing venv and/or requirements). Launching setup in a new terminal window..."
  run_setup_in_new_terminal_and_wait
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
