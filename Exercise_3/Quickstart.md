# Quick Start Guide

## Setup (5 minutes)

### 1. Project Structure


### 2. Create Missing Files

If you're missing any files, run:
```bash
bash setup_project.sh
```

### 3. Install Dependencies

**Option A: Using pip directly**
```bash
pip install -r requirements.txt
```

**Option B: Using virtual environment (recommended)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

### Quick Run - All Models
```bash
python src/main.py --model all --seed 42
```

### Run Individual Models

**Traditional Methods:**
```bash
# HOG + SVM
python src/main.py --model hog_svm --seed 42

# HOG + Logistic Regression
python src/main.py --model hog_logreg --seed 42
```

**Deep Learning Methods:**
```bash
# CNN-Small (faster, fewer parameters)
python src/main.py --model cnn_small --device cuda --seed 42 --epochs 10

# CNN-Medium (more parameters, potentially better accuracy)
python src/main.py --model cnn_medium --device cuda --seed 42 --epochs 10
```

### Command Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--model` | `hog_svm`, `hog_logreg`, `cnn_small`, `cnn_medium`, `all` | `all` | Which model to run |
| `--seed` | Any integer | `42` | Random seed for reproducibility |
| `--device` | `cuda`, `cpu` | `cuda` | Device for deep learning |
| `--epochs` | Any integer | `10` | Number of training epochs |
| `--lr` | Any float | `0.001` | Learning rate |
| `--batch-size` | Any integer | `64` | Batch size |

### Examples

**CPU only (no GPU):**
```bash
python src/main.py --model all --device cpu --seed 42
```

**Longer training:**
```bash
python src/main.py --model cnn_small --epochs 20 --lr 0.0005 --seed 42
```

**Quick test run:**
```bash
python src/main.py --model cnn_small --epochs 5 --seed 42
```

## Results

After running, you'll find:

1. **CSV Results**: `results/tables/results.csv`
   - Contains accuracy, F1-score, timing information

2. **Confusion Matrices**: `results/figures/`
   - One PNG file per model
   - Example: `cm_fashion_mnist_cnn_small.png`

## Typical Runtime

On a modern GPU (e.g., RTX 3080):
- **HOG + SVM**: ~5-10 minutes (feature extraction is slow)
- **HOG + LogReg**: ~5-10 minutes
- **CNN-Small**: ~2-3 minutes (10 epochs)
- **CNN-Medium**: ~3-5 minutes (10 epochs)

On CPU:
- **Traditional methods**: Similar to GPU
- **Deep learning**: 10-30x slower (30-60 minutes per model)

## Troubleshooting

### CUDA out of memory
```bash
# Reduce batch size
python src/main.py --model cnn_small --batch-size 32 --device cuda
```

### No GPU available
```bash
# Use CPU
python src/main.py --model all --device cpu
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Dataset download fails
```bash
# Manually download from:
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
# Place files in: fashion_mnist_data/
```

## Expected Results

**Approximate performance (10 epochs):**
- HOG + SVM: ~87-89% accuracy
- HOG + LogReg: ~84-86% accuracy
- CNN-Small: ~89-91% accuracy
- CNN-Medium: ~91-93% accuracy

## Next Steps

1. Run all experiments: `python src/main.py --model all --seed 42`
2. Check results in `results/tables/results.csv`
3. Review confusion matrices in `results/figures/`
4. Write your report using the template in `README.md`
5. Analyze which classes are most confused
6. Compare traditional vs deep learning approaches

## For the Report

Key things to analyze:
1. **Performance**: Which method worked best? Why?
2. **Speed**: Training time vs accuracy tradeoff
3. **Confusion patterns**: Which classes are hard to distinguish?
4. **Traditional vs Deep Learning**: When would you use each approach?
5. **Error analysis**: Look at specific misclassifications

Good luck! 🚀