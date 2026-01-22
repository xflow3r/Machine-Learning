# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies

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

### Quick Run - All Models on Both Datasets
```bash
python src/main.py --model all --dataset both --seed 42
```

### Run Specific Dataset

**Fashion-MNIST only:**
```bash
python src/main.py --model all --dataset fashion_mnist --seed 42
```

**CIFAR-10 only:**
```bash
python src/main.py --model all --dataset cifar10 --seed 42
```

### Run Individual Models

**Traditional Methods (Simple Baseline):**
```bash
# Color Histogram + SVM
python src/main.py --model hist_svm --dataset both --seed 42

# Color Histogram + Logistic Regression
python src/main.py --model hist_logreg --dataset both --seed 42
```

**Traditional Methods (Powerful Approach):**
```bash
# HOG + SVM
python src/main.py --model hog_svm --dataset both --seed 42

# HOG + Logistic Regression
python src/main.py --model hog_logreg --dataset both --seed 42
```

**Deep Learning Methods:**
```bash
# CNN-Small (faster, fewer parameters)
python src/main.py --model cnn_small --dataset both --device cuda --seed 42 --epochs 10

# CNN-Medium (more parameters, potentially better accuracy)
python src/main.py --model cnn_medium --dataset both --device cuda --seed 42 --epochs 10
```

### With Data Augmentation (Deep Learning Only)

```bash
# Run with data augmentation
python src/main.py --model cnn_small --dataset both --augment --seed 42

# Compare with and without augmentation
python src/main.py --model cnn_small --dataset cifar10 --seed 42
python src/main.py --model cnn_small --dataset cifar10 --augment --seed 42
```

### Command Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--model` | `hist_svm`, `hist_logreg`, `hog_svm`, `hog_logreg`, `cnn_small`, `cnn_medium`, `all` | `all` | Which model to run |
| `--dataset` | `fashion_mnist`, `cifar10`, `both` | `both` | Which dataset to use |
| `--seed` | Any integer | `42` | Random seed for reproducibility |
| `--device` | `cuda`, `cpu` | `cuda` | Device for deep learning |
| `--epochs` | Any integer | `10` | Number of training epochs |
| `--lr` | Any float | `0.001` | Learning rate |
| `--batch-size` | Any integer | `64` | Batch size |
| `--augment` | Flag | `False` | Use data augmentation (DL only) |

### Examples

**CPU only (no GPU):**
```bash
python src/main.py --model all --dataset both --device cpu --seed 42
```

**Quick test on one dataset:**
```bash
python src/main.py --model cnn_small --dataset fashion_mnist --epochs 5 --seed 42
```

**Full experiment with augmentation:**
```bash
python src/main.py --model cnn_medium --dataset cifar10 --augment --epochs 20 --lr 0.0005 --seed 42
```

**Compare simple vs powerful features:**
```bash
# Simple baseline
python src/main.py --model hist_svm --dataset fashion_mnist --seed 42

# Powerful approach
python src/main.py --model hog_svm --dataset fashion_mnist --seed 42
```

## Results

After running, you'll find:

1. **CSV Results**: `results/tables/results.csv`
   - Contains accuracy, F1-score, timing information
   - One row per experiment

2. **Confusion Matrices**: `results/figures/`
   - One PNG file per model-dataset combination
   - Example: `cm_fashion_mnist_cnn_small.png`
   - With augmentation: `cm_cifar10_cnn_medium_augmented.png`

## Typical Runtime

### Fashion-MNIST (28x28 grayscale)

On a modern GPU (e.g., RTX 3080):
- **Hist + SVM/LogReg**: ~1-2 minutes (feature extraction is fast)
- **HOG + SVM/LogReg**: ~5-10 minutes (HOG extraction is slower)
- **CNN-Small**: ~2-3 minutes (10 epochs)
- **CNN-Medium**: ~3-5 minutes (10 epochs)

On CPU:
- **Traditional methods**: Similar to GPU
- **Deep learning**: 10-30x slower (20-60 minutes per model)

### CIFAR-10 (32x32 RGB)

On a modern GPU (e.g., RTX 3080):
- **Hist + SVM/LogReg**: ~2-3 minutes (3-channel histograms)
- **HOG + SVM/LogReg**: ~10-15 minutes (RGB HOG is slower)
- **CNN-Small**: ~3-5 minutes (10 epochs)
- **CNN-Medium**: ~5-8 minutes (10 epochs)

On CPU:
- **Traditional methods**: Similar to GPU
- **Deep learning**: 10-30x slower (30-90 minutes per model)

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

**Fashion-MNIST:**
- Automatically downloaded to `fashion_mnist_data/`
- Manual download: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/

**CIFAR-10:**
- Automatically downloaded to `cifar10_data/`
- Manual download: https://www.cs.toronto.edu/~kriz/cifar.html

## Expected Results

**Approximate performance (10 epochs, no augmentation):**

### Fashion-MNIST
- Hist + SVM: ~79-82% accuracy (simple baseline)
- Hist + LogReg: ~76-79% accuracy (simple baseline)
- HOG + SVM: ~87-89% accuracy (powerful approach)
- HOG + LogReg: ~84-86% accuracy (powerful approach)
- CNN-Small: ~89-91% accuracy
- CNN-Medium: ~91-93% accuracy

### CIFAR-10
- Hist + SVM: ~30-35% accuracy (simple baseline)
- Hist + LogReg: ~28-32% accuracy (simple baseline)
- HOG + SVM: ~45-50% accuracy (powerful approach)
- HOG + LogReg: ~42-47% accuracy (powerful approach)
- CNN-Small: ~65-70% accuracy
- CNN-Medium: ~72-77% accuracy
- CNN-Medium (with augmentation): ~75-80% accuracy

## Next Steps

1. Run all experiments on both datasets
2. Compare simple (histogram) vs powerful (HOG) features
3. Compare traditional vs deep learning approaches
4. Test data augmentation on CNNs
5. Analyze confusion matrices to see which classes are difficult
6. Write your report using the provided template

## For the Report

Key things to analyze:

### 1. Feature Comparison
- How does color histogram (simple) compare to HOG (powerful)?
- Which features work better for which dataset?

### 2. Dataset Comparison
- Why does HOG work well on Fashion-MNIST but struggle on CIFAR-10?
- How do CNNs perform on both datasets?

### 3. Traditional vs Deep Learning
- When would you use traditional methods?
- What are the accuracy vs speed tradeoffs?

### 4. Data Augmentation
- Does augmentation help? How much?
- Which dataset benefits more from augmentation?

### 5. Error Analysis
- Look at confusion matrices
- Which classes are most confused?
- Are errors different between traditional and deep learning?

Good luck! 🚀