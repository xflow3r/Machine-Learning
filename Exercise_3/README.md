# Fashion-MNIST and CIFAR-10 Classification Project

This project compares traditional machine learning methods with deep learning approaches on two image classification datasets: Fashion-MNIST and CIFAR-10.

## Our Approach to this Exercise - What We Did

### Two Datasets
1. **Fashion-MNIST**: 28×28 grayscale images of fashion items
2. **CIFAR-10**: 32×32 RGB images of natural objects

### Traditional Methods
1. **Simple Feature**: Color Histogram + SVM/Logistic Regression
2. **Powerful Feature**: HOG (keypoint-based) + SVM/Logistic Regression

### Deep Learning
1. **CNN-Small**: Lightweight architecture (~100-130K parameters)
2. **CNN-Medium**: Deeper architecture (~300-330K parameters)
3. **Data Augmentation**: Optional flag to enable augmentation

### Evaluation
- Confusion matrices per class
- Performance metrics (accuracy, F1-macro)
- Runtime comparison (feature extraction, training, testing)

## AI Disclaimer
AI agents were used in generating the skeletons of certain files and in debugging experiments. It was also used to write some of this README and the setup scripts.

## Project Structure

```
.
├── src/
│   ├── fashion_mnist_downloader.py  # Fashion-MNIST downloader
│   ├── data.py                       # Multi-dataset loader with augmentation
│   ├── metrics.py                    # Evaluation metrics
│   ├── timing.py                     # Timing utilities
│   ├── utils.py                      # Helper functions
│   ├── main.py                       # Main entry point
│   └── models/
│       ├── traditional.py            # Color Histogram + HOG features
│       └── deep.py                   # CNN models (grayscale + RGB support)
├── results/
│   ├── tables/
│   │   └── results.csv              # Consolidated results
│   └── figures/
│       └── cm_*.png                 # Confusion matrices
├── fashion_mnist_data/              # Fashion-MNIST (auto-downloaded)
├── cifar10_data/                    # CIFAR-10 (auto-downloaded)
├── requirements.txt
└── README.md                        # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or with virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Download Datasets

Datasets are automatically downloaded on first run:
- **Fashion-MNIST**: Downloaded to `fashion_mnist_data/`
- **CIFAR-10**: Downloaded to `cifar10_data/` via torchvision

## Usage

### Quick Start - Run Everything

```bash
# Run all models on both datasets
python src/main.py --model all --dataset both --seed 42
```

### Run Specific Combinations

**Specific model on both datasets:**
```bash
python src/main.py --model hog_svm --dataset both --seed 42
```

**All models on one dataset:**
```bash
python src/main.py --model all --dataset fashion_mnist --seed 42
```

**Single model + dataset:**
```bash
python src/main.py --model cnn_medium --dataset cifar10 --device cuda --seed 42
```

### Available Models

**Traditional Methods:**
- `hist_svm`: Color Histogram + SVM (simple baseline)
- `hist_logreg`: Color Histogram + Logistic Regression (simple baseline)
- `hog_svm`: HOG + SVM (powerful keypoint-based)
- `hog_logreg`: HOG + Logistic Regression (powerful keypoint-based)

**Deep Learning:**
- `cnn_small`: Small CNN (~100K parameters)
- `cnn_medium`: Medium CNN (~300K parameters)

### Command Line Arguments

```bash
python src/main.py \
  --model {hist_svm,hist_logreg,hog_svm,hog_logreg,cnn_small,cnn_medium,all} \
  --dataset {fashion_mnist,cifar10,both} \
  --seed 42 \
  --device {cuda,cpu} \
  --epochs 10 \
  --lr 0.001 \
  --batch-size 64 \
  --augment  # Flag for data augmentation (DL only)
```

### Data Augmentation Example

```bash
# Without augmentation
python src/main.py --model cnn_medium --dataset cifar10 --seed 42

# With augmentation
python src/main.py --model cnn_medium --dataset cifar10 --augment --seed 42
```

## Results

### Output Files

1. **CSV Table**: `results/tables/results.csv`
   - All experiments in one file
   - Columns: dataset, model, accuracy, f1_macro, feature_time, train_time, test_time, total_time, seed, augmented, notes

2. **Confusion Matrices**: `results/figures/`
   - Format: `cm_{dataset}_{model}.png`
   - With augmentation: `cm_{dataset}_{model}_augmented.png`

### Expected Performance

**Fashion-MNIST (easier dataset):**
- Color Histogram: ~76-82% (simple baseline)
- HOG: ~84-89% (powerful traditional)
- CNNs: ~89-93% (deep learning)

**CIFAR-10 (harder dataset):**
- Color Histogram: ~28-35% (simple baseline struggles with complex images)
- HOG: ~42-50% (better but still limited)
- CNNs: ~65-80% (significant improvement, especially with augmentation)

## For Collaborators

### Traditional Methods (`src/models/traditional.py`)

Implements:
1. **Color Histogram Features**:
   - Simple baseline (32 bins per channel)
   - Grayscale: 32 features
   - RGB: 96 features (3×32)

2. **HOG Features**:
   - Powerful keypoint-based approach
   - Handles both grayscale and RGB
   - Orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)

3. **Classifiers**:
   - SVM with RBF kernel (C=10.0)
   - Logistic Regression (multinomial, lbfgs)

### Deep Learning (`src/models/deep.py`)

Implements:
1. **CNN-Small**:
   - 2 conv layers (32, 64 filters)
   - Adaptive pooling for different input sizes
   - ~100-130K parameters

2. **CNN-Medium**:
   - 3 conv layers (32, 64, 128 filters)
   - Batch normalization
   - ~300-330K parameters

3. **Features**:
   - Supports both grayscale (1 channel) and RGB (3 channels)
   - Data augmentation (rotation, flip, crop, color jitter)
   - Adam optimizer

## Hardware Requirements

### GPU Acceleration

**Important**: Only **deep learning models (CNNs)** can use GPU. Traditional methods are CPU-only.

| Method | GPU Support | Library |
|--------|-------------|---------|
| Color Histogram | ❌ CPU only | NumPy |
| HOG Features | ❌ CPU only | scikit-image |
| SVM | ❌ CPU only | scikit-learn |
| Logistic Regression | ❌ CPU only | scikit-learn |
| CNN-Small | ✅ GPU supported | PyTorch |
| CNN-Medium | ✅ GPU supported | PyTorch |

**Why?** scikit-learn doesn't support GPU acceleration. To use GPU for traditional ML, you'd need to switch to alternatives like cuML (NVIDIA Rapids), which requires significant code changes.

### Minimum
- CPU: Any modern processor (4+ cores recommended)
- RAM: 8 GB
- Disk: 5 GB (for datasets)

### Recommended
- GPU: NVIDIA GPU with 4+ GB VRAM (for CNNs only)
- RAM: 16 GB
- CUDA 11.0+

## Reproducibility

- **Python**: 3.8+
- **Random seed**: 42 (default)
- **Dependencies**: Listed in `requirements.txt`
- **Datasets**: Automatically downloaded with fixed versions

To reproduce results:
```bash
python src/main.py --model all --dataset both --seed 42 --epochs 10
python src/main.py --model cnn-medium --dataset both --seed 42 --epochs 10
```
