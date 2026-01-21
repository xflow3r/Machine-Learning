# ============================================================================
# File: README.md
# ============================================================================

# Fashion-MNIST Classification Project

This project compares traditional machine learning methods (HOG features + SVM/Logistic Regression) with deep learning approaches (CNNs) on the Fashion-MNIST dataset.

## Project Structure

```
.
├── src/
│   ├── fashion_mnist_downloader.py  # Dataset downloader
│   ├── data.py                       # Data loading utilities
│   ├── metrics.py                    # Evaluation metrics
│   ├── timing.py                     # Timing utilities
│   ├── utils.py                      # Helper functions
│   ├── main.py                       # Main entry point
│   └── models/
│       ├── traditional.py            # HOG + SVM/LogReg (Person A)
│       └── deep.py                   # CNN models (Person B)
├── results/
│   ├── tables/
│   │   └── results.csv              # Results table
│   └── figures/
│       └── cm_*.png                 # Confusion matrices
├── report/                           # Report and figures
├── fashion_mnist_data/              # Downloaded dataset (auto-created)
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

The dataset will be automatically downloaded when you run the code for the first time.

## Usage

### Run All Models

```bash
python src/main.py --model all --seed 42
```

### Run Specific Models

Traditional methods:
```bash
python src/main.py --model hog_svm --seed 42
python src/main.py --model hog_logreg --seed 42
```

Deep learning methods:
```bash
python src/main.py --model cnn_small --device cuda --seed 42
python src/main.py --model cnn_medium --device cuda --seed 42
```

### Command Line Arguments

- `--model`: Model to run (`hog_svm`, `hog_logreg`, `cnn_small`, `cnn_medium`, `all`)
- `--dataset`: Dataset name (default: `fashion_mnist`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device for deep learning (`cuda` or `cpu`, default: `cuda`)
- `--batch-size`: Batch size for deep learning (default: 64)

## For Collaborators

### Person A (Traditional Methods)

Implement in `src/models/traditional.py`:
- HOG feature extraction
- SVM classifier (`hog_svm`)
- Logistic Regression classifier (`hog_logreg`)

**Required function signature:**
```python
def run_traditional(model_name, X_train, y_train, X_test, y_test) -> dict
```

**Required return dictionary:**
```python
var = {
    'accuracy': float,
    'f1_macro': float,
    'train_time': float,
    'test_time': float,
    'feature_time': float,  # HOG extraction time
    'y_true': np.array,
    'y_pred': np.array,
    'model_name': str
}
```

### Person B (Deep Learning Methods)

Implement in `src/models/deep.py`:
- CNN-small architecture
- CNN-medium architecture
- Training loop
- Evaluation loop

**Required function signature:**
```python
def run_deep(model_name, train_loader, test_loader, device='cuda') -> dict
```

**Required return dictionary:**
```python
var = {
    'accuracy': float,
    'f1_macro': float,
    'train_time': float,
    'test_time': float,
    'y_true': np.array,
    'y_pred': np.array,
    'model_name': str
}
```

## Results

Results are automatically saved to:
- **CSV table**: `results/tables/results.csv`
- **Confusion matrices**: `results/figures/cm_fashion_mnist_<model>.png`

## Reproducibility

- Python version: 3.8+
- Random seed: 42 (default)
- All dependencies listed in `requirements.txt`
- Run with `--seed 42` for reproducible results

## Hardware

Experiments were run on:
- CPU: [Fill in]
- GPU: [Fill in]
- RAM: [Fill in]

---

# ============================================================================
# File: requirements.txt
# ============================================================================

# Core dependencies
numpy>=1.21.0
torch>=2.0.0
torchvision>=0.15.0

# Traditional ML
scikit-learn>=1.0.0
scikit-image>=0.19.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
pandas>=1.3.0
tqdm>=4.62.0

---

# ============================================================================
# File: setup_project.sh (Bash script to create directory structure)
# ============================================================================

#!/bin/bash

# Create directory structure
mkdir -p src/models
mkdir -p results/tables
mkdir -p results/figures
mkdir -p report
mkdir -p fashion_mnist_data

# Create __init__.py files
touch src/__init__.py
touch src/models/__init__.py

echo "Project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Copy the code from the artifact into the respective files"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Run: python src/main.py --model all"

---

# ============================================================================
# File: report/report_template.md
# ============================================================================

# Fashion-MNIST Classification: Traditional vs Deep Learning

## 1. Setup

### 1.1 Dataset
- **Dataset**: Fashion-MNIST
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image size**: 28×28 grayscale
- **Classes**: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

### 1.2 Preprocessing
- Normalization: [0, 255] → [0, 1]
- No data augmentation used

### 1.3 Hardware
- CPU: [Fill in]
- GPU: [Fill in]
- RAM: [Fill in]

### 1.4 Software
- Python: [version]
- PyTorch: [version]
- scikit-learn: [version]
- See `requirements.txt` for full dependencies

---

## 2. Traditional Methods

### 2.1 Feature Extraction
- **Method**: Histogram of Oriented Gradients (HOG)
- **Parameters**: [Fill in]

### 2.2 Classifiers

#### 2.2.1 SVM
- **Kernel**: [Fill in]
- **C**: [Fill in]
- **Other parameters**: [Fill in]

#### 2.2.2 Logistic Regression
- **Solver**: [Fill in]
- **Regularization**: [Fill in]
- **Max iterations**: [Fill in]

---

## 3. Deep Learning Methods

### 3.1 CNN-Small
- **Architecture**:
  - [Fill in layers]
- **Parameters**: [Total params]
- **Training**:
  - Optimizer: [e.g., Adam]
  - Learning rate: [e.g., 0.001]
  - Epochs: [e.g., 10]
  - Batch size: 64

### 3.2 CNN-Medium
- **Architecture**:
  - [Fill in layers]
- **Parameters**: [Total params]
- **Training**: [Same as CNN-Small or specify differences]

---

## 4. Results

### 4.1 Performance Metrics

| Model | Accuracy | F1-Macro | Feature Time (s) | Train Time (s) | Test Time (s) | Total Time (s) |
|-------|----------|----------|------------------|----------------|---------------|----------------|
| HOG+SVM | [Fill] | [Fill] | [Fill] | [Fill] | [Fill] | [Fill] |
| HOG+LogReg | [Fill] | [Fill] | [Fill] | [Fill] | [Fill] | [Fill] |
| CNN-Small | [Fill] | [Fill] | N/A | [Fill] | [Fill] | [Fill] |
| CNN-Medium | [Fill] | [Fill] | N/A | [Fill] | [Fill] | [Fill] |

### 4.2 Confusion Matrices

[Insert confusion matrix figures here]

---

## 5. Discussion

### 5.1 Performance Comparison
[Discuss which models performed best and why]

### 5.2 Confusion Patterns
[Discuss which classes were frequently confused and why]

### 5.3 Computational Efficiency
[Discuss the tradeoff between accuracy and training/inference time]

### 5.4 Traditional vs Deep Learning
[Compare the two approaches: when would you use each?]

### 5.5 Limitations
[Discuss limitations of the experiments]

---

## 6. Reproducibility

### 6.1 Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python src/main.py --model all --seed 42

# Run specific model
python src/main.py --model cnn_small --device cuda --seed 42
```

### 6.2 Random Seed
All experiments use seed=42 for reproducibility.

### 6.3 Package Versions
See `requirements.txt` for exact versions used.

---

## 7. Conclusion

[Brief summary of findings and key takeaways]

---

## References

[Add any references if needed]