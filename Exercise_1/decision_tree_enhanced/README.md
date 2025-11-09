# Enhanced Decision Tree Analysis

This directory contains comprehensive decision tree analysis scripts for all datasets in Exercise 1, with enhanced metrics, parameter sensitivity analysis, and LaTeX-ready visualizations.

## Directory Structure

```
decision_tree_enhanced/
├── phishing/
│   ├── decision_tree_enhanced.py       # Enhanced analysis script
│   ├── phishing_results.csv            # 20 experiments x 20 metrics
│   ├── phishing_accuracy_comparison.png
│   ├── phishing_depth_sensitivity.png
│   ├── phishing_overfitting_analysis.png
│   ├── phishing_training_time.png
│   ├── phishing_complexity_analysis.png
│   └── phishing_confusion_matrix.png
│
├── voting/
│   ├── decision_tree_enhanced.py
│   ├── voting_results.csv
│   ├── voting_accuracy_comparison.png
│   ├── voting_depth_sensitivity.png
│   ├── voting_overfitting_analysis.png
│   ├── voting_training_time.png
│   └── voting_complexity_analysis.png  (no confusion matrix - no test labels)
│
├── road_safety/
│   ├── decision_tree_enhanced.py
│   ├── road_safety_results.csv
│   ├── road_safety_accuracy_comparison.png
│   ├── road_safety_depth_sensitivity.png
│   ├── road_safety_overfitting_analysis.png
│   ├── road_safety_training_time.png
│   └── road_safety_complexity_analysis.png  (no confusion matrix - 11 classes too large)
│
└── reviews/
    ├── decision_tree_enhanced.py
    ├── reviews_results.csv
    ├── reviews_accuracy_comparison.png
    ├── reviews_depth_sensitivity.png
    ├── reviews_overfitting_analysis.png
    ├── reviews_training_time.png
    └── reviews_complexity_analysis.png  (no confusion matrix - no test labels)
```

## Features

### Comprehensive Performance Measures
- **Accuracy**: Train, validation, and test set performance
- **Precision**: Macro and weighted averages
- **Recall**: Macro and weighted averages
- **F1-Score**: Macro and weighted averages
- **Overfitting Metrics**: Train-val and train-test gaps
- **Model Complexity**: Tree depth and number of leaves
- **Training Time**: Computational cost analysis

### 20 Systematic Experiments

Each script runs 20 experiments organized into 5 categories:

1. **Baseline Experiments (6)**:
   - Holdout 80/20 (default parameters)
   - Holdout 70/30 (default parameters)
   - Holdout 80/20 (depth=5)
   - 5-Fold CV (default parameters)
   - 10-Fold CV (default parameters)
   - 5-Fold CV (depth=5)

2. **Depth Sensitivity (5)**: max_depth = 3, 5, 10, 15, 20, None

3. **Split Sensitivity (4)**: min_samples_split = 2, 5/10, 10/20, 20/50, 50/100
   - Values scale with dataset size

4. **Leaf Sensitivity (4)**: min_samples_leaf = 1, 5/10, 10/20, 20/50, 30/100
   - Values scale with dataset size

5. **Criterion Comparison (1)**: gini vs entropy

### Visualizations (PNG @ 300 DPI)

All visualizations are LaTeX-compatible with serif fonts:

1. **Accuracy Comparison**: Train/val/test across all 20 experiments
2. **Depth Sensitivity**: Performance vs tree depth
3. **Overfitting Analysis**: Train-val and train-test gaps
4. **Training Time**: Computational cost comparison
5. **Complexity Analysis**: Tree depth and number of leaves
6. **Confusion Matrix**: Best model (when applicable)

## Dataset Summaries

### Phishing Dataset
- **Size**: 1,082 train / 271 test
- **Features**: 9 (after dropping Result_Label)
- **Classes**: 3 (Legitimate: -1, Suspicious: 0, Phishing: 1)
- **Best Result**: 
  - Validation: 89.46% (Exp10: 5-Fold CV, depth=10, split=5)
  - Test: 89.30%
  - Test F1 (macro): 86.67%
- **Key Finding**: Moderate overfitting, optimal depth around 10

### Voting Dataset
- **Size**: 218 train / 217 test (no test labels)
- **Features**: 16 (after dropping ID)
- **Classes**: 2 (Democrat, Republican)
- **Best Result**:
  - Validation: 97.73% (Exp1: Holdout 80/20)
  - Test: N/A (no labels)
- **Key Finding**: High accuracy, sensitive to validation split; handles NaN natively

### Road Safety Dataset
- **Size**: 261,899 train / 65,475 test
- **Features**: 53 (after preprocessing: dropped 14 columns, engineered 4 temporal features)
- **Classes**: 11 (age bands: 0-5, 6-10, ..., Over 75)
- **Best Result**:
  - Validation: 51.28% (Exp15: 5-Fold CV, split=100)
  - Test: 53.11% (Exp9: 5-Fold CV, depth=15)
  - Test F1 (macro): 62.73%
- **Key Finding**: Imbalanced classes (23% in 26-35 band, 0.02% in 0-5 band); uses 50k sample for CV

### Reviews Dataset
- **Size**: 750 train / 750 test (no test labels)
- **Features**: 10,001 (TF-IDF features from review text)
- **Classes**: 50 (reviewer names - author attribution task)
- **Best Result**:
  - Validation: 39.33% (Exp5: 10-Fold CV)
  - Test: N/A (no labels)
  - Validation F1 (macro): 36.32%
- **Key Finding**: High-dimensional sparse features; 50-class problem is challenging (39% >> 2% random baseline); benefits from deeper trees

## Usage

Run any enhanced script from the Exercise_1 directory:

```bash
cd /home/lucas/projects/Machine-Learning/Exercise_1
venv/bin/python decision_tree_enhanced/phishing/decision_tree_enhanced.py
venv/bin/python decision_tree_enhanced/voting/decision_tree_enhanced.py
venv/bin/python decision_tree_enhanced/road_safety/decision_tree_enhanced.py
venv/bin/python decision_tree_enhanced/reviews/decision_tree_enhanced.py
```

Each script will:
1. Load and preprocess the dataset
2. Run 20 experiments (~10-60 seconds depending on dataset size)
3. Save results to CSV (all metrics for all experiments)
4. Generate 5-6 PNG visualizations (300 DPI, LaTeX-ready)
5. Print summary statistics

## Output Files

### CSV Format
Each `*_results.csv` contains 20 rows (experiments) × ~20 columns:
- Experiment metadata: name, validation_strategy, max_depth, min_samples_split, min_samples_leaf, criterion
- Performance metrics: train_accuracy, val_accuracy, val_accuracy_sd, test_accuracy
- Precision/Recall/F1: macro and weighted averages for val and test
- Overfitting: overfitting_gap_train_val, overfitting_gap_train_test
- Complexity: tree_depth, n_leaves, training_time

### PNG Files
- 300 DPI resolution
- Serif font (LaTeX-compatible)
- Transparent background
- Professional color schemes

## Key Insights

### Parameter Sensitivity
- **Depth**: Most important hyperparameter; optimal range 5-15 depending on dataset
- **Split/Leaf**: Less impact; larger values reduce overfitting but may underfit
- **Criterion**: Minimal difference between gini and entropy

### Validation Strategy
- **Holdout**: Faster but higher variance
- **5-Fold CV**: Good balance of reliability and speed
- **10-Fold CV**: Most reliable but 2x slower than 5-fold

### Overfitting Patterns
- **Phishing**: Moderate (train-test gap ~5-10%)
- **Voting**: Low (train-test gap ~0-2%)
- **Road Safety**: Low (train-test gap ~0-5%, benefits from large dataset)

## Next Steps

1. ~~Complete reviews dataset analysis~~ ✓ **COMPLETED**
2. Create preprocessing comparison scripts (voting NaN handling, road_safety feature engineering)
3. Cross-dataset comparison analysis
4. Generate LaTeX tables for report

## Notes

- **Reviews Dataset**: The target variable is reviewer name (author attribution), not star rating. This is a challenging 50-class text classification problem where the model learns to identify which of 50 reviewers wrote a given review based on writing style captured in TF-IDF features.

## Dependencies

- Python 3.12.3
- scikit-learn 1.7.2
- pandas 2.3.3
- matplotlib
- seaborn
- numpy

## Random State

All scripts use `RANDOM_STATE = 2742` for reproducibility.
