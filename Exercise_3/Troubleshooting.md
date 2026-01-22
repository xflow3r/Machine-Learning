# Troubleshooting Guide
This file was AI-Generated
## Common Issues and Solutions

### 1. Module Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Make sure you've installed all dependencies
pip install -r requirements.txt

# If using virtual environment, make sure it's activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

---

### 2. CUDA/GPU Issues

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Option 1: Reduce batch size
python src/main.py --model cnn_small --batch-size 32 --device cuda

# Option 2: Use CPU instead
python src/main.py --model cnn_small --device cpu

# Option 3: Only run small model
python src/main.py --model cnn_small --device cuda
```

**Problem:**
```
UserWarning: CUDA not available, using CPU
```

**Solution:**
This is just a warning. The code will automatically use CPU. To verify CUDA availability:
```python
import torch
print(torch.cuda.is_available())  # Should print True if CUDA works
print(torch.cuda.get_device_name(0))  # Print GPU name
```

---

### 3. Dataset Download Issues

**Problem:**
```
URLError: Connection refused
```

**Solution:**
```bash
# Manual download:
# 1. Go to: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
# 2. Download these 4 files:
#    - train-images-idx3-ubyte.gz
#    - train-labels-idx1-ubyte.gz
#    - t10k-images-idx3-ubyte.gz
#    - t10k-labels-idx1-ubyte.gz
# 3. Place them in: fashion_mnist_data/

# Or use wget/curl:
cd fashion_mnist_data
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```

---

### 4. HOG Feature Extraction Too Slow

**Problem:**
HOG feature extraction takes >30 minutes

**Solution:**
```bash
# This is normal behavior. HOG extraction is CPU-intensive.
# To speed up:

# Option 1: Use fewer training samples for testing
# Modify data.py to load subset

# Option 2: Use multiple cores (if scikit-image supports it)
# Check scikit-image version and multiprocessing options

# Option 3: Just be patient - it only needs to run once!
# The code shows progress bar, so you can monitor it
```

---

### 5. scikit-image HOG Warning

**Problem:**
```
UserWarning: The default value of `block_norm`...
```

**Solution:**
This is just a warning, not an error. The code will still work fine. To suppress:
```python
import warnings
warnings.filterwarnings('ignore')
```

---

### 6. Results Directory Not Created

**Problem:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'results/tables/results.csv'
```

**Solution:**
```bash
# Run setup script
bash setup_project.sh

# Or manually create directories
mkdir -p results/tables
mkdir -p results/figures
mkdir -p fashion_mnist_data
```

---

### 7. Confusion Matrix Plot Not Showing

**Problem:**
Confusion matrix saved but not displayed

**Solution:**
The code automatically saves plots instead of showing them. This is intentional for non-interactive environments. To view:
```bash
# Linux/Mac
open results/figures/cm_fashion_mnist_cnn_small.png

# Windows
start results/figures/cm_fashion_mnist_cnn_small.png

# Or just navigate to the folder
```

---

### 8. PyTorch Installation Issues

**Problem:**
torch not installing correctly or wrong version

**Solution:**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Check installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

### 9. Script Won't Run (Permission Denied)

**Problem:**
```bash
bash: ./run_experiments.sh: Permission denied
```

**Solution:**
```bash
# Make script executable
chmod +x run_experiments.sh
chmod +x setup_project.sh

# Then run
./run_experiments.sh
```

---

### 10. Different Results Each Run

**Problem:**
Accuracy varies between runs even with same seed

**Solution:**
This can happen due to:
- Non-deterministic CUDA operations
- Different hardware
- Floating point precision

To ensure better reproducibility:
```bash
# Always use same seed
python src/main.py --model all --seed 42

# Set environment variable for more determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python src/main.py --model all --seed 42
```

---

### 11. Very Low Accuracy (<50%)

**Problem:**
Models achieving much worse than expected accuracy

**Possible Causes and Solutions:**

1. **Data not normalized:**
   - Check that images are divided by 255.0
   - Verify in `data.py`: `normalize=True`

2. **Wrong labels:**
   - Print some samples: `print(y_train[:10])`
   - Should be 0-9, not 1-10

3. **Model not training:**
   - Check loss is decreasing
   - Verify optimizer is updating weights

4. **Bug in evaluation:**
   - Compare train and test accuracy
   - Check confusion matrix for patterns

---

### 12. Training Takes Forever

**Problem:**
CNN training not progressing or extremely slow

**Solutions:**

1. **Check GPU usage:**
   ```bash
   # Linux
   nvidia-smi

   # Should show python process using GPU
   ```

2. **Reduce epochs for testing:**
   ```bash
   python src/main.py --model cnn_small --epochs 2 --device cuda
   ```

3. **Verify data on GPU:**
   - Check `device` parameter in train function
   - Ensure data moved to GPU in training loop

---

### 13. RAM/Memory Issues

**Problem:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Reduce batch size
python src/main.py --model cnn_small --batch-size 32

# Or reduce dataset size for testing
# Modify data.py to load subset
```

---

### 14. Windows-Specific Issues

**Problem:**
Path separators or encoding issues on Windows

**Solution:**
```python
# Use os.path.join for paths
import os
path = os.path.join('results', 'tables', 'results.csv')

# Or use pathlib
from pathlib import Path
path = Path('results') / 'tables' / 'results.csv'
```

---

### 15. Can't Find main.py

**Problem:**
```
python: can't open file 'main.py': No such file or directory
```

**Solution:**
```bash
# Make sure you're in the right directory
cd Exercise_3

# Run from project root
python src/main.py --model all

# Or navigate to src
cd src
python main.py --model all
```

---

## Still Having Issues?

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Verify all files exist:**
   ```bash
   ls -la src/
   ls -la src/models/
   ```

3. **Check package versions:**
   ```bash
   pip list | grep torch
   pip list | grep sklearn
   pip list | grep scikit-image
   ```

4. **Create minimal test:**
   ```python
   # test.py
   import torch
   import numpy as np
   from sklearn.svm import SVC
   from skimage.feature import hog

   print("✓ All imports successful!")
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

5. **Start fresh:**
   ```bash
   # Remove virtual environment
   rm -rf venv/

   # Recreate
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Getting Help

If you're still stuck:

1. **Check error message carefully** - often contains the solution
2. **Search error message** - Stack Overflow is your friend
3. **Check package documentation:**
   - PyTorch: https://pytorch.org/docs/
   - scikit-learn: https://scikit-learn.org/
4. **Ask in course forum** - other students may have same issue
5. **Contact TA** - mayer@ifs.tuwien.ac.at

---

## Tips for Smooth Running

✅ **Before starting:**
- Verify Python 3.8+
- Check ~5GB free disk space
- Install dependencies first
- Test with single model before running all

✅ **During experiments:**
- Don't interrupt training mid-epoch
- Monitor GPU/CPU usage
- Save intermediate results
- Keep terminal output logs

✅ **After completion:**
- Backup results folder
- Check all CSV entries
- Verify all confusion matrices generated
- Review outputs for sanity

Good luck! 🚀