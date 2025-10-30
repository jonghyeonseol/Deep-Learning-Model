# 🆘 Troubleshooting Guide

**Having problems?** Don't worry! Most issues are easy to fix. This guide covers the most common problems and their solutions.

---

## 🎯 Quick Diagnosis

**Start here**: What kind of problem are you having?

| Problem Type | Jump to Section |
|--------------|-----------------|
| 🔴 **Error messages** | [Common Errors](#-common-errors) |
| ⚙️ **Installation issues** | [Installation](#-installation-issues) |
| 🐌 **Training is slow** | [Performance](#-performance-issues) |
| 📊 **Bad accuracy** | [Training Quality](#-training-quality-issues) |
| 💾 **Out of memory** | [Memory](#-memory-issues) |
| 🖥️ **GPU not working** | [GPU](#-gpu-issues) |

---

## 🔴 Common Errors

### Error: "python3: command not found"

**What it means**: Python 3 isn't installed or isn't in your PATH.

**Solutions**:

```bash
# Check if Python is installed
which python3

# Try python instead of python3
python --version
```

**If not installed**:
- **Mac**: Download from https://www.python.org/downloads/
- **Linux (Ubuntu/Debian)**: `sudo apt-get install python3`
- **Linux (Fedora)**: `sudo dnf install python3`

---

### Error: "No module named 'torch'"

**What it means**: PyTorch isn't installed.

**Solution**:

```bash
# Make sure you're in the right directory
cd Deep-Learning-Model

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**If that doesn't work**:

```bash
# Install PyTorch directly
pip install torch torchvision torchaudio
```

---

### Error: "No module named 'models'" or "No module named 'utils'"

**What it means**: You're running the script from the wrong directory.

**Solution**:

```bash
# Check current directory
pwd

# Should see: .../Deep-Learning-Model

# If not, navigate to the project folder
cd path/to/Deep-Learning-Model

# Verify files are here
ls
# Should see: main.py, models/, utils/, etc.
```

---

### Error: "CUDA out of memory"

**What it means**: Your GPU doesn't have enough memory for the current batch size.

**Solutions** (try in order):

**Solution 1: Reduce batch size**
```bash
# Instead of default (128)
python3 main_modern.py --model resnet18 --batch-size 64

# If still failing
python3 main_modern.py --model resnet18 --batch-size 32
```

**Solution 2: Use a smaller model**
```bash
# Try a smaller variant
python3 main_modern.py --model resnet_tiny
```

**Solution 3: Train on CPU (slower but works)**
```bash
# Basic training works on CPU
python3 main.py --activation relu --epochs 10
```

---

### Error: "RuntimeError: DataLoader worker (pid XXXX) is killed by signal"

**What it means**: Not enough shared memory (common in Docker/containers).

**Solution**:

```bash
# Add num_workers=0 to data loader
# Or edit data_loader.py and set num_workers=0
```

**Quick fix**: Reduce batch size:
```bash
python3 main.py --batch-size 64 --quick
```

---

### Error: "FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/...'"

**What it means**: Checkpoint directory doesn't exist yet.

**Solution** (automatic): The code should create it automatically on first run.

**Manual fix**:
```bash
mkdir -p checkpoints
```

---

### Error: "ImportError: cannot import name 'X' from 'models'"

**What it means**: Code might be outdated or files are missing.

**Solutions**:

```bash
# 1. Make sure you have all files
ls models/
# Should see: __init__.py, network.py, resnet.py, etc.

# 2. Check __init__.py exists
cat models/__init__.py

# 3. Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

### Error: "AttributeError: 'NoneType' object has no attribute..."

**What it means**: Something failed to load properly.

**Solutions**:

```bash
# 1. Check if data downloaded
ls data/
# Should see: cifar-10-batches-py/

# 2. Re-download dataset
rm -rf data/cifar-10-batches-py
python3 main.py --activation relu --quick
# Will re-download automatically
```

---

### Error: "ValueError: num_samples should be a positive integer"

**What it means**: Data loading issue.

**Solution**:

```bash
# Clear old data and re-download
rm -rf data/
python3 main.py --activation relu --quick
```

---

### Error: "RuntimeError: Function 'X' returned nan values"

**What it means**: Numerical instability during training.

**Solutions**:

**Solution 1: Lower learning rate**
```bash
python3 main.py --lr 0.0001  # Instead of default 0.001
```

**Solution 2: Check for NaN in data**
```bash
# Use different dataset
# Or check data preprocessing
```

**Solution 3: Use gradient clipping**
```bash
python3 main_modern.py --model resnet18  # Has gradient clipping built-in
```

---

## ⚙️ Installation Issues

### Problem: "pip: command not found"

**Solution**:

```bash
# Try pip3 instead
pip3 install -r requirements.txt

# Or use python3 -m pip
python3 -m pip install -r requirements.txt
```

---

### Problem: "Permission denied" during installation

**Solutions**:

**Option 1: Use virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Option 2: Install for current user**
```bash
pip install -r requirements.txt --user
```

**Option 3: Use sudo (not recommended)**
```bash
sudo pip install -r requirements.txt
```

---

### Problem: "Could not find a version that satisfies the requirement torch"

**What it means**: Your Python version might be incompatible.

**Solutions**:

```bash
# Check Python version (needs 3.8+)
python3 --version

# If <3.8, upgrade Python
# Then install PyTorch from official website
# https://pytorch.org/get-started/locally/
```

---

### Problem: Virtual environment won't activate

**macOS/Linux**:
```bash
# Use source, not just venv/bin/activate
source venv/bin/activate

# If that fails, try
. venv/bin/activate
```

**Windows**:
```bash
# PowerShell
venv\Scripts\Activate.ps1

# CMD
venv\Scripts\activate.bat
```

---

### Problem: "SSL: CERTIFICATE_VERIFY_FAILED" when downloading data

**Solution**:

```bash
# macOS
/Applications/Python\ 3.*/Install\ Certificates.command

# Or download CIFAR-10 manually from:
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Extract to: data/cifar-10-batches-py/
```

---

## 🐌 Performance Issues

### Problem: Training is extremely slow

**Diagnosis**: Check if you're using GPU or CPU:

```bash
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

**If False (using CPU):**

**Solution 1**: That's expected! CPU is 10-20x slower than GPU.
- Use `--quick` mode for testing
- Use smaller models (basic CNN)
- Be patient for longer training

**Solution 2**: Get GPU access
- Use Google Colab (free GPU)
- Use cloud services (AWS, GCP, Azure)

**If True (using GPU but still slow):**

**Solution 1**: Enable AMP (mixed precision)
```bash
python3 main_modern.py --model resnet18 --amp
```

**Solution 2**: Increase batch size (if memory allows)
```bash
python3 main_modern.py --model resnet18 --batch-size 256 --amp
```

**Solution 3**: Use smaller model
```bash
python3 main_modern.py --model resnet_tiny
```

---

### Problem: First epoch is very slow, then normal

**What it means**: This is normal! First epoch includes:
- Data loading and preprocessing
- Model initialization
- JIT compilation (if using torch.compile)

**Solution**: Wait it out. Subsequent epochs will be faster.

---

### Problem: Training stops/hangs

**Possible causes**:

**Cause 1: Out of memory**
- Check GPU memory: `nvidia-smi`
- Solution: Reduce batch size

**Cause 2: Deadlock in data loading**
- Solution: Set `num_workers=0` in data loader

**Cause 3: System went to sleep**
- Solution: Adjust power settings

---

## 📊 Training Quality Issues

### Problem: Accuracy is very low (<50% on CIFAR-10)

**Diagnosis**:

```bash
# Check if you're using --quick mode
# Quick mode only trains 2 epochs - accuracy will be low!

# Solution: Train longer
python3 main.py --activation relu --epochs 20
```

**Other causes**:

**Cause 1: Learning rate too high**
```bash
# Try lower learning rate
python3 main.py --lr 0.0001
```

**Cause 2: Model is too simple**
```bash
# Use modern model
python3 main_modern.py --model resnet18 --epochs 50
```

---

### Problem: Accuracy stops improving

**What it means**: Model has converged.

**Solutions**:

**Solution 1: Train longer**
```bash
--epochs 100  # Instead of --epochs 50
```

**Solution 2: Use learning rate scheduling**
```bash
# Modern trainer has built-in scheduling
python3 main_modern.py --model resnet18 --epochs 100
```

**Solution 3: Add data augmentation**
```bash
python3 main_modern.py --model resnet18 --use-mixup --use-cutmix
```

**Solution 4: Use better model**
```bash
python3 main_modern.py --model resnet50  # Bigger model
```

---

### Problem: Validation accuracy much lower than training accuracy

**What it means**: Overfitting - model memorized training data.

**Solutions**:

**Solution 1: Add augmentation**
```bash
python3 main_modern.py --model resnet18 --use-mixup
```

**Solution 2: Add dropout**
```bash
python3 main_modern.py --model resnet18 --dropout-rate 0.3
```

**Solution 3: Use label smoothing**
```bash
python3 main_modern.py --model resnet18 --label-smoothing 0.1
```

**Solution 4: Train less epochs**
```bash
# Stop before overfitting occurs
# Or use early stopping (automatic in modern_trainer)
```

---

### Problem: Loss is NaN

**What it means**: Numerical instability.

**Solutions**:

**Solution 1: Lower learning rate**
```bash
python3 main.py --lr 0.0001
```

**Solution 2: Use gradient clipping**
```bash
python3 main_modern.py --model resnet18  # Has gradient clipping
```

**Solution 3: Check for data issues**
```bash
# Re-download dataset
rm -rf data/
python3 main.py --quick
```

---

## 💾 Memory Issues

### Problem: "RuntimeError: CUDA out of memory"

**Quick fixes** (try in order):

```bash
# 1. Reduce batch size
python3 main_modern.py --model resnet18 --batch-size 64

# 2. Use smaller model
python3 main_modern.py --model resnet_tiny

# 3. Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"

# 4. Use CPU
python3 main.py --activation relu --epochs 10
```

**Permanent solution**:
- Get GPU with more memory
- Use gradient accumulation (not implemented yet)

---

### Problem: System runs out of RAM

**Symptoms**:
- Computer freezes
- Process killed
- "Killed" message

**Solutions**:

```bash
# 1. Close other programs
# 2. Reduce number of data loader workers
# Edit data_loader.py: num_workers=2 (or 0)

# 3. Use smaller batch size
python3 main.py --batch-size 32
```

---

## 🖥️ GPU Issues

### Problem: GPU not detected

**Check if CUDA is available**:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

**If False**:

**Step 1: Check GPU exists**
```bash
# NVIDIA
nvidia-smi

# AMD
rocm-smi

# If command not found, drivers aren't installed
```

**Step 2: Install CUDA-enabled PyTorch**
```bash
# Go to https://pytorch.org/get-started/locally/
# Select your CUDA version
# Install appropriate version

# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 3: Verify**
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

### Problem: "CUDA capability sm_XX is not compatible"

**What it means**: Your GPU is too old for the installed PyTorch version.

**Solution**: Install older PyTorch or use CPU:

```bash
# Option 1: Use CPU
python3 main.py --activation relu --epochs 10

# Option 2: Install older PyTorch
pip install torch==1.13.0 torchvision==0.14.0
```

---

### Problem: Multiple GPUs detected, want to use specific one

**Solution**:

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python3 main_modern.py --model resnet18

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 python3 main_modern.py --model resnet18

# Use CPU only
CUDA_VISIBLE_DEVICES= python3 main.py --activation relu
```

---

## 🎨 Visualization Issues

### Problem: "No display found" when using --visualize

**What it means**: You're on a server without a display.

**Solutions**:

**Option 1: Remove --visualize**
```bash
# Just train without visualization
python3 main.py --activation relu --epochs 10
```

**Option 2: Use --monitor instead**
```bash
# Save plots to file instead of displaying
python3 main.py --activation relu --epochs 10 --monitor
# Plots saved in checkpoints/
```

**Option 3: Set up X11 forwarding (advanced)**
```bash
# SSH with X11
ssh -X user@server
```

---

### Problem: Plots don't update in real-time

**What it means**: Matplotlib backend issue.

**Solution**: Plots are saved to files anyway! Check:
```bash
ls checkpoints/relu/
# Look for .png files
```

---

## 📁 File & Directory Issues

### Problem: "Read-only file system"

**What it means**: You don't have write permissions.

**Solutions**:

```bash
# Check permissions
ls -la

# Change to your home directory
cd ~
git clone <repo>
cd Deep-Learning-Model
```

---

### Problem: Disk space full

**Solutions**:

```bash
# Check disk space
df -h

# Clear old checkpoints
rm -rf checkpoints/*

# Clear PyTorch cache
rm -rf ~/.cache/torch

# Clear pip cache
pip cache purge
```

---

## 🔧 General Debugging Tips

### Tip 1: Start Simple

```bash
# Always test with --quick first
python3 main.py --activation relu --quick

# If that works, scale up
python3 main.py --activation relu --epochs 10
```

---

### Tip 2: Check Basics

```bash
# 1. Right directory?
pwd
# Should end in: /Deep-Learning-Model

# 2. Environment activated?
which python
# Should show: .../venv/bin/python

# 3. Dependencies installed?
pip list | grep torch
# Should show: torch, torchvision
```

---

### Tip 3: Read Error Messages Carefully

Most error messages tell you exactly what's wrong!

**Common patterns**:
- "No module named X" → Install X
- "No such file" → Wrong directory
- "Out of memory" → Reduce batch size
- "Command not found" → Not installed or not in PATH

---

### Tip 4: Test Components Separately

```bash
# Test data loading
python3 -c "from utils.data_loader import CIFAR10DataLoader; loader = CIFAR10DataLoader(); print('Data loaded OK!')"

# Test model loading
python3 -c "from models.network import ConvNeuralNetwork; model = ConvNeuralNetwork(); print('Model created OK!')"

# Test GPU
python3 -c "import torch; print('GPU:', torch.cuda.is_available())"
```

---

## 🆘 Still Stuck?

### Before Asking for Help, Collect This Info:

```bash
# 1. System info
uname -a

# 2. Python version
python3 --version

# 3. PyTorch version
python3 -c "import torch; print(torch.__version__)"

# 4. CUDA version (if GPU)
nvidia-smi

# 5. Full error message
# Copy the entire traceback

# 6. What command you ran
# Exact command that failed
```

---

### Where to Get Help:

1. **Re-read this guide** - Most issues are covered here
2. **Check [BEGINNER_START.md FAQ](BEGINNER_START.md#-frequently-asked-questions-faq)**
3. **Check [NAVIGATION.md](NAVIGATION.md)** - Find relevant documentation
4. **Search error message** on Google/Stack Overflow
5. **GitHub Issues** - Check if others had the same problem

---

## 📊 Quick Reference: Error → Solution

| Error | Quick Fix |
|-------|-----------|
| "python3: command not found" | Install Python 3 |
| "No module named torch" | `pip install -r requirements.txt` |
| "CUDA out of memory" | `--batch-size 64` |
| "No such file or directory" | Check you're in right folder |
| "Permission denied" | Use virtual environment |
| Training very slow | Normal on CPU, use `--quick` |
| Accuracy < 50% | Train longer, use `--epochs 20` |
| "No display found" | Remove `--visualize` |
| Virtual env won't activate | `source venv/bin/activate` |
| GPU not detected | Install CUDA-enabled PyTorch |

---

## 🎉 Problem Solved?

Great! Now get back to learning:
- 📖 [BEGINNER_START.md](BEGINNER_START.md) - Continue learning
- 🚀 [CHEAT_SHEET.md](CHEAT_SHEET.md) - Try more commands
- 🏗️ [ARCHITECTURES.md](ARCHITECTURES.md) - Explore models

---

**Remember**: Everyone encounters these problems. You're doing great! 💪
