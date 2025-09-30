# Deep Learning Model Testing Manual
*A Beginner's Guide to Testing Activation Functions*

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding Activation Functions](#understanding-activation-functions)
3. [Quick Testing Commands](#quick-testing-commands)
4. [Detailed Testing Guide](#detailed-testing-guide)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites
- macOS with zsh shell (which you have)
- Python virtual environment set up in your project

### Basic Setup
Before running any commands, you need to activate your Python environment:

```zsh
# Navigate to your project directory
cd /Users/seoljonghyeon/Documents/GitHub/Automated-LC-MS-MS-analaysis_ver2/Deep-Learning-Model

# Activate your virtual environment
source venv/bin/activate
```

**Important**: You must run `source venv/bin/activate` before every session!

---

## Understanding Activation Functions

### What are Activation Functions?
Activation functions determine how neurons in your neural network "activate" or respond to inputs. Different activation functions can significantly impact your model's performance.

### Available Activation Functions

#### **Classic Functions** (Traditional, well-tested)
- **ReLU** - Fast, prevents vanishing gradients, most popular
- **Tanh** - Smooth, outputs between -1 and 1
- **Sigmoid** - Smooth, outputs between 0 and 1
- **Step** - Binary output (0 or 1)
- **Softmax** - Converts to probabilities

#### **Modern Functions** (State-of-the-art, often better performance)
- **GELU** - Used in BERT and GPT models
- **Swish** - Developed by Google, often outperforms ReLU
- **Mish** - Self-regularizing, smooth
- **SiLU** - Similar to Swish, used in EfficientNet
- **Hardswish** - Efficient version of Swish

#### **Other Useful Functions**
- **LeakyReLU** - Fixes "dying ReLU" problem
- **ELU** - Smooth alternative to ReLU
- **PReLU** - Learnable version of LeakyReLU
- **SELU** - Self-normalizing

---

## Quick Testing Commands

### 1. **See What's Available**
```zsh
source venv/bin/activate && python3 main.py --list-activations
```
This shows all activation functions organized by category.

### 2. **Quick Demo (No Training Required)**
```zsh
source venv/bin/activate && python3 demo.py
```
- Tests activation functions with sample data
- Shows model architectures
- Displays inference speeds
- **Takes 1-2 minutes**

### 3. **Test One Activation Function (Fast)**
```zsh
source venv/bin/activate && python3 main.py --activation relu --quick
```
- Trains for only 2 epochs
- **Takes 3-5 minutes**
- Replace `relu` with any activation name

### 4. **Compare Similar Functions**
```zsh
# Test all modern functions
source venv/bin/activate && python3 main.py --activation modern --epochs 3

# Test all classic functions
source venv/bin/activate && python3 main.py --activation classic --epochs 3
```

---

## Detailed Testing Guide

### Step 1: Start with Demo
Always begin with the demo to ensure everything works:
```zsh
source venv/bin/activate && python3 demo.py
```

**What to expect:**
- Activation function outputs
- Model parameter counts
- Inference speed comparisons

### Step 2: Test Individual Functions
Pick one activation function to test thoroughly:

```zsh
# Test a modern function (recommended)
source venv/bin/activate && python3 main.py --activation swish --epochs 5

# Test a classic function
source venv/bin/activate && python3 main.py --activation relu --epochs 5
```

**What happens:**
- Model trains for 5 epochs
- Creates visualizations
- Saves best model
- **Takes 10-15 minutes**

### Step 3: Compare Multiple Functions
Compare functions within the same category:

```zsh
# Compare modern functions (recommended)
source venv/bin/activate && python3 main.py --activation modern --epochs 5
```

**What happens:**
- Tests: GELU, Swish, Mish, SiLU, Hardswish
- Shows comparison table
- Identifies best performer
- **Takes 45-60 minutes**

### Step 4: Full Comparison (Advanced)
```zsh
# Compare ALL functions (long process)
source venv/bin/activate && python3 main.py --activation all --epochs 3
```
- Tests all 14 activation functions
- **Takes 2-3 hours**

---

## Understanding Results

### During Training
You'll see output like this:
```
Epoch 1/5
--------------------------------------------------
Batch 0/1407, Loss: 2.2854, Acc: 28.12%
Batch 100/1407, Loss: 1.4673, Acc: 29.61%
...
Train Loss: 1.3621, Train Acc: 51.07%
Val Loss: 1.0918, Val Acc: 61.20%
```

**Key Metrics:**
- **Loss**: Lower is better (measures error)
- **Accuracy**: Higher is better (% correct predictions)
- **Train Acc**: Performance on training data
- **Val Acc**: Performance on validation data (more important)

### Final Comparison Table
```
Activation   Test Loss    Test Acc (%)  Val Acc (%)   Parameters
swish        1.0234       65.23         63.45         1,276,234
relu         1.1456       62.78         61.20         1,276,234
mish         1.0156       66.12         64.89         1,276,234
```

**How to interpret:**
- **Best Test Acc**: Highest percentage = best performing
- **Parameters**: All same (architecture unchanged)
- **Lower Loss + Higher Accuracy = Better Function**

### Generated Files
After training, check the `checkpoints/` folder:
```
checkpoints/
├── swish/
│   ├── best_model.pth          # Saved model
│   ├── training_history.png    # Loss/accuracy curves
│   ├── predictions.png         # Sample predictions
│   └── confusion_matrix.png    # Detailed performance
└── relu/
    └── ...
```

---

## Advanced Usage

### Custom Parameters
```zsh
# Different learning rate
source venv/bin/activate && python3 main.py --activation swish --lr 0.01 --epochs 10

# Different batch size
source venv/bin/activate && python3 main.py --activation mish --batch-size 64 --epochs 8

# Full training session
source venv/bin/activate && python3 main.py --activation swish --epochs 20 --batch-size 32 --lr 0.001
```

### Recommended Testing Sequence for Beginners:

#### **Day 1: Get Familiar (30 minutes)**
```zsh
# 1. See what's available
source venv/bin/activate && python3 main.py --list-activations

# 2. Run demo
source venv/bin/activate && python3 demo.py

# 3. Quick test
source venv/bin/activate && python3 main.py --activation relu --quick
```

#### **Day 2: Compare Modern Functions (1 hour)**
```zsh
source venv/bin/activate && python3 main.py --activation modern --epochs 5
```

#### **Day 3: Test Best Performer (30 minutes)**
```zsh
# Use the best function from Day 2
source venv/bin/activate && python3 main.py --activation [best_function] --epochs 10
```

---

## Troubleshooting

### Common Issues

#### **Error: "command not found: python"**
**Solution:** Use `python3` instead of `python`

#### **Error: "ModuleNotFoundError: No module named 'torch'"**
**Solution:** Activate virtual environment first:
```zsh
source venv/bin/activate
```

#### **Training seems stuck**
- **Normal**: First epoch takes longest (downloading CIFAR-10 dataset)
- **Wait**: 5-10 minutes for first batch
- **Check**: Terminal should show progress every 100 batches

#### **Out of memory error**
**Solution:** Reduce batch size:
```zsh
source venv/bin/activate && python3 main.py --activation relu --batch-size 16 --quick
```

#### **Want to stop training**
Press `Ctrl + C` to stop training safely.

### Getting Help
```zsh
# See all available options
source venv/bin/activate && python3 main.py --help
```

---

## Quick Reference Commands

### Essential Commands
```zsh
# Activate environment (always first!)
source venv/bin/activate

# List functions
python3 main.py --list-activations

# Quick demo
python3 demo.py

# Test one function (fast)
python3 main.py --activation swish --quick

# Compare modern functions
python3 main.py --activation modern --epochs 5

# See help
python3 main.py --help
```

### Recommended First Steps
1. `python3 demo.py` - Understand the system
2. `python3 main.py --activation swish --quick` - Quick test
3. `python3 main.py --activation modern --epochs 5` - Compare functions
4. Use the best performing function for your research

---

## Tips for Success

1. **Always activate environment first**
2. **Start with demo.py** to verify setup
3. **Use --quick for initial testing**
4. **Compare modern functions first** (usually best performance)
5. **Check generated visualizations** in checkpoints folder
6. **Higher validation accuracy = better function**
7. **Save your best models** for future use

**Happy Deep Learning!** 🚀