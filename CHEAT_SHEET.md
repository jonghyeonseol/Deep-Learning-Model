# 🚀 Deep Learning Command Cheat Sheet

**Quick reference for all commands** - bookmark this page!

---

## 🔧 Setup Commands (First Time Only)

```bash
# 1. Go to project folder
cd Deep-Learning-Model

# 2. Activate environment (DO THIS EVERY TIME!)
source venv/bin/activate

# 3. Install dependencies (ONCE ONLY)
pip install -r requirements.txt
```

**Tip**: Add ` (venv)` should appear in your terminal when activated!

---

## ⚡ Quick Start (5 Minutes)

```bash
# Fastest way to train your first model
python3 main.py --activation relu --quick
```

**Result**: 55-65% accuracy in 3-5 minutes

---

## 🎯 Basic Training Commands

### Compare Activation Functions

```bash
# Try different activation functions (5-10 minutes each)
python3 main.py --activation relu --quick
python3 main.py --activation swish --quick
python3 main.py --activation gelu --quick

# Compare ALL modern activations (15-20 minutes)
python3 main.py --activation modern --epochs 5

# Compare ALL classic activations (15-20 minutes)
python3 main.py --activation classic --epochs 5

# Compare EVERYTHING (2-3 hours)
python3 main.py --activation all --epochs 3
```

---

### Full Training (Better Results)

```bash
# Train for 10 epochs (10-15 minutes) - ~85-90% accuracy
python3 main.py --activation relu --epochs 10

# Train for 20 epochs (20-30 minutes) - ~87-92% accuracy
python3 main.py --activation swish --epochs 20

# With live monitoring (see progress in real-time)
python3 main.py --activation swish --epochs 10 --monitor
```

---

## 🏆 Modern Models (Best Results)

### ResNet (Recommended for Beginners)

```bash
# ResNet-18: 93-95% accuracy (30-45 minutes)
python3 main_modern.py --model resnet18 --epochs 50

# ResNet-18 with turbo boost (45-60 minutes)
python3 main_modern.py --model resnet18 --epochs 100 --amp

# ResNet-18 with ALL features (60-90 minutes) - ~95-96%
python3 main_modern.py --model resnet18 --epochs 100 --amp \
  --use-mixup --use-cutmix --use-randaugment

# ResNet-50 (bigger, slower) (60-90 minutes)
python3 main_modern.py --model resnet50 --epochs 100 --amp
```

---

### EfficientNet (Best Efficiency)

```bash
# EfficientNet-B0 (fewer parameters, good accuracy)
python3 main_modern.py --model efficientnet_b0 --epochs 100 --amp

# With augmentation (best results)
python3 main_modern.py --model efficientnet_b0 --epochs 100 --amp \
  --use-cutmix --use-randaugment --label-smoothing 0.1
```

---

### Vision Transformer (Advanced)

```bash
# Vision Transformer (needs lots of data/epochs)
python3 main_modern.py --model vit --epochs 200 --amp \
  --use-mixup --use-randaugment

# Smaller batch size if out of memory
python3 main_modern.py --model vit --epochs 200 --batch-size 64 --amp
```

---

### ConvNeXt (Modern CNN)

```bash
# ConvNeXt optimized for CIFAR-10
python3 main_modern.py --model convnext_cifar --epochs 200 --amp \
  --use-mixup --use-randaugment

# ConvNeXt-Tiny (more powerful, slower)
python3 main_modern.py --model convnext_tiny --epochs 200 --amp
```

---

## 🎨 Visualization Commands

```bash
# Live training monitor (real-time plots)
python3 main.py --activation relu --epochs 10 --monitor

# Visualize network structure (cool animation!)
python3 main.py --visualize

# Both together
python3 main.py --activation swish --quick --monitor --visualize
```

---

## 🔍 Comparison & Benchmarking

```bash
# Compare multiple modern models (2-3 hours)
python3 main_modern.py --compare-models --epochs 50

# Quick benchmark (2 epochs each, ~10 minutes)
python3 benchmark_all.py --quick

# Full benchmark (all models, all settings)
python3 benchmark_all.py --full
```

---

## ⚙️ Common Options Explained

### Model Selection
```bash
--model resnet18          # ResNet-18 (recommended)
--model resnet50          # ResNet-50 (bigger)
--model efficientnet_b0   # EfficientNet-B0 (efficient)
--model vit               # Vision Transformer
--model convnext_cifar    # ConvNeXt for CIFAR-10
```

### Training Options
```bash
--epochs 50               # Number of training rounds
--batch-size 128          # Images per batch (reduce if out of memory)
--lr 0.001                # Learning rate
--quick                   # Fast mode (2 epochs only)
```

### Speed & Optimization
```bash
--amp                     # Mixed precision (2x faster on GPU)
```

### Data Augmentation
```bash
--use-mixup               # MixUp augmentation
--use-cutmix              # CutMix augmentation
--use-randaugment         # RandAugment (automated)
```

### Other Options
```bash
--monitor                 # Live training visualization
--visualize               # Network structure visualization
--label-smoothing 0.1     # Label smoothing (better generalization)
```

---

## 💡 Recommended Combos for Different Goals

### "I Want Results FAST"
```bash
python3 main.py --activation swish --quick
```
**Time**: 3-5 minutes | **Accuracy**: 55-65%

---

### "I Want GOOD Results Without Waiting Too Long"
```bash
python3 main_modern.py --model resnet18 --epochs 50 --amp
```
**Time**: 30-45 minutes | **Accuracy**: 93-95%

---

### "I Want the BEST Results"
```bash
python3 main_modern.py --model resnet18 --epochs 200 --amp \
  --use-mixup --use-cutmix --use-randaugment \
  --optimizer adamw --scheduler cosine --weight-decay 1e-4
```
**Time**: 2-3 hours | **Accuracy**: 95-97%

---

### "I Want to EXPERIMENT and Learn"
```bash
# Try all activation functions
python3 main.py --activation all --epochs 5

# Compare modern models
python3 main_modern.py --compare-models --epochs 30
```
**Time**: 1-3 hours | **Learning**: High!

---

### "I Have Limited GPU Memory"
```bash
python3 main_modern.py --model resnet18 --epochs 50 --batch-size 64 --amp
```
**Smaller batch size** = less memory usage

---

### "I Want to See Everything Happening"
```bash
python3 main.py --activation swish --epochs 10 --monitor --visualize
```
**Cool visualizations** of training progress!

---

## 🎓 Learning Path Commands

### Level 1: Complete Beginner
```bash
# Step 1: First model
python3 main.py --activation relu --quick

# Step 2: Try variations
python3 main.py --activation swish --quick
python3 main.py --activation gelu --quick

# Step 3: Full training
python3 main.py --activation swish --epochs 10
```

---

### Level 2: Understanding Basics
```bash
# Compare activations
python3 main.py --activation modern --epochs 5

# With visualization
python3 main.py --activation swish --epochs 10 --monitor
```

---

### Level 3: Modern Models
```bash
# ResNet-18
python3 main_modern.py --model resnet18 --epochs 50 --amp

# With augmentation
python3 main_modern.py --model resnet18 --epochs 100 --amp --use-mixup
```

---

### Level 4: Advanced Techniques
```bash
# Full pipeline
python3 main_modern.py --model resnet18 --epochs 200 --amp \
  --use-mixup --use-cutmix --use-randaugment

# Try EfficientNet
python3 main_modern.py --model efficientnet_b0 --epochs 100 --amp

# Try ConvNeXt
python3 main_modern.py --model convnext_cifar --epochs 200 --amp
```

---

## 🆘 Troubleshooting Commands

### Check Python Version
```bash
python3 --version
# Should be 3.8 or higher
```

### Check PyTorch Installation
```bash
python3 -c "import torch; print(torch.__version__)"
# Should print version number
```

### Check CUDA (GPU) Availability
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# True = GPU available, False = CPU only
```

### List Available Models
```bash
python3 main_modern.py --help | grep "model"
```

### List Available Activations
```bash
python3 main.py --list-activations
```

### Clear Checkpoints (Free Space)
```bash
rm -rf checkpoints/*
# Warning: This deletes all saved models!
```

---

## 📊 Expected Training Times

| Command | CPU Time | GPU Time | Expected Accuracy |
|---------|----------|----------|-------------------|
| `--quick` | 5-10 min | 2-3 min | 55-65% |
| `--epochs 10` (basic) | 20-30 min | 10-15 min | 85-90% |
| `resnet18 --epochs 50` | 4-6 hours | 30-45 min | 93-95% |
| `resnet18 --epochs 100` | 8-12 hours | 60-90 min | 94-96% |
| `convnext_tiny --epochs 200` | 20-30 hours | 2-3 hours | 97%+ |

**Note**: Times are approximate and vary by hardware

---

## 🎯 Quick Decision Tree

**Start here**: What's your goal?

```
Do you want to learn? ──YES──> Start with Level 1 commands
    │
    NO
    │
    ↓
Do you have < 10 minutes? ──YES──> Use --quick
    │
    NO
    │
    ↓
Do you have a GPU? ──NO──> Use basic CNN (main.py)
    │
    YES
    ↓
Do you want best accuracy? ──YES──> ResNet-18 with all features
    │
    NO
    ↓
Do you want fast results? ──YES──> ResNet-18 --epochs 50 --amp
    │
    NO
    ↓
Want to experiment? ──YES──> Try --compare-models
```

---

## 💾 File Locations

### Your Results Are Saved Here:
```
checkpoints/
├── relu/                    # Results for ReLU activation
│   ├── best_model.pth      # Trained model
│   ├── training_history.png
│   ├── predictions.png
│   └── confusion_matrix.png
├── swish/                   # Results for Swish activation
└── resnet18/                # Results for ResNet-18
```

### Dataset Location:
```
data/cifar-10-batches-py/   # CIFAR-10 dataset (auto-downloaded)
```

---

## 🔄 Common Workflows

### Workflow 1: Compare Activation Functions
```bash
python3 main.py --activation relu --epochs 10
python3 main.py --activation swish --epochs 10
python3 main.py --activation gelu --epochs 10

# Compare results in checkpoints/*/training_history.png
```

---

### Workflow 2: Find Best Model
```bash
python3 main_modern.py --compare-models --epochs 50

# Check which model performed best
```

---

### Workflow 3: Optimize for Accuracy
```bash
# Step 1: Baseline
python3 main_modern.py --model resnet18 --epochs 50

# Step 2: Add augmentation
python3 main_modern.py --model resnet18 --epochs 50 --use-mixup

# Step 3: Add more augmentation
python3 main_modern.py --model resnet18 --epochs 50 --use-mixup --use-cutmix

# Step 4: Full pipeline
python3 main_modern.py --model resnet18 --epochs 100 --amp \
  --use-mixup --use-cutmix --use-randaugment
```

---

## 📱 Copy-Paste Commands (Ready to Use!)

### Absolute Beginner
```bash
source venv/bin/activate && python3 main.py --activation relu --quick
```

### Quick Test of Modern Model
```bash
source venv/bin/activate && python3 main_modern.py --model resnet18 --epochs 10 --amp
```

### Best Results in 1 Hour
```bash
source venv/bin/activate && python3 main_modern.py --model resnet18 --epochs 100 --amp --use-mixup
```

### Ultimate Training (Leave Running Overnight)
```bash
source venv/bin/activate && python3 main_modern.py --model convnext_cifar --epochs 300 --amp --use-mixup --use-cutmix --use-randaugment
```

---

## 🎨 Visual Indicator: Command Complexity

| Symbol | Meaning | Example |
|--------|---------|---------|
| 🟢 | Beginner-friendly | `python3 main.py --quick` |
| 🟡 | Intermediate | `python3 main_modern.py --model resnet18` |
| 🔴 | Advanced | Full training with all options |
| ⚡ | Fast (< 10 min) | `--quick` |
| 🐢 | Slow (> 1 hour) | `--epochs 100+` |
| 🎓 | Educational | `--compare-models` |
| 🏆 | Best accuracy | Full pipeline with augmentation |

---

## 🎉 Most Popular Commands (Top 10)

1. 🥇 `python3 main.py --activation relu --quick` - First model
2. 🥈 `python3 main.py --activation swish --epochs 10` - Better results
3. 🥉 `python3 main_modern.py --model resnet18 --epochs 50 --amp` - Modern model
4. 📊 `python3 main.py --activation modern --epochs 5` - Compare activations
5. 🎨 `python3 main.py --activation swish --monitor` - Live visualization
6. 🚀 `python3 main_modern.py --model resnet18 --epochs 100 --amp --use-mixup` - High accuracy
7. ⚙️ `python3 main_modern.py --compare-models --epochs 30` - Compare models
8. 🎯 `python3 main_modern.py --model efficientnet_b0 --epochs 100 --amp` - Efficient model
9. 📈 `python3 benchmark_all.py --quick` - Quick benchmark
10. 🏆 `python3 main_modern.py --model convnext_cifar --epochs 200 --amp --use-mixup --use-cutmix` - Best results

---

**Bookmark this page** for quick reference! 🔖

**Need help?** Check `BEGINNER_START.md` for detailed explanations.
