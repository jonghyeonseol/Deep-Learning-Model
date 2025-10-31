# Quick Start Guide

Get up and running with the CIFAR-10 Deep Learning Framework in 5 minutes.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- 4GB RAM minimum
- 2GB disk space for dataset

## Installation

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd Deep-Learning-Model

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```python
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.1.0+cu121
CUDA available: True
```

## Your First Training Run

### Option 1: Quick Test (5 minutes)

```bash
# Train for 2 epochs to verify everything works
python3 main.py --activation relu --quick
```

This will:
1. Download CIFAR-10 dataset (~170MB, first run only)
2. Train a CNN for 2 epochs
3. Save results to `checkpoints/relu/`

### Option 2: Using Configuration (Recommended)

```bash
# Use pre-made quick test configuration
python3 train_with_config.py --config configs/quick_test.yaml
```

### Option 3: Modern Training

```bash
# Train ResNet-18 with modern techniques (30 minutes)
python3 main_modern.py --model resnet18 --epochs 100
```

## Check Your Results

After training completes:

```bash
# View training curves
ls checkpoints/relu/training_history.png

# Check logs
cat logs/training.log

# View TensorBoard
tensorboard --logdir checkpoints/relu/logs
# Open http://localhost:6006
```

## Next Steps

### 1. Try Different Models

```bash
# ResNet-18 (30 minutes)
python3 train_with_config.py --config configs/resnet18_basic.yaml

# EfficientNet with all techniques (2 hours)
python3 train_with_config.py --config configs/efficientnet_modern.yaml

# Vision Transformer (2-3 hours)
python3 train_with_config.py --config configs/vit_transformer.yaml
```

### 2. Customize Your Training

```bash
# Copy an example configuration
cp configs/resnet18_basic.yaml configs/my_experiment.yaml

# Edit the configuration
nano configs/my_experiment.yaml

# Train with your config
python3 train_with_config.py --config configs/my_experiment.yaml
```

### 3. Explore Advanced Features

```bash
# Train with live visualization
python3 main.py --activation swish --epochs 10 --monitor

# Visualize network structure
python3 main.py --visualize --activation relu

# Use mixed precision for 2x speedup
python3 main_modern.py --model resnet18 --amp

# Apply strong augmentation
python3 main_modern.py --model resnet18 --use-mixup --use-cutmix --use-randaugment
```

## Common Commands Reference

### Training Commands

```bash
# Basic training
python3 main.py --activation <name> --epochs <N>

# Modern training
python3 main_modern.py --model <name> --epochs <N>

# Config-based training
python3 train_with_config.py --config <path>

# Quick test (2 epochs)
python3 main.py --activation relu --quick

# List all activation functions
python3 main.py --list-activations
```

### Monitoring and Visualization

```bash
# Real-time training monitor
python3 main.py --activation swish --monitor

# Network structure visualization
python3 main.py --visualize

# TensorBoard
tensorboard --logdir checkpoints/<model_name>/logs

# View logs
cat logs/training.log
```

### Model Comparison

```bash
# Compare modern activations (GELU, Swish, Mish, SiLU, Hardswish)
python3 main.py --activation modern --epochs 5

# Compare classic activations (ReLU, Tanh, Sigmoid, etc.)
python3 main.py --activation classic --epochs 5

# Compare all activations (2-3 hours)
python3 main.py --activation all --epochs 3

# Benchmark all models
python3 benchmark_all.py
```

## Configuration Examples

### Minimal Configuration

```yaml
# configs/minimal.yaml
model:
  architecture: resnet18
  num_classes: 10

training:
  epochs: 10
  batch_size: 128
  learning_rate: 0.1
  optimizer: sgd
```

### Production Configuration

```yaml
# configs/production.yaml
model:
  architecture: efficientnet_b0
  num_classes: 10
  drop_rate: 0.2

training:
  epochs: 300
  batch_size: 128
  learning_rate: 0.001
  optimizer: adamw
  weight_decay: 0.0001
  use_amp: true
  use_ema: true
  label_smoothing: 0.1

scheduler:
  type: cosine_warmup
  warmup_epochs: 20

augmentation:
  use_randaugment: true
  use_mixup: true
  use_cutmix: true

early_stopping:
  enabled: true
  patience: 30
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size

```yaml
training:
  batch_size: 64  # or 32
  use_amp: true   # Enable mixed precision
```

### Issue: Training is Too Slow

**Solution**: Enable optimizations

```yaml
training:
  use_amp: true   # 2x speedup

data:
  num_workers: 8  # More parallel data loading
  pin_memory: true
```

### Issue: Model is Overfitting

**Solution**: Add regularization

```yaml
training:
  weight_decay: 0.001
  label_smoothing: 0.1

augmentation:
  use_randaugment: true
  use_mixup: true
```

### Issue: Can't Find Configuration

**Error**: `ConfigurationError: Configuration file not found`

**Solution**: Use absolute or relative path

```bash
# Absolute path
python3 train_with_config.py --config /full/path/to/config.yaml

# Relative path from project root
python3 train_with_config.py --config configs/my_config.yaml
```

## Python API Quick Reference

### Basic Training

```python
from utils.config import load_config
from utils.data_loader import CIFAR10DataLoader
from models.resnet import ResNet18
from utils.trainer import Trainer

# Load configuration
config = load_config('configs/quick_test.yaml')

# Create data loaders
loader = CIFAR10DataLoader(batch_size=128)
train_loader, val_loader, test_loader = loader.get_loaders()

# Create model
model = ResNet18(num_classes=10)

# Create trainer and train
trainer = Trainer(model, train_loader, val_loader)
trainer.configure_optimizer(optimizer_name='adam', lr=0.001)
trainer.train(epochs=10)
```

### Modern Training

```python
from utils.modern_trainer import ModernTrainer

trainer = ModernTrainer(model, train_loader, val_loader)

# Enable modern features
trainer.configure_amp(enabled=True)
trainer.configure_ema(enabled=True, decay=0.9999)
trainer.configure_gradient_clipping(max_norm=1.0)

# Train
trainer.train(epochs=100)
```

## Expected Performance

| Model | Epochs | Time (RTX 3090) | Test Accuracy |
|-------|--------|-----------------|---------------|
| Quick Test | 2 | 3-5 min | ~60% |
| ResNet-18 Basic | 100 | 30 min | ~95.5% |
| EfficientNet Modern | 300 | 2 hours | ~96%+ |
| Vision Transformer | 300 | 2-3 hours | ~96.5%+ |

## Getting Help

### Documentation
- **Full Guide**: `docs/USER_GUIDE.md`
- **Configuration**: `configs/README.md`
- **Project Overview**: `CLAUDE.md`
- **API Reference**: `docs/API_REFERENCE.md`

### Community
- Report bugs on GitHub Issues
- Ask questions in Discussions
- Contribute improvements via Pull Requests

## What's Next?

1. **Read the Full User Guide**: `docs/USER_GUIDE.md`
2. **Explore Configurations**: `configs/README.md`
3. **Try Advanced Techniques**: Mixed precision, EMA, augmentation
4. **Experiment with Models**: ResNet, EfficientNet, ViT, ConvNeXt
5. **Monitor Training**: TensorBoard, live plots, attention maps

Happy training! 🚀
