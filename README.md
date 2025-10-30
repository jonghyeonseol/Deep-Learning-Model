# Modern Deep Learning Framework for Image Classification

A state-of-the-art PyTorch-based deep learning framework for CIFAR-10 image classification, featuring modern architectures, advanced training techniques, and comprehensive visualization tools.

---

## 👋 New to Deep Learning? Start Here!

**Welcome!** This framework is designed to be **beginner-friendly** while teaching state-of-the-art techniques.

### 🎯 Choose Your Path:

| I am... | Start with... | Time Needed |
|---------|---------------|-------------|
| 🆕 **Complete Beginner** | 📖 [BEGINNER_START.md](BEGINNER_START.md) | 5 minutes to first model |
| 📚 **Quick Learner** | 🚀 [CHEAT_SHEET.md](CHEAT_SHEET.md) | Copy-paste commands |
| 🏗️ **Model Explorer** | 🎨 [ARCHITECTURES.md](ARCHITECTURES.md) | Compare architectures |
| 🔬 **Researcher** | 📊 [MODERN_DL_GUIDE.md](MODERN_DL_GUIDE.md) | Latest techniques 2024-2025 |
| 🆕 **What's New?** | ✨ [UPDATE_SUMMARY_2025.md](UPDATE_SUMMARY_2025.md) | 2025 updates |

### ⚡ Super Quick Start (60 Seconds):

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Train your first model (3-5 minutes)
python3 main.py --activation relu --quick

# 3. See your results in checkpoints/relu/
```

**Done!** You just trained a neural network! 🎉

👉 **Never trained a model before?** Read the [Beginner's Guide](BEGINNER_START.md) for step-by-step instructions.

---

## Features

### 🏗️ Modern Architectures (2024-2025)
- **ResNet** (18/34/50/101) - Residual networks with skip connections
- **EfficientNet** (B0/B1) - Compound scaling with MBConv blocks
- **Vision Transformer** - Patch-based attention mechanism
- **ConvNeXt** - Modernized CNN competing with transformers
- **Custom CNN** - Flexible architecture with 14+ activation functions

### 🚀 Advanced Training Techniques
- **AdamW Optimizer** - Decoupled weight decay
- **Cosine Annealing with Warmup** - Advanced LR scheduling
- **Mixed Precision Training (AMP)** - 2x faster training
- **Label Smoothing** - Improved generalization
- **Exponential Moving Average (EMA)** - Stable inference
- **Gradient Clipping** - Training stability

### 🎨 Data Augmentation
- **RandAugment** - Automated augmentation policies
- **MixUp** - Sample mixing for regularization
- **CutMix** - Patch-based sample mixing
- **Cutout** - Random erasing
- **Standard Augmentations** - Flip, crop, color jitter

### 📊 Visualization & Monitoring
- Real-time training plots (loss, accuracy, LR)
- Live network structure visualization
- Layer activation monitoring
- Confusion matrices
- Sample predictions with ground truth

## Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd Deep-Learning-Model

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Training
```bash
# Quick test with ReLU activation (2 epochs, 3-5 minutes)
python3 main.py --activation relu --quick

# Train with specific activation function
python3 main.py --activation swish --epochs 10

# Compare all modern activation functions
python3 main.py --activation modern --epochs 5
```

### Modern Training Pipeline
```bash
# Train ResNet-18 with modern techniques
python3 main_modern.py --model resnet18 --epochs 100 --amp

# Train EfficientNet with full augmentation
python3 main_modern.py --model efficientnet_b0 --epochs 200 \
  --use-mixup --use-cutmix --use-randaugment --amp

# Train Vision Transformer
python3 main_modern.py --model vit --epochs 300 --amp

# Train ConvNeXt (state-of-the-art)
python3 main_modern.py --model convnext_cifar --epochs 200 --amp
```

### Live Visualization
```bash
# Real-time network structure
python3 main.py --activation relu --visualize

# Live training monitoring
python3 main.py --activation swish --monitor --epochs 5

# Combined visualization and monitoring
python3 main.py --activation relu --monitor --visualize --quick
```

## Architecture Comparison

| Model | Parameters | CIFAR-10 Acc | Training Time (RTX 3090) |
|-------|-----------|--------------|--------------------------|
| **Custom CNN** | ~2M | ~85-90% | ~10 min (10 epochs) |
| **ResNet-18** | 11.2M | ~95.5% | ~30 min (200 epochs) |
| **ResNet-50** | 23.5M | ~96.0% | ~1 hour (200 epochs) |
| **EfficientNet-B0** | 4.0M | ~95.8% | ~45 min (300 epochs) |
| **Vision Transformer** | 5.7M | ~96.5% | ~2 hours (300 epochs) |
| **ConvNeXt-CIFAR** | 12.5M | ~96-97% | ~1 hour (200 epochs) |

*Results with modern training techniques (AdamW, augmentation, AMP)*

## Available Activation Functions

- **Modern (2017-2020)**: GELU, Swish, Mish, SiLU, Hardswish
- **Classic**: ReLU, Tanh, Sigmoid, LeakyReLU, ELU, PReLU, SELU
- **Other**: Step, Softmax

All activation functions are custom implementations (not using `torch.nn` built-ins) for educational purposes.

## Project Structure

```
Deep-Learning-Model/
├── models/                  # Neural network architectures
│   ├── network.py           # Custom CNN
│   ├── activations.py       # 14+ activation functions
│   ├── resnet.py            # ResNet family
│   ├── efficientnet.py      # EfficientNet
│   ├── cnn_transformer.py   # ViT and hybrid models
│   └── convnext.py          # ConvNeXt (NEW)
├── utils/                   # Training utilities
│   ├── trainer.py           # Basic trainer
│   ├── modern_trainer.py    # Modern techniques
│   ├── data_loader.py       # CIFAR-10 data loading
│   ├── augmentation.py      # RandAugment, MixUp, CutMix
│   ├── regularization.py    # DropBlock, Stochastic Depth
│   ├── visualization.py     # Plotting tools
│   └── monitor.py           # Real-time monitoring
├── main.py                  # Basic training script
├── main_modern.py           # Modern training pipeline
├── benchmark_all.py         # Performance comparison
├── requirements.txt         # Python dependencies
└── checkpoints/             # Saved models and results
```

## Modern Deep Learning Techniques (2024-2025)

### Implemented ✅
- Residual connections (ResNet)
- Attention mechanisms (SE, Multi-head)
- Depthwise separable convolutions (EfficientNet)
- Inverted bottleneck design (EfficientNet, ConvNeXt)
- AdamW optimizer with decoupled weight decay
- Cosine annealing LR schedule with warmup
- Mixed precision training (AMP)
- Label smoothing
- RandAugment, MixUp, CutMix
- Stochastic depth / Drop path
- DropBlock regularization
- Layer normalization (ViT, ConvNeXt)
- GELU activation (Transformers, ConvNeXt)

### Coming Soon 🚧
- Enhanced Vision Transformers (DeiT, Swin)
- Sharpness-Aware Minimization (SAM optimizer)
- Stochastic Weight Averaging (SWA)
- Layer-wise Learning Rate Decay (LLRD)
- TrivialAugment, AugMax
- Knowledge distillation
- Model quantization
- Advanced metrics module

## Dataset Information

- **CIFAR-10**: 60,000 32×32 RGB images in 10 classes
  - Training: 45,000 images (after 10% validation split)
  - Validation: 5,000 images
  - Test: 10,000 images
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Normalization**: Mean=[0.4914, 0.4822, 0.4465], Std=[0.2023, 0.1994, 0.2010]
- Auto-downloaded on first run (~170MB)

## Performance Tips

1. **Use modern architectures**: ResNet-18 or EfficientNet-B0 for best accuracy
2. **Enable AMP**: `--amp` flag for 2x speedup
3. **Apply strong augmentation**: `--use-randaugment --use-mixup`
4. **Train longer**: 200-300 epochs for modern models
5. **Use larger batch sizes**: 128-256 with linear LR scaling
6. **Monitor validation**: Early stopping prevents overfitting
7. **GPU recommended**: CUDA-enabled GPU for efficient training

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- NumPy, Matplotlib, tqdm
- (Optional) CUDA-capable GPU for faster training

## Citation

If you use this framework in your research, please consider citing:

```bibtex
@software{deep_learning_framework_2024,
  title={Modern Deep Learning Framework for Image Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Deep-Learning-Model}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- PyTorch team for excellent deep learning framework
- Papers: ResNet, EfficientNet, Vision Transformer, ConvNeXt, RandAugment, MixUp, CutMix
- CIFAR-10 dataset by Alex Krizhevsky

## Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: October 2024 | **Status**: Active Development
