# Architecture Comparison Guide

Comprehensive comparison of all available neural network architectures in this framework.

## Table of Contents
1. [Quick Comparison](#quick-comparison)
2. [Custom CNN](#custom-cnn)
3. [ResNet](#resnet)
4. [EfficientNet](#efficientnet)
5. [Vision Transformer](#vision-transformer)
6. [ConvNeXt](#convnext)
7. [When to Use Each Architecture](#when-to-use-each-architecture)

---

## Quick Comparison

| Architecture | Params | CIFAR-10 Acc | Speed | Memory | Difficulty | Year |
|-------------|---------|--------------|-------|--------|------------|------|
| **Custom CNN** | ~2M | ~85-90% | ⚡⚡⚡⚡⚡ | Low | Easy | 2012 |
| **ResNet-18** | 11.2M | ~95.5% | ⚡⚡⚡⚡ | Low | Easy | 2015 |
| **ResNet-50** | 23.5M | ~96.0% | ⚡⚡⚡ | Medium | Easy | 2015 |
| **EfficientNet-B0** | 4.0M | ~95.8% | ⚡⚡⚡⚡ | Low | Medium | 2019 |
| **EfficientNet-B1** | 6.5M | ~96.2% | ⚡⚡⚡ | Medium | Medium | 2019 |
| **Vision Transformer** | 5.7M | ~96.5% | ⚡⚡ | High | Hard | 2020 |
| **ConvNeXt-CIFAR** | 12.5M | ~96-97% | ⚡⚡⚡ | Medium | Medium | 2022 |
| **ConvNeXt-Tiny** | 27.8M | ~97%+ | ⚡⚡ | High | Medium | 2022 |

*Accuracy with modern training techniques (200-300 epochs, AdamW, augmentation)*
*Speed: ⚡⚡⚡⚡⚡ = Fastest, ⚡ = Slowest*

---

## Custom CNN

### Overview
Classic convolutional neural network with configurable activation functions. Simple 3-layer CNN designed for CIFAR-10.

### Architecture
```
Input (32x32x3)
  ↓
Conv2d(3→32, 3x3) + Activation + MaxPool(2x2)  [16x16x32]
  ↓
Conv2d(32→64, 3x3) + Activation + MaxPool(2x2)  [8x8x64]
  ↓
Conv2d(64→128, 3x3) + Activation + MaxPool(2x2)  [4x4x128]
  ↓
Flatten → Linear(2048→512) → Linear(512→256) → Linear(256→10)
  ↓
Output (10 classes)
```

### Key Features
- **14+ Activation Functions**: GELU, ReLU, Swish, Mish, Tanh, etc.
- **Simple Design**: Easy to understand and modify
- **Fast Training**: Only 10 epochs needed for decent results
- **Educational**: Custom implementations (not using `torch.nn` built-ins)

### Strengths
✅ Very fast training (minutes)
✅ Small model size (~2M parameters)
✅ Easy to understand and debug
✅ Great for learning and experimentation

### Weaknesses
❌ Lower accuracy (~85-90% on CIFAR-10)
❌ No modern techniques (residuals, attention)
❌ Limited depth (only 3 conv layers)

### Usage
```bash
# Train with ReLU activation
python3 main.py --activation relu --epochs 10

# Compare activations
python3 main.py --activation modern --epochs 5

# Quick test
python3 main.py --activation swish --quick
```

### Best For
- Learning deep learning fundamentals
- Quick experiments with activation functions
- Baseline comparison
- Resource-constrained environments

---

## ResNet

### Overview
Residual Networks (ResNet) introduced skip connections, enabling training of very deep networks. Revolutionary architecture that won ImageNet 2015.

### Architecture (ResNet-18)
```
Input (32x32x3)
  ↓
Conv2d(3→64, 7x7, stride=2) + BatchNorm + ReLU + MaxPool  [8x8x64]
  ↓
Layer 1: 2 × BasicBlock(64)   [8x8x64]
Layer 2: 2 × BasicBlock(128)  [4x4x128]  ← Downsample
Layer 3: 2 × BasicBlock(256)  [2x2x256]  ← Downsample
Layer 4: 2 × BasicBlock(512)  [1x1x512]  ← Downsample
  ↓
Global Average Pooling → Linear(512→10)
  ↓
Output (10 classes)
```

**BasicBlock:**
```
x ────────────────────┐
  ↓                   │
  Conv(3x3) + BN      │
  ↓                   │
  ReLU                │  (Skip Connection)
  ↓                   │
  Conv(3x3) + BN      │
  ↓                   │
  Add ←───────────────┘
  ↓
  ReLU
```

### Variants
- **ResNet-18**: 2 blocks per layer, 11.2M params, ~95.5% accuracy
- **ResNet-34**: 3-6 blocks per layer, 21.3M params, ~95.8% accuracy
- **ResNet-50**: Bottleneck blocks, 23.5M params, ~96.0% accuracy
- **ResNet-101**: Deeper, 42.5M params, ~96.2% accuracy

### Key Features
- **Skip Connections**: Direct paths for gradient flow
- **Batch Normalization**: After each convolution
- **Global Average Pooling**: Instead of fully-connected layers
- **Deep Architecture**: 18-152 layers possible

### Strengths
✅ Excellent accuracy (~95-96% on CIFAR-10)
✅ Stable training (skip connections prevent vanishing gradients)
✅ Well-studied and reliable
✅ Fast training with good convergence
✅ Scales well to very deep networks

### Weaknesses
❌ More parameters than EfficientNet for same accuracy
❌ Fixed architecture (less flexible)
❌ Requires longer training (100-200 epochs)

### Usage
```bash
# ResNet-18 (recommended)
python3 main_modern.py --model resnet18 --epochs 200 --amp

# ResNet-50 (more accurate)
python3 main_modern.py --model resnet50 --epochs 200 --amp --use-mixup

# With full augmentation
python3 main_modern.py --model resnet18 --epochs 200 --amp \
  --use-mixup --use-cutmix --use-randaugment
```

### Best For
- Production deployments
- Transfer learning
- General-purpose image classification
- When stability and reliability are priorities

---

## EfficientNet

### Overview
EfficientNet scales networks optimally across depth, width, and resolution using compound scaling. Achieves better accuracy with fewer parameters.

### Architecture (EfficientNet-B0)
```
Input (32x32x3)
  ↓
Conv2d(3→32, 3x3, stride=2) + BatchNorm + Swish  [16x16x32]
  ↓
Stage 1: 1 × MBConv1(32,  16)   [16x16x16]
Stage 2: 2 × MBConv6(16,  24)   [8x8x24]
Stage 3: 2 × MBConv6(24,  40)   [4x4x40]
Stage 4: 3 × MBConv6(40,  80)   [2x2x80]
Stage 5: 3 × MBConv6(80,  112)  [2x2x112]
Stage 6: 4 × MBConv6(112, 192)  [1x1x192]
Stage 7: 1 × MBConv6(192, 320)  [1x1x320]
  ↓
Conv2d(320→1280, 1x1) + BatchNorm + Swish
  ↓
Global Average Pooling → Dropout → Linear(1280→10)
  ↓
Output (10 classes)
```

**MBConv (Mobile Inverted Bottleneck Convolution):**
```
x ────────────────────┐ (if in_channels == out_channels)
  ↓                   │
  Conv1x1 (expand)    │
  ↓                   │
  DepthwiseConv3x3    │  (Efficient spatial filtering)
  ↓                   │
  SE (Squeeze-Excite) │  (Channel attention)
  ↓                   │
  Conv1x1 (project)   │
  ↓                   │
  Add ←───────────────┘
```

### Variants
- **EfficientNet-B0**: 4.0M params, ~95.8% accuracy (baseline)
- **EfficientNet-B1**: 6.5M params, ~96.2% accuracy (scale up)

### Key Features
- **Compound Scaling**: Balanced depth, width, resolution scaling
- **MBConv Blocks**: Efficient mobile-inspired convolutions
- **Squeeze-and-Excitation**: Channel-wise attention mechanism
- **Swish Activation**: Smooth, non-monotonic activation
- **Stochastic Depth**: Randomly drop layers during training

### Strengths
✅ Best parameter efficiency (high accuracy with few params)
✅ SE attention improves feature quality
✅ Depthwise convolutions are computationally efficient
✅ Scales well (B0 to B7 for different compute budgets)
✅ Fast inference

### Weaknesses
❌ Slower training than ResNet
❌ More complex architecture (harder to modify)
❌ Requires more epochs for convergence (200-300)
❌ SE blocks add some computational overhead

### Usage
```bash
# EfficientNet-B0 (efficient)
python3 main_modern.py --model efficientnet_b0 --epochs 300 --amp

# EfficientNet-B1 (more accurate)
python3 main_modern.py --model efficientnet_b1 --epochs 300 --amp \
  --use-cutmix --use-randaugment

# With label smoothing
python3 main_modern.py --model efficientnet_b0 --epochs 300 --amp \
  --label-smoothing 0.1 --use-mixup
```

### Best For
- Resource-constrained environments (mobile, edge)
- When parameter efficiency is critical
- Transfer learning with limited compute
- Production deployments with strict latency requirements

---

## Vision Transformer

### Overview
Vision Transformer (ViT) applies the Transformer architecture (from NLP) directly to images by treating image patches as tokens. No convolutions used.

### Architecture
```
Input (32x32x3)
  ↓
Patch Embedding: Split into 16 patches (8x8 each) → Linear(192→768)  [16 × 768]
  ↓
Add Positional Embeddings + [CLS] token  [17 × 768]
  ↓
Transformer Encoder × 12:
  ├─ Multi-Head Self-Attention (8 heads)
  ├─ LayerNorm
  ├─ Feed-Forward Network (MLP)
  └─ LayerNorm
  ↓
Extract [CLS] token → Linear(768→10)
  ↓
Output (10 classes)
```

**Multi-Head Self-Attention:**
```
Each patch attends to all other patches:
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Learns global relationships between patches
(unlike CNNs which start with local receptive fields)
```

### Key Features
- **Patch Embeddings**: 8x8 or 16x16 image patches as tokens
- **Self-Attention**: Global receptive field from layer 1
- **Positional Embeddings**: Learnable position encodings
- **[CLS] Token**: Special token for classification
- **Layer Normalization**: Before each sub-layer

### Strengths
✅ Captures long-range dependencies natively
✅ Highly scalable (scales better than CNNs with more data)
✅ State-of-the-art on large datasets
✅ Interpretable attention maps
✅ Flexible (can handle variable-length sequences)

### Weaknesses
❌ Requires large datasets or strong augmentation
❌ Slow training (attention is O(n²) complexity)
❌ High memory usage
❌ Less effective on small datasets like CIFAR-10
❌ Needs very long training (300+ epochs)

### Usage
```bash
# Vision Transformer (requires strong augmentation)
python3 main_modern.py --model vit --epochs 300 --amp \
  --use-mixup --use-randaugment

# With label smoothing and long training
python3 main_modern.py --model vit --epochs 500 --amp \
  --use-mixup --use-cutmix --label-smoothing 0.1

# Smaller batch size (high memory usage)
python3 main_modern.py --model vit --epochs 300 --batch-size 64 --amp
```

### Best For
- Large-scale datasets (ImageNet, JFT-300M)
- When compute resources are abundant
- Research and experimentation
- Transfer learning from pre-trained models
- When interpretability (attention maps) is desired

---

## ConvNeXt

### Overview
ConvNeXt (2022) modernizes the standard CNN by incorporating design choices from Vision Transformers, achieving competitive performance while maintaining CNN efficiency.

### Architecture (ConvNeXt-CIFAR)
```
Input (32x32x3)
  ↓
Stem: Conv2d(3→64, 2x2, stride=2) + LayerNorm  [16x16x64]
  ↓
Stage 1: 3 × ConvNeXt Block(64)    [16x16x64]
  ↓ Downsample (2x2 conv)
Stage 2: 3 × ConvNeXt Block(128)   [8x8x128]
  ↓ Downsample (2x2 conv)
Stage 3: 9 × ConvNeXt Block(256)   [4x4x256]
  ↓ Downsample (2x2 conv)
Stage 4: 3 × ConvNeXt Block(512)   [2x2x512]
  ↓
LayerNorm + Global Average Pooling → Linear(512→10)
  ↓
Output (10 classes)
```

**ConvNeXt Block:**
```
x ────────────────────┐
  ↓                   │
  DepthwiseConv7x7    │  (Large kernel!)
  ↓                   │
  LayerNorm           │  (Not BatchNorm)
  ↓                   │
  Conv1x1 (expand 4x) │  (Inverted bottleneck)
  ↓                   │
  GELU                │  (Not ReLU)
  ↓                   │
  Conv1x1 (compress)  │
  ↓                   │
  Layer Scale         │  (Learnable per-channel scaling)
  ↓                   │
  Stochastic Depth    │  (Randomly drop block)
  ↓                   │
  Add ←───────────────┘
```

### Variants
- **ConvNeXt-CIFAR**: 12.5M params, ~96-97% accuracy (optimized for 32x32)
- **ConvNeXt-Tiny**: 27.8M params, ~97%+ accuracy (full-sized)
- **ConvNeXt-Small**: 49.5M params, ~97%+ accuracy

**Note**: ConvNeXt V2 (2023-2024) adds:
- Global Response Normalization (GRN) layer for better inter-channel feature competition
- Fully Convolutional Masked Autoencoder (FCMAE) for self-supervised pre-training
- Status: ⚠️ V2 features not yet implemented (V1 available)

### Key Features
- **Large Kernels (7x7)**: Larger receptive fields like transformers
- **LayerNorm**: Instead of BatchNorm (from transformers)
- **GELU Activation**: Smooth, non-monotonic (from transformers)
- **Inverted Bottleneck**: Expand then compress (from MobileNet/EfficientNet)
- **Depthwise Convolutions**: Efficient spatial mixing
- **Layer Scale**: Improves training stability
- **Stochastic Depth**: Regularization via random layer dropping

### Strengths
✅ State-of-the-art CNN performance
✅ Simpler than transformers (pure convolutions)
✅ Better than ResNet and comparable to ViT
✅ Efficient training (faster than ViT)
✅ Good trade-off between accuracy and speed
✅ Works well on small datasets (unlike ViT)

### Weaknesses
❌ More parameters than EfficientNet
❌ Requires long training (200-300 epochs)
❌ Large kernels (7x7) have some overhead
❌ Relatively new (less battle-tested)

### Usage
```bash
# ConvNeXt-CIFAR (optimized for CIFAR-10)
python3 main_modern.py --model convnext_cifar --epochs 200 --amp \
  --use-mixup --use-randaugment

# ConvNeXt-Tiny (more powerful)
python3 main_modern.py --model convnext_tiny --epochs 200 --amp \
  --use-cutmix --label-smoothing 0.1

# Full modern training
python3 main_modern.py --model convnext_cifar --epochs 300 --amp \
  --use-mixup --use-cutmix --use-randaugment \
  --optimizer adamw --scheduler cosine
```

### Best For
- State-of-the-art results with CNNs
- When you want transformer-level performance without attention
- Research and production
- When you need a balance of accuracy and efficiency
- Modern baseline for CNN research

---

## When to Use Each Architecture

### Choose **Custom CNN** if:
- 🎓 Learning deep learning basics
- ⚡ Need quick experiments
- 💻 Limited compute resources
- 📊 Baseline comparison

### Choose **ResNet** if:
- 🏆 Need reliable, proven architecture
- 🚀 Production deployment
- 📈 Good accuracy with reasonable compute
- 🔧 Want to fine-tune pre-trained models

### Choose **EfficientNet** if:
- 📱 Mobile/edge deployment
- 💰 Parameter efficiency is critical
- ⚡ Need fast inference
- 🎯 Best accuracy per parameter

### Choose **Vision Transformer** if:
- 🔬 Research experiments
- 💾 Have large datasets
- 🖼️ Need interpretable attention
- 🚀 Can afford long training

### Choose **ConvNeXt** if:
- 🏅 Want state-of-the-art CNN
- ⚖️ Need balance of accuracy and speed
- 🔍 Modern baseline for research
- 📊 Competing with transformers

---

## Training Time Comparison

On NVIDIA RTX 3090 (24GB), batch size 128, CIFAR-10:

| Model | 10 Epochs | 100 Epochs | 300 Epochs |
|-------|-----------|------------|------------|
| Custom CNN | 2 min | 15 min | 45 min |
| ResNet-18 | 8 min | 1.3 hours | 4 hours |
| ResNet-50 | 15 min | 2.5 hours | 7.5 hours |
| EfficientNet-B0 | 12 min | 2 hours | 6 hours |
| Vision Transformer | 25 min | 4 hours | 12 hours |
| ConvNeXt-CIFAR | 15 min | 2.5 hours | 7.5 hours |

*With mixed precision (AMP) enabled*

---

## Memory Usage Comparison

Peak GPU memory (batch size 128, CIFAR-10):

| Model | Memory | Max Batch Size (24GB GPU) |
|-------|--------|---------------------------|
| Custom CNN | ~1 GB | 2048+ |
| ResNet-18 | ~3 GB | 512 |
| ResNet-50 | ~5 GB | 256 |
| EfficientNet-B0 | ~4 GB | 384 |
| Vision Transformer | ~8 GB | 128 |
| ConvNeXt-CIFAR | ~5 GB | 256 |

---

## Accuracy vs Parameters

```
ConvNeXt-Tiny ────────┐  ~97%
Vision Transformer ──┐│  ~96.5%
ResNet-50 ──────────┐││  ~96%
EfficientNet-B1 ───┐│││  ~96.2%
ResNet-34 ────────┐││││  ~95.8%
EfficientNet-B0 ─┐│││││  ~95.8%
ResNet-18 ──────┐││││││  ~95.5%
ConvNeXt-CIFAR ┐│││││││  ~96-97%
Custom CNN ───┐││││││││  ~85-90%
             │││││││││
             10M  30M   Parameters
```

**Sweet Spot**: ResNet-18, EfficientNet-B0, or ConvNeXt-CIFAR

---

## Recommended Starting Points

### Beginner
```bash
python3 main.py --activation relu --quick
```

### Intermediate
```bash
python3 main_modern.py --model resnet18 --epochs 100 --amp
```

### Advanced
```bash
python3 main_modern.py --model convnext_cifar --epochs 200 --amp \
  --use-mixup --use-cutmix --use-randaugment \
  --optimizer adamw --scheduler cosine --weight-decay 1e-4
```

---

## Emerging Architectures (2024-2025) 🆕

These architectures represent cutting-edge research and potential future additions to the framework:

### LaViT (CVPR 2024)
**Efficient Vision Transformers via Attention Reuse**

**Key Innovation**:
- Computes self-attention only in initial layers
- Reuses attention scores through lightweight linear transformations
- Dramatically reduces computational cost

**Benefits**:
- 30-40% faster than standard ViT
- Minimal accuracy loss
- Better for resource-constrained environments

**Status**: ⚠️ Not yet implemented - advanced ViT optimization

---

### Hybrid CNN-Transformer (2025 Research)
**Best of Both Worlds**

**Recent Findings (2025)**:
- Hybrid CNN-Transformer achieves superior accuracy vs standalone models
- CNN extracts local features efficiently
- Transformer captures global dependencies
- Trade-off: Higher computational cost

**Current Implementation**:
- ✅ Basic CNN-Transformer hybrid available
- ⚠️ Latest 2025 optimizations not yet included

---

### ConvNeXt V2 (2023-2024)
**Co-designing CNNs with Self-Supervised Learning**

**What's New in V2**:
1. **Global Response Normalization (GRN)**
   - Enhances inter-channel feature competition
   - Improves feature quality without attention
2. **Fully Convolutional Masked Autoencoder (FCMAE)**
   - Self-supervised pre-training for CNNs
   - Sparse convolution on visible patches only
   - Transfer learning without ImageNet

**Status**:
- ✅ ConvNeXt V1 fully implemented
- ⚠️ V2 features (GRN, FCMAE) not yet added
- 🎓 Great intermediate project for learners

---

## Educational Roadmap

For learners looking to expand this framework, here's a suggested progression:

### Beginner Projects
1. **Test-Time Augmentation**: Simple ensemble technique
2. **TrivialAugment**: Simpler than RandAugment
3. **Model Ensembling**: Average predictions from multiple models

### Intermediate Projects
4. **ConvNeXt V2 GRN Layer**: Add to existing ConvNeXt
5. **Knowledge Distillation**: Compress large models to small ones
6. **Lion Optimizer**: Alternative to AdamW

### Advanced Projects
7. **LaViT-style Attention Optimization**: Efficient transformers
8. **Masked Autoencoders**: Self-supervised pre-training
9. **Sophia Optimizer**: Second-order optimization

---

## Architecture Evolution Timeline

```
2012: AlexNet (CNN revolution)
2015: ResNet (skip connections)
2017: Squeeze-and-Excitation (channel attention)
2019: EfficientNet (compound scaling)
2020: Vision Transformer (pure attention)
2022: ConvNeXt (modernized CNN)
2023: ConvNeXt V2 (self-supervised CNNs)
2024: LaViT (efficient ViT)
2025: Hybrid models dominate
```

**Current State of the Art (2025)**:
- Best accuracy: Vision Transformers with massive pre-training
- Best efficiency: ConvNeXt V2 and EfficientNetV2
- Best for CIFAR-10: Hybrid CNN-Transformer or ConvNeXt
- Future direction: Hybrid architectures + self-supervised learning

---

**For detailed usage, see `QUICK_START.md` and `README.md`**

**For latest techniques, see `MODERN_DL_GUIDE.md`**
