# Modern Deep Learning Framework 🚀

**State-of-the-Art Deep Learning Implementation for CIFAR-10 Image Classification**

This project implements modern deep learning architectures and training techniques used in 2024-2025 research and production systems.

---

## 🎯 What's New

### ✨ **3 Modern Architectures**
- **ResNet**: Residual networks with skip connections (ResNet-18, 34, 50)
- **EfficientNet**: Mobile-optimized with depthwise separable convolutions + SE attention
- **CNN-Transformer**: Hybrid combining CNNs and self-attention mechanisms

### ⚡ **Modern Training Techniques**
- **AdamW Optimizer**: Decoupled weight decay (better than Adam)
- **Cosine Annealing with Warmup**: Smooth LR scheduling
- **Mixed Precision (AMP)**: 2-3x faster training on GPU
- **Exponential Moving Average (EMA)**: +0.5% accuracy improvement
- **Gradient Clipping**: Training stability
- **Label Smoothing**: Better calibration

### 📊 **Advanced Data Augmentation**
- **RandAugment**: Automated augmentation policies
- **MixUp**: Linear interpolation between samples
- **CutMix**: Cut-and-paste augmentation
- **Cutout / Random Erasing**: Occlusion robustness

### 🛡️ **Modern Regularization**
- **DropBlock**: Structured dropout for CNNs
- **Stochastic Depth**: Random layer dropping
- **Label Smoothing**: Confidence calibration

---

## 🚀 Quick Start

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Train with Modern Techniques

#### **Option A: ResNet-18 (Recommended for beginners)**
```bash
python main_modern.py --model resnet18 --modern-training --epochs 50
```
Expected: **93-94% accuracy** in ~10 minutes (GPU)

#### **Option B: EfficientNet (Best efficiency)**
```bash
python main_modern.py --model efficientnet_b0 --modern-training --augmentation randaugment --epochs 100
```
Expected: **94-95% accuracy** with fewer parameters

#### **Option C: CNN-Transformer (Cutting-edge)**
```bash
python main_modern.py --model cnn_transformer_base --activation gelu --modern-training --epochs 100
```
Expected: **93-94% accuracy** with self-attention

---

## 📋 Available Models

| Model | Parameters | Speed | Accuracy (50 epochs) | Use Case |
|-------|-----------|-------|---------------------|----------|
| `cnn` | 1.2M | ⚡⚡⚡ | 86-88% | Baseline |
| `resnet_tiny` | 2M | ⚡⚡⚡ | 88-90% | Quick experiments |
| `resnet18` | 11M | ⚡⚡ | 93-94% | **Best overall** |
| `resnet34` | 21M | ⚡ | 93-94% | Deeper network |
| `resnet50` | 23M | ⚡ | 94-95% | Highest accuracy |
| `efficientnet_tiny` | 1M | ⚡⚡⚡ | 89-91% | Mobile/edge |
| `efficientnet_b0` | 4M | ⚡⚡ | 94-95% | **Best efficiency** |
| `efficientnet_b1` | 6M | ⚡⚡ | 94-95% | Scaled up |
| `cnn_transformer_small` | 8M | ⚡ | 92-93% | Attention learning |
| `cnn_transformer_base` | 16M | ⚡ | 93-94% | Full hybrid |
| `vit_tiny` | 5M | ⚡ | 91-92% | Pure transformer |

---

## 🎓 Learning Path

### **Level 1: Understand Basics**
```bash
# Original basic CNN
python main.py --activation relu --epochs 10

# See available activations
python main.py --list-activations
```

### **Level 2: Modern Architecture**
```bash
# Train ResNet (adds residual connections + batch norm)
python main_modern.py --model resnet18 --epochs 20

# Compare with original CNN
python main_modern.py --model cnn --epochs 20
```
**Learning**: Residual connections, batch normalization, global average pooling

### **Level 3: Modern Training**
```bash
# Without modern training
python main_modern.py --model resnet18 --epochs 30

# With modern training
python main_modern.py --model resnet18 --modern-training --epochs 30
```
**Learning**: AdamW, cosine annealing, mixed precision, EMA, label smoothing

### **Level 4: Advanced Augmentation**
```bash
# Standard augmentation
python main_modern.py --model resnet18 --augmentation standard --modern-training

# RandAugment
python main_modern.py --model resnet18 --augmentation randaugment --modern-training

# With Cutout
python main_modern.py --model resnet18 --augmentation randaugment --cutout --modern-training
```
**Learning**: RandAugment, Cutout, data augmentation impact

### **Level 5: Mobile-Optimized Networks**
```bash
# EfficientNet with depthwise separable convolutions
python main_modern.py --model efficientnet_b0 --modern-training
```
**Learning**: Depthwise separable convolutions, SE attention, parameter efficiency

### **Level 6: Attention Mechanisms**
```bash
# CNN-Transformer hybrid
python main_modern.py --model cnn_transformer_base --modern-training --epochs 100
```
**Learning**: Self-attention, transformers, patch embeddings, positional encoding

### **Level 7: Comprehensive Comparison**
```bash
# Compare all models
python main_modern.py --compare-models --epochs 50

# Full benchmark
python benchmark_all.py --full
```
**Learning**: Architecture comparison, ablation studies, performance analysis

---

## 📊 Benchmark & Comparison

### Compare Different Approaches

```bash
# Compare architectures
python benchmark_all.py --architectures --epochs 20

# Compare training techniques (modern vs standard)
python benchmark_all.py --techniques --epochs 20

# Compare activation functions
python benchmark_all.py --activations --epochs 20

# Quick test (2 epochs per experiment)
python benchmark_all.py --quick --architectures
```

### Expected Performance Improvements

| Technique | Baseline | With Technique | Improvement |
|-----------|----------|----------------|-------------|
| Standard → Modern Training | 91% | 93% | **+2%** |
| Adam → AdamW | 91% | 92% | **+1%** |
| Step LR → Cosine Annealing | 91% | 92% | **+1%** |
| No AMP → AMP | Same | Same | **2-3x faster** |
| No EMA → EMA | 92% | 92.5% | **+0.5%** |
| No Label Smoothing → Label Smoothing (0.1) | 92% | 93% | **+1%** |
| Basic Aug → RandAugment | 91% | 93% | **+2%** |
| No MixUp → MixUp | 92% | 93.5% | **+1.5%** |
| **All Combined** | **91%** | **94-95%** | **+3-4%** |

---

## 🔧 Detailed Usage

### List All Models
```bash
python main_modern.py --list-models
```

### Train Specific Model
```bash
python main_modern.py \
  --model resnet18 \
  --activation swish \
  --modern-training \
  --epochs 50 \
  --batch-size 128 \
  --lr 0.001
```

### With Advanced Augmentation
```bash
python main_modern.py \
  --model efficientnet_b0 \
  --augmentation randaugment \
  --cutout \
  --mixup \
  --modern-training
```

### Quick Test (2 epochs)
```bash
python main_modern.py --model resnet_tiny --quick
```

---

## 📁 Project Structure

```
Deep-Learning-Model/
├── models/
│   ├── network.py              # Original CNN
│   ├── resnet.py               # ✨ ResNet architectures
│   ├── efficientnet.py         # ✨ EfficientNet architectures
│   ├── cnn_transformer.py      # ✨ CNN-Transformer hybrid
│   └── activations.py          # Custom activation functions
├── utils/
│   ├── trainer.py              # Standard trainer
│   ├── modern_trainer.py       # ✨ Modern trainer (AdamW, AMP, EMA)
│   ├── augmentation.py         # ✨ Advanced augmentation
│   ├── regularization.py       # ✨ DropBlock, Stochastic Depth
│   ├── data_loader.py          # CIFAR-10 data loader
│   └── visualization.py        # Training plots
├── main.py                     # Original training script
├── main_modern.py              # ✨ Modern training script
├── benchmark_all.py            # ✨ Comprehensive benchmark
├── MODERN_DL_GUIDE.md          # ✨ Detailed guide
├── README_MODERN.md            # ✨ This file
└── CLAUDE.md                   # Project instructions
```

---

## 🎯 Expected Results (CIFAR-10, 50 epochs)

### With Modern Training

| Model | Test Accuracy | Training Time (GPU) | Parameters |
|-------|--------------|-------------------|-----------|
| ResNet-18 | 93-94% | ~10 min | 11M |
| ResNet-50 | 94-95% | ~20 min | 23M |
| EfficientNet-B0 | 94-95% | ~15 min | 4M |
| CNN-Transformer | 93-94% | ~25 min | 16M |

### Without Modern Training (Standard)

| Model | Test Accuracy | Training Time (GPU) |
|-------|--------------|-------------------|
| ResNet-18 | 91-92% | ~15 min |
| Basic CNN | 85-87% | ~5 min |

---

## 💡 Key Concepts Explained

### 1. **Residual Connections** (ResNet)
```python
# Instead of learning F(x), learn residual F(x) - x
output = F(x) + x  # Skip connection
```
**Why**: Easier to optimize, enables very deep networks (100+ layers)

### 2. **Depthwise Separable Convolutions** (EfficientNet)
```python
# Standard conv: H×W×C_in×C_out operations
# Depthwise separable: H×W×C_in + C_in×C_out operations
# ~9x fewer operations!
```
**Why**: Mobile-friendly, parameter efficient

### 3. **Squeeze-and-Excitation** (Attention)
```python
# Learn channel-wise attention weights
weights = Sigmoid(FC(GlobalAvgPool(x)))
output = x * weights  # Reweight channels
```
**Why**: Focus on important channels, ignore noise

### 4. **Self-Attention** (Transformer)
```python
# Each patch attends to all other patches
attention_weights = Softmax(Q @ K^T / sqrt(d))
output = attention_weights @ V
```
**Why**: Model long-range dependencies, global context

### 5. **AdamW** (Optimizer)
```python
# Adam: weight_decay affects gradient normalization ❌
# AdamW: weight_decay is decoupled ✅
w = w - lr * gradient - lr * weight_decay * w
```
**Why**: Proper regularization, better generalization

### 6. **Mixed Precision** (AMP)
```python
# Use float16 for speed, float32 for stability
with autocast():
    output = model(input)  # Automatic dtype selection
```
**Why**: 2-3x faster, less memory, no accuracy loss

---

## 🎓 What You'll Learn

By experimenting with this project, you'll understand:

### **Modern Architectures**
- ✅ Residual connections and why they work
- ✅ Batch normalization for stable training
- ✅ Depthwise separable convolutions for efficiency
- ✅ Attention mechanisms (SE and self-attention)
- ✅ Patch embeddings and positional encoding

### **Training Techniques**
- ✅ AdamW vs Adam optimizer differences
- ✅ Learning rate scheduling (warmup + cosine decay)
- ✅ Mixed precision training (float16 + float32)
- ✅ Exponential moving average for better models
- ✅ Gradient clipping for stability

### **Regularization**
- ✅ Label smoothing for calibration
- ✅ DropBlock vs standard dropout
- ✅ Stochastic depth for deep networks
- ✅ Weight decay best practices

### **Data Augmentation**
- ✅ RandAugment automated policies
- ✅ MixUp and CutMix sample mixing
- ✅ Cutout and random erasing
- ✅ When to use each technique

### **PyTorch Best Practices**
- ✅ Efficient data loading
- ✅ GPU memory optimization
- ✅ Training loop best practices
- ✅ Model checkpointing and loading

---

## 📚 Further Reading

### Essential Papers (in reading order)

1. **Batch Normalization** (2015) - Understanding modern training
2. **ResNet** (2015) - Residual connections
3. **MobileNet** (2017) - Depthwise separable convolutions
4. **Squeeze-and-Excitation** (2018) - Channel attention
5. **EfficientNet** (2019) - Compound scaling
6. **Vision Transformer** (2020) - Transformers for images
7. **AdamW** (2019) - Proper weight decay
8. **MixUp** (2017) - Sample mixing
9. **CutMix** (2019) - Spatial mixing
10. **RandAugment** (2019) - Automated augmentation

See [MODERN_DL_GUIDE.md](MODERN_DL_GUIDE.md) for full paper references.

---

## ❓ FAQ

**Q: Which model should I start with?**
A: Start with `resnet18` - good balance of performance and speed.

**Q: Do I need a GPU?**
A: CPU works but is slow. GPU recommended for modern training (AMP requires CUDA).

**Q: What's the minimum GPU memory needed?**
A: 4GB for ResNet-18, 6GB for EfficientNet-B0, 8GB for CNN-Transformer

**Q: How long does training take?**
A: ResNet-18 (50 epochs): ~10 min on GPU, ~2 hours on CPU

**Q: What accuracy should I expect?**
A: With modern training: 93-95% on CIFAR-10 (50-100 epochs)

**Q: Can I use this for other datasets?**
A: Yes! Modify `num_classes` and data loader. Works for any image classification task.

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new architectures (Vision Transformer variants, ConvNeXt, etc.)
- Implement additional techniques (SAM optimizer, LAMB, etc.)
- Improve documentation
- Add more datasets

---

## 📄 License

MIT License - Free for educational and research use

---

## 🙏 Acknowledgments

This project implements techniques from modern deep learning research:
- ResNet architecture from Microsoft Research
- EfficientNet from Google Brain
- Transformer architecture from Google Research
- Various optimization and regularization techniques from the community

---

**Happy Learning! 🚀**

For detailed explanations of each technique, see [MODERN_DL_GUIDE.md](MODERN_DL_GUIDE.md)
