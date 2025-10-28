# Quick Start Guide

Get started with the Modern Deep Learning Framework in under 5 minutes!

## 1. Installation (2 minutes)

```bash
# Navigate to project directory
cd Deep-Learning-Model

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Your First Training Run (3 minutes)

### Quick Test with ReLU
```bash
# Train for 2 epochs (~3-5 minutes on GPU, ~10 minutes on CPU)
python3 main.py --activation relu --quick
```

**Expected Output:**
```
Epoch 1/2: 100%|████████| Train Loss: 1.234 | Train Acc: 55.2% | Val Acc: 58.3%
Epoch 2/2: 100%|████████| Train Loss: 0.987 | Train Acc: 65.1% | Val Acc: 67.8%

Training complete! Best validation accuracy: 67.8%
Model saved to: checkpoints/relu/best_model.pth
```

### View Results
```bash
ls checkpoints/relu/
# Output: best_model.pth  training_history.png  confusion_matrix.png  predictions.png
```

Open the generated PNG files to see:
- Training/validation curves
- Confusion matrix
- Sample predictions

## 3. Try Modern Architectures (10-30 minutes)

### ResNet-18 (Recommended for beginners)
```bash
# Train with modern techniques
python3 main_modern.py --model resnet18 --epochs 10 --amp

# Expected: ~88-90% accuracy after 10 epochs
```

### EfficientNet-B0 (Best efficiency)
```bash
python3 main_modern.py --model efficientnet_b0 --epochs 10 --amp --use-randaugment

# Expected: ~87-89% accuracy after 10 epochs
```

### ConvNeXt-CIFAR (State-of-the-art)
```bash
python3 main_modern.py --model convnext_cifar --epochs 10 --amp --use-mixup

# Expected: ~89-91% accuracy after 10 epochs
```

## 4. Live Visualization

### Real-time Training Monitor
```bash
# Watch training progress in real-time
python3 main.py --activation swish --epochs 5 --monitor
```

This opens a live plot window showing:
- Training/validation loss
- Training/validation accuracy
- Learning rate schedule

### Network Structure Visualization
```bash
# Visualize neural network architecture
python3 main.py --activation relu --visualize --quick
```

## 5. Compare Activation Functions

### Modern Activations (GELU, Swish, Mish, SiLU, Hardswish)
```bash
python3 main.py --activation modern --epochs 5

# Trains 5 models in sequence, comparing performance
```

### Classic Activations (ReLU, Tanh, Sigmoid, LeakyReLU, ELU)
```bash
python3 main.py --activation classic --epochs 5
```

### All Available Activations
```bash
# Warning: This takes 2-3 hours!
python3 main.py --activation all --epochs 3
```

## 6. Understanding the Output

### Checkpoints Directory
```
checkpoints/
├── relu/
│   ├── best_model.pth           # Saved model weights
│   ├── training_history.png     # Loss/accuracy curves
│   ├── confusion_matrix.png     # Per-class performance
│   └── predictions.png          # Sample predictions
├── swish/
│   └── ...
└── resnet18/
    └── ...
```

### Training Logs
```
Epoch 5/10: 100%|██████████████| 351/351 [00:45<00:00,  7.73it/s]
  Train Loss: 0.512 | Train Acc: 82.3%
  Val Loss: 0.487 | Val Acc: 83.1%
  ✓ New best validation accuracy!
```

## 7. Common Issues & Solutions

### Issue: CUDA out of memory
```bash
# Solution: Reduce batch size
python3 main.py --activation relu --batch-size 64 --quick
```

### Issue: Training too slow
```bash
# Solution: Enable mixed precision (AMP)
python3 main_modern.py --model resnet18 --amp --epochs 10
```

### Issue: Low accuracy after a few epochs
```
# Normal! Modern models need longer training:
# - 50-100 epochs for custom CNN
# - 100-200 epochs for ResNet/EfficientNet
# - 200-300 epochs for Vision Transformer
```

## 8. Next Steps

### For Better Accuracy
```bash
# Train longer with strong augmentation
python3 main_modern.py \
  --model resnet18 \
  --epochs 200 \
  --batch-size 128 \
  --use-mixup \
  --use-cutmix \
  --use-randaugment \
  --amp
```

### For Research/Experimentation
```bash
# Customize hyperparameters
python3 main_modern.py \
  --model efficientnet_b0 \
  --epochs 100 \
  --lr 0.001 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --label-smoothing 0.1 \
  --amp
```

### Benchmark All Architectures
```bash
# Compare all models
python3 benchmark_all.py --epochs 50
```

## 9. Tips for Success

1. **Start Small**: Use `--quick` flag to test code changes rapidly
2. **Use GPU**: 10-50x faster than CPU training
3. **Monitor Validation**: If val loss increases while train loss decreases → overfitting
4. **Try Different Models**: ResNet-18 is a good starting point
5. **Enable AMP**: Almost free 2x speedup with `--amp`
6. **Use Augmentation**: `--use-randaugment --use-mixup` significantly improves accuracy
7. **Train Longer**: Modern models need 100-300 epochs for best results

## 10. Command Reference

### Basic Training
```bash
python3 main.py --activation <name> [--epochs N] [--batch-size N] [--lr LR]
```

### Modern Training
```bash
python3 main_modern.py --model <name> [--epochs N] [--amp] [--use-mixup]
```

### Flags
- `--quick`: 2 epochs for rapid testing
- `--amp`: Mixed precision training (2x speedup)
- `--monitor`: Real-time training plots
- `--visualize`: Network structure visualization
- `--use-mixup`: MixUp augmentation
- `--use-cutmix`: CutMix augmentation
- `--use-randaugment`: RandAugment
- `--label-smoothing ALPHA`: Label smoothing (default: 0.0)

## 11. Troubleshooting

### Check PyTorch Installation
```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Verify Dataset Download
```bash
ls data/cifar-10-batches-py/
# Should show: batches.meta  data_batch_1  data_batch_2  ...
```

### Test Model Import
```bash
python3 -c "from models import resnet18, convnext_cifar; print('Models imported successfully!')"
```

## 12. Getting Help

- **Documentation**: See `README.md` and `CLAUDE.md`
- **Architecture Details**: See `ARCHITECTURES.md`
- **Code Examples**: Check `main.py` and `main_modern.py`
- **Issues**: Open an issue on GitHub

---

**Happy Training! 🚀**

For more details, see the full documentation in `README.md` and `CLAUDE.md`.
