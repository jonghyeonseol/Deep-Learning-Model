# Deep Learning Framework - User Guide

Complete guide for using the CIFAR-10 image classification framework with modern deep learning techniques.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Configuration System](#configuration-system)
4. [Training Models](#training-models)
5. [Visualization and Monitoring](#visualization-and-monitoring)
6. [Advanced Techniques](#advanced-techniques)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Deep-Learning-Model

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Training Run (5 minutes)

```bash
# Quick test with 2 epochs
python3 main.py --activation relu --quick

# Or use configuration file
python3 train_with_config.py --config configs/quick_test.yaml
```

### Check Results

Training outputs are saved to:
- **Checkpoints**: `checkpoints/{model_name}/best_model.pth`
- **Logs**: `logs/{model_name}/training.log`
- **Visualizations**: `checkpoints/{model_name}/*.png`
- **TensorBoard**: `checkpoints/{model_name}/logs/`

---

## Basic Usage

### 1. Training with Command Line Arguments

```bash
# Train ResNet-18 with Swish activation
python3 main_modern.py \
    --model resnet18 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --optimizer sgd

# Train EfficientNet with all features
python3 main_modern.py \
    --model efficientnet_b0 \
    --epochs 300 \
    --use-mixup \
    --use-cutmix \
    --use-randaugment \
    --amp
```

### 2. Training with Configuration Files

```bash
# Use pre-made configuration
python3 train_with_config.py --config configs/resnet18_basic.yaml

# Customize and use your own config
cp configs/resnet18_basic.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
python3 train_with_config.py --config configs/my_experiment.yaml
```

### 3. Python API Usage

```python
from utils.config import load_config, validate_training_config
from utils.data_loader import CIFAR10DataLoader
from models.resnet import ResNet18
from utils.modern_trainer import ModernTrainer

# Load and validate configuration
config = load_config('configs/resnet18_basic.yaml')
validate_training_config(config)

# Create data loaders
data_loader = CIFAR10DataLoader(
    batch_size=config.training.batch_size,
    num_workers=config.data.num_workers
)
train_loader, val_loader, test_loader = data_loader.get_loaders()

# Create model
model = ResNet18(
    num_classes=config.model.num_classes,
    activation=config.model.activation
)

# Create trainer
trainer = ModernTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    save_dir=config.checkpoint.save_dir
)

# Configure training
trainer.configure_optimizer(
    optimizer_name=config.training.optimizer,
    lr=config.training.learning_rate,
    weight_decay=config.training.weight_decay
)

trainer.configure_scheduler(
    scheduler_name=config.scheduler.type,
    T_max=config.training.epochs
)

# Train
trainer.train(
    epochs=config.training.epochs,
    save_best=True
)

# Test
test_loss, test_acc = trainer.test()
print(f'Test Accuracy: {test_acc:.2f}%')
```

---

## Configuration System

### Configuration File Structure

```yaml
# my_config.yaml
model:
  architecture: resnet18
  activation: swish
  num_classes: 10

training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.1
  optimizer: sgd
  weight_decay: 0.0005

scheduler:
  type: cosine
  min_lr: 0.000001

augmentation:
  use_randaugment: true
  use_mixup: false
  use_cutmix: false

data:
  dataset: cifar10
  data_dir: ./data
  num_workers: 4

checkpoint:
  save_dir: ./checkpoints
  save_best: true

logging:
  log_dir: ./logs
  tensorboard: true
```

### Loading Configurations

```python
from utils.config import load_config, save_config, Config

# Load from file
config = load_config('configs/my_config.yaml')

# Access with dot notation
print(config.model.architecture)  # 'resnet18'
print(config.training.batch_size)  # 128

# Access nested values with string path
lr = config.get('training.learning_rate', default=0.001)

# Modify configuration
config.training.epochs = 200
config.update({'training': {'batch_size': 256}})

# Save modified configuration
save_config(config, 'configs/modified_config.yaml')
```

### Creating Configurations Programmatically

```python
from utils.config import Config, save_config, EXAMPLE_CONFIGS

# Use example template
config = Config(EXAMPLE_CONFIGS['resnet18_basic'])

# Modify as needed
config.training.epochs = 50
config.model.activation = 'mish'

# Save
save_config(config, 'configs/custom_config.yaml')
```

### Validating Configurations

```python
from utils.config import load_config, validate_training_config
from utils.exceptions import ConfigurationError

try:
    config = load_config('configs/my_config.yaml')
    validate_training_config(config)
    print("Configuration is valid!")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

---

## Training Models

### Available Models

1. **ResNet** (resnet18, resnet34, resnet50, resnet101, resnet152)
   - Classic residual networks
   - Best for: Baseline experiments, fast training

2. **EfficientNet** (efficientnet_b0 through efficientnet_b7)
   - State-of-the-art CNNs with compound scaling
   - Best for: Maximum efficiency and accuracy

3. **Vision Transformer** (vit)
   - Transformer-based architecture
   - Best for: Research, attention visualization

4. **ConvNeXt** (convnext_tiny, convnext_small, convnext_base)
   - Modern CNN architecture
   - Best for: Matching transformer performance with CNN efficiency

### Training Strategies

#### Strategy 1: Quick Baseline (30 minutes)

```bash
python3 train_with_config.py --config configs/resnet18_basic.yaml
```

**Expected Results**:
- Training time: ~30 minutes on RTX 3090
- Test accuracy: ~95.5%
- Good for: Initial experiments, baseline comparisons

#### Strategy 2: Maximum Performance (2-3 hours)

```bash
python3 train_with_config.py --config configs/efficientnet_modern.yaml
```

**Expected Results**:
- Training time: ~2 hours on RTX 3090
- Test accuracy: ~96%+
- Good for: Production models, competitions

#### Strategy 3: Transformer Research (2-3 hours)

```bash
python3 train_with_config.py --config configs/vit_transformer.yaml
```

**Expected Results**:
- Training time: ~2-3 hours on RTX 3090
- Test accuracy: ~96.5%+
- Good for: Attention visualization, transformer research

### Resuming Training

```python
from utils.trainer import Trainer

# Create trainer
trainer = Trainer(model, train_loader, val_loader, save_dir='checkpoints')

# Load checkpoint
trainer.load_checkpoint('best_model.pth')

# Continue training
trainer.train(epochs=50, save_best=True)
```

### Transfer Learning

```python
import torch
from models.resnet import ResNet50

# Load pretrained model
model = ResNet50(num_classes=10)
checkpoint = torch.load('path/to/pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze early layers
for name, param in model.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        param.requires_grad = False

# Fine-tune
trainer = Trainer(model, train_loader, val_loader)
trainer.configure_optimizer(lr=0.001)  # Lower LR for fine-tuning
trainer.train(epochs=20)
```

---

## Visualization and Monitoring

### Real-Time Training Monitoring

```bash
# Enable live training plots
python3 main.py --activation swish --epochs 10 --monitor
```

Features:
- Live loss and accuracy curves
- Gradient flow visualization
- Learning rate tracking
- Updates every epoch

### TensorBoard Visualization

```bash
# Start TensorBoard server
tensorboard --logdir checkpoints/{model_name}/logs

# Open browser to http://localhost:6006
```

Available metrics:
- Training/validation loss curves
- Training/validation accuracy
- Learning rate schedule
- Gradient norms
- Model graph structure

### Network Structure Visualization

```bash
# Visualize network architecture
python3 main.py --visualize --activation relu
```

Shows:
- Neuron connections
- Layer structure
- Activation patterns
- Forward pass flow

### Post-Training Analysis

```python
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_predictions

# Plot training curves
plot_training_history(
    trainer.history,
    save_path='analysis/training_curves.png'
)

# Confusion matrix
plot_confusion_matrix(
    model, test_loader, device,
    save_path='analysis/confusion_matrix.png'
)

# Sample predictions
plot_predictions(
    model, test_loader, device, num_samples=16,
    save_path='analysis/predictions.png'
)
```

---

## Advanced Techniques

### 1. Mixed Precision Training (2x Speedup)

```python
from utils.modern_trainer import ModernTrainer

trainer = ModernTrainer(model, train_loader, val_loader)
trainer.configure_amp(enabled=True)  # Enable mixed precision
trainer.train(epochs=100)
```

Or in configuration:
```yaml
training:
  use_amp: true
```

**Benefits**:
- 2x faster training
- Lower GPU memory usage
- Minimal accuracy loss (<0.1%)

### 2. Exponential Moving Average (EMA)

```python
trainer.configure_ema(enabled=True, decay=0.9999)
trainer.train(epochs=100)

# Use EMA model for inference
trainer.use_ema_for_eval(True)
test_loss, test_acc = trainer.test()
```

Or in configuration:
```yaml
training:
  use_ema: true
  ema_decay: 0.9999
```

**Benefits**:
- More stable inference
- Better generalization
- Standard in modern training

### 3. Advanced Data Augmentation

#### RandAugment (Simple but Effective)

```python
from utils.augmentation import RandAugment

augment = RandAugment(n=2, m=9)  # n operations with magnitude m
augmented_image = augment(image)
```

Configuration:
```yaml
augmentation:
  use_randaugment: true
  randaugment_n: 2  # Number of operations
  randaugment_m: 9  # Magnitude (0-30)
```

#### MixUp (Mix Images and Labels)

```python
from utils.augmentation import mixup_data

# In training loop
inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)
outputs = model(inputs)
loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
```

Configuration:
```yaml
augmentation:
  use_mixup: true
  mixup_alpha: 1.0
```

#### CutMix (Cut and Paste Patches)

```python
from utils.augmentation import cutmix_data

inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0)
outputs = model(inputs)
loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
```

Configuration:
```yaml
augmentation:
  use_cutmix: true
  cutmix_alpha: 1.0
  cutmix_prob: 0.5
```

### 4. Advanced Optimizers

#### AdamW (Adam with Decoupled Weight Decay)

```python
trainer.configure_optimizer(
    optimizer_name='adamw',
    lr=0.001,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)
```

**When to use**:
- Transformer models
- Long training runs (200+ epochs)
- When Adam with L2 regularization fails

#### SGD with Momentum (Classic but Effective)

```python
trainer.configure_optimizer(
    optimizer_name='sgd',
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)
```

**When to use**:
- ResNet, VGG architectures
- Short to medium training (50-150 epochs)
- When you want interpretable training

### 5. Learning Rate Schedules

#### Cosine Annealing with Warmup

```python
trainer.configure_scheduler(
    scheduler_name='cosine_warmup',
    T_max=200,
    warmup_epochs=20,
    min_lr=1e-6
)
```

**Benefits**:
- Smooth learning rate decay
- Periodic restarts prevent local minima
- Standard for transformers

#### ReduceLROnPlateau (Adaptive)

```python
trainer.configure_scheduler(
    scheduler_name='plateau',
    patience=10,
    factor=0.1,
    min_lr=1e-6
)
```

**Benefits**:
- Automatically adapts to training
- Good for unknown optimal schedules
- Reduces LR when stuck

### 6. Regularization Techniques

#### Label Smoothing

```python
trainer.configure_criterion(
    criterion_name='crossentropy',
    label_smoothing=0.1
)
```

**Benefits**:
- Prevents overconfident predictions
- Better generalization
- Standard in modern training

#### DropBlock (for CNNs)

```python
from utils.regularization import DropBlock2D

model = ResNet18(num_classes=10)
model.layer3.register_forward_hook(DropBlock2D(block_size=5, drop_prob=0.1))
```

**Benefits**:
- Better than standard dropout for CNNs
- Removes contiguous regions
- Improves feature learning

#### Stochastic Depth (for Deep Networks)

```python
from models.resnet import ResNet50

model = ResNet50(
    num_classes=10,
    stochastic_depth=True,
    survival_prob=0.8
)
```

**Benefits**:
- Enables training very deep networks (1000+ layers)
- Improves gradient flow
- Acts as implicit ensemble

### 7. Model Ensemble

```python
from torch.nn import functional as F

# Train multiple models
models = [
    train_model('resnet18'),
    train_model('efficientnet_b0'),
    train_model('vit')
]

# Ensemble prediction
def ensemble_predict(models, inputs):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            output = F.softmax(model(inputs), dim=1)
            predictions.append(output)
    return torch.stack(predictions).mean(0)
```

**Benefits**:
- 0.5-2% accuracy improvement
- More robust predictions
- Reduces variance

---

## API Reference

### Core Classes

#### Config
```python
from utils.config import Config, load_config, save_config

config = Config({'model': {'architecture': 'resnet18'}})
config.model.architecture  # Access with dot notation
config.get('model.learning_rate', 0.001)  # Nested access with default
save_config(config, 'config.yaml')  # Save to file
```

#### Trainer
```python
from utils.trainer import Trainer

trainer = Trainer(
    model,              # nn.Module
    train_loader,       # DataLoader
    val_loader,         # DataLoader (optional)
    test_loader,        # DataLoader (optional)
    device,             # torch.device (optional)
    save_dir            # str, checkpoint directory
)

trainer.configure_optimizer(optimizer_name, lr, weight_decay, **kwargs)
trainer.configure_scheduler(scheduler_name, **kwargs)
trainer.configure_criterion(criterion_name)
trainer.train(epochs, early_stopping_patience, save_best)
trainer.test()
trainer.save_checkpoint(filename)
trainer.load_checkpoint(filename)
```

#### ModernTrainer (extends Trainer)
```python
from utils.modern_trainer import ModernTrainer

trainer = ModernTrainer(model, train_loader, val_loader, test_loader, save_dir)
trainer.configure_amp(enabled=True)
trainer.configure_ema(enabled=True, decay=0.9999)
trainer.configure_gradient_clipping(max_norm=1.0)
trainer.use_ema_for_eval(use_ema=True)
```

#### CIFAR10DataLoader
```python
from utils.data_loader import CIFAR10DataLoader

loader = CIFAR10DataLoader(
    data_dir='./data',
    batch_size=128,
    num_workers=4,
    validation_split=0.1,
    augmentation=True
)

train_loader, val_loader, test_loader = loader.get_loaders()
```

### Model Constructors

```python
from models.resnet import ResNet18, ResNet34, ResNet50
from models.efficientnet import EfficientNet_B0
from models.cnn_transformer import VisionTransformer
from models.convnext import ConvNeXt_Tiny

# ResNet
model = ResNet18(num_classes=10, activation='swish')

# EfficientNet
model = EfficientNet_B0(num_classes=10, drop_rate=0.2)

# Vision Transformer
model = VisionTransformer(
    image_size=32,
    patch_size=4,
    num_classes=10,
    embed_dim=384,
    depth=7,
    num_heads=6
)

# ConvNeXt
model = ConvNeXt_Tiny(num_classes=10)
```

### Utility Functions

```python
from utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_predictions,
    visualize_attention_maps
)

from utils.metrics import (
    calculate_accuracy,
    calculate_top_k_accuracy,
    compute_confusion_matrix
)

from utils.logger import get_logger, setup_training_logger

# Logging
logger = get_logger(__name__, level=logging.INFO, log_file='training.log')
logger.info("Training started")

# Visualization
plot_training_history(history, save_path='curves.png')
plot_confusion_matrix(model, test_loader, device, save_path='cm.png')
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `batch_size: 64` → `batch_size: 32`
- Enable mixed precision: `use_amp: true`
- Reduce model size: `resnet50` → `resnet18`
- Reduce number of workers: `num_workers: 8` → `num_workers: 4`

#### 2. Training Loss is NaN

**Error**: Loss becomes NaN during training

**Solutions**:
- Reduce learning rate: `lr: 0.1` → `lr: 0.01`
- Enable gradient clipping: `gradient_clip: 1.0`
- Check data normalization
- Reduce label smoothing: `label_smoothing: 0.1` → `label_smoothing: 0.0`

#### 3. Model is Overfitting

**Symptoms**: Training accuracy >> Validation accuracy

**Solutions**:
- Increase weight decay: `weight_decay: 5e-4` → `weight_decay: 1e-3`
- Enable augmentation: `use_randaugment: true`
- Add label smoothing: `label_smoothing: 0.1`
- Use dropout: `drop_rate: 0.2`
- Train for fewer epochs with early stopping

#### 4. Model is Underfitting

**Symptoms**: Both training and validation accuracy are low

**Solutions**:
- Increase model capacity: `resnet18` → `resnet50`
- Increase training duration: `epochs: 100` → `epochs: 200`
- Increase learning rate: `lr: 0.001` → `lr: 0.01`
- Reduce weight decay: `weight_decay: 1e-3` → `weight_decay: 5e-4`
- Reduce augmentation strength

#### 5. Training is Too Slow

**Solutions**:
- Enable mixed precision: `use_amp: true` (2x speedup)
- Increase batch size if memory allows
- Increase number of workers: `num_workers: 4` → `num_workers: 8`
- Use smaller model for prototyping
- Disable TensorBoard during training
- Use `pin_memory: true` and `persistent_workers: true`

#### 6. Validation Accuracy Oscillates

**Solutions**:
- Use EMA: `use_ema: true`
- Reduce learning rate
- Use cosine annealing: `scheduler: cosine`
- Increase batch size for more stable gradients

#### 7. Configuration Loading Errors

**Error**: `ConfigurationError: Missing required field`

**Solutions**:
```python
from utils.config import load_config, validate_training_config

try:
    config = load_config('my_config.yaml')
    validate_training_config(config)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Check error message for missing/invalid fields
```

#### 8. Checkpoint Loading Errors

**Error**: `CheckpointNotFoundError` or `InvalidCheckpointPathError`

**Solutions**:
```python
from utils.exceptions import CheckpointNotFoundError, InvalidCheckpointPathError

try:
    trainer.load_checkpoint('best_model.pth')
except CheckpointNotFoundError:
    print("Checkpoint not found. Train from scratch.")
except InvalidCheckpointPathError as e:
    print(f"Invalid path: {e}")
```

---

## Best Practices

### 1. Start Simple, Then Scale

```bash
# Step 1: Verify code works (2 epochs)
python3 train_with_config.py --config configs/quick_test.yaml

# Step 2: Establish baseline (100 epochs)
python3 train_with_config.py --config configs/resnet18_basic.yaml

# Step 3: Apply advanced techniques (300 epochs)
python3 train_with_config.py --config configs/efficientnet_modern.yaml
```

### 2. Monitor Everything

```yaml
logging:
  tensorboard: true
  log_lr: true
  log_grad_norm: true
  wandb: true  # Optional: Weights & Biases
```

### 3. Use Version Control for Configs

```bash
git add configs/experiment_v1.yaml
git commit -m "Add experiment v1 config"
```

### 4. Document Your Experiments

```python
# Add experiment description to config
description: "Testing impact of MixUp on ResNet-18"
experiment_id: "exp_001"
date: "2024-01-15"
```

### 5. Validate Before Long Training

```python
from utils.config import validate_training_config

config = load_config('my_config.yaml')
validate_training_config(config)  # Catch errors early!
```

### 6. Use Early Stopping

```yaml
early_stopping:
  enabled: true
  patience: 30
  metric: val_acc
  mode: max
```

### 7. Save Experiment Results

```bash
# Create results directory
mkdir -p results/experiment_001

# Copy config
cp configs/my_config.yaml results/experiment_001/

# Save logs
cp -r logs/my_model results/experiment_001/

# Save checkpoints
cp checkpoints/my_model/best_model.pth results/experiment_001/
```

---

## Additional Resources

### Documentation
- `CLAUDE.md`: Project overview and architecture
- `configs/README.md`: Configuration file guide
- `IMPROVEMENTS_APPLIED.md`: Recent improvements documentation

### Example Scripts
- `main.py`: Basic training script
- `main_modern.py`: Modern training with advanced techniques
- `train_with_config.py`: Training with YAML configuration
- `benchmark_all.py`: Compare all models

### Research Papers
- ResNet: [He et al., 2015](https://arxiv.org/abs/1512.03385)
- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Vision Transformer: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)
- ConvNeXt: [Liu et al., 2022](https://arxiv.org/abs/2201.03545)

### Community
- Report issues on GitHub
- Contribute improvements via pull requests
- Share your experiment results

---

## Changelog

### Version 2.0 (Latest)
- ✅ Added YAML configuration system
- ✅ Implemented custom exception classes
- ✅ Added comprehensive logging framework
- ✅ Created example configurations
- ✅ Added security improvements (path validation)
- ✅ Comprehensive unit tests (37+ tests)

### Version 1.0
- Initial release with ResNet, EfficientNet, ViT
- Modern training techniques (AMP, EMA, augmentation)
- Visualization and monitoring tools
