# Configuration Files

This directory contains example YAML configuration files for training different models with various techniques.

## Available Configurations

### 1. `quick_test.yaml` - Quick Testing (2 epochs, ~3-5 minutes)
**Purpose**: Rapid testing and debugging
- **Model**: ResNet-18 with ReLU
- **Training**: 2 epochs, batch size 64, Adam optimizer
- **Augmentation**: None (for speed)
- **Use Case**: Verify code works, test new features, quick experiments

**Run**:
```bash
python3 train_with_config.py --config configs/quick_test.yaml
```

### 2. `resnet18_basic.yaml` - Baseline Training (~30 minutes)
**Purpose**: Simple, effective baseline training
- **Model**: ResNet-18 with Swish activation
- **Training**: 100 epochs, SGD with momentum, cosine annealing
- **Augmentation**: RandAugment only
- **Use Case**: Baseline comparisons, understand basics
- **Expected Accuracy**: ~95.5% on CIFAR-10

**Run**:
```bash
python3 train_with_config.py --config configs/resnet18_basic.yaml
```

### 3. `efficientnet_modern.yaml` - State-of-the-Art (~2 hours)
**Purpose**: Maximum performance with all modern techniques
- **Model**: EfficientNet-B0 with Swish activation
- **Training**: 300 epochs, AdamW, cosine warmup, AMP, EMA, label smoothing
- **Augmentation**: RandAugment + MixUp + CutMix + Cutout
- **Regularization**: DropBlock + Stochastic Depth
- **Use Case**: Production models, competitions, research
- **Expected Accuracy**: ~96%+ on CIFAR-10

**Run**:
```bash
python3 train_with_config.py --config configs/efficientnet_modern.yaml
```

### 4. `vit_transformer.yaml` - Vision Transformer (~2-3 hours)
**Purpose**: Transformer-based image classification
- **Model**: Vision Transformer (ViT) with 7 layers, 6 heads
- **Training**: 300 epochs, AdamW, long warmup, AMP, EMA
- **Augmentation**: RandAugment + MixUp (high magnitude)
- **Regularization**: Stochastic Depth for deep networks
- **Use Case**: Transformer research, attention visualization
- **Expected Accuracy**: ~96.5%+ on CIFAR-10

**Run**:
```bash
python3 train_with_config.py --config configs/vit_transformer.yaml
```

## Configuration Structure

All configuration files follow this structure:

```yaml
model:                    # Model architecture settings
  architecture: str       # Model name: resnet18, efficientnet_b0, vit, etc.
  activation: str         # Activation function
  num_classes: int        # Number of output classes
  pretrained: bool        # Use pretrained weights
  drop_rate: float        # Dropout rate (optional)
  drop_path_rate: float   # Drop path rate (optional)

training:                 # Training hyperparameters
  epochs: int             # Number of training epochs
  batch_size: int         # Batch size
  learning_rate: float    # Initial learning rate
  optimizer: str          # sgd, adam, adamw
  weight_decay: float     # L2 regularization

  # Modern techniques
  use_amp: bool           # Mixed precision training
  use_ema: bool           # Exponential moving average
  label_smoothing: float  # Label smoothing value
  gradient_clip: float    # Gradient clipping threshold

scheduler:                # Learning rate schedule
  type: str               # step, cosine, cosine_warmup, plateau
  warmup_epochs: int      # Warmup duration (for cosine_warmup)
  min_lr: float           # Minimum learning rate

augmentation:             # Data augmentation techniques
  use_randaugment: bool   # RandAugment
  use_mixup: bool         # MixUp augmentation
  use_cutmix: bool        # CutMix augmentation
  use_cutout: bool        # Cutout / Random Erasing

regularization:           # Regularization techniques (optional)
  use_dropblock: bool     # DropBlock for CNNs
  use_stochastic_depth: bool  # Stochastic Depth

data:                     # Data loading settings
  dataset: str            # Dataset name
  data_dir: str           # Data directory path
  num_workers: int        # Number of data loading workers
  pin_memory: bool        # Pin memory for faster GPU transfer

checkpoint:               # Checkpoint settings
  save_dir: str           # Checkpoint save directory
  save_best: bool         # Save best model
  save_frequency: int     # Save every N epochs

logging:                  # Logging configuration
  log_dir: str            # Log directory
  tensorboard: bool       # Enable TensorBoard
  console_log_level: str  # Console log level: DEBUG, INFO, WARNING, ERROR
  file_log_level: str     # File log level

early_stopping:           # Early stopping (optional)
  enabled: bool           # Enable early stopping
  patience: int           # Number of epochs to wait
  metric: str             # Metric to monitor: val_loss, val_acc
  mode: str               # min or max
```

## Creating Custom Configurations

### Step 1: Copy a Template
```bash
cp configs/resnet18_basic.yaml configs/my_experiment.yaml
```

### Step 2: Modify Settings
Edit `my_experiment.yaml` to customize:
- Model architecture and hyperparameters
- Training duration and optimizer
- Augmentation strategies
- Logging and checkpointing

### Step 3: Validate Configuration
```python
from utils.config import load_config, validate_training_config

config = load_config('configs/my_experiment.yaml')
validate_training_config(config)  # Raises errors if invalid
```

### Step 4: Run Training
```bash
python3 train_with_config.py --config configs/my_experiment.yaml
```

## Configuration Best Practices

### For Quick Experiments
- Use fewer epochs (2-10)
- Smaller batch size (32-64)
- Disable expensive features (AMP, augmentation)
- Example: `quick_test.yaml`

### For Baseline Models
- Standard epochs (100-150)
- Medium batch size (128)
- Basic augmentation (RandAugment)
- Classic optimizers (SGD with momentum)
- Example: `resnet18_basic.yaml`

### For Maximum Performance
- Long training (200-300 epochs)
- Modern optimizer (AdamW)
- All augmentation techniques
- Mixed precision training
- EMA for stable inference
- Example: `efficientnet_modern.yaml`

### For Transformers
- Very long training (300+ epochs)
- Longer warmup (20+ epochs)
- Smaller batch sizes (32-64)
- Strong augmentation (high magnitude)
- Stochastic depth regularization
- Example: `vit_transformer.yaml`

## Hyperparameter Guidelines

### Learning Rate
- **SGD**: 0.1 (with warmup) or 0.01 (without warmup)
- **Adam/AdamW**: 0.001 to 0.0001
- **Rule of thumb**: LR scales linearly with batch size

### Weight Decay
- **SGD**: 5e-4 (CIFAR-10 standard)
- **AdamW**: 1e-4 to 1e-5
- **Too high**: Underfitting, slow convergence
- **Too low**: Overfitting, poor generalization

### Batch Size
- **Small (32-64)**: Better generalization, slower training
- **Medium (128)**: Good balance
- **Large (256+)**: Faster training, may need LR adjustment

### Augmentation Strength
- **Light**: Basic flips and crops
- **Medium**: + RandAugment (N=2, M=9)
- **Strong**: + MixUp/CutMix (transformers)

## Troubleshooting

### Training is too slow
- Enable `use_amp: true` for 2x speedup
- Increase `batch_size` if GPU memory allows
- Increase `num_workers` for faster data loading
- Disable TensorBoard during training

### Model is overfitting
- Increase `weight_decay` (5e-4 → 1e-3)
- Enable `label_smoothing` (0.1)
- Add stronger augmentation
- Enable `use_ema: true`
- Reduce model size

### Model is underfitting
- Decrease `weight_decay`
- Increase `learning_rate`
- Train for more `epochs`
- Increase model size
- Reduce augmentation strength

### Training is unstable (NaN loss)
- Reduce `learning_rate` (divide by 10)
- Enable `gradient_clip: 1.0`
- Check data normalization
- Reduce `label_smoothing`

### Out of memory
- Reduce `batch_size` (128 → 64 → 32)
- Disable `use_amp` (or enable it if disabled)
- Reduce `num_workers`
- Use smaller model architecture

## References

- **RandAugment**: [Cubuk et al., 2020](https://arxiv.org/abs/1909.13719)
- **MixUp**: [Zhang et al., 2018](https://arxiv.org/abs/1710.09412)
- **CutMix**: [Yun et al., 2019](https://arxiv.org/abs/1905.04899)
- **EMA**: [Polyak & Juditsky, 1992](https://epubs.siam.org/doi/10.1137/0330046)
- **Label Smoothing**: [Szegedy et al., 2016](https://arxiv.org/abs/1512.00567)
- **Mixed Precision**: [Micikevicius et al., 2018](https://arxiv.org/abs/1710.03740)
