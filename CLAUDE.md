# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a modern deep learning framework for image classification:

**Image Classification** (CIFAR-10): State-of-the-art deep learning framework for image classification on CIFAR-10 dataset. Features custom activation functions, modern architectures (ResNet, EfficientNet, Vision Transformers, ConvNeXt), advanced training techniques (AdamW, Mixed Precision, Label Smoothing), comprehensive augmentation (RandAugment, MixUp, CutMix), and real-time visualization tools.

## Common Commands

### Environment Setup
```bash
# Activate virtual environment (required before running any Python commands)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Training
```bash
# List all available activation functions
python3 main.py --list-activations

# Quick training test with 2 epochs (3-5 minutes)
python3 main.py --activation relu --quick

# Train with specific activation function
python3 main.py --activation [name] --epochs [N] --batch-size [N] --lr [float]

# Compare all modern activation functions (GELU, Swish, Mish, SiLU, Hardswish)
python3 main.py --activation modern --epochs 5

# Compare all classic activation functions (ReLU, Tanh, Sigmoid, LeakyReLU, ELU)
python3 main.py --activation classic --epochs 5

# Compare all available activation functions (2-3 hours)
python3 main.py --activation all --epochs 3
```

### Modern Training (State-of-the-Art Techniques)
```bash
# Train ResNet-18 with modern techniques
python3 main_modern.py --model resnet18 --epochs 100

# Train EfficientNet-B0
python3 main_modern.py --model efficientnet_b0 --epochs 100

# Train Vision Transformer
python3 main_modern.py --model vit --epochs 100

# Train ConvNeXt (modernized CNN)
python3 main_modern.py --model convnext_tiny --epochs 100

# Enable advanced augmentation
python3 main_modern.py --model resnet18 --use-mixup --use-cutmix --use-randaugment

# Train with mixed precision (2x speedup)
python3 main_modern.py --model resnet18 --amp

# Full modern training pipeline
python3 main_modern.py --model resnet18 --epochs 200 --amp --use-mixup --use-cutmix --use-randaugment
```

### Live Visualization
```bash
# Visualize network structure in real-time (neurons and connections)
python3 main.py --visualize

# Train with live monitoring (real-time loss/accuracy plots)
python3 main.py --activation swish --epochs 5 --monitor

# Combine visualization and monitoring
python3 main.py --activation relu --monitor --quick
```


## Code Architecture

### Module Organization

**models/**: Neural network architectures and activation functions
- `network.py`: `NeuralNetwork` (fully-connected) and `ConvNeuralNetwork` (CNN for CIFAR-10)
- `activations.py`: Custom implementations of 14+ activation functions (GELU, ReLU, Tanh, Sigmoid, Swish, Mish, etc.)
- `resnet.py`: ResNet-18/34/50/101 with residual blocks and batch normalization
- `efficientnet.py`: EfficientNet-B0/B1 with MBConv blocks and SE attention
- `cnn_transformer.py`: Hybrid CNN-Transformer and Vision Transformer (ViT)
- `convnext.py`: ConvNeXt architecture (modernized CNN matching transformer performance)

**utils/**: Training utilities and visualization tools
- `trainer.py`: Basic `Trainer` class for training loop, validation, checkpointing
- `modern_trainer.py`: Modern training with AdamW, cosine annealing, AMP, EMA, label smoothing
- `data_loader.py`: `CIFAR10DataLoader` with train/val/test splits
- `visualization.py`: Training plots, confusion matrices, prediction samples
- `augmentation.py`: RandAugment, MixUp, CutMix, Cutout, AutoAugment
- `regularization.py`: DropBlock, Stochastic Depth, Drop Path
- `monitor.py`: Real-time monitoring (`LayerMonitor`, `ActivationAnalyzer`)
- `realtime_monitor.py`: Live training monitors with dynamic plotting
- `metrics.py`: Comprehensive evaluation metrics

**Entry Points**:
- `main.py`: Basic training script for activation function comparison
- `main_modern.py`: Modern training pipeline with state-of-the-art techniques
- `benchmark_all.py`: Performance comparison across all architectures

### Data Flow

1. **Data Loading**: `CIFAR10DataLoader` downloads CIFAR-10 (if needed), applies normalization, creates train/val/test splits
2. **Model Creation**: `ConvNeuralNetwork` builds CNN with specified activation function via `get_activation(name)`
3. **Training**: `Trainer` manages training loop, optimizer (Adam), scheduler (StepLR), loss computation, validation
4. **Checkpointing**: Best models saved to `checkpoints/{model_name}/best_model.pth`
5. **Visualization**: Training history, confusion matrices, and sample predictions saved as PNG files

### Key Design Patterns

- **Activation Function Factory**: `get_activation(name)` returns activation module by string name
- **Modular Architecture**: Models, training, data loading, and visualization are fully decoupled
- **Checkpoint Organization**: Each activation function gets its own subdirectory in `checkpoints/`
- **Training History**: Stored in `Trainer.history` as dict with keys: `train_loss`, `train_acc`, `val_loss`, `val_acc`

### Model Architecture Details

**ConvNeuralNetwork** (for CIFAR-10):
- 3 Conv layers: Conv2d(3→32) → Conv2d(32→64) → Conv2d(64→128)
- MaxPool2d(2,2) after each conv layer
- 3 FC layers: Linear(2048→512) → Linear(512→256) → Linear(256→10)
- Activation function applied after each layer (except final output)
- Dropout (default 0.2) applied in FC layers

**Parameter Initialization**:
- Convolutional layers: Kaiming normal initialization
- Fully connected layers: Xavier uniform initialization

## Dataset Information

- **CIFAR-10**: 60,000 32×32 RGB images in 10 classes
- **Training Set**: 45,000 images (after 10% validation split)
- **Validation Set**: 5,000 images
- **Test Set**: 10,000 images
- **Normalization**: Mean=[0.4914, 0.4822, 0.4465], Std=[0.2023, 0.1994, 0.2010]
- **Data Location**: Downloaded to `./data/cifar-10-batches-py/` on first run

## Output Directories

- `checkpoints/{model_name}/`: Training checkpoints and visualizations for each model
  - `best_model.pth`: Saved model weights
  - `training_history.png`: Loss and accuracy curves
  - `predictions.png`: Sample predictions with ground truth
  - `confusion_matrix.png`: Classification confusion matrix
- `data/cifar-10-batches-py/`: CIFAR-10 dataset (auto-downloaded)
- `visualizations/`: Demo visualization outputs

## Available Activation Functions

**Modern** (2017-2020): gelu, swish, mish, silu, hardswish
**Classic**: relu, tanh, sigmoid, leakyrelu, elu, prelu, selu
**Other**: step, softmax

## Important Implementation Notes

- Always activate the virtual environment before running scripts
- First run downloads CIFAR-10 (~170MB) which may take 5-10 minutes
- GPU training is automatic if CUDA is available (check with `torch.cuda.is_available()`)
- Use `--quick` flag for rapid testing (2 epochs instead of default 10)
- Use `--visualize` flag to see live network structure (neurons and connections)
- Use `--monitor` flag to enable real-time training plots (loss, accuracy, gradients)
- Validation accuracy is more important than training accuracy (indicates generalization)
- The `Trainer` class includes early stopping (patience=5 epochs by default)
- All custom activation functions in `activations.py` are implemented from scratch for educational purposes


## Modern Deep Learning Features (2024-2025)

### State-of-the-Art Architectures

1. **ResNet (Residual Networks)**
   - ResNet-18/34/50/101/152 variants
   - Skip connections (residual blocks) solve vanishing gradient problem
   - Batch normalization after each convolution
   - Global average pooling instead of fully-connected layers
   - BasicBlock (3x3 conv) for shallow, BottleneckBlock (1x1, 3x3, 1x1) for deep networks

2. **EfficientNet**
   - EfficientNet-B0 through B7 with compound scaling
   - MBConv (Mobile Inverted Bottleneck) blocks with depthwise separable convolutions
   - Squeeze-and-Excitation (SE) attention for channel-wise feature recalibration
   - Swish activation function (SiLU)
   - Stochastic depth for regularization

3. **Vision Transformer (ViT)**
   - Patch-based image tokenization (16x16 or 32x32 patches)
   - Multi-head self-attention mechanism
   - Positional embeddings for spatial information
   - Transformer encoder blocks with layer normalization
   - Classification token ([CLS]) for global representation

4. **ConvNeXt (Coming Soon)**
   - Modernized pure CNN architecture (2022)
   - Competes with transformers while maintaining CNN efficiency
   - Depthwise convolutions, LayerNorm, GELU activation
   - Larger kernels (7x7) and inverted bottleneck design
   - Layer scale and stochastic depth

### Modern Training Techniques

**Optimizers:**
- **AdamW**: Adam with decoupled weight decay (fixes L2 regularization in Adam)
- **SGD with Momentum**: Classic but effective with proper tuning

**Learning Rate Schedulers:**
- **Cosine Annealing with Warmup**: Smooth LR decay from max to min with linear warmup phase
- **Warm Restarts**: Periodic LR resets to escape local minima
- **Step Decay**: Reduce LR by factor at specified epochs

**Training Enhancements:**
- **Mixed Precision Training (AMP)**: FP16 computation with FP32 master weights for 2x speedup
- **Gradient Clipping**: Prevent exploding gradients by clipping to max norm
- **Label Smoothing**: Soften one-hot labels (0.1 smoothing) to improve generalization
- **Exponential Moving Average (EMA)**: Maintain shadow model weights for more stable inference
- **Early Stopping**: Monitor validation loss and stop when no improvement

### Advanced Data Augmentation

**Standard Augmentation:**
- Random horizontal flip, crop, rotation
- Color jitter (brightness, contrast, saturation, hue)
- Normalization with dataset statistics

**Modern Augmentation (2019-2024):**

1. **RandAugment** (2020)
   - Simplified search space with only 2 hyperparameters (N, M)
   - Randomly applies N transformations with magnitude M
   - Transformations: ShearX/Y, TranslateX/Y, Rotate, Solarize, Posterize, etc.

2. **MixUp** (2018)
   - Linear interpolation between pairs of training samples
   - Mix images: x_mixed = λ*x1 + (1-λ)*x2
   - Mix labels: y_mixed = λ*y1 + (1-λ)*y2
   - λ ~ Beta(α, α) where α=0.2-1.0

3. **CutMix** (2019)
   - Cut and paste image patches between training samples
   - Mix labels proportional to patch area
   - Better than MixUp for localization tasks

4. **Cutout / Random Erasing** (2017)
   - Randomly mask square regions of input
   - Forces model to use full context, not just discriminative parts
   - Improves robustness to occlusion

5. **TrivialAugment** (2021, Coming Soon)
   - Simpler than RandAugment with one operation per image
   - Sample magnitude uniformly per operation
   - Achieves similar performance with less hyperparameter tuning

6. **AugMax** (2021, Coming Soon)
   - Adversarial augmentation that maximizes training loss
   - Finds hardest augmentation for current model state

### Regularization Techniques

1. **DropBlock** (2018)
   - Structured dropout for convolutional layers
   - Drops contiguous regions instead of random pixels
   - More effective than standard dropout for CNNs
   - Applied with increasing probability during training

2. **Stochastic Depth** (2016)
   - Randomly drop entire residual blocks during training
   - Reduces effective network depth, improves gradient flow
   - Enables training of very deep networks (1000+ layers)
   - Linear decay schedule from 0.0 (early layers) to 0.5 (deep layers)

3. **Drop Path** (2018)
   - Similar to stochastic depth, drops paths in residual connections
   - Applied in EfficientNet and Vision Transformers

4. **Weight Decay**
   - L2 regularization on model weights
   - Typical values: 1e-4 to 5e-4
   - Decoupled in AdamW for better performance

### Attention Mechanisms

1. **Squeeze-and-Excitation (SE) Blocks** (2018)
   - Channel-wise attention via global pooling and FC layers
   - Learns to emphasize informative channels
   - Used in EfficientNet, ResNet-SE

2. **Multi-Head Self-Attention** (2017)
   - Core of Transformer models
   - Captures long-range dependencies
   - Parallel attention heads learn different relationships

3. **Coordinate Attention** (Coming Soon)
   - Encodes positional information into channel attention
   - Better for mobile applications

### Loss Functions

- **Cross-Entropy Loss**: Standard classification loss
- **Label Smoothing Cross-Entropy**: Softens target distribution
- **Focal Loss** (Coming Soon): Handles class imbalance by down-weighting easy examples
- **Contrastive Loss** (Coming Soon): For self-supervised learning

### Model Compression & Efficiency (Coming Soon)

- **Knowledge Distillation**: Train student network from teacher predictions
- **Quantization**: Convert FP32 → INT8 for 4x speedup
- **Pruning**: Remove redundant weights/channels
- **Neural Architecture Search (NAS)**: Automated architecture design

### Performance Monitoring

- **Real-time Training Visualization**: Live plots of loss, accuracy, learning rate
- **Layer Activation Monitoring**: Visualize intermediate feature maps
- **Gradient Flow Analysis**: Detect vanishing/exploding gradients
- **Confusion Matrix**: Per-class performance breakdown
- **Top-k Accuracy**: Measure if true class is in top-k predictions

### Best Practices (2024-2025)

1. **Start Simple**: Begin with ResNet-18 or EfficientNet-B0
2. **Use Modern Training**: AdamW + Cosine Annealing + AMP
3. **Apply Strong Augmentation**: RandAugment + MixUp or CutMix
4. **Monitor Overfitting**: Use validation set, early stopping
5. **Longer Training**: Modern models benefit from 200-300 epochs
6. **Batch Size**: Larger batches (128-256) with linear LR scaling
7. **Test Time Augmentation**: Average predictions over multiple augmented versions
8. **Model Ensemble**: Combine multiple models for best accuracy

### Recommended Hyperparameters (CIFAR-10)

**ResNet-18/34:**
```bash
python3 main_modern.py \
  --model resnet18 \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --scheduler cosine \
  --weight-decay 5e-4 \
  --use-mixup \
  --use-randaugment \
  --amp
```

**EfficientNet-B0:**
```bash
python3 main_modern.py \
  --model efficientnet_b0 \
  --epochs 300 \
  --batch-size 128 \
  --lr 0.001 \
  --optimizer adamw \
  --scheduler cosine \
  --weight-decay 1e-4 \
  --use-cutmix \
  --use-randaugment \
  --label-smoothing 0.1 \
  --amp
```

**Vision Transformer:**
```bash
python3 main_modern.py \
  --model vit \
  --epochs 300 \
  --batch-size 64 \
  --lr 0.001 \
  --optimizer adamw \
  --scheduler cosine_warmup \
  --warmup-epochs 20 \
  --weight-decay 1e-4 \
  --use-mixup \
  --use-randaugment \
  --label-smoothing 0.1 \
  --amp
```

### Expected Performance (CIFAR-10 Test Accuracy)

| Model | Parameters | Top-1 Acc | Training Time (RTX 3090) |
|-------|-----------|-----------|--------------------------|
| ResNet-18 | 11.2M | ~95.5% | ~30 min (200 epochs) |
| ResNet-50 | 23.5M | ~96.0% | ~1 hour (200 epochs) |
| EfficientNet-B0 | 4.0M | ~95.8% | ~45 min (300 epochs) |
| Vision Transformer | 5.7M | ~96.5% | ~2 hours (300 epochs) |
| ConvNeXt-Tiny | 27.8M | ~97.0%+ | ~1.5 hours (300 epochs) |

*Performance depends on hyperparameters, augmentation, and hardware*

### Recent Research Trends (2024-2025)

**Latest Architectures (2024-2025):**
1. **ConvNeXt V2** (2023-2024): Co-designing with Masked Autoencoders
   - Fully Convolutional Masked Autoencoder (FCMAE) framework
   - Global Response Normalization (GRN) layer for inter-channel competition
   - Sparse convolution-based encoder for self-supervised pre-training
2. **LaViT** (CVPR 2024): Efficient Vision Transformers
   - Calculates self-attention only in initial layers
   - Reuses attention scores through lightweight linear operations
   - Significantly reduces computational costs
3. **DC-AE** (ICLR 2025): Deep Compression Autoencoder
   - Lightweight ViTs for high-resolution diffusion models
   - Spatial compression ratios up to 128x
   - Dramatically reduces token count for processing

**Advanced Augmentation (2024-2025):**
1. **TrivialAugment** (2021, validated 2024-2025): Simplest automated augmentation
   - Randomly selects one augmentation per image
   - Samples magnitude uniformly per operation
   - Outperforms RandAugment in recent medical imaging studies (2024)
   - Tuning-free approach (no hyperparameter search needed)
2. **Generative AI Augmentation** (2024): Using diffusion models
   - Generate synthetic training data with controlled variations
   - Improves dataset diversity for rare classes
   - Particularly effective for small datasets

**Modern Optimizers (2024-2025):**
1. **Lion Optimizer** (Google Brain, 2023, validated 2024-2025)
   - Discovered via genetic algorithms
   - 3-10x smaller learning rate than AdamW
   - 50% less memory (single momentum buffer)
   - Fastest initial convergence but may lag in final performance
2. **Sophia Optimizer** (2023, validated 2024-2025)
   - Scalable stochastic second-order optimizer
   - 2x speedup over Adam for language models
   - Better sample efficiency (50% fewer steps for same loss)
   - Particularly effective for large-scale pre-training

**Knowledge Distillation (2024-2025):**
1. **Student-Centered KD**: Learning from human educational wisdom
2. **Cluster-Quantized KD** (CQKD): Unified compression framework
3. **ViT-to-CNN Distillation**: Transfer transformer knowledge to efficient CNNs
4. **Privacy-Preserving KD**: Distillation under limited data scenarios

**Test-Time Techniques (2024-2025):**
1. **Test-Time Augmentation (TTA)**: Proven to reduce expected error
   - Ensemble predictions across multiple augmented versions
   - Soft voting (averaging class probabilities)
   - 0.2-0.5% accuracy improvement with minimal cost
2. **Diffusion-Enhanced TTA** (2025): Multi-modal test-time adaptation
   - Uses pre-trained vision and language models
   - Adapts to unknown domains at inference time

**Other Emerging Trends:**
1. **Vision-Language Models**: CLIP-style contrastive learning
2. **Self-Supervised Learning**: SimCLR, MoCo, DINO, MAE
3. **Efficient Transformers**: Swin, CrossViT, Twins, LaViT
4. **Hybrid Architectures**: ConvNeXt V2, CoAtNet (Conv + Attention)
5. **Neural Architecture Search**: EfficientNetV2, RegNet
6. **Foundation Models**: Large pre-trained models for transfer learning

### Educational Roadmap: Potential Future Implementations

The following techniques are not yet implemented but would be valuable additions for learners:

**Ready for Implementation (Beginner-Friendly):**
1. **TrivialAugment**: Simpler than RandAugment, easier to understand
   - Single augmentation per image with uniform magnitude sampling
   - No hyperparameter tuning required
   - Great for teaching automated augmentation concepts
2. **Test-Time Augmentation (TTA)**: Simple ensemble technique
   - Augment test images multiple times and average predictions
   - Easy to implement (10-20 lines of code)
   - Demonstrates ensemble learning without training multiple models
3. **Lion Optimizer**: Alternative to AdamW with interesting properties
   - Simpler update rule than Adam/AdamW
   - Good for teaching optimizer mechanics
   - Demonstrates memory-efficient optimization

**Intermediate Implementations:**
1. **ConvNeXt V2**: Evolution of ConvNeXt (already implemented)
   - Add Global Response Normalization (GRN) layer
   - Demonstrates latest CNN improvements (2024)
   - Shows progression from V1 to V2
2. **Knowledge Distillation**: Teacher-student learning framework
   - Train large model (teacher), transfer to small model (student)
   - Teaches model compression and knowledge transfer
   - Practical for deployment scenarios
3. **Sophia Optimizer**: Second-order optimizer (advanced)
   - Demonstrates Hessian diagonal approximation
   - More complex but more efficient than first-order methods
   - Good for understanding second-order optimization

**Advanced Implementations (Research-Level):**
1. **Masked Autoencoders (MAE)**: Self-supervised pre-training
   - Mask random patches and reconstruct
   - Demonstrates self-supervised learning
   - Can improve performance with limited labeled data
2. **Diffusion-Enhanced Augmentation**: Generative AI for data augmentation
   - Use stable diffusion for synthetic data generation
   - Cutting-edge technique (2024-2025)
   - Bridges generative AI and discriminative models
3. **Efficient ViT variants** (LaViT-style): Attention optimization
   - Reduce attention computation via reuse
   - Demonstrates efficiency techniques for transformers
   - Balances accuracy and computational cost

**Why These Additions Matter for Education:**
- **Progressive Learning**: From simple (TTA) to advanced (MAE)
- **Current Relevance**: All techniques validated in 2024-2025 research
- **Practical Value**: Used in production systems and competitions
- **Conceptual Coverage**: Spans optimization, augmentation, compression, and self-supervision

### Useful Resources

- **Papers with Code**: https://paperswithcode.com/sota/image-classification-on-cifar-10
- **PyTorch Image Models (timm)**: https://github.com/rwightman/pytorch-image-models
- **Awesome Deep Learning**: https://github.com/ChristosChristofidis/awesome-deep-learning
- **Deep Learning Book**: http://www.deeplearningbook.org/
- **Lion Optimizer**: https://github.com/lucidrains/lion-pytorch
- **TrivialAugment Paper**: https://arxiv.org/abs/2103.10158
- **Sophia Optimizer Paper**: https://arxiv.org/abs/2305.14342
- **ConvNeXt V2 Paper**: https://arxiv.org/abs/2301.00808
