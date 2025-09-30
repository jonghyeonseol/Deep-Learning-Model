# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep learning framework for testing and comparing different activation functions on image classification tasks (CIFAR-10). The project implements custom activation functions from scratch and provides comprehensive training, visualization, and monitoring tools.

## Common Commands

### Environment Setup
```bash
# Activate virtual environment (required before running any Python commands)
source venv/bin/activate
```

### Training and Testing
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
- `network.py`: Contains `NeuralNetwork` (fully-connected) and `ConvNeuralNetwork` (CNN for CIFAR-10)
- `activations.py`: Custom implementations of 14+ activation functions (GELU, ReLU, Tanh, Sigmoid, Swish, Mish, etc.)
- All activation functions are custom PyTorch modules, not using `torch.nn` built-ins

**utils/**: Training utilities and visualization tools
- `trainer.py`: `Trainer` class handles training loop, validation, checkpointing, early stopping
- `data_loader.py`: `CIFAR10DataLoader` wraps PyTorch's CIFAR-10 dataset with train/val/test splits
- `visualization.py`: `Visualizer` creates training plots, confusion matrices, prediction samples
- `monitor.py`: Real-time monitoring tools (`PerceptronVisualizer`, `LayerMonitor`, `ActivationAnalyzer`)
- `realtime_monitor.py`: Live training monitors with dynamic plotting capabilities

**Entry Points**:
- `main.py`: Unified script for training, visualization, and monitoring with CLI arguments

### Data Flow

1. **Data Loading**: `CIFAR10DataLoader` downloads CIFAR-10 (if needed), applies normalization, creates train/val/test splits
2. **Model Creation**: `ConvNeuralNetwork` builds CNN with specified activation function via `get_activation(name)`
3. **Training**: `Trainer` manages training loop, optimizer (Adam), scheduler (StepLR), loss computation, validation
4. **Checkpointing**: Best models saved to `checkpoints/{activation_name}/best_model.pth`
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

- `checkpoints/{activation}/`: Training checkpoints and visualizations for each activation function
  - `best_model.pth`: Saved model weights
  - `training_history.png`: Loss and accuracy curves
  - `predictions.png`: Sample predictions with ground truth
  - `confusion_matrix.png`: Classification confusion matrix
- `data/`: CIFAR-10 dataset (auto-downloaded)
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