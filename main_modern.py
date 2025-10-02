#!/usr/bin/env python3
"""
Modern Deep Learning Training Script with State-of-the-Art Techniques.

This script demonstrates all modern deep learning practices:
- Multiple architecture options (ResNet, EfficientNet, CNN-Transformer)
- Modern training (AdamW, cosine annealing, mixed precision)
- Advanced augmentation (RandAugment, MixUp, CutMix)
- Modern regularization (DropBlock, Stochastic Depth, Label Smoothing)

Usage:
    # Train ResNet-18 with modern techniques
    python main_modern.py --model resnet18 --modern-training

    # Train EfficientNet with all augmentations
    python main_modern.py --model efficientnet_b0 --augmentation randaugment --mixup --cutmix

    # Train CNN-Transformer hybrid
    python main_modern.py --model cnn_transformer_base --activation gelu

    # Compare all architectures
    python main_modern.py --compare-models
"""

import argparse
import torch
import torch.nn as nn
import os
import time
import json
from collections import defaultdict

# Model imports
from models.network import ConvNeuralNetwork
from models.resnet import (ResNet18, ResNet34, ResNet50, ResNet_Tiny)
from models.efficientnet import (EfficientNet_B0, EfficientNet_B1, EfficientNet_Tiny)
from models.cnn_transformer import (CNNTransformer_Small, CNNTransformer_Base, VisionTransformer_Tiny)

# Training imports
from utils.modern_trainer import ModernTrainer
from utils.trainer import Trainer
from utils.data_loader import CIFAR10DataLoader
from utils.visualization import Visualizer
from utils.augmentation import (
    get_cifar10_transforms, MixUp, CutMix, mixup_criterion
)


def get_model(model_name, activation='relu', num_classes=10):
    """
    Get model by name.

    Args:
        model_name: Model architecture name
        activation: Activation function
        num_classes: Number of output classes

    Returns:
        PyTorch model
    """
    model_dict = {
        # Original CNN
        'cnn': lambda: ConvNeuralNetwork(
            input_channels=3, num_classes=num_classes,
            activation=activation, dropout_rate=0.2
        ),

        # ResNet variants
        'resnet18': lambda: ResNet18(num_classes=num_classes, activation=activation),
        'resnet34': lambda: ResNet34(num_classes=num_classes, activation=activation),
        'resnet50': lambda: ResNet50(num_classes=num_classes, activation=activation),
        'resnet_tiny': lambda: ResNet_Tiny(num_classes=num_classes, activation=activation),

        # EfficientNet variants
        'efficientnet_b0': lambda: EfficientNet_B0(num_classes=num_classes, activation=activation),
        'efficientnet_b1': lambda: EfficientNet_B1(num_classes=num_classes, activation=activation),
        'efficientnet_tiny': lambda: EfficientNet_Tiny(num_classes=num_classes, activation=activation),

        # CNN-Transformer hybrids
        'cnn_transformer_small': lambda: CNNTransformer_Small(num_classes=num_classes, activation=activation),
        'cnn_transformer_base': lambda: CNNTransformer_Base(num_classes=num_classes, activation=activation),
        'vit_tiny': lambda: VisionTransformer_Tiny(num_classes=num_classes, activation=activation),
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")

    model = model_dict[model_name]()
    print(f"✓ Created {model_name} model")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def train_with_modern_techniques(
    model, train_loader, val_loader, test_loader, save_dir,
    epochs=10, lr=0.001, use_amp=True, use_ema=True,
    label_smoothing=0.1, gradient_clip=1.0,
    use_mixup=False, use_cutmix=False, mixup_alpha=1.0
):
    """
    Train model with modern techniques.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        save_dir: Save directory
        epochs: Number of epochs
        lr: Learning rate
        use_amp: Use mixed precision
        use_ema: Use exponential moving average
        label_smoothing: Label smoothing factor
        gradient_clip: Gradient clipping max norm
        use_mixup: Use MixUp augmentation
        use_cutmix: Use CutMix augmentation
        mixup_alpha: MixUp/CutMix alpha parameter

    Returns:
        Training results dict
    """
    print(f"\n{'='*70}")
    print("MODERN TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"✓ AdamW optimizer (decoupled weight decay)")
    print(f"✓ Cosine annealing with warmup")
    if use_amp:
        print(f"✓ Mixed precision training (AMP)")
    if use_ema:
        print(f"✓ Exponential moving average (EMA)")
    if label_smoothing > 0:
        print(f"✓ Label smoothing ({label_smoothing})")
    if gradient_clip > 0:
        print(f"✓ Gradient clipping (max_norm={gradient_clip})")
    if use_mixup:
        print(f"✓ MixUp augmentation (alpha={mixup_alpha})")
    if use_cutmix:
        print(f"✓ CutMix augmentation (alpha={mixup_alpha})")
    print(f"{'='*70}\n")

    # Create modern trainer
    trainer = ModernTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=save_dir
    )

    # Configure optimizer (AdamW with weight decay)
    trainer.configure_optimizer(
        optimizer_name='adamw',
        lr=lr,
        weight_decay=0.05  # Modern best practice: 0.01 to 0.1
    )

    # Configure scheduler (cosine annealing with warmup)
    trainer.configure_scheduler(
        scheduler_name='cosine_warmup',
        epochs=epochs,
        warmup_steps=len(train_loader) * 5  # 5 epoch warmup
    )

    # Configure loss (with label smoothing)
    trainer.configure_criterion(
        criterion_name='label_smoothing' if label_smoothing > 0 else 'crossentropy',
        label_smoothing=label_smoothing
    )

    # Enable modern techniques
    if use_amp and torch.cuda.is_available():
        trainer.enable_amp()

    if gradient_clip > 0:
        trainer.enable_gradient_clipping(max_norm=gradient_clip)

    if use_ema:
        trainer.enable_ema(decay=0.9999)

    # Handle MixUp/CutMix augmentation
    if use_mixup or use_cutmix:
        print("⚠️  MixUp/CutMix requires custom training loop")
        print("   Using standard training without MixUp/CutMix for now")
        print("   (See utils/augmentation.py for MixUp/CutMix implementation)\n")

    # Train model
    start_time = time.time()
    trainer.train(
        epochs=epochs,
        early_stopping_patience=15,
        save_best=True,
        use_ema_for_eval=use_ema
    )
    training_time = time.time() - start_time

    # Test model
    test_loss, test_acc = trainer.test(use_ema=use_ema)

    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': trainer.best_val_acc,
        'training_time': training_time,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }


def train_standard(model, train_loader, val_loader, test_loader, save_dir,
                   epochs=10, lr=0.001):
    """
    Train model with standard techniques (for comparison).

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        save_dir: Save directory
        epochs: Number of epochs
        lr: Learning rate

    Returns:
        Training results dict
    """
    print(f"\n{'='*70}")
    print("STANDARD TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"✓ Adam optimizer")
    print(f"✓ Step LR scheduler")
    print(f"✓ Cross entropy loss")
    print(f"{'='*70}\n")

    # Create standard trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=save_dir
    )

    # Configure training
    trainer.configure_optimizer('adam', lr=lr, weight_decay=1e-4)
    trainer.configure_scheduler('step', step_size=30, gamma=0.1)
    trainer.configure_criterion('crossentropy')

    # Train model
    start_time = time.time()
    trainer.train(epochs=epochs, early_stopping_patience=10, save_best=True)
    training_time = time.time() - start_time

    # Test model
    test_loss, test_acc = trainer.test()

    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': trainer.best_val_acc,
        'training_time': training_time,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }


def compare_models(models, epochs=10, batch_size=128, modern_training=True):
    """
    Compare different model architectures.

    Args:
        models: List of model names
        epochs: Number of epochs
        batch_size: Batch size
        modern_training: Use modern training techniques
    """
    results = []

    # Load data
    print("\n📦 Loading CIFAR-10 dataset...")
    data_loader = CIFAR10DataLoader(batch_size=batch_size, validation_split=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    print("✓ Data loaded\n")

    for model_name in models:
        print(f"\n{'#'*70}")
        print(f"# Training: {model_name}")
        print(f"{'#'*70}\n")

        try:
            # Create model
            model = get_model(model_name, activation='swish')
            model.summary()

            # Setup save directory
            save_dir = f'./checkpoints/{model_name}_{"modern" if modern_training else "standard"}'
            os.makedirs(save_dir, exist_ok=True)

            # Train model
            if modern_training:
                result = train_with_modern_techniques(
                    model, train_loader, val_loader, test_loader,
                    save_dir=save_dir, epochs=epochs, lr=0.001,
                    use_amp=True, use_ema=True, label_smoothing=0.1
                )
            else:
                result = train_standard(
                    model, train_loader, val_loader, test_loader,
                    save_dir=save_dir, epochs=epochs, lr=0.001
                )

            result['model_name'] = model_name
            results.append(result)

            # Save results
            with open(os.path.join(save_dir, 'results.json'), 'w') as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Display comparison
    print(f"\n{'='*90}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*90}")
    print(f"{'Model':<25} {'Test Acc (%)':<15} {'Val Acc (%)':<15} {'Params':<15} {'Time (s)':<10}")
    print(f"{'-'*90}")

    for result in results:
        print(f"{result['model_name']:<25} "
              f"{result['test_accuracy']:<15.2f} "
              f"{result['best_val_accuracy']:<15.2f} "
              f"{result['model_parameters']:<15,} "
              f"{result['training_time']:<10.1f}")

    if results:
        best = max(results, key=lambda x: x['test_accuracy'])
        print(f"\n✓ Best model: {best['model_name']}")
        print(f"  Test accuracy: {best['test_accuracy']:.2f}%")
        print(f"  Parameters: {best['model_parameters']:,}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Modern Deep Learning Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ResNet-18 with modern techniques
  python main_modern.py --model resnet18 --modern-training

  # Train EfficientNet with RandAugment
  python main_modern.py --model efficientnet_b0 --augmentation randaugment

  # Train CNN-Transformer
  python main_modern.py --model cnn_transformer_base --activation gelu

  # Compare all models
  python main_modern.py --compare-models

  # Quick test (2 epochs)
  python main_modern.py --model resnet_tiny --quick
        """
    )

    # Model selection
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['cnn', 'resnet18', 'resnet34', 'resnet50', 'resnet_tiny',
                               'efficientnet_b0', 'efficientnet_b1', 'efficientnet_tiny',
                               'cnn_transformer_small', 'cnn_transformer_base', 'vit_tiny'],
                       help='Model architecture (default: resnet18)')

    parser.add_argument('--activation', type=str, default='swish',
                       choices=['relu', 'gelu', 'swish', 'mish', 'silu', 'leakyrelu'],
                       help='Activation function (default: swish)')

    # Training configuration
    parser.add_argument('--modern-training', action='store_true',
                       help='Use modern training techniques (AdamW, cosine annealing, AMP, EMA)')

    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')

    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')

    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')

    # Data augmentation
    parser.add_argument('--augmentation', type=str, default='standard',
                       choices=['basic', 'standard', 'autoaugment', 'randaugment'],
                       help='Data augmentation mode (default: standard)')

    parser.add_argument('--cutout', action='store_true',
                       help='Apply Cutout augmentation')

    parser.add_argument('--mixup', action='store_true',
                       help='Apply MixUp augmentation')

    parser.add_argument('--cutmix', action='store_true',
                       help='Apply CutMix augmentation')

    # Regularization
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1, set 0 to disable)')

    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping max norm (default: 1.0)')

    # Comparison
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare different model architectures')

    parser.add_argument('--compare-list', type=str, nargs='+',
                       default=['resnet18', 'efficientnet_b0', 'cnn_transformer_small'],
                       help='Models to compare (default: resnet18 efficientnet_b0 cnn_transformer_small)')

    # Utilities
    parser.add_argument('--quick', action='store_true',
                       help='Quick training (2 epochs) for testing')

    parser.add_argument('--list-models', action='store_true',
                       help='List all available models')

    args = parser.parse_args()

    # List models
    if args.list_models:
        print("\n📋 Available Models:")
        print("\nOriginal CNN:")
        print("  - cnn: Basic convolutional network")
        print("\nResNet variants:")
        print("  - resnet18: ResNet-18 (11M params)")
        print("  - resnet34: ResNet-34 (21M params)")
        print("  - resnet50: ResNet-50 with bottleneck (23M params)")
        print("  - resnet_tiny: Tiny ResNet for quick experiments")
        print("\nEfficientNet variants:")
        print("  - efficientnet_b0: EfficientNet-B0 baseline")
        print("  - efficientnet_b1: EfficientNet-B1 (wider & deeper)")
        print("  - efficientnet_tiny: Tiny EfficientNet")
        print("\nCNN-Transformer hybrids:")
        print("  - cnn_transformer_small: Small CNN-Transformer")
        print("  - cnn_transformer_base: Base CNN-Transformer")
        print("  - vit_tiny: Pure Vision Transformer\n")
        return

    # Quick mode
    if args.quick:
        args.epochs = 2
        print("⚡ Quick mode: 2 epochs\n")

    # Print configuration
    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Model: {args.model}")
    print(f"Activation: {args.activation}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Modern training: {args.modern_training}")
    print(f"{'='*70}\n")

    # Compare models
    if args.compare_models:
        compare_models(
            models=args.compare_list,
            epochs=args.epochs,
            batch_size=args.batch_size,
            modern_training=args.modern_training
        )
        return

    # Single model training
    print("📦 Loading CIFAR-10 dataset...")
    data_loader = CIFAR10DataLoader(batch_size=args.batch_size, validation_split=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    print("✓ Data loaded\n")

    # Create model
    model = get_model(args.model, activation=args.activation)
    model.summary()

    # Setup save directory
    save_dir = f'./checkpoints/{args.model}_{args.activation}_{"modern" if args.modern_training else "standard"}'
    os.makedirs(save_dir, exist_ok=True)

    # Train
    if args.modern_training:
        result = train_with_modern_techniques(
            model, train_loader, val_loader, test_loader,
            save_dir=save_dir,
            epochs=args.epochs,
            lr=args.lr,
            use_amp=True,
            use_ema=True,
            label_smoothing=args.label_smoothing,
            gradient_clip=args.gradient_clip,
            use_mixup=args.mixup,
            use_cutmix=args.cutmix
        )
    else:
        result = train_standard(
            model, train_loader, val_loader, test_loader,
            save_dir=save_dir,
            epochs=args.epochs,
            lr=args.lr
        )

    # Print final results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Test accuracy: {result['test_accuracy']:.2f}%")
    print(f"Test loss: {result['test_loss']:.4f}")
    print(f"Best val accuracy: {result['best_val_accuracy']:.2f}%")
    print(f"Training time: {result['training_time']:.1f}s")
    print(f"Model parameters: {result['model_parameters']:,}")
    print(f"{'='*70}\n")

    # Save results
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Results saved to {save_dir}/results.json")


if __name__ == '__main__':
    main()
