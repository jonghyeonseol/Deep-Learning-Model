#!/usr/bin/env python3
"""
Comprehensive benchmark script to compare all models and techniques.

This script runs systematic experiments to compare:
1. Different architectures (CNN, ResNet, EfficientNet, CNN-Transformer)
2. Modern vs Standard training
3. Different augmentation strategies
4. Different activation functions

Usage:
    python benchmark_all.py --full          # Run all experiments (very long)
    python benchmark_all.py --quick         # Quick benchmark (2 epochs each)
    python benchmark_all.py --architectures # Compare architectures only
    python benchmark_all.py --techniques    # Compare training techniques
"""

import argparse
import torch
import os
import json
import time
import pandas as pd
from datetime import datetime

from models.network import ConvNeuralNetwork
from models.resnet import ResNet18, ResNet_Tiny
from models.efficientnet import EfficientNet_Tiny
from models.cnn_transformer import CNNTransformer_Small

from utils.modern_trainer import ModernTrainer
from utils.trainer import Trainer
from utils.data_loader import CIFAR10DataLoader


def run_experiment(experiment_name, model, train_loader, val_loader, test_loader,
                   use_modern=True, epochs=10, lr=0.001):
    """
    Run a single experiment.

    Args:
        experiment_name: Name of experiment
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        use_modern: Use modern training techniques
        epochs: Number of epochs
        lr: Learning rate

    Returns:
        Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")

    save_dir = f'./benchmarks/{experiment_name}'
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()

    try:
        if use_modern:
            trainer = ModernTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                save_dir=save_dir
            )
            trainer.configure_optimizer('adamw', lr=lr, weight_decay=0.05)
            trainer.configure_scheduler('cosine_warmup', epochs=epochs)
            trainer.configure_criterion('label_smoothing', label_smoothing=0.1)
            if torch.cuda.is_available():
                trainer.enable_amp()
            trainer.enable_gradient_clipping(max_norm=1.0)
            trainer.enable_ema(decay=0.9999)
        else:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                save_dir=save_dir
            )
            trainer.configure_optimizer('adam', lr=lr, weight_decay=1e-4)
            trainer.configure_scheduler('step', step_size=30, gamma=0.1)
            trainer.configure_criterion('crossentropy')

        # Train
        trainer.train(epochs=epochs, early_stopping_patience=15, save_best=True)

        # Test
        if use_modern:
            test_loss, test_acc = trainer.test(use_ema=True)
        else:
            test_loss, test_acc = trainer.test()

        training_time = time.time() - start_time

        result = {
            'experiment': experiment_name,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'best_val_accuracy': trainer.best_val_acc,
            'training_time': training_time,
            'parameters': sum(p.numel() for p in model.parameters()),
            'use_modern': use_modern,
            'epochs': epochs,
            'status': 'success'
        }

        # Save result
        with open(os.path.join(save_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ {experiment_name} completed successfully")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Training Time: {training_time:.1f}s")

        return result

    except Exception as e:
        print(f"\n✗ {experiment_name} failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            'experiment': experiment_name,
            'status': 'failed',
            'error': str(e)
        }


def benchmark_architectures(epochs=10, batch_size=128):
    """
    Compare different model architectures.

    Args:
        epochs: Number of epochs
        batch_size: Batch size

    Returns:
        List of results
    """
    print(f"\n{'#'*70}")
    print("# BENCHMARK: Model Architectures")
    print(f"{'#'*70}\n")

    # Load data
    data_loader = CIFAR10DataLoader(batch_size=batch_size, validation_split=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    experiments = [
        ('CNN_Basic', ConvNeuralNetwork(3, 10, 'swish', 0.2)),
        ('ResNet18', ResNet18(10, 'swish', 0.2)),
        ('ResNet_Tiny', ResNet_Tiny(10, 'swish', 0.2)),
        ('EfficientNet_Tiny', EfficientNet_Tiny(10, 'swish', 0.2)),
        ('CNNTransformer_Small', CNNTransformer_Small(10, 'gelu')),
    ]

    results = []
    for name, model in experiments:
        result = run_experiment(
            f'arch_{name}',
            model,
            train_loader, val_loader, test_loader,
            use_modern=True,
            epochs=epochs
        )
        results.append(result)

    return results


def benchmark_training_techniques(epochs=10, batch_size=128):
    """
    Compare modern vs standard training techniques.

    Args:
        epochs: Number of epochs
        batch_size: Batch size

    Returns:
        List of results
    """
    print(f"\n{'#'*70}")
    print("# BENCHMARK: Training Techniques")
    print(f"{'#'*70}\n")

    # Load data
    data_loader = CIFAR10DataLoader(batch_size=batch_size, validation_split=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    experiments = [
        ('Standard_Training', ResNet_Tiny(10, 'swish', 0.2), False),
        ('Modern_Training', ResNet_Tiny(10, 'swish', 0.2), True),
    ]

    results = []
    for name, model, use_modern in experiments:
        result = run_experiment(
            f'technique_{name}',
            model,
            train_loader, val_loader, test_loader,
            use_modern=use_modern,
            epochs=epochs
        )
        results.append(result)

    return results


def benchmark_activations(epochs=10, batch_size=128):
    """
    Compare different activation functions.

    Args:
        epochs: Number of epochs
        batch_size: Batch size

    Returns:
        List of results
    """
    print(f"\n{'#'*70}")
    print("# BENCHMARK: Activation Functions")
    print(f"{'#'*70}\n")

    # Load data
    data_loader = CIFAR10DataLoader(batch_size=batch_size, validation_split=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    activations = ['relu', 'gelu', 'swish', 'mish', 'silu']
    results = []

    for activation in activations:
        model = ResNet_Tiny(10, activation, 0.2)
        result = run_experiment(
            f'activation_{activation}',
            model,
            train_loader, val_loader, test_loader,
            use_modern=True,
            epochs=epochs
        )
        results.append(result)

    return results


def save_benchmark_report(all_results, output_dir='./benchmarks'):
    """
    Save comprehensive benchmark report.

    Args:
        all_results: Dictionary of all benchmark results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create summary report
    report = {
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'results': all_results
    }

    # Save JSON report
    report_path = os.path.join(output_dir, 'benchmark_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Benchmark report saved to {report_path}")

    # Create CSV for each benchmark type
    for benchmark_name, results in all_results.items():
        if not results:
            continue

        # Filter successful results
        successful_results = [r for r in results if r.get('status') == 'success']
        if not successful_results:
            continue

        df = pd.DataFrame(successful_results)
        csv_path = os.path.join(output_dir, f'{benchmark_name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ {benchmark_name} results saved to {csv_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    for benchmark_name, results in all_results.items():
        successful = [r for r in results if r.get('status') == 'success']
        if not successful:
            continue

        print(f"\n{benchmark_name.upper()}:")
        print(f"{'Experiment':<30} {'Test Acc (%)':<15} {'Time (s)':<15}")
        print(f"{'-'*60}")

        for result in successful:
            print(f"{result['experiment']:<30} "
                  f"{result['test_accuracy']:<15.2f} "
                  f"{result['training_time']:<15.1f}")

        best = max(successful, key=lambda x: x['test_accuracy'])
        print(f"\n  Best: {best['experiment']} ({best['test_accuracy']:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive deep learning benchmark')

    parser.add_argument('--full', action='store_true',
                       help='Run all benchmarks (takes a long time)')

    parser.add_argument('--quick', action='store_true',
                       help='Quick benchmark with 2 epochs (for testing)')

    parser.add_argument('--architectures', action='store_true',
                       help='Benchmark different architectures')

    parser.add_argument('--techniques', action='store_true',
                       help='Benchmark training techniques')

    parser.add_argument('--activations', action='store_true',
                       help='Benchmark activation functions')

    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')

    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')

    args = parser.parse_args()

    # Set epochs
    epochs = 2 if args.quick else args.epochs

    print(f"\n{'='*70}")
    print("COMPREHENSIVE DEEP LEARNING BENCHMARK")
    print(f"{'='*70}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Epochs per experiment: {epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*70}\n")

    all_results = {}

    # Run selected benchmarks
    if args.full or args.architectures:
        results = benchmark_architectures(epochs, args.batch_size)
        all_results['architectures'] = results

    if args.full or args.techniques:
        results = benchmark_training_techniques(epochs, args.batch_size)
        all_results['techniques'] = results

    if args.full or args.activations:
        results = benchmark_activations(epochs, args.batch_size)
        all_results['activations'] = results

    # If no specific benchmark selected, show help
    if not any([args.full, args.architectures, args.techniques, args.activations]):
        print("Please specify which benchmarks to run:")
        print("  --full          : Run all benchmarks")
        print("  --architectures : Compare architectures")
        print("  --techniques    : Compare training techniques")
        print("  --activations   : Compare activation functions")
        print("  --quick         : Quick mode (2 epochs)\n")
        return

    # Save report
    if all_results:
        save_benchmark_report(all_results)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETED")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
