#!/usr/bin/env python3
"""
Main script to train neural networks with different activation functions on CIFAR-10.
Usage: python main.py --activation all
"""

import argparse
import torch
import os
from models import ConvNeuralNetwork, get_activation
from utils import CIFAR10DataLoader, Trainer, Visualizer


def train_model(activation, epochs=10, batch_size=32, lr=0.001):
    """
    Train a model with specified activation function.

    Args:
        activation (str): Activation function name
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate

    Returns:
        dict: Training results
    """
    print(f"\n{'='*60}")
    print(f"Training model with {activation.upper()} activation")
    print(f"{'='*60}")

    # Create data loader
    data_loader = CIFAR10DataLoader(batch_size=batch_size, validation_split=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Print dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"Dataset: CIFAR-10")
    print(f"Train samples: {dataset_info['train_size']:,}")
    print(f"Validation samples: {dataset_info['val_size']:,}")
    print(f"Test samples: {dataset_info['test_size']:,}")
    print(f"Number of classes: {dataset_info['num_classes']}")

    # Create model
    model = ConvNeuralNetwork(
        input_channels=3,
        num_classes=10,
        activation=activation,
        dropout_rate=0.2
    )

    # Print model summary
    model.summary()

    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = f'./checkpoints/{activation}'
    os.makedirs(save_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        save_dir=save_dir
    )

    # Configure training
    trainer.configure_optimizer('adam', lr=lr, weight_decay=1e-4)
    trainer.configure_scheduler('step', step_size=5, gamma=0.1)
    trainer.configure_criterion('crossentropy')

    # Train model
    trainer.train(epochs=epochs, early_stopping_patience=5, save_best=True)

    # Test model
    test_loss, test_acc = trainer.test()

    # Create visualizations
    visualizer = Visualizer(class_names=data_loader.classes)

    # Plot training history
    visualizer.plot_training_history(
        trainer.history,
        save_path=os.path.join(save_dir, 'training_history.png')
    )

    # Plot sample predictions
    visualizer.plot_model_predictions(
        model, test_loader, device, num_samples=8,
        denormalize_fn=data_loader.denormalize,
        save_path=os.path.join(save_dir, 'predictions.png')
    )

    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        model, test_loader, device,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )

    return {
        'activation': activation,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': trainer.best_val_acc,
        'model_parameters': model.get_num_parameters()
    }


def compare_activations(activations, epochs=10, batch_size=32, lr=0.001):
    """
    Compare different activation functions.

    Args:
        activations (list): List of activation function names
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
    """
    results = []

    for activation in activations:
        try:
            result = train_model(activation, epochs, batch_size, lr)
            results.append(result)
        except Exception as e:
            print(f"Error training with {activation}: {e}")
            continue

    # Display comparison results
    print(f"\n{'='*80}")
    print("ACTIVATION FUNCTION COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Activation':<12} {'Test Loss':<12} {'Test Acc (%)':<12} {'Val Acc (%)':<12} {'Parameters':<12}")
    print(f"{'-'*80}")

    for result in results:
        print(f"{result['activation']:<12} "
              f"{result['test_loss']:<12.4f} "
              f"{result['test_accuracy']:<12.2f} "
              f"{result['best_val_accuracy']:<12.2f} "
              f"{result['model_parameters']:<12,}")

    # Find best performing activation
    if results:
        best_result = max(results, key=lambda x: x['test_accuracy'])
        print(f"\nBest performing activation: {best_result['activation'].upper()}")
        print(f"Test accuracy: {best_result['test_accuracy']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train neural networks with different activation functions')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'gelu', 'tanh', 'all'],
                       help='Activation function to use (default: relu)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with fewer epochs for testing')

    args = parser.parse_args()

    # Adjust epochs for quick training
    if args.quick:
        args.epochs = 2
        print("Quick training mode: using 2 epochs")

    print("CIFAR-10 Neural Network Training")
    print(f"Arguments: {args}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    if args.activation == 'all':
        # Train with all activation functions
        activations = ['relu', 'gelu', 'tanh']
        print(f"\nTraining with all activation functions: {activations}")
        compare_activations(activations, args.epochs, args.batch_size, args.lr)
    else:
        # Train with single activation function
        print(f"\nTraining with {args.activation} activation")
        result = train_model(args.activation, args.epochs, args.batch_size, args.lr)
        print(f"\nFinal results:")
        print(f"Test accuracy: {result['test_accuracy']:.2f}%")
        print(f"Test loss: {result['test_loss']:.4f}")


if __name__ == '__main__':
    main()