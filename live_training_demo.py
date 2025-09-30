#!/usr/bin/env python3
"""
🧠 LIVE Neural Network Training Demo
=====================================

This script demonstrates REAL-TIME monitoring of your ANN model during training.
You will see your neural network "thinking" and learning live!

Features:
- 📈 Live training loss and accuracy curves
- 🧠 Real-time layer activation monitoring
- ⚡ Live gradient flow visualization
- 🎯 Neural network structure visualization
- 📊 Training statistics and performance metrics

Usage:
    python3 live_training_demo.py --activation swish --epochs 3
    python3 live_training_demo.py --activation modern --epochs 2
"""

import argparse
import torch
import os
import time
from models import ConvNeuralNetwork, get_available_activations
from utils import CIFAR10DataLoader
from utils.live_trainer import LiveTrainer
from utils.realtime_monitor import create_comprehensive_monitor


def live_training_demo(activation, epochs=3, batch_size=32, lr=0.001):
    """
    Demonstrate live training with real-time monitoring.

    Args:
        activation (str): Activation function name
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
    """
    print("🚀" + "="*70 + "🚀")
    print("   🧠 LIVE NEURAL NETWORK TRAINING DEMONSTRATION 🧠")
    print("🚀" + "="*70 + "🚀")
    print()
    print(f"🎯 Training with {activation.upper()} activation")
    print(f"📊 Monitoring: REAL-TIME (you'll see live plots!)")
    print(f"⏱️  Duration: {epochs} epochs (~{epochs * 3} minutes)")
    print()
    print("📺 WHAT YOU'LL SEE:")
    print("   • Live loss and accuracy curves updating in real-time")
    print("   • Layer activation patterns changing as the network learns")
    print("   • Gradient flow through the network")
    print("   • Training progress and performance metrics")
    print()
    input("👀 Press ENTER when ready to start live monitoring... ")
    print()

    # Create data loader
    print("📁 Loading CIFAR-10 dataset...")
    data_loader = CIFAR10DataLoader(batch_size=batch_size, validation_split=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Print dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"✅ Dataset loaded:")
    print(f"   Train samples: {dataset_info['train_size']:,}")
    print(f"   Validation samples: {dataset_info['val_size']:,}")
    print(f"   Test samples: {dataset_info['test_size']:,}")
    print()

    # Create model
    print(f"🏗️  Building neural network with {activation.upper()} activation...")
    model = ConvNeuralNetwork(
        input_channels=3,
        num_classes=10,
        activation=activation,
        dropout_rate=0.2
    )

    # Print model summary
    model.summary()
    print()

    # Create live trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = f'./live_checkpoints/{activation}'
    os.makedirs(save_dir, exist_ok=True)

    print(f"🔧 Setting up LIVE trainer...")
    print(f"   Device: {device}")
    print(f"   Save directory: {save_dir}")
    print(f"   Live monitoring: ENABLED")
    print()

    # Initialize live trainer with monitoring
    trainer = LiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        save_dir=save_dir,
        enable_live_monitoring=True  # This enables real-time monitoring!
    )

    # Configure training
    trainer.configure_optimizer('adam', lr=lr, weight_decay=1e-4)
    trainer.configure_scheduler('step', step_size=2, gamma=0.5)
    trainer.configure_criterion('crossentropy')

    print("🎬 STARTING LIVE TRAINING...")
    print("📺 Watch for popup windows showing real-time monitoring!")
    print("   (Multiple plot windows will appear)")
    print()
    time.sleep(2)

    # Train model with live monitoring
    try:
        trainer.train(
            epochs=epochs,
            early_stopping_patience=None,  # No early stopping for demo
            save_best=True
        )
    except KeyboardInterrupt:
        print("\n⏹️  Training stopped by user")

    print()
    print("🧪 Testing trained model...")
    test_loss, test_acc = trainer.test()

    # Get detailed training statistics
    stats = trainer.get_training_stats()

    print()
    print("📊" + "="*50 + "📊")
    print("           LIVE TRAINING RESULTS")
    print("📊" + "="*50 + "📊")
    print()
    print(f"🎯 Activation Function: {activation.upper()}")
    print(f"🏆 Final Test Accuracy: {test_acc:.2f}%")
    print(f"📉 Final Test Loss: {test_loss:.4f}")
    print(f"🥇 Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print()
    print("⚡ Training Efficiency:")
    print(f"   Average gradient norm: {stats['avg_gradient_norm']:.4f}")
    print(f"   Average batch time: {stats['avg_batch_time']:.3f} seconds")
    print(f"   Training efficiency: {stats['training_efficiency']:.2f} batches/second")
    print()

    # Performance assessment
    if test_acc > 70:
        print("🎉 EXCELLENT! Your neural network learned very well!")
    elif test_acc > 60:
        print("👍 GOOD! Solid learning performance!")
    elif test_acc > 50:
        print("📈 DECENT! Network is learning, could train longer!")
    else:
        print("🤔 LEARNING... Network needs more training or tuning!")

    print()
    print(f"💾 Model and visualizations saved to: {save_dir}")
    print()

    return {
        'activation': activation,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': trainer.best_val_acc,
        'training_stats': stats
    }


def compare_activations_live(activations, epochs=2, batch_size=32, lr=0.001):
    """
    Compare multiple activation functions with live monitoring.

    Args:
        activations (list): List of activation function names
        epochs (int): Number of training epochs per activation
        batch_size (int): Batch size
        lr (float): Learning rate
    """
    print("🏁" + "="*60 + "🏁")
    print("   🧠 LIVE ACTIVATION FUNCTION COMPARISON 🧠")
    print("🏁" + "="*60 + "🏁")
    print()
    print(f"🎯 Testing {len(activations)} activation functions:")
    for i, act in enumerate(activations, 1):
        print(f"   {i}. {act.upper()}")
    print()
    print(f"⏱️  Total time: ~{len(activations) * epochs * 3} minutes")
    print("📺 Each activation will show live monitoring!")
    print()

    input("🚀 Press ENTER to start activation comparison... ")
    print()

    results = []

    for i, activation in enumerate(activations, 1):
        print(f"\n🔄 [{i}/{len(activations)}] Testing {activation.upper()} activation...")
        print("─" * 50)

        try:
            result = live_training_demo(activation, epochs, batch_size, lr)
            results.append(result)

            print(f"✅ {activation.upper()} completed!")
            print(f"   Test Accuracy: {result['test_accuracy']:.2f}%")
            print()

            if i < len(activations):
                print("⏳ Preparing next activation function...")
                time.sleep(3)  # Brief pause between tests

        except Exception as e:
            print(f"❌ Error with {activation}: {e}")
            continue

    # Display comparison results
    print("\n🏆" + "="*70 + "🏆")
    print("              ACTIVATION FUNCTION COMPARISON RESULTS")
    print("🏆" + "="*70 + "🏆")
    print()

    if results:
        print(f"{'Rank':<4} {'Activation':<12} {'Test Acc':<10} {'Val Acc':<10} {'Efficiency':<12}")
        print("─" * 60)

        # Sort by test accuracy
        sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)

        for rank, result in enumerate(sorted_results, 1):
            efficiency = result['training_stats']['training_efficiency']
            print(f"{rank:<4} {result['activation'].upper():<12} "
                  f"{result['test_accuracy']:<10.2f}% "
                  f"{result['best_val_accuracy']:<10.2f}% "
                  f"{efficiency:<12.2f}")

        print()
        winner = sorted_results[0]
        print(f"🥇 WINNER: {winner['activation'].upper()}")
        print(f"   Test Accuracy: {winner['test_accuracy']:.2f}%")
        print(f"   Validation Accuracy: {winner['best_val_accuracy']:.2f}%")
        print()


def main():
    """Main function for live training demonstration."""
    available_activations = get_available_activations()
    modern_activations = ['gelu', 'swish', 'mish', 'silu']
    classic_activations = ['relu', 'tanh', 'sigmoid']

    parser = argparse.ArgumentParser(description='🧠 Live Neural Network Training Demo')
    parser.add_argument('--activation', type=str, default='swish',
                       choices=available_activations + ['modern', 'classic', 'compare'],
                       help='Activation function to demo (default: swish)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick demo with fewer epochs')

    args = parser.parse_args()

    # Adjust for quick demo
    if args.quick:
        args.epochs = 2
        print("⚡ Quick demo mode: using 2 epochs per activation")

    print("🧠 Live Neural Network Training Demo")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()

    if args.activation == 'modern':
        # Compare modern activation functions
        compare_activations_live(modern_activations, args.epochs, args.batch_size, args.lr)
    elif args.activation == 'classic':
        # Compare classic activation functions
        compare_activations_live(classic_activations, args.epochs, args.batch_size, args.lr)
    elif args.activation == 'compare':
        # Compare a mix of activations
        comparison_set = ['relu', 'gelu', 'swish', 'mish']
        compare_activations_live(comparison_set, args.epochs, args.batch_size, args.lr)
    else:
        # Single activation demo
        result = live_training_demo(args.activation, args.epochs, args.batch_size, args.lr)

    print()
    print("🎬" + "="*50 + "🎬")
    print("    LIVE TRAINING DEMO COMPLETED!")
    print("🎬" + "="*50 + "🎬")
    print()
    print("📊 What you experienced:")
    print("   • Real-time loss and accuracy updates")
    print("   • Live layer activation monitoring")
    print("   • Gradient flow visualization")
    print("   • Neural network learning process")
    print()
    print("💡 Tips for understanding the visualizations:")
    print("   • Decreasing loss = network learning")
    print("   • Increasing accuracy = better performance")
    print("   • Layer activations show what neurons are doing")
    print("   • Gradient flow shows learning health")
    print()
    print("🔬 Try different activation functions to see how they affect learning!")


if __name__ == '__main__':
    main()