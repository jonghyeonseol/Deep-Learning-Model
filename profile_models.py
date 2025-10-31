#!/usr/bin/env python3
"""
Model Profiling Script

Profile and compare model architectures for:
- Parameter count
- FLOPs (floating point operations)
- Inference speed (latency and throughput)
- Memory footprint
- Training step breakdown

Usage:
    # Profile single model
    python3 profile_models.py --model resnet18

    # Compare multiple models
    python3 profile_models.py --models resnet18 efficientnet_b0 vit

    # Profile training step
    python3 profile_models.py --model resnet18 --profile-training

    # Detailed profiling with custom batch size
    python3 profile_models.py --model resnet18 --batch-size 64 --detailed
"""

import torch
import argparse
from utils.profiler import (
    profile_model,
    compare_models,
    print_model_comparison,
    Profiler,
    profile_training_step
)
from utils.data_loader import CIFAR10DataLoader
from models.resnet import ResNet18, ResNet34, ResNet50
from models.efficientnet import EfficientNet_B0
from models.cnn_transformer import VisionTransformer
from utils.logger import get_logger

logger = get_logger(__name__)


def get_model(model_name: str, num_classes: int = 10):
    """Get model instance by name."""
    model_dict = {
        'resnet18': ResNet18(num_classes=num_classes),
        'resnet34': ResNet34(num_classes=num_classes),
        'resnet50': ResNet50(num_classes=num_classes),
        'efficientnet_b0': EfficientNet_B0(num_classes=num_classes),
        'vit': VisionTransformer(
            image_size=32,
            patch_size=4,
            num_classes=num_classes,
            embed_dim=384,
            depth=7,
            num_heads=6
        )
    }

    if model_name not in model_dict:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(model_dict.keys())}"
        )

    return model_dict[model_name]


def profile_single_model(
    model_name: str,
    batch_size: int = 1,
    detailed: bool = False
):
    """Profile a single model."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROFILING: {model_name.upper()}")
    logger.info(f"{'=' * 80}\n")

    model = get_model(model_name)
    stats = profile_model(
        model,
        input_shape=(3, 32, 32),
        batch_size=batch_size
    )

    # Print basic stats
    logger.info(f"Model: {model_name}")
    logger.info(f"  Total Parameters: {stats['total_params']:,}")
    logger.info(f"  Trainable Parameters: {stats['trainable_params']:,}")
    logger.info(f"  Parameter Memory: {stats['memory_mb']:.2f} MB")
    logger.info(f"\nComputational Complexity:")
    logger.info(f"  FLOPs: {stats['flops'] / 1e9:.4f} GFLOPs")
    logger.info(f"  FLOPs per parameter: {stats['flops'] / stats['total_params']:.2f}")
    logger.info(f"\nInference Performance (batch_size={batch_size}):")
    logger.info(f"  Latency: {stats['inference_time_ms']:.2f} ms")
    logger.info(f"  Throughput: {stats['throughput_fps']:.2f} images/sec")
    logger.info(f"  Time per image: {stats['inference_time_ms'] / batch_size:.2f} ms")

    if detailed:
        # Additional detailed profiling
        logger.info(f"\nDetailed Analysis:")

        # Memory efficiency
        params_per_mb = stats['total_params'] / stats['memory_mb']
        logger.info(f"  Parameters per MB: {params_per_mb:,.0f}")

        # Computational efficiency
        flops_per_ms = (stats['flops'] / 1e6) / stats['inference_time_ms']
        logger.info(f"  MFLOPs per millisecond: {flops_per_ms:.2f}")

        # Parameter efficiency
        params_per_gflop = stats['total_params'] / (stats['flops'] / 1e9)
        logger.info(f"  Parameters per GFLOPs: {params_per_gflop:,.0f}")

    logger.info(f"\n{'=' * 80}\n")


def compare_multiple_models(
    model_names: list,
    batch_size: int = 1
):
    """Compare multiple models."""
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPARING MODELS")
    logger.info(f"{'=' * 80}\n")

    models = {name: get_model(name) for name in model_names}
    results = compare_models(
        models,
        input_shape=(3, 32, 32),
        batch_size=batch_size
    )

    print_model_comparison(results)

    # Print efficiency rankings
    logger.info("\nEFFICIENCY RANKINGS:")
    logger.info("-" * 80)

    # Sort by different metrics
    metrics = [
        ('total_params', 'Fewest Parameters', False),
        ('flops', 'Lowest FLOPs', False),
        ('inference_time_ms', 'Fastest Inference', False),
        ('throughput_fps', 'Highest Throughput', True)
    ]

    for metric, title, reverse in metrics:
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1][metric],
            reverse=reverse
        )
        logger.info(f"\n{title}:")
        for rank, (name, stats) in enumerate(sorted_models, 1):
            logger.info(f"  {rank}. {name}")


def profile_training(
    model_name: str,
    batch_size: int = 128,
    num_batches: int = 10
):
    """Profile training step breakdown."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING STEP PROFILING: {model_name.upper()}")
    logger.info(f"{'=' * 80}\n")

    # Create model and data
    model = get_model(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    data_loader = CIFAR10DataLoader(
        batch_size=batch_size,
        num_workers=2
    )
    train_loader, _, _ = data_loader.get_loaders()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Profile training step
    profiler = profile_training_step(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_batches=num_batches
    )

    # Print results
    profiler.print_stats(sort_by='total_time')

    # Calculate breakdown percentages
    stats = profiler.get_stats()
    total_time = sum(s['total_time'] for s in stats.values())

    logger.info("\nTRAINING STEP BREAKDOWN:")
    logger.info("-" * 80)
    for section, section_stats in stats.items():
        percentage = (section_stats['total_time'] / total_time) * 100
        logger.info(
            f"{section:<25} {section_stats['total_time']:>10.4f}s "
            f"({percentage:>5.1f}%)"
        )

    logger.info(f"\nTotal training time: {total_time:.4f}s")
    logger.info(f"Time per batch: {total_time / num_batches:.4f}s")
    logger.info(f"Images per second: {(batch_size * num_batches) / total_time:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Profile deep learning models'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Single model to profile'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Multiple models to compare'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for profiling (default: 1)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed profiling information'
    )
    parser.add_argument(
        '--profile-training',
        action='store_true',
        help='Profile training step breakdown'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Profile all available models'
    )

    args = parser.parse_args()

    # Determine which models to profile
    if args.all:
        model_names = [
            'resnet18', 'resnet34', 'resnet50',
            'efficientnet_b0', 'vit'
        ]
    elif args.models:
        model_names = args.models
    elif args.model:
        model_names = [args.model]
    else:
        # Default: profile ResNet-18
        model_names = ['resnet18']

    # Single model detailed profiling
    if len(model_names) == 1 and not args.all:
        model_name = model_names[0]
        profile_single_model(
            model_name,
            batch_size=args.batch_size,
            detailed=args.detailed
        )

        if args.profile_training:
            profile_training(
                model_name,
                batch_size=args.batch_size
            )
    else:
        # Multiple model comparison
        compare_multiple_models(
            model_names,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()
