#!/usr/bin/env python3
"""
Main entry point for proteomics deep learning experiments.

Usage:
    python proteomics/main_proteomics.py --data data/proteomics/mzml/sample.mzML --model cnn --epochs 10
"""

import argparse
import logging
import yaml
from pathlib import Path
import torch
import torch.nn as nn

# Import proteomics modules
from data_loaders import SpectrumDataset, create_dataloaders
from models import SpectrumCNN, SpectrumTransformer, RetentionTimePredictor, LightweightSpectrumCNN
from training import ProteomicsTrainer, get_loss_function
from utils import plot_training_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: dict, input_dim: int, num_classes: int):
    """Create model based on configuration."""
    model_arch = config['model']['architecture']
    task = config['model']['task']

    if model_arch == 'cnn':
        model = SpectrumCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            task=task,
            **config['model']['cnn']
        )
    elif model_arch == 'lightweight_cnn':
        model = LightweightSpectrumCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            task=task,
            dropout=config['model']['cnn']['dropout']
        )
    elif model_arch == 'transformer':
        model = SpectrumTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            task=task,
            **config['model']['transformer']
        )
    elif model_arch == 'rt_predictor':
        model = RetentionTimePredictor(
            input_dim=input_dim,
            **config['model'].get('rt_predictor', {})
        )
    else:
        raise ValueError(f"Unknown architecture: {model_arch}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Proteomics Deep Learning')

    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--val-data', type=str, help='Path to validation data')
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'raw', 'mzml', 'csv'], help='Data format')
    parser.add_argument('--labels', type=str, help='Path to labels file (CSV)')

    # Model arguments
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'lightweight_cnn', 'transformer', 'rt_predictor'],
                       help='Model architecture')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression'], help='Task type')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    # Configuration
    parser.add_argument('--config', type=str,
                       default='proteomics/config/default_config.yaml',
                       help='Path to config file')

    # Output
    parser.add_argument('--checkpoint-dir', type=str,
                       default='./checkpoints/proteomics',
                       help='Checkpoint directory')
    parser.add_argument('--experiment-name', type=str, default='experiment',
                       help='Experiment name')

    args = parser.parse_args()

    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}. Using defaults.")
        config = {
            'data': {'format': 'auto', 'mz_range': [50, 2000], 'bin_size': 0.1},
            'model': {'architecture': args.model, 'task': args.task,
                     'cnn': {'num_conv_layers': 5, 'dropout': 0.3, 'use_residual': True},
                     'transformer': {'d_model': 256, 'nhead': 8, 'num_encoder_layers': 4}},
            'training': {'batch_size': 32, 'num_workers': 4}
        }

    # Override config with command-line arguments
    config['model']['architecture'] = args.model
    config['model']['task'] = args.task
    config['data']['format'] = args.format
    config['training']['num_epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    config['training']['patience'] = args.patience

    logger.info("=" * 60)
    logger.info("PROTEOMICS DEEP LEARNING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    # Load labels if provided
    labels = None
    if args.labels:
        import pandas as pd
        labels_df = pd.read_csv(args.labels)
        if 'label' in labels_df.columns:
            labels = labels_df['label'].values
        logger.info(f"Loaded {len(labels)} labels")

    # Create dataset
    logger.info("Creating dataset...")
    dataset = SpectrumDataset(
        data_path=args.data,
        format=config['data']['format'],
        task=config['model']['task'],
        labels=labels,
        mz_range=tuple(config['data']['mz_range']),
        bin_size=config['data']['bin_size'],
        normalization=config['data'].get('normalization', 'tic'),
        ms_level=config['data'].get('ms_level', 2),
        augment=True
    )

    logger.info(f"Dataset size: {len(dataset)}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    if args.val_data:
        dataloaders = create_dataloaders(
            train_path=args.data,
            val_path=args.val_data,
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 4),
            format=config['data']['format'],
            task=config['model']['task'],
            labels=labels
        )
    else:
        # Use single dataloader for training (no validation)
        from torch.utils.data import DataLoader
        dataloaders = {
            'train': DataLoader(dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['training'].get('num_workers', 4))
        }

    # Get input dimension from dataset
    input_dim = dataset.get_spectrum_shape()[0]
    logger.info(f"Input dimension: {input_dim}")

    # Create model
    logger.info("Creating model...")
    model = create_model(config, input_dim, args.num_classes)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create loss function
    loss_fn_name = config['training'].get('loss_function', 'cross_entropy')
    criterion = get_loss_function(loss_fn_name)
    logger.info(f"Loss function: {loss_fn_name}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Create scheduler
    scheduler = None
    if config['training'].get('scheduler') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif config['training'].get('scheduler') == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Create trainer
    logger.info("Initializing trainer...")
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    trainer = ProteomicsTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('val'),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['training'].get('device', 'auto'),
        checkpoint_dir=str(checkpoint_dir),
        patience=config['training']['patience'],
        use_amp=config['training'].get('use_amp', True)
    )

    # Train
    logger.info("Starting training...")
    trainer.train(num_epochs=config['training']['num_epochs'], verbose=True)

    # Save training history plot
    logger.info("Saving training history...")
    plot_training_history(
        trainer.history,
        save_path=str(checkpoint_dir / 'training_history.png')
    )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Best epoch: {trainer.best_epoch}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
