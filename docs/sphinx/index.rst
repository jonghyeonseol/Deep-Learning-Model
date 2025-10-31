Deep Learning Framework Documentation
=====================================

Welcome to the comprehensive API documentation for the Deep Learning Framework - a modern, production-ready PyTorch-based image classification system designed for CIFAR-10 dataset.

**Version**: 2.0

Overview
--------

This framework provides:

* **State-of-the-art architectures**: ResNet, EfficientNet, Vision Transformers, ConvNeXt
* **Modern training techniques**: AdamW, Mixed Precision (AMP), EMA, Label Smoothing
* **Advanced data augmentation**: RandAugment, MixUp, CutMix, Cutout
* **Custom activation functions**: 14+ implementations including GELU, Swish, Mish
* **Comprehensive utilities**: Logging, profiling, configuration management
* **Production-ready code**: Type hints, custom exceptions, extensive testing

Quick Start
-----------

Basic training with default settings::

    python3 main.py --activation relu --epochs 10

Modern training with state-of-the-art techniques::

    python3 main_modern.py --model resnet18 --epochs 200 --amp

Benchmark all architectures::

    python3 benchmark_all.py --architectures --quick

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   configuration
   training_guide

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   models
   utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   examples
   best_practices
   troubleshooting


Models Package
--------------

The models package contains all neural network architectures and activation functions.

**Architectures:**

* ``models.network`` - Basic CNN and fully-connected networks
* ``models.resnet`` - ResNet-18/34/50/101 with residual blocks
* ``models.efficientnet`` - EfficientNet-B0/B1 with MBConv blocks
* ``models.cnn_transformer`` - Hybrid CNN-Transformer and Vision Transformer
* ``models.convnext`` - ConvNeXt (modernized CNN architecture)

**Components:**

* ``models.activations`` - Custom activation functions (14+ implementations)


Utils Package
-------------

The utils package provides training, data loading, visualization, and utility functions.

**Core Training:**

* ``utils.trainer`` - Basic training loop with checkpointing
* ``utils.modern_trainer`` - Modern training with AdamW, AMP, EMA
* ``utils.data_loader`` - CIFAR-10 data loading with augmentation

**Augmentation & Regularization:**

* ``utils.augmentation`` - RandAugment, MixUp, CutMix, Cutout
* ``utils.regularization`` - DropBlock, Stochastic Depth, Drop Path

**Monitoring & Profiling:**

* ``utils.monitor`` - Layer monitoring and activation analysis
* ``utils.realtime_monitor`` - Live training visualization
* ``utils.profiler`` - Performance profiling (FLOPs, memory, inference)

**Configuration & Utilities:**

* ``utils.config`` - YAML/JSON configuration management
* ``utils.logger`` - Centralized logging framework
* ``utils.exceptions`` - Custom exception hierarchy
* ``utils.visualization`` - Training plots and confusion matrices


Performance & Features
----------------------

**Expected Results on CIFAR-10:**

* ResNet-18: ~95.5% test accuracy (200 epochs)
* ResNet-50: ~96.0% test accuracy (200 epochs)
* EfficientNet-B0: ~95.8% test accuracy (300 epochs)
* Vision Transformer: ~96.5% test accuracy (300 epochs)
* ConvNeXt-Tiny: ~97.0%+ test accuracy (300 epochs)

**Training Techniques:**

* Mixed Precision Training (AMP) - 2x speedup
* Exponential Moving Average (EMA) - improved stability
* Label Smoothing - better generalization
* Gradient Clipping - prevent exploding gradients
* Cosine Annealing with Warmup - optimal LR schedule


Installation
------------

Requirements::

    pip install torch torchvision numpy matplotlib pyyaml tensorboard pytest

Or install all dependencies::

    pip install -r requirements.txt


Project Structure
-----------------

::

    ├── models/              # Neural network architectures
    │   ├── activations.py   # Custom activation functions
    │   ├── network.py       # Basic CNN and FC networks
    │   ├── resnet.py        # ResNet variants
    │   ├── efficientnet.py  # EfficientNet variants
    │   ├── cnn_transformer.py  # Hybrid and ViT models
    │   └── convnext.py      # ConvNeXt architecture
    │
    ├── utils/               # Training utilities
    │   ├── trainer.py       # Basic training loop
    │   ├── modern_trainer.py  # Modern training techniques
    │   ├── data_loader.py   # Data loading and preprocessing
    │   ├── augmentation.py  # Data augmentation
    │   ├── regularization.py  # Regularization techniques
    │   ├── monitor.py       # Training monitoring
    │   ├── profiler.py      # Performance profiling
    │   ├── logger.py        # Logging framework
    │   ├── config.py        # Configuration management
    │   ├── exceptions.py    # Custom exceptions
    │   └── visualization.py  # Plotting utilities
    │
    ├── main.py              # Basic training script
    ├── main_modern.py       # Modern training script
    ├── benchmark_all.py     # Comprehensive benchmarking
    └── profile_models.py    # Model profiling script


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
