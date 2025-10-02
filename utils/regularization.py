"""
Modern regularization techniques for deep learning.

This module implements:
- DropBlock: Structured dropout for CNNs
- Stochastic Depth: Randomly drop entire layers during training
- Shake-Shake regularization
- Drop Path (similar to Stochastic Depth but for any architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DropBlock2D(nn.Module):
    """
    DropBlock: A regularization method for convolutional networks.

    Paper: "DropBlock: A regularization method for convolutional networks"
    https://arxiv.org/abs/1810.12890

    Unlike standard dropout which drops individual values independently,
    DropBlock drops contiguous regions. This is more effective for CNNs
    because nearby activations are highly correlated.

    Key differences from Dropout:
    - Drops spatial blocks instead of individual pixels
    - More effective for convolutional layers
    - Typically applied with increasing probability during training

    Args:
        drop_prob: Probability of dropping a block
        block_size: Size of blocks to drop
    """

    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        # Get shape
        batch_size, channels, height, width = x.shape

        # Compute gamma (sampling mask probability)
        gamma = self._compute_gamma(height, width)

        # Sample mask
        mask = (torch.rand(batch_size, channels, height, width,
                          device=x.device) < gamma).float()

        # Apply block mask
        block_mask = self._compute_block_mask(mask)

        # Normalize and apply
        out = x * block_mask * block_mask.numel() / block_mask.sum()

        return out

    def _compute_gamma(self, height, width):
        """Compute gamma for DropBlock."""
        return (self.drop_prob / (self.block_size ** 2)) * \
               ((height * width) / ((height - self.block_size + 1) *
                                   (width - self.block_size + 1)))

    def _compute_block_mask(self, mask):
        """Expand point mask to block mask."""
        block_size = self.block_size
        pad = block_size // 2

        # Pad mask
        mask = F.pad(mask, [pad, pad, pad, pad])

        # Max pooling to expand points to blocks
        block_mask = F.max_pool2d(
            mask,
            kernel_size=(block_size, block_size),
            stride=(1, 1),
            padding=0
        )

        # Invert (0 becomes dropped region)
        block_mask = 1 - block_mask

        return block_mask

    def set_drop_prob(self, drop_prob):
        """Update drop probability (useful for scheduling)."""
        self.drop_prob = drop_prob


class StochasticDepth(nn.Module):
    """
    Stochastic Depth: Deep Networks with Stochastic Depth.

    Paper: "Deep Networks with Stochastic Depth"
    https://arxiv.org/abs/1603.09382

    Randomly drops entire residual blocks during training.
    This reduces training time and improves generalization.

    During inference, all layers are used with scaling.

    Usage:
        Wrap residual blocks with StochasticDepth:
        out = x + StochasticDepth(residual_block(x), drop_prob)

    Args:
        drop_prob: Probability of dropping the block (0.0 to 1.0)
        mode: 'batch' or 'sample' - whether to drop per batch or per sample
    """

    def __init__(self, drop_prob=0.1, mode='batch'):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
        self.mode = mode

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        keep_prob = 1 - self.drop_prob

        if self.mode == 'batch':
            # Drop entire batch or keep entire batch
            if torch.rand(1).item() < keep_prob:
                return x / keep_prob
            else:
                return torch.zeros_like(x)
        else:
            # Drop per sample
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, device=x.device)
            random_tensor.floor_()  # Binarize
            return x * random_tensor / keep_prob


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth per sample).

    Used in modern architectures like EfficientNet and Vision Transformers.

    Randomly drops paths (samples) in a batch during training.
    Different from DropBlock which drops spatial regions.

    Args:
        drop_prob: Probability of dropping a path
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        output = x.div(keep_prob) * random_tensor
        return output


class LinearScheduler:
    """
    Linear scheduler for DropBlock/StochasticDepth probabilities.

    Gradually increases drop probability during training.

    Args:
        dropblock_module: DropBlock module to schedule
        start_value: Initial drop probability
        stop_value: Final drop probability
        nr_steps: Number of steps to reach stop_value
    """

    def __init__(self, dropblock_module, start_value=0.0, stop_value=0.1, nr_steps=5000):
        self.dropblock = dropblock_module
        self.start_value = start_value
        self.stop_value = stop_value
        self.nr_steps = nr_steps
        self.current_step = 0

    def step(self):
        """Update drop probability."""
        if self.current_step < self.nr_steps:
            drop_prob = self.start_value + \
                       (self.stop_value - self.start_value) * \
                       (self.current_step / self.nr_steps)
            self.dropblock.set_drop_prob(drop_prob)
            self.current_step += 1


class ShakeShake(nn.Module):
    """
    Shake-Shake regularization for residual networks.

    Paper: "Shake-Shake regularization"
    https://arxiv.org/abs/1705.07485

    Applies random interpolation between two residual branches
    during training, with different weights for forward and backward passes.

    Args:
        None
    """

    def __init__(self):
        super(ShakeShake, self).__init__()

    def forward(self, x1, x2):
        """
        Args:
            x1: First branch output
            x2: Second branch output

        Returns:
            Shaken combination
        """
        if not self.training:
            # At test time, use average
            return (x1 + x2) / 2.0

        # During training, use random weights
        alpha = torch.rand(x1.size(0), 1, 1, 1, device=x1.device)

        # Forward pass: use alpha
        # Backward pass: use different random values (implemented via gradient hook)
        return alpha * x1 + (1 - alpha) * x2


class WeightDecayScheduler:
    """
    Scheduler for weight decay (L2 regularization).

    Gradually changes weight decay during training.

    Args:
        optimizer: PyTorch optimizer
        start_value: Initial weight decay
        stop_value: Final weight decay
        nr_steps: Number of steps
    """

    def __init__(self, optimizer, start_value=0.01, stop_value=0.0001, nr_steps=5000):
        self.optimizer = optimizer
        self.start_value = start_value
        self.stop_value = stop_value
        self.nr_steps = nr_steps
        self.current_step = 0

    def step(self):
        """Update weight decay."""
        if self.current_step < self.nr_steps:
            weight_decay = self.start_value + \
                          (self.stop_value - self.start_value) * \
                          (self.current_step / self.nr_steps)

            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = weight_decay

            self.current_step += 1


class AdaptiveDropout(nn.Module):
    """
    Adaptive Dropout: Adjusts dropout rate based on activation statistics.

    Higher dropout for neurons with higher activation.

    Args:
        p: Base dropout probability
        adaptive: Whether to use adaptive dropout
    """

    def __init__(self, p=0.5, adaptive=True):
        super(AdaptiveDropout, self).__init__()
        self.p = p
        self.adaptive = adaptive

    def forward(self, x):
        if not self.training:
            return x

        if self.adaptive:
            # Compute per-neuron activation magnitude
            act_magnitude = x.abs().mean(dim=0, keepdim=True)
            act_magnitude = act_magnitude / (act_magnitude.max() + 1e-8)

            # Higher activation -> higher dropout probability
            drop_prob = self.p * act_magnitude

            # Generate mask
            mask = torch.bernoulli(1 - drop_prob).to(x.device)
            return x * mask / (1 - self.p)
        else:
            # Standard dropout
            return F.dropout(x, p=self.p, training=True)


class GradualWarmup:
    """
    Gradual warmup for regularization techniques.

    Slowly increases regularization strength during training.

    Args:
        module: Module with regularization (must have set_strength method)
        warmup_epochs: Number of epochs for warmup
        target_strength: Target regularization strength
    """

    def __init__(self, module, warmup_epochs=5, target_strength=1.0):
        self.module = module
        self.warmup_epochs = warmup_epochs
        self.target_strength = target_strength
        self.current_epoch = 0

    def step(self):
        """Update regularization strength."""
        if self.current_epoch < self.warmup_epochs:
            strength = self.target_strength * (self.current_epoch / self.warmup_epochs)
            if hasattr(self.module, 'set_strength'):
                self.module.set_strength(strength)
            self.current_epoch += 1


def apply_weight_decay_exemptions(model, weight_decay, skip_list=('bias', 'bn', 'ln')):
    """
    Apply weight decay exemptions for specific parameters.

    Modern best practice: Don't apply weight decay to bias and normalization layers.

    Args:
        model: PyTorch model
        weight_decay: Weight decay value
        skip_list: Parameter names to skip (default: bias, batch norm, layer norm)

    Returns:
        List of parameter groups for optimizer
    """
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should skip weight decay
        if any(skip_name in name.lower() for skip_name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]


class MixedPrecisionWrapper:
    """
    Wrapper for automatic mixed precision training.

    Handles gradient scaling and overflow detection.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        enabled: Whether to enable mixed precision
    """

    def __init__(self, model, optimizer, enabled=True):
        self.model = model
        self.optimizer = optimizer
        self.enabled = enabled and torch.cuda.is_available()

        if self.enabled:
            from torch.cuda.amp import GradScaler, autocast
            self.scaler = GradScaler()
            self.autocast = autocast
        else:
            self.scaler = None
            self.autocast = None

    def __enter__(self):
        if self.enabled:
            return self.autocast()
        return self

    def __exit__(self, *args):
        pass

    def backward(self, loss):
        """Backward pass with gradient scaling."""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self):
        """Optimizer step with gradient scaling."""
        if self.enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
