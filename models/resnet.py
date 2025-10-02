"""
Modern ResNet-style architecture with residual blocks and batch normalization.

This implementation demonstrates:
- Residual connections (skip connections) for training deeper networks
- Batch normalization for faster convergence and better generalization
- Bottleneck blocks for efficiency
- Global average pooling instead of fully connected layers
"""

import torch
import torch.nn as nn
from .activations import get_activation


class BasicBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions.

    Architecture:
        x -> Conv3x3 -> BN -> Activation -> Conv3x3 -> BN -> (+) -> Activation
        |                                                      ^
        |______________________________________________________|
                            (skip connection)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, activation='relu', dropout_rate=0.0):
        super(BasicBlock, self).__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = get_activation(activation)

        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If dimensions change, use 1x1 conv to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.activation2 = get_activation(activation)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out += identity
        out = self.activation2(out)

        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block with 1x1 -> 3x3 -> 1x1 convolutions.

    More efficient than BasicBlock for deeper networks.
    Uses 1x1 convolutions to reduce/expand channels (bottleneck design).

    Architecture:
        x -> Conv1x1 -> BN -> Act -> Conv3x3 -> BN -> Act -> Conv1x1 -> BN -> (+) -> Act
        |                                                                        ^
        |________________________________________________________________________|
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, activation='relu', dropout_rate=0.0):
        super(BottleneckBlock, self).__init__()

        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv for feature extraction
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv to expand dimensions
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)

        out = self.activation(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        out = self.bn3(self.conv3(out))

        # Add skip connection
        out += identity
        out = self.activation(out)

        return out


class ModernResNet(nn.Module):
    """
    Modern ResNet architecture for CIFAR-10.

    Features:
    - Residual blocks with skip connections
    - Batch normalization for stable training
    - Global average pooling instead of FC layers (reduces parameters)
    - Flexible depth and width

    Args:
        block: Block type (BasicBlock or BottleneckBlock)
        num_blocks: List of number of blocks in each stage
        num_classes: Number of output classes
        activation: Activation function name
        dropout_rate: Dropout rate for regularization
        width_multiplier: Channel width multiplier (default 1.0)
    """

    def __init__(self, block, num_blocks, num_classes=10, activation='relu',
                 dropout_rate=0.2, width_multiplier=1.0):
        super(ModernResNet, self).__init__()

        self.in_channels = int(64 * width_multiplier)
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Initial convolution layer (for CIFAR-10, we use 3x3 instead of 7x7)
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.activation = get_activation(activation)

        # Residual stages
        self.stage1 = self._make_stage(block, int(64 * width_multiplier),
                                        num_blocks[0], stride=1, activation=activation)
        self.stage2 = self._make_stage(block, int(128 * width_multiplier),
                                        num_blocks[1], stride=2, activation=activation)
        self.stage3 = self._make_stage(block, int(256 * width_multiplier),
                                        num_blocks[2], stride=2, activation=activation)
        self.stage4 = self._make_stage(block, int(512 * width_multiplier),
                                        num_blocks[3], stride=2, activation=activation)

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier
        self.fc = nn.Linear(int(512 * width_multiplier) * block.expansion, num_classes)

        self._initialize_weights()

    def _make_stage(self, block, out_channels, num_blocks, stride, activation):
        """Create a stage with multiple residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride,
                               activation, self.dropout_rate))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using He initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv
        out = self.activation(self.bn1(self.conv1(x)))

        # Residual stages
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # Global average pooling
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        # Classifier
        out = self.fc(out)

        return out

    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print a summary of the network architecture."""
        print(f"Modern ResNet Summary:")
        print(f"Number of classes: {self.num_classes}")
        print(f"Activation: {self.activation_name}")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Total parameters: {self.get_num_parameters():,}")


# Predefined ResNet architectures
def ResNet18(num_classes=10, activation='relu', dropout_rate=0.2):
    """ResNet-18 (11M parameters)"""
    return ModernResNet(BasicBlock, [2, 2, 2, 2], num_classes, activation, dropout_rate)


def ResNet34(num_classes=10, activation='relu', dropout_rate=0.2):
    """ResNet-34 (21M parameters)"""
    return ModernResNet(BasicBlock, [3, 4, 6, 3], num_classes, activation, dropout_rate)


def ResNet50(num_classes=10, activation='relu', dropout_rate=0.2):
    """ResNet-50 with bottleneck blocks (23M parameters)"""
    return ModernResNet(BottleneckBlock, [3, 4, 6, 3], num_classes, activation, dropout_rate)


def ResNet101(num_classes=10, activation='relu', dropout_rate=0.2):
    """ResNet-101 with bottleneck blocks (42M parameters)"""
    return ModernResNet(BottleneckBlock, [3, 4, 23, 3], num_classes, activation, dropout_rate)


def ResNet_Tiny(num_classes=10, activation='relu', dropout_rate=0.2):
    """Tiny ResNet for quick experiments (fewer parameters)"""
    return ModernResNet(BasicBlock, [1, 1, 1, 1], num_classes, activation,
                       dropout_rate, width_multiplier=0.5)
