"""
EfficientNet-style architecture with depthwise separable convolutions and SE attention.

This implementation demonstrates:
- Depthwise Separable Convolutions (MobileNet-style) for efficiency
- Squeeze-and-Excitation (SE) blocks for channel attention
- Inverted residual blocks (MobileNetV2-style)
- Compound scaling (depth, width, resolution)
"""

import torch
import torch.nn as nn
import math
from .activations import get_activation


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention.

    The SE block adaptively recalibrates channel-wise feature responses.

    Architecture:
        Input -> Global Avg Pool -> FC (reduce) -> Activation -> FC (expand) -> Sigmoid -> Scale Input

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default 4)
    """

    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()

        reduced_channels = max(1, channels // reduction)

        # Squeeze: Global average pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation: Two FC layers with bottleneck
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # Squeeze: [B, C, H, W] -> [B, C, 1, 1]
        squeeze = self.squeeze(x).view(batch_size, channels)

        # Excitation: [B, C] -> [B, C]
        excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)

        # Scale: Element-wise multiplication
        return x * excitation


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution (MobileNet-style).

    Splits standard convolution into:
    1. Depthwise convolution: Applies a single filter per input channel
    2. Pointwise convolution: 1x1 conv to combine channels

    This reduces parameters and computation significantly:
    - Standard conv: H * W * C_in * C_out
    - Depthwise separable: H * W * C_in + C_in * C_out

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depthwise conv
        stride: Stride for depthwise conv
        activation: Activation function name
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(DepthwiseSeparableConv, self).__init__()

        padding = (kernel_size - 1) // 2

        # Depthwise convolution: each input channel is convolved separately
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.activation1 = get_activation(activation)

        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = get_activation(activation)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation2(x)

        return x


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block (MBConv).

    Used in MobileNetV2 and EfficientNet.

    Architecture:
        Input -> Expand (1x1) -> Depthwise (3x3/5x5) -> SE -> Project (1x1) -> (+) -> Output
        |                                                                        ^
        |________________________________________________________________________|
                                    (skip connection if applicable)

    Key features:
    - Inverted residual: Expands channels, then compresses
    - Depthwise convolution for efficiency
    - Squeeze-and-Excitation for attention
    - Skip connection when input/output dims match

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depthwise conv (3 or 5)
        stride: Stride for depthwise conv
        expand_ratio: Channel expansion ratio (typically 6)
        se_ratio: SE reduction ratio (typically 0.25)
        activation: Activation function name
        dropout_rate: Stochastic depth dropout rate
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 expand_ratio=6, se_ratio=0.25, activation='swish', dropout_rate=0.2):
        super(MBConvBlock, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_channels = int(in_channels * expand_ratio)
        self.expand = expand_ratio != 1

        layers = []

        # Expansion phase (if needed)
        if self.expand:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                get_activation(activation)
            ])

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            get_activation(activation)
        ])

        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.append(SqueezeExcitation(hidden_channels, reduction=int(1/se_ratio)))

        # Projection phase
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        if self.use_residual:
            return x + self.dropout(self.conv(x))
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    """
    EfficientNet-style architecture for CIFAR-10.

    EfficientNet uses compound scaling to balance network depth, width, and resolution.

    Features:
    - MBConv blocks with inverted residuals
    - Depthwise separable convolutions
    - Squeeze-and-Excitation attention
    - Efficient parameter usage

    Args:
        width_multiplier: Scale for channel dimensions (default 1.0)
        depth_multiplier: Scale for network depth (default 1.0)
        num_classes: Number of output classes
        activation: Activation function (typically 'swish' for EfficientNet)
        dropout_rate: Dropout rate
    """

    def __init__(self, width_multiplier=1.0, depth_multiplier=1.0, num_classes=10,
                 activation='swish', dropout_rate=0.2):
        super(EfficientNet, self).__init__()

        self.num_classes = num_classes
        self.activation_name = activation

        def round_channels(channels, multiplier=1.0, divisor=8):
            """Round number of channels to nearest divisor."""
            channels *= multiplier
            new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
            if new_channels < 0.9 * channels:
                new_channels += divisor
            return int(new_channels)

        def round_repeats(repeats, multiplier=1.0):
            """Round number of block repeats."""
            return int(math.ceil(multiplier * repeats))

        # Initial stem
        out_channels = round_channels(32, width_multiplier)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(activation)
        )

        # MBConv blocks configuration: [expand_ratio, channels, repeats, stride, kernel_size]
        blocks_config = [
            [1, 16, 1, 1, 3],   # Stage 1
            [6, 24, 2, 2, 3],   # Stage 2
            [6, 40, 2, 2, 5],   # Stage 3
            [6, 80, 3, 2, 3],   # Stage 4
            [6, 112, 3, 1, 5],  # Stage 5
            [6, 192, 4, 2, 5],  # Stage 6
            [6, 320, 1, 1, 3],  # Stage 7
        ]

        # Build blocks
        in_channels = out_channels
        stages = []

        for expand_ratio, channels, repeats, stride, kernel_size in blocks_config:
            out_channels = round_channels(channels, width_multiplier)
            num_repeats = round_repeats(repeats, depth_multiplier)

            for i in range(num_repeats):
                stages.append(MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    activation=activation,
                    dropout_rate=dropout_rate
                ))
                in_channels = out_channels

        self.stages = nn.Sequential(*stages)

        # Head
        final_channels = round_channels(1280, width_multiplier)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
            get_activation(activation),
            nn.AdaptiveAvgPool2d(1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(final_channels, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
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
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print a summary of the network architecture."""
        print(f"EfficientNet Summary:")
        print(f"Number of classes: {self.num_classes}")
        print(f"Activation: {self.activation_name}")
        print(f"Total parameters: {self.get_num_parameters():,}")


# Predefined EfficientNet variants
def EfficientNet_B0(num_classes=10, activation='swish', dropout_rate=0.2):
    """EfficientNet-B0: Baseline model"""
    return EfficientNet(width_multiplier=1.0, depth_multiplier=1.0,
                       num_classes=num_classes, activation=activation, dropout_rate=dropout_rate)


def EfficientNet_B1(num_classes=10, activation='swish', dropout_rate=0.2):
    """EfficientNet-B1: Wider and deeper"""
    return EfficientNet(width_multiplier=1.0, depth_multiplier=1.1,
                       num_classes=num_classes, activation=activation, dropout_rate=dropout_rate)


def EfficientNet_Tiny(num_classes=10, activation='swish', dropout_rate=0.2):
    """Tiny EfficientNet for quick experiments"""
    return EfficientNet(width_multiplier=0.5, depth_multiplier=0.5,
                       num_classes=num_classes, activation=activation, dropout_rate=dropout_rate)
