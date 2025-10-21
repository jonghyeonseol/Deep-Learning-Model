"""
Spectrum CNN Model

1D Convolutional Neural Network for mass spectrum analysis.
Designed for classification and regression tasks on binned spectra.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectrumCNN(nn.Module):
    """
    1D CNN for mass spectrum classification/regression.

    Architecture:
    - Multiple 1D convolutional layers with residual connections
    - Batch normalization and dropout for regularization
    - Adaptive pooling before FC layers
    - Flexible output for classification or regression
    """

    def __init__(
        self,
        input_dim: int = 19500,  # Number of m/z bins
        num_classes: int = 10,  # For classification
        task: str = "classification",  # 'classification' or 'regression'
        num_conv_layers: int = 5,
        conv_channels: list = [32, 64, 128, 256, 512],
        kernel_size: int = 5,
        dropout: float = 0.3,
        use_residual: bool = True,
    ):
        """
        Initialize SpectrumCNN.

        Args:
            input_dim: Input spectrum dimension (number of bins)
            num_classes: Number of output classes (for classification)
            task: Task type ('classification' or 'regression')
            num_conv_layers: Number of convolutional layers
            conv_channels: List of output channels for each conv layer
            kernel_size: Convolutional kernel size
            dropout: Dropout probability
            use_residual: Use residual connections
        """
        super(SpectrumCNN, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task = task
        self.use_residual = use_residual

        # Ensure conv_channels matches num_conv_layers
        if len(conv_channels) != num_conv_layers:
            # Auto-generate channels
            conv_channels = [32 * (2 ** i) for i in range(num_conv_layers)]
            conv_channels = [min(c, 512) for c in conv_channels]  # Cap at 512

        # Input layer (expand to multi-channel)
        self.input_conv = nn.Conv1d(1, conv_channels[0], kernel_size=1)
        self.input_bn = nn.BatchNorm1d(conv_channels[0])

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.residual_convs = nn.ModuleList()  # For residual connections

        for i in range(num_conv_layers):
            in_channels = conv_channels[i]
            out_channels = conv_channels[i + 1] if i < num_conv_layers - 1 else conv_channels[i]

            # Main conv layer
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False
                )
            )

            # Batch norm
            self.bn_layers.append(nn.BatchNorm1d(out_channels))

            # Residual connection (1x1 conv if channel size changes)
            if use_residual and in_channels != out_channels:
                self.residual_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            else:
                self.residual_convs.append(None)

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Adaptive pooling to fixed size before FC
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)

        # Fully connected layers
        fc_input_dim = conv_channels[-1] * 128
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)

        # Output layer
        if task == "classification":
            self.output = nn.Linear(256, num_classes)
        else:  # regression
            self.output = nn.Linear(256, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input spectrum tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, num_classes) for classification
            or (batch_size, 1) for regression
        """
        # Reshape: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)

        # Input layer
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)

        # Convolutional layers
        for i, (conv, bn, res_conv) in enumerate(zip(self.conv_layers, self.bn_layers, self.residual_convs)):
            identity = x

            # Convolution
            x = conv(x)
            x = bn(x)

            # Residual connection
            if self.use_residual:
                if res_conv is not None:
                    identity = res_conv(identity)
                x = x + identity

            x = F.relu(x)

            # Pooling every 2 layers
            if (i + 1) % 2 == 0:
                x = self.pool(x)

            x = self.dropout(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output
        x = self.output(x)

        return x

    def get_feature_embeddings(self, x):
        """
        Extract feature embeddings before output layer.

        Args:
            x: Input spectrum tensor

        Returns:
            Feature embeddings (batch_size, 256)
        """
        # Same as forward, but stop before output layer
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)

        for i, (conv, bn, res_conv) in enumerate(zip(self.conv_layers, self.bn_layers, self.residual_convs)):
            identity = x
            x = conv(x)
            x = bn(x)
            if self.use_residual:
                if res_conv is not None:
                    identity = res_conv(identity)
                x = x + identity
            x = F.relu(x)
            if (i + 1) % 2 == 0:
                x = self.pool(x)
            x = self.dropout(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)

        return x


class LightweightSpectrumCNN(nn.Module):
    """
    Lightweight 1D CNN for faster training on smaller datasets.

    Fewer layers and parameters than SpectrumCNN.
    """

    def __init__(
        self,
        input_dim: int = 19500,
        num_classes: int = 10,
        task: str = "classification",
        dropout: float = 0.3,
    ):
        """Initialize LightweightSpectrumCNN."""
        super(LightweightSpectrumCNN, self).__init__()

        self.task = task

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)

        # FC layers
        self.fc1 = nn.Linear(128 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes if task == "classification" else 1)

    def forward(self, x):
        """Forward pass."""
        x = x.unsqueeze(1)

        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Example usage
if __name__ == "__main__":
    # Create model
    model = SpectrumCNN(
        input_dim=19500,
        num_classes=10,
        task="classification",
        num_conv_layers=5,
        dropout=0.3
    )

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 19500)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Get embeddings
    embeddings = model.get_feature_embeddings(x)
    print(f"Embedding shape: {embeddings.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
