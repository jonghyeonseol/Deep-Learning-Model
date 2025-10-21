"""
Retention Time Predictor Model

Predicts peptide/compound retention time from mass spectra.
Combines CNN and regression head for RT prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RetentionTimePredictor(nn.Module):
    """
    CNN-based retention time predictor.

    Predicts retention time (continuous value) from mass spectrum.
    """

    def __init__(
        self,
        input_dim: int = 19500,
        num_conv_layers: int = 4,
        conv_channels: list = [32, 64, 128, 256],
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        """
        Initialize RetentionTimePredictor.

        Args:
            input_dim: Input spectrum dimension
            num_conv_layers: Number of convolutional layers
            conv_channels: Output channels for each conv layer
            kernel_size: Convolutional kernel size
            dropout: Dropout probability
        """
        super(RetentionTimePredictor, self).__init__()

        self.input_dim = input_dim

        # Ensure conv_channels matches num_conv_layers
        if len(conv_channels) != num_conv_layers:
            conv_channels = [32 * (2 ** i) for i in range(num_conv_layers)]

        # Input layer
        self.input_conv = nn.Conv1d(1, conv_channels[0], kernel_size=1)
        self.input_bn = nn.BatchNorm1d(conv_channels[0])

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(num_conv_layers):
            in_channels = conv_channels[i]
            out_channels = conv_channels[i + 1] if i < num_conv_layers - 1 else conv_channels[i]

            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            self.bn_layers.append(nn.BatchNorm1d(out_channels))

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(64)

        # Regression head
        fc_input_dim = conv_channels[-1] * 64
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, 1)  # Single RT value

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
            Predicted retention time (batch_size, 1)
        """
        # Reshape
        x = x.unsqueeze(1)

        # Input layer
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)

        # Convolutional layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

            # Pooling every 2 layers
            if (i + 1) % 2 == 0:
                x = self.pool(x)

            x = self.dropout(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Regression head
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output RT
        x = self.output(x)

        return x


class MultiTaskSpectrumModel(nn.Module):
    """
    Multi-task model for joint classification and RT prediction.

    Shares feature extraction layers, with separate heads for
    classification and RT regression.
    """

    def __init__(
        self,
        input_dim: int = 19500,
        num_classes: int = 10,
        num_conv_layers: int = 5,
        conv_channels: list = [32, 64, 128, 256, 512],
        kernel_size: int = 5,
        dropout: float = 0.3,
    ):
        """
        Initialize MultiTaskSpectrumModel.

        Args:
            input_dim: Input spectrum dimension
            num_classes: Number of classes for classification
            num_conv_layers: Number of convolutional layers
            conv_channels: Output channels for each conv layer
            kernel_size: Convolutional kernel size
            dropout: Dropout probability
        """
        super(MultiTaskSpectrumModel, self).__init__()

        if len(conv_channels) != num_conv_layers:
            conv_channels = [32 * (2 ** i) for i in range(num_conv_layers)]
            conv_channels = [min(c, 512) for c in conv_channels]

        # Shared feature extraction
        self.input_conv = nn.Conv1d(1, conv_channels[0], kernel_size=1)
        self.input_bn = nn.BatchNorm1d(conv_channels[0])

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(num_conv_layers):
            in_channels = conv_channels[i]
            out_channels = conv_channels[i + 1] if i < num_conv_layers - 1 else conv_channels[i]

            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            self.bn_layers.append(nn.BatchNorm1d(out_channels))

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)

        # Shared FC layers
        fc_input_dim = conv_channels[-1] * 128
        self.shared_fc = nn.Linear(fc_input_dim, 512)
        self.shared_bn = nn.BatchNorm1d(512)

        # Classification head
        self.class_fc1 = nn.Linear(512, 256)
        self.class_fc1_bn = nn.BatchNorm1d(256)
        self.class_output = nn.Linear(256, num_classes)

        # Regression head (RT prediction)
        self.reg_fc1 = nn.Linear(512, 256)
        self.reg_fc1_bn = nn.BatchNorm1d(256)
        self.reg_output = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
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

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x: Input spectrum tensor (batch_size, input_dim)
            return_features: If True, return shared features

        Returns:
            If return_features is False: (class_logits, rt_prediction)
            If return_features is True: (class_logits, rt_prediction, features)
        """
        # Reshape
        x = x.unsqueeze(1)

        # Input layer
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)

        # Convolutional layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

            if (i + 1) % 2 == 0:
                x = self.pool(x)

            x = self.dropout(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Shared FC
        x = self.shared_fc(x)
        x = self.shared_bn(x)
        x = F.relu(x)
        shared_features = x

        # Classification head
        class_x = self.class_fc1(x)
        class_x = self.class_fc1_bn(class_x)
        class_x = F.relu(class_x)
        class_x = self.dropout(class_x)
        class_logits = self.class_output(class_x)

        # Regression head
        reg_x = self.reg_fc1(x)
        reg_x = self.reg_fc1_bn(reg_x)
        reg_x = F.relu(reg_x)
        reg_x = self.dropout(reg_x)
        rt_prediction = self.reg_output(reg_x)

        if return_features:
            return class_logits, rt_prediction, shared_features
        else:
            return class_logits, rt_prediction


# Example usage
if __name__ == "__main__":
    # Test RetentionTimePredictor
    print("Testing RetentionTimePredictor...")
    rt_model = RetentionTimePredictor(
        input_dim=19500,
        num_conv_layers=4,
        conv_channels=[32, 64, 128, 256]
    )

    batch_size = 8
    x = torch.randn(batch_size, 19500)
    rt_pred = rt_model(x)
    print(f"Input shape: {x.shape}")
    print(f"RT prediction shape: {rt_pred.shape}")
    print(f"RT predictions: {rt_pred.squeeze().tolist()}")

    num_params = sum(p.numel() for p in rt_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Test MultiTaskSpectrumModel
    print("\nTesting MultiTaskSpectrumModel...")
    mt_model = MultiTaskSpectrumModel(
        input_dim=19500,
        num_classes=10,
        num_conv_layers=5
    )

    class_logits, rt_pred = mt_model(x)
    print(f"Classification output shape: {class_logits.shape}")
    print(f"RT prediction shape: {rt_pred.shape}")

    num_params = sum(p.numel() for p in mt_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
