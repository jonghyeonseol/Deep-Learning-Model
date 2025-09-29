import torch
import torch.nn as nn
from .activations import get_activation


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout_rate=0.2):
        """
        A flexible neural network implementation.

        Args:
            input_size (int): Size of input features
            hidden_sizes (list): List of hidden layer sizes
            output_size (int): Size of output layer
            activation (str): Activation function name ('gelu', 'relu', 'tanh')
            dropout_rate (float): Dropout rate for regularization
        """
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.dropout_rate = dropout_rate

        # Build layers
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer to first hidden layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Add linear layer
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add activation (except for output layer)
            if i < len(layer_sizes) - 2:
                self.activations.append(get_activation(activation))
                self.dropouts.append(nn.Dropout(dropout_rate))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Flatten input if needed (for images)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Pass through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)

        # Output layer (no activation, no dropout)
        x = self.layers[-1](x)

        return x

    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print a summary of the network architecture."""
        print(f"Neural Network Summary:")
        print(f"Input size: {self.input_size}")
        print(f"Hidden layers: {self.hidden_sizes}")
        print(f"Output size: {self.output_size}")
        print(f"Activation: {self.activation_name}")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Total parameters: {self.get_num_parameters():,}")

        print("\nLayer details:")
        for i, layer in enumerate(self.layers):
            if i == 0:
                print(f"Input -> Hidden_{i+1}: {layer.in_features} -> {layer.out_features}")
            elif i == len(self.layers) - 1:
                print(f"Hidden_{i} -> Output: {layer.in_features} -> {layer.out_features}")
            else:
                print(f"Hidden_{i} -> Hidden_{i+1}: {layer.in_features} -> {layer.out_features}")


class ConvNeuralNetwork(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, activation='relu', dropout_rate=0.2):
        """
        Convolutional Neural Network for image classification (e.g., CIFAR-10).

        Args:
            input_channels (int): Number of input channels (3 for RGB)
            num_classes (int): Number of output classes
            activation (str): Activation function name
            dropout_rate (float): Dropout rate
        """
        super(ConvNeuralNetwork, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.activation_name = activation
        self.dropout_rate = dropout_rate

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Activation functions
        self.activation = get_activation(activation)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        # For CIFAR-10 (32x32 images), after 3 pooling operations: 32/8 = 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers with pooling
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print a summary of the network architecture."""
        print(f"Convolutional Neural Network Summary:")
        print(f"Input channels: {self.input_channels}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Activation: {self.activation_name}")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Total parameters: {self.get_num_parameters():,}")