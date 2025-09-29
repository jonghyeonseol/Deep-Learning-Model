import torch
import torch.nn as nn
import math


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return torch.maximum(torch.zeros_like(x), x)


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x)


def get_activation(activation_name):
    """
    Factory function to get activation function by name.

    Args:
        activation_name (str): Name of the activation function ('gelu', 'relu', 'tanh')

    Returns:
        nn.Module: The corresponding activation function
    """
    activation_map = {
        'gelu': GELU(),
        'relu': ReLU(),
        'tanh': Tanh()
    }

    if activation_name.lower() not in activation_map:
        raise ValueError(f"Unsupported activation function: {activation_name}. "
                        f"Supported functions: {list(activation_map.keys())}")

    return activation_map[activation_name.lower()]