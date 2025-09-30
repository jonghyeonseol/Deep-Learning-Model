import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class Step(nn.Module):
    def __init__(self, threshold=0.0):
        super(Step, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return (x > self.threshold).float()


class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class ELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.elu(x, alpha=self.alpha)


class PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.full((num_parameters,), init))

    def forward(self, x):
        return F.prelu(x, self.weight)


class SELU(nn.Module):
    def __init__(self):
        super(SELU, self).__init__()

    def forward(self, x):
        return F.selu(x)


class Hardswish(nn.Module):
    def __init__(self):
        super(Hardswish, self).__init__()

    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return F.silu(x)


def get_activation(activation_name):
    """
    Factory function to get activation function by name.

    Args:
        activation_name (str): Name of the activation function

    Returns:
        nn.Module: The corresponding activation function
    """
    activation_map = {
        'gelu': GELU(),
        'relu': ReLU(),
        'tanh': Tanh(),
        'sigmoid': Sigmoid(),
        'step': Step(),
        'softmax': Softmax(),
        'swish': Swish(),
        'mish': Mish(),
        'leakyrelu': LeakyReLU(),
        'elu': ELU(),
        'prelu': PReLU(),
        'selu': SELU(),
        'hardswish': Hardswish(),
        'silu': SiLU()
    }

    if activation_name.lower() not in activation_map:
        raise ValueError(f"Unsupported activation function: {activation_name}. "
                        f"Supported functions: {list(activation_map.keys())}")

    return activation_map[activation_name.lower()]


def get_available_activations():
    """
    Get list of all available activation functions.

    Returns:
        list: List of available activation function names
    """
    return ['gelu', 'relu', 'tanh', 'sigmoid', 'step', 'softmax', 'swish',
            'mish', 'leakyrelu', 'elu', 'prelu', 'selu', 'hardswish', 'silu']