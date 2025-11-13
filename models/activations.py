import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

# Use absolute import - utils package should be in PYTHONPATH
try:
    from utils.exceptions import InvalidActivationError
except ImportError:
    # Fallback for direct script execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.exceptions import InvalidActivationError


class GELU(nn.Module):
    def __init__(self) -> None:
        super(GELU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ReLU(nn.Module):
    def __init__(self) -> None:
        super(ReLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.zeros_like(x), x)


class Tanh(nn.Module):
    def __init__(self) -> None:
        super(Tanh, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class Sigmoid(nn.Module):
    def __init__(self) -> None:
        super(Sigmoid, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class Step(nn.Module):
    def __init__(self, threshold: float = 0.0) -> None:
        super(Step, self).__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.threshold).float()


class Softmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __init__(self) -> None:
        super(Mish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class ELU(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha)


class PReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.full((num_parameters,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, self.weight)


class SELU(nn.Module):
    def __init__(self) -> None:
        super(SELU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.selu(x)


class Hardswish(nn.Module):
    def __init__(self) -> None:
        super(Hardswish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3) / 6


class SiLU(nn.Module):
    def __init__(self) -> None:
        super(SiLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


def get_activation(activation_name: str) -> nn.Module:
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
        available = list(activation_map.keys())
        raise InvalidActivationError(activation_name, available=available)

    return activation_map[activation_name.lower()]


def get_available_activations() -> List[str]:
    """
    Get list of all available activation functions.

    Returns:
        list: List of available activation function names
    """
    return ['gelu', 'relu', 'tanh', 'sigmoid', 'step', 'softmax', 'swish',
            'mish', 'leakyrelu', 'elu', 'prelu', 'selu', 'hardswish', 'silu']