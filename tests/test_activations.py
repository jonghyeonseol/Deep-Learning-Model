"""
Unit tests for activation functions.

Tests cover:
- Output shape preservation
- Value ranges
- Gradient computation
- Edge cases (zeros, negatives, large values)
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.activations import (
    GELU, ReLU, Tanh, Sigmoid, Swish, Mish,
    LeakyReLU, ELU, PReLU, SELU, Hardswish, SiLU,
    get_activation, get_available_activations
)


class TestActivationFunctions(unittest.TestCase):
    """Test suite for activation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.features = 128
        self.test_input = torch.randn(self.batch_size, self.features)

    def test_gelu_output_shape(self):
        """Test GELU maintains input shape."""
        gelu = GELU()
        output = gelu(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_gelu_values(self):
        """Test GELU produces reasonable values."""
        gelu = GELU()
        x = torch.tensor([0.0, 1.0, -1.0])
        output = gelu(x)
        # GELU(0) ≈ 0, GELU(1) ≈ 0.84, GELU(-1) ≈ -0.16
        self.assertAlmostEqual(output[0].item(), 0.0, places=2)
        self.assertGreater(output[1].item(), 0.8)
        self.assertLess(output[2].item(), 0.0)

    def test_relu_output_shape(self):
        """Test ReLU maintains input shape."""
        relu = ReLU()
        output = relu(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_relu_non_negative(self):
        """Test ReLU outputs are non-negative."""
        relu = ReLU()
        output = relu(self.test_input)
        self.assertTrue(torch.all(output >= 0))

    def test_relu_zero_for_negatives(self):
        """Test ReLU zeros out negative values."""
        relu = ReLU()
        x = torch.tensor([-1.0, -2.0, -3.0])
        output = relu(x)
        self.assertTrue(torch.all(output == 0))

    def test_sigmoid_output_range(self):
        """Test Sigmoid outputs are in (0, 1)."""
        sigmoid = Sigmoid()
        output = sigmoid(self.test_input)
        self.assertTrue(torch.all(output > 0))
        self.assertTrue(torch.all(output < 1))

    def test_tanh_output_range(self):
        """Test Tanh outputs are in (-1, 1)."""
        tanh = Tanh()
        output = tanh(self.test_input)
        self.assertTrue(torch.all(output > -1))
        self.assertTrue(torch.all(output < 1))

    def test_swish_output_shape(self):
        """Test Swish maintains input shape."""
        swish = Swish()
        output = swish(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_mish_output_shape(self):
        """Test Mish maintains input shape."""
        mish = Mish()
        output = mish(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_leaky_relu_negative_slope(self):
        """Test LeakyReLU preserves some negative values."""
        leaky_relu = LeakyReLU(negative_slope=0.01)
        x = torch.tensor([-1.0])
        output = leaky_relu(x)
        self.assertAlmostEqual(output.item(), -0.01, places=2)

    def test_elu_output_shape(self):
        """Test ELU maintains input shape."""
        elu = ELU()
        output = elu(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_prelu_learnable_parameters(self):
        """Test PReLU has learnable parameters."""
        prelu = PReLU(num_parameters=1)
        self.assertEqual(len(list(prelu.parameters())), 1)

    def test_selu_output_shape(self):
        """Test SELU maintains input shape."""
        selu = SELU()
        output = selu(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_hardswish_output_shape(self):
        """Test Hardswish maintains input shape."""
        hardswish = Hardswish()
        output = hardswish(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_silu_output_shape(self):
        """Test SiLU maintains input shape."""
        silu = SiLU()
        output = silu(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_get_activation_factory(self):
        """Test activation factory function."""
        relu = get_activation('relu')
        self.assertIsInstance(relu, ReLU)

        gelu = get_activation('gelu')
        self.assertIsInstance(gelu, GELU)

    def test_get_activation_case_insensitive(self):
        """Test activation factory is case-insensitive."""
        relu_lower = get_activation('relu')
        relu_upper = get_activation('RELU')
        relu_mixed = get_activation('ReLU')

        self.assertIsInstance(relu_lower, ReLU)
        self.assertIsInstance(relu_upper, ReLU)
        self.assertIsInstance(relu_mixed, ReLU)

    def test_get_activation_invalid(self):
        """Test activation factory raises error for invalid name."""
        with self.assertRaises(ValueError):
            get_activation('invalid_activation')

    def test_get_available_activations(self):
        """Test getting list of available activations."""
        activations = get_available_activations()
        self.assertIsInstance(activations, list)
        self.assertGreater(len(activations), 0)
        self.assertIn('relu', activations)
        self.assertIn('gelu', activations)

    def test_gradient_flow(self):
        """Test activations allow gradient flow."""
        for name in ['relu', 'gelu', 'swish']:
            activation = get_activation(name)
            x = torch.randn(10, requires_grad=True)
            output = activation(x)
            loss = output.sum()
            loss.backward()
            self.assertIsNotNone(x.grad)

    def test_zero_input(self):
        """Test activations handle zero input."""
        zeros = torch.zeros(self.batch_size, self.features)
        for name in ['relu', 'gelu', 'swish', 'mish']:
            activation = get_activation(name)
            output = activation(zeros)
            self.assertEqual(output.shape, zeros.shape)

    def test_large_values(self):
        """Test activations handle large values without NaN/Inf."""
        large_values = torch.tensor([100.0, -100.0, 1000.0])
        for name in ['relu', 'tanh', 'sigmoid']:
            activation = get_activation(name)
            output = activation(large_values)
            self.assertFalse(torch.any(torch.isnan(output)))
            self.assertFalse(torch.any(torch.isinf(output)))


if __name__ == '__main__':
    unittest.main()
