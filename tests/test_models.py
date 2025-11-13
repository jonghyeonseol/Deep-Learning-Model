"""
Unit tests for model architectures.

Tests cover:
- Output shape correctness
- Forward pass functionality
- Parameter count validation
- Model initialization
- Different input sizes
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network import ConvNeuralNetwork, NeuralNetwork
from models.resnet import ResNet18, ResNet34, ResNet_Tiny
from models.efficientnet import EfficientNet_B0, EfficientNet_Tiny
from models.cnn_transformer import VisionTransformer_Tiny


class TestBasicModels(unittest.TestCase):
    """Test suite for basic neural network models."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.num_classes = 10

    def test_neural_network_forward(self):
        """Test NeuralNetwork forward pass."""
        model = NeuralNetwork(
            input_size=784,
            hidden_sizes=[128, 64],
            output_size=self.num_classes,
            activation='relu'
        )
        x = torch.randn(self.batch_size, 784)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_conv_neural_network_cifar10(self):
        """Test ConvNeuralNetwork with CIFAR-10 input size."""
        model = ConvNeuralNetwork(
            input_channels=3,
            num_classes=self.num_classes,
            activation='relu'
        )
        x = torch.randn(self.batch_size, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_conv_network_parameters(self):
        """Test ConvNeuralNetwork has trainable parameters."""
        model = ConvNeuralNetwork(num_classes=self.num_classes)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(num_params, 0)
        self.assertLess(num_params, 10_000_000)  # Reasonable upper bound

    def test_conv_network_different_activations(self):
        """Test ConvNeuralNetwork with different activation functions."""
        activations = ['relu', 'gelu', 'swish', 'mish']
        x = torch.randn(2, 3, 32, 32)

        for activation in activations:
            with self.subTest(activation=activation):
                model = ConvNeuralNetwork(activation=activation)
                output = model(x)
                self.assertEqual(output.shape, (2, 10))
                self.assertFalse(torch.any(torch.isnan(output)))


class TestResNetModels(unittest.TestCase):
    """Test suite for ResNet architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.num_classes = 10
        self.input_shape = (self.batch_size, 3, 32, 32)

    def test_resnet18_output_shape(self):
        """Test ResNet-18 output shape."""
        model = ResNet18(num_classes=self.num_classes)
        x = torch.randn(*self.input_shape)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_resnet34_output_shape(self):
        """Test ResNet-34 output shape."""
        model = ResNet34(num_classes=self.num_classes)
        x = torch.randn(*self.input_shape)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_resnet_tiny_output_shape(self):
        """Test ResNet-Tiny output shape."""
        model = ResNet_Tiny(num_classes=self.num_classes)
        x = torch.randn(*self.input_shape)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_resnet18_parameter_count(self):
        """Test ResNet-18 has expected parameter count."""
        model = ResNet18(num_classes=self.num_classes)
        num_params = sum(p.numel() for p in model.parameters())
        # ResNet-18 should have ~11M parameters
        self.assertGreater(num_params, 10_000_000)
        self.assertLess(num_params, 15_000_000)

    def test_resnet_different_activations(self):
        """Test ResNet with different activation functions."""
        activations = ['relu', 'gelu', 'swish']
        x = torch.randn(2, 3, 32, 32)

        for activation in activations:
            with self.subTest(activation=activation):
                model = ResNet_Tiny(activation=activation)
                output = model(x)
                self.assertEqual(output.shape, (2, 10))

    def test_resnet_gradient_flow(self):
        """Test ResNet allows gradient flow."""
        model = ResNet_Tiny()
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestEfficientNetModels(unittest.TestCase):
    """Test suite for EfficientNet architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.num_classes = 10
        self.input_shape = (self.batch_size, 3, 32, 32)

    def test_efficientnet_b0_output_shape(self):
        """Test EfficientNet-B0 output shape."""
        model = EfficientNet_B0(num_classes=self.num_classes)
        x = torch.randn(*self.input_shape)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_efficientnet_tiny_output_shape(self):
        """Test EfficientNet-Tiny output shape."""
        model = EfficientNet_Tiny(num_classes=self.num_classes)
        x = torch.randn(*self.input_shape)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_efficientnet_parameter_efficiency(self):
        """Test EfficientNet has fewer parameters than ResNet."""
        efficientnet = EfficientNet_B0(num_classes=self.num_classes)
        resnet = ResNet18(num_classes=self.num_classes)

        eff_params = sum(p.numel() for p in efficientnet.parameters())
        resnet_params = sum(p.numel() for p in resnet.parameters())

        # EfficientNet should be more parameter-efficient
        self.assertLess(eff_params, resnet_params)

    def test_efficientnet_no_nan_output(self):
        """Test EfficientNet produces valid outputs."""
        model = EfficientNet_Tiny()
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        self.assertFalse(torch.any(torch.isnan(output)))
        self.assertFalse(torch.any(torch.isinf(output)))


class TestTransformerModels(unittest.TestCase):
    """Test suite for Transformer-based architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.num_classes = 10
        self.input_shape = (self.batch_size, 3, 32, 32)

    def test_vision_transformer_output_shape(self):
        """Test Vision Transformer output shape."""
        model = VisionTransformer_Tiny(num_classes=self.num_classes)
        x = torch.randn(*self.input_shape)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_vision_transformer_parameters(self):
        """Test Vision Transformer has reasonable parameter count."""
        model = VisionTransformer_Tiny()
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 1_000_000)
        self.assertLess(num_params, 20_000_000)

    def test_vision_transformer_gradient_flow(self):
        """Test Vision Transformer allows gradient flow."""
        model = VisionTransformer_Tiny()
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestModelCompatibility(unittest.TestCase):
    """Test cross-model compatibility and consistency."""

    def test_all_models_same_output_shape(self):
        """Test all models produce same output shape for CIFAR-10."""
        models = [
            ConvNeuralNetwork(num_classes=10),
            ResNet_Tiny(num_classes=10),
            EfficientNet_Tiny(num_classes=10),
            VisionTransformer_Tiny(num_classes=10),
        ]

        x = torch.randn(2, 3, 32, 32)
        expected_shape = (2, 10)

        for model in models:
            with self.subTest(model=model.__class__.__name__):
                output = model(x)
                self.assertEqual(output.shape, expected_shape)

    def test_models_eval_mode(self):
        """Test models can switch to evaluation mode."""
        models = [
            ConvNeuralNetwork(),
            ResNet_Tiny(),
            EfficientNet_Tiny(),
        ]

        for model in models:
            with self.subTest(model=model.__class__.__name__):
                model.eval()
                self.assertFalse(model.training)

    def test_models_train_mode(self):
        """Test models can switch to training mode."""
        models = [
            ConvNeuralNetwork(),
            ResNet_Tiny(),
            EfficientNet_Tiny(),
        ]

        for model in models:
            with self.subTest(model=model.__class__.__name__):
                model.train()
                self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
