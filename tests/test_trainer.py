"""
Unit tests for Trainer class.

Tests cover:
- Initialization
- Checkpoint save/load
- Path validation and security
- Configuration methods
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trainer import Trainer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestTrainer(unittest.TestCase):
    """Test suite for Trainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.model = SimpleModel()
        self.train_data = [(torch.randn(4, 10), torch.randint(0, 5, (4,))) for _ in range(2)]
        self.train_loader = self.train_data

        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            save_dir=self.test_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_trainer_initialization(self):
        """Test Trainer initializes correctly."""
        self.assertIsNotNone(self.trainer)
        self.assertIsInstance(self.trainer.model, nn.Module)
        self.assertEqual(self.trainer.save_dir, self.test_dir)

    def test_trainer_device_selection(self):
        """Test Trainer selects appropriate device."""
        device = self.trainer.device
        self.assertIsInstance(device, torch.device)
        # Should be either cuda or cpu
        self.assertIn(device.type, ['cuda', 'cpu'])

    def test_trainer_history_initialization(self):
        """Test training history is initialized."""
        self.assertIsNotNone(self.trainer.history)
        self.assertEqual(len(self.trainer.history), 0)

    def test_configure_optimizer_adam(self):
        """Test configuring Adam optimizer."""
        self.trainer.configure_optimizer('adam', lr=0.001)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsInstance(self.trainer.optimizer, torch.optim.Adam)

    def test_configure_optimizer_sgd(self):
        """Test configuring SGD optimizer."""
        self.trainer.configure_optimizer('sgd', lr=0.01, momentum=0.9)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsInstance(self.trainer.optimizer, torch.optim.SGD)

    def test_configure_optimizer_invalid(self):
        """Test configuring invalid optimizer raises error."""
        with self.assertRaises(ValueError):
            self.trainer.configure_optimizer('invalid_optimizer')

    def test_configure_scheduler_step(self):
        """Test configuring step scheduler."""
        self.trainer.configure_optimizer('adam')
        self.trainer.configure_scheduler('step', step_size=10)
        self.assertIsNotNone(self.trainer.scheduler)

    def test_configure_scheduler_before_optimizer(self):
        """Test configuring scheduler before optimizer raises error."""
        with self.assertRaises(ValueError):
            self.trainer.configure_scheduler('step')

    def test_configure_criterion_crossentropy(self):
        """Test configuring cross entropy loss."""
        self.trainer.configure_criterion('crossentropy')
        self.assertIsNotNone(self.trainer.criterion)
        self.assertIsInstance(self.trainer.criterion, nn.CrossEntropyLoss)

    def test_configure_criterion_mse(self):
        """Test configuring MSE loss."""
        self.trainer.configure_criterion('mse')
        self.assertIsNotNone(self.trainer.criterion)
        self.assertIsInstance(self.trainer.criterion, nn.MSELoss)

    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        self.trainer.configure_optimizer('adam')
        self.trainer.save_checkpoint('test_checkpoint.pth')

        checkpoint_path = os.path.join(self.test_dir, 'test_checkpoint.pth')
        self.assertTrue(os.path.exists(checkpoint_path))

    def test_save_checkpoint_path_sanitization(self):
        """Test checkpoint save sanitizes path."""
        self.trainer.configure_optimizer('adam')
        # Should remove directory traversal
        self.trainer.save_checkpoint('../malicious/path.pth')

        # Should save as 'path.pth' in save_dir
        checkpoint_path = os.path.join(self.test_dir, 'path.pth')
        self.assertTrue(os.path.exists(checkpoint_path))

        # Should NOT create malicious directory
        malicious_path = os.path.join(os.path.dirname(self.test_dir), 'malicious')
        self.assertFalse(os.path.exists(malicious_path))

    def test_save_checkpoint_rejects_hidden_files(self):
        """Test checkpoint save rejects hidden files."""
        self.trainer.configure_optimizer('adam')
        with self.assertRaises(ValueError):
            self.trainer.save_checkpoint('.hidden_file.pth')

    def test_save_checkpoint_rejects_empty_filename(self):
        """Test checkpoint save rejects empty filename."""
        self.trainer.configure_optimizer('adam')
        with self.assertRaises(ValueError):
            self.trainer.save_checkpoint('')

    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        self.trainer.configure_optimizer('adam')
        self.trainer.best_val_acc = 95.5
        self.trainer.current_epoch = 10

        # Save checkpoint
        self.trainer.save_checkpoint('test_load.pth')

        # Create new trainer and load
        new_trainer = Trainer(
            model=SimpleModel(),
            train_loader=self.train_loader,
            save_dir=self.test_dir
        )
        new_trainer.configure_optimizer('adam')
        new_trainer.load_checkpoint('test_load.pth')

        # Verify loaded correctly
        self.assertEqual(new_trainer.best_val_acc, 95.5)
        self.assertEqual(new_trainer.current_epoch, 10)

    def test_load_checkpoint_path_sanitization(self):
        """Test checkpoint load sanitizes path."""
        self.trainer.configure_optimizer('adam')
        self.trainer.save_checkpoint('valid_file.pth')

        # Try to load with path traversal
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_checkpoint('../other_dir/file.pth')

    def test_load_checkpoint_rejects_hidden_files(self):
        """Test checkpoint load rejects hidden files."""
        with self.assertRaises(ValueError):
            self.trainer.load_checkpoint('.hidden_file.pth')

    def test_load_checkpoint_nonexistent_file(self):
        """Test loading nonexistent checkpoint raises error."""
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_checkpoint('nonexistent.pth')

    def test_load_checkpoint_invalid_filename(self):
        """Test loading with invalid filename raises error."""
        with self.assertRaises(ValueError):
            self.trainer.load_checkpoint('')

    def test_checkpoint_preserves_model_state(self):
        """Test checkpoint preserves model parameters."""
        self.trainer.configure_optimizer('adam')

        # Get original weights
        original_weights = self.model.fc.weight.data.clone()

        # Save checkpoint
        self.trainer.save_checkpoint('state_test.pth')

        # Modify weights
        with torch.no_grad():
            self.model.fc.weight.fill_(0)

        # Load checkpoint
        self.trainer.load_checkpoint('state_test.pth')

        # Verify weights restored
        loaded_weights = self.model.fc.weight.data
        self.assertTrue(torch.allclose(loaded_weights, original_weights))

    def test_checkpoint_preserves_optimizer_state(self):
        """Test checkpoint preserves optimizer state."""
        self.trainer.configure_optimizer('adam', lr=0.001)

        # Take a training step to initialize optimizer state
        self.trainer.configure_criterion('crossentropy')
        data, target = self.train_data[0]
        data, target = data.to(self.trainer.device), target.to(self.trainer.device)
        self.trainer.optimizer.zero_grad()
        output = self.model(data)
        loss = self.trainer.criterion(output, target)
        loss.backward()
        self.trainer.optimizer.step()

        # Save checkpoint
        self.trainer.save_checkpoint('optimizer_test.pth')

        # Create new trainer and load
        new_trainer = Trainer(
            model=SimpleModel(),
            train_loader=self.train_loader,
            save_dir=self.test_dir
        )
        new_trainer.configure_optimizer('adam', lr=0.001)
        new_trainer.load_checkpoint('optimizer_test.pth')

        # Verify optimizer state loaded
        self.assertIsNotNone(new_trainer.optimizer.state)

    def test_get_learning_rate(self):
        """Test getting learning rate."""
        self.trainer.configure_optimizer('adam', lr=0.001)
        lr = self.trainer.get_learning_rate()
        self.assertAlmostEqual(lr, 0.001, places=6)

    def test_get_learning_rate_no_optimizer(self):
        """Test getting learning rate without optimizer."""
        lr = self.trainer.get_learning_rate()
        self.assertIsNone(lr)


if __name__ == '__main__':
    unittest.main()
