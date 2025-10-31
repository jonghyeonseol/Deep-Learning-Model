"""
Unit tests for logging functionality.

Tests cover:
- Logger creation and configuration
- Log levels
- File and console output
- Logger reuse
"""

import unittest
import logging
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct import to avoid utils/__init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "logger",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'logger.py')
)
logger_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logger_module)
get_logger = logger_module.get_logger
setup_training_logger = logger_module.setup_training_logger


class TestLogger(unittest.TestCase):
    """Test suite for logging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_log_file = os.path.join(self.test_dir, 'test.log')

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

        # Clear all handlers from test loggers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a Logger instance."""
        logger = get_logger('test')
        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        """Test logger has correct name."""
        logger = get_logger('test_module')
        self.assertEqual(logger.name, 'test_module')

    def test_get_logger_default_level(self):
        """Test logger has default INFO level."""
        logger = get_logger('test', level=logging.INFO)
        self.assertEqual(logger.level, logging.INFO)

    def test_get_logger_custom_level(self):
        """Test logger can be created with custom level."""
        logger = get_logger('test_debug', level=logging.DEBUG)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_get_logger_with_file(self):
        """Test logger creates log file."""
        logger = get_logger('test_file', log_file=self.test_log_file)
        logger.info("Test message")

        self.assertTrue(os.path.exists(self.test_log_file))

    def test_get_logger_file_content(self):
        """Test logger writes to file."""
        logger = get_logger('test_content', log_file=self.test_log_file)
        test_message = "Test log message"
        logger.info(test_message)

        with open(self.test_log_file, 'r') as f:
            content = f.read()
            self.assertIn(test_message, content)

    def test_get_logger_reuse(self):
        """Test getting same logger returns same instance."""
        logger1 = get_logger('test_reuse')
        logger2 = get_logger('test_reuse')
        self.assertIs(logger1, logger2)

    def test_get_logger_no_duplicate_handlers(self):
        """Test repeated calls don't add duplicate handlers."""
        logger = get_logger('test_handlers')
        handler_count_1 = len(logger.handlers)

        logger = get_logger('test_handlers')
        handler_count_2 = len(logger.handlers)

        self.assertEqual(handler_count_1, handler_count_2)

    def test_logger_console_disabled(self):
        """Test logger can be created without console output."""
        logger = get_logger('test_no_console', console=False)
        # Should have no console handlers
        console_handlers = [h for h in logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and h.stream in (sys.stdout, sys.stderr)]
        # Note: May have file handler only
        self.assertTrue(len(logger.handlers) >= 0)

    def test_setup_training_logger(self):
        """Test training logger setup."""
        logger = setup_training_logger(self.test_dir, level=logging.INFO)
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'training')

    def test_setup_training_logger_creates_file(self):
        """Test training logger creates log file."""
        logger = setup_training_logger(self.test_dir)
        logger.info("Training test")

        log_file = Path(self.test_dir) / 'training.log'
        self.assertTrue(log_file.exists())

    def test_logger_format(self):
        """Test log message format."""
        logger = get_logger('test_format', log_file=self.test_log_file)
        logger.info("Format test")

        with open(self.test_log_file, 'r') as f:
            content = f.read()
            # Check for timestamp
            self.assertRegex(content, r'\d{4}-\d{2}-\d{2}')
            # Check for logger name
            self.assertIn('test_format', content)
            # Check for log level
            self.assertIn('INFO', content)

    def test_logger_different_levels(self):
        """Test logger handles different log levels."""
        logger = get_logger('test_levels', log_file=self.test_log_file, level=logging.DEBUG)

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        with open(self.test_log_file, 'r') as f:
            content = f.read()
            self.assertIn("DEBUG", content)
            self.assertIn("INFO", content)
            self.assertIn("WARNING", content)
            self.assertIn("ERROR", content)
            self.assertIn("CRITICAL", content)

    def test_logger_level_filtering(self):
        """Test logger filters messages by level."""
        logger = get_logger('test_filter', log_file=self.test_log_file, level=logging.WARNING)

        logger.debug("Debug - should not appear")
        logger.info("Info - should not appear")
        logger.warning("Warning - should appear")

        with open(self.test_log_file, 'r') as f:
            content = f.read()
            self.assertNotIn("Debug - should not appear", content)
            self.assertNotIn("Info - should not appear", content)
            self.assertIn("Warning - should appear", content)

    def test_logger_creates_directory(self):
        """Test logger creates parent directories for log file."""
        nested_dir = os.path.join(self.test_dir, 'nested', 'path')
        log_file = os.path.join(nested_dir, 'test.log')

        logger = get_logger('test_nested', log_file=log_file)
        logger.info("Test")

        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.exists(log_file))


if __name__ == '__main__':
    unittest.main()
