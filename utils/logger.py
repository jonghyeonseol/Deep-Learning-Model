"""
Centralized logging configuration for the Deep Learning framework.

This module provides a consistent logging interface across all components.
Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        console: Whether to log to console

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Model training started")
        >>> logger.error("Training failed: %s", error_msg)
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_training_logger(
    save_dir: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger for training runs with file and console output.

    Args:
        save_dir: Directory to save log files
        level: Logging level

    Returns:
        Configured logger instance
    """
    log_file = Path(save_dir) / 'training.log'
    return get_logger('training', level=level, log_file=str(log_file))


# Convenience functions for common log levels
def debug(msg: str, *args, **kwargs) -> None:
    """Log debug message to root logger."""
    logging.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log info message to root logger."""
    logging.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log warning message to root logger."""
    logging.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log error message to root logger."""
    logging.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log critical message to root logger."""
    logging.critical(msg, *args, **kwargs)
