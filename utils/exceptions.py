"""
Custom exceptions for the Deep Learning framework.

This module provides specific exception classes for better error handling
and more informative error messages.
"""


class DLFrameworkError(Exception):
    """Base exception for all Deep Learning framework errors."""
    pass


class ConfigurationError(DLFrameworkError):
    """Raised when there's an error in configuration."""

    def __init__(self, message: str, param_name: str = None):
        self.param_name = param_name
        if param_name:
            message = f"Configuration error for '{param_name}': {message}"
        super().__init__(message)


class CheckpointError(DLFrameworkError):
    """Raised when there's an error with checkpoint operations."""
    pass


class InvalidCheckpointPathError(CheckpointError):
    """Raised when checkpoint path is invalid or insecure."""

    def __init__(self, path: str, reason: str):
        message = f"Invalid checkpoint path '{path}': {reason}"
        super().__init__(message)
        self.path = path
        self.reason = reason


class CheckpointNotFoundError(CheckpointError, FileNotFoundError):
    """Raised when checkpoint file is not found."""

    def __init__(self, path: str):
        message = (
            f"Checkpoint file not found: {path}\n"
            f"Make sure the file exists and the path is correct."
        )
        super().__init__(message)
        self.path = path


class CheckpointLoadError(CheckpointError):
    """Raised when checkpoint cannot be loaded."""

    def __init__(self, path: str, reason: str):
        message = (
            f"Failed to load checkpoint from '{path}': {reason}\n"
            f"The file may be corrupted or incompatible."
        )
        super().__init__(message)
        self.path = path
        self.reason = reason


class ModelError(DLFrameworkError):
    """Raised when there's an error with model operations."""
    pass


class InvalidArchitectureError(ModelError):
    """Raised when model architecture is invalid."""

    def __init__(self, architecture_name: str, available: list = None):
        message = f"Invalid architecture: '{architecture_name}'"
        if available:
            message += f"\nAvailable architectures: {', '.join(available)}"
        super().__init__(message)
        self.architecture_name = architecture_name
        self.available = available


class InvalidActivationError(ModelError):
    """Raised when activation function is invalid."""

    def __init__(self, activation_name: str, available: list = None):
        message = f"Invalid activation function: '{activation_name}'"
        if available:
            message += f"\nAvailable activations: {', '.join(available)}"
        super().__init__(message)
        self.activation_name = activation_name
        self.available = available


class TrainingError(DLFrameworkError):
    """Raised when there's an error during training."""
    pass


class OptimizerNotConfiguredError(TrainingError):
    """Raised when optimizer is not configured before training."""

    def __init__(self):
        message = (
            "Optimizer not configured. "
            "Call trainer.configure_optimizer() before training."
        )
        super().__init__(message)


class CriterionNotConfiguredError(TrainingError):
    """Raised when loss criterion is not configured before training."""

    def __init__(self):
        message = (
            "Loss criterion not configured. "
            "Call trainer.configure_criterion() before training."
        )
        super().__init__(message)


class SchedulerConfigurationError(TrainingError):
    """Raised when scheduler configuration fails."""

    def __init__(self, reason: str):
        message = f"Failed to configure scheduler: {reason}"
        super().__init__(message)
        self.reason = reason


class DataError(DLFrameworkError):
    """Raised when there's an error with data operations."""
    pass


class DataLoaderError(DataError):
    """Raised when data loader creation fails."""

    def __init__(self, reason: str):
        message = f"Failed to create data loader: {reason}"
        super().__init__(message)
        self.reason = reason


class InvalidDatasetError(DataError):
    """Raised when dataset is invalid or corrupted."""

    def __init__(self, dataset_name: str, reason: str):
        message = f"Invalid dataset '{dataset_name}': {reason}"
        super().__init__(message)
        self.dataset_name = dataset_name
        self.reason = reason


class AugmentationError(DataError):
    """Raised when data augmentation fails."""

    def __init__(self, augmentation_name: str, reason: str):
        message = f"Augmentation '{augmentation_name}' failed: {reason}"
        super().__init__(message)
        self.augmentation_name = augmentation_name
        self.reason = reason


class DeviceError(DLFrameworkError):
    """Raised when there's an error with device operations."""
    pass


class CUDANotAvailableError(DeviceError):
    """Raised when CUDA is required but not available."""

    def __init__(self):
        message = (
            "CUDA is not available. "
            "Make sure PyTorch is installed with CUDA support or use CPU."
        )
        super().__init__(message)


# Convenience functions for raising common errors

def require_optimizer(optimizer):
    """Raise error if optimizer is not configured."""
    if optimizer is None:
        raise OptimizerNotConfiguredError()


def require_criterion(criterion):
    """Raise error if criterion is not configured."""
    if criterion is None:
        raise CriterionNotConfiguredError()


def validate_checkpoint_path(filename: str):
    """
    Validate checkpoint filename.

    Args:
        filename: Checkpoint filename

    Raises:
        InvalidCheckpointPathError: If filename is invalid
    """
    import os

    if not filename:
        raise InvalidCheckpointPathError(filename, "Filename cannot be empty")

    if filename.startswith('.'):
        raise InvalidCheckpointPathError(filename, "Hidden files are not allowed")

    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        raise InvalidCheckpointPathError(
            filename,
            "Path traversal detected. Use basename only."
        )


def check_checkpoint_exists(filepath: str):
    """
    Check if checkpoint file exists.

    Args:
        filepath: Full path to checkpoint file

    Raises:
        CheckpointNotFoundError: If file does not exist
    """
    import os

    if not os.path.exists(filepath):
        raise CheckpointNotFoundError(filepath)
