"""
Configuration management for training experiments.

This module provides utilities for loading and validating configuration files
in YAML or JSON format.

Example YAML config:
    model:
      architecture: resnet18
      activation: swish
      num_classes: 10

    training:
      epochs: 100
      batch_size: 128
      learning_rate: 0.001
      optimizer: adamw
      weight_decay: 0.05

    scheduler:
      type: cosine_warmup
      warmup_epochs: 5

    augmentation:
      use_randaugment: true
      use_mixup: true
      use_cutmix: false
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
from .logger import get_logger
from .exceptions import ConfigurationError

logger = get_logger(__name__)


class Config:
    """
    Configuration container with dot notation access.

    Examples:
        >>> config = Config({'model': {'architecture': 'resnet18'}})
        >>> config.model.architecture
        'resnet18'
        >>> config.get('model.learning_rate', 0.001)
        0.001
    """

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Dictionary containing configuration
        """
        self._config = config_dict or {}
        self._frozen = False

    def __getattr__(self, name: str) -> Any:
        """Allow dot notation access to config values."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value

        raise AttributeError(f"Config has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment."""
        if self._frozen:
            raise ConfigurationError(
                "Cannot modify frozen configuration",
                param_name=key
            )
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value with default fallback.

        Supports nested keys with dot notation.

        Args:
            key: Config key (supports 'nested.key.path')
            default: Default value if key not found

        Returns:
            Config value or default

        Examples:
            >>> config.get('model.learning_rate', 0.001)
            >>> config.get('nonexistent', 'default_value')
        """
        if '.' in key:
            keys = key.split('.')
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._config.copy()

    def update(self, other: Union[Dict, 'Config']):
        """
        Update configuration with another dict or Config.

        Args:
            other: Dictionary or Config to merge
        """
        if self._frozen:
            raise ConfigurationError("Cannot modify frozen configuration")

        if isinstance(other, Config):
            other = other.to_dict()

        self._deep_update(self._config, other)

    def _deep_update(self, base: Dict, updates: Dict):
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def freeze(self):
        """Freeze configuration to prevent modifications."""
        self._frozen = True

    def is_frozen(self) -> bool:
        """Check if configuration is frozen."""
        return self._frozen

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self._config})"


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object

    Raises:
        ConfigurationError: If file cannot be loaded or parsed

    Examples:
        >>> config = load_config('configs/train_resnet.yaml')
        >>> print(config.model.architecture)
        resnet18
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            param_name="config_path"
        )

    try:
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported file format: {config_path.suffix}. "
                    "Use .yaml, .yml, or .json",
                    param_name="config_path"
                )

        logger.info(f"Loaded configuration from {config_path}")
        return Config(config_dict)

    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML file: {e}",
            param_name="config_path"
        )
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Failed to parse JSON file: {e}",
            param_name="config_path"
        )
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration: {e}",
            param_name="config_path"
        )


def save_config(config: Union[Config, Dict], save_path: Union[str, Path]):
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Configuration to save
        save_path: Path to save configuration

    Raises:
        ConfigurationError: If file cannot be saved
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config

    try:
        with open(save_path, 'w') as f:
            if save_path.suffix in ['.yaml', '.yml']:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif save_path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ConfigurationError(
                    f"Unsupported file format: {save_path.suffix}",
                    param_name="save_path"
                )

        logger.info(f"Saved configuration to {save_path}")

    except Exception as e:
        raise ConfigurationError(
            f"Failed to save configuration: {e}",
            param_name="save_path"
        )


def validate_training_config(config: Config) -> None:
    """
    Validate training configuration has required fields.

    Args:
        config: Configuration to validate

    Raises:
        ConfigurationError: If required fields are missing or invalid
    """
    required_fields = {
        'model.architecture': str,
        'training.epochs': int,
        'training.batch_size': int,
        'training.learning_rate': (int, float),
    }

    for field, expected_type in required_fields.items():
        value = config.get(field)
        if value is None:
            raise ConfigurationError(
                f"Missing required field: {field}",
                param_name=field
            )

        if not isinstance(value, expected_type):
            raise ConfigurationError(
                f"Invalid type for {field}: expected {expected_type}, got {type(value)}",
                param_name=field
            )

    # Validate value ranges
    if config.get('training.epochs') <= 0:
        raise ConfigurationError(
            "epochs must be positive",
            param_name="training.epochs"
        )

    if config.get('training.batch_size') <= 0:
        raise ConfigurationError(
            "batch_size must be positive",
            param_name="training.batch_size"
        )

    if config.get('training.learning_rate') <= 0:
        raise ConfigurationError(
            "learning_rate must be positive",
            param_name="training.learning_rate"
        )

    logger.info("Configuration validation passed")


# Example configuration templates
EXAMPLE_CONFIGS = {
    'resnet18_basic': {
        'model': {
            'architecture': 'resnet18',
            'activation': 'swish',
            'num_classes': 10
        },
        'training': {
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 0.1,
            'optimizer': 'sgd',
            'momentum': 0.9,
            'weight_decay': 5e-4
        },
        'scheduler': {
            'type': 'cosine',
            'min_lr': 1e-6
        },
        'augmentation': {
            'use_randaugment': True,
            'use_mixup': False,
            'use_cutmix': False
        }
    },
    'efficientnet_modern': {
        'model': {
            'architecture': 'efficientnet_b0',
            'activation': 'swish',
            'num_classes': 10
        },
        'training': {
            'epochs': 300,
            'batch_size': 128,
            'learning_rate': 0.001,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'use_amp': True,
            'use_ema': True,
            'label_smoothing': 0.1,
            'gradient_clip': 1.0
        },
        'scheduler': {
            'type': 'cosine_warmup',
            'warmup_epochs': 20,
            'min_lr': 1e-6
        },
        'augmentation': {
            'use_randaugment': True,
            'use_mixup': True,
            'use_cutmix': True,
            'mixup_alpha': 1.0
        }
    }
}


def create_example_config(name: str, save_path: Union[str, Path]):
    """
    Create example configuration file.

    Args:
        name: Example config name ('resnet18_basic' or 'efficientnet_modern')
        save_path: Path to save configuration

    Raises:
        ConfigurationError: If example name is invalid
    """
    if name not in EXAMPLE_CONFIGS:
        available = ', '.join(EXAMPLE_CONFIGS.keys())
        raise ConfigurationError(
            f"Unknown example config: {name}. Available: {available}",
            param_name="name"
        )

    config = Config(EXAMPLE_CONFIGS[name])
    save_config(config, save_path)
    logger.info(f"Created example configuration: {name} -> {save_path}")
