"""
Proteomics Module

Deep learning framework for proteomics mass spectrometry data analysis.

Supports:
- Data loading from .raw, mzML, and CSV formats
- CNN and Transformer models for spectrum analysis
- Classification and regression tasks
- Retention time prediction
"""

__version__ = '1.0.0'

from . import data_loaders
from . import models
from . import training
from . import utils

__all__ = ['data_loaders', 'models', 'training', 'utils']
