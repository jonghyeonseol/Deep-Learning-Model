"""Evaluation metrics for proteomics models"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def spectral_similarity(spec1: np.ndarray, spec2: np.ndarray, method: str = 'cosine') -> float:
    """Calculate spectral similarity."""
    if method == 'cosine':
        dot = np.dot(spec1, spec2)
        norm = np.linalg.norm(spec1) * np.linalg.norm(spec2)
        return dot / norm if norm > 0 else 0.0
    elif method == 'euclidean':
        dist = np.linalg.norm(spec1 - spec2)
        return 1.0 / (1.0 + dist)
    else:
        raise ValueError(f"Unknown method: {method}")
