"""Visualization utilities for proteomics spectra"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

sns.set_style("whitegrid")


def plot_spectrum(mz: np.ndarray, intensity: np.ndarray, title: str = "Mass Spectrum",
                  precursor_mz: Optional[float] = None, save_path: Optional[str] = None):
    """Plot a single mass spectrum."""
    plt.figure(figsize=(12, 4))
    plt.stem(mz, intensity, linefmt='b-', markerfmt=' ', basefmt=' ')

    if precursor_mz:
        plt.axvline(precursor_mz, color='r', linestyle='--', label=f'Precursor: {precursor_mz:.2f}')
        plt.legend()

    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Plot training history (loss and metrics)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_metric'], label='Train Accuracy')
    axes[1].plot(history['val_metric'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_spectrum_comparison(mz: np.ndarray, intensity_true: np.ndarray,
                             intensity_pred: np.ndarray, title: str = "Spectrum Comparison",
                             save_path: Optional[str] = None):
    """Compare true vs predicted spectra."""
    plt.figure(figsize=(12, 5))
    plt.stem(mz, intensity_true, linefmt='b-', markerfmt=' ', basefmt=' ', label='True')
    plt.stem(mz, intensity_pred, linefmt='r-', markerfmt=' ', basefmt=' ', label='Predicted', alpha=0.6)
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
