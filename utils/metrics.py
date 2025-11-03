"""
Comprehensive metrics module for deep learning model evaluation.

Provides various classification metrics including:
- Accuracy, Precision, Recall, F1 Score
- PR Curve, AP (Average Precision), mAP (mean Average Precision)
- IoU (Intersection over Union) for segmentation tasks
- Confusion Matrix
- Per-class metrics
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve
)


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics for classification tasks."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes in the dataset
            class_names: Optional list of class names for better readability
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]

        # Storage for predictions and targets
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []

    def reset(self):
        """Reset stored predictions and targets."""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               probabilities: Optional[torch.Tensor] = None):
        """
        Update metrics with new predictions.

        Args:
            predictions: Model predictions (class indices)
            targets: Ground truth labels
            probabilities: Class probabilities (for PR curves and AP)
        """
        # Convert to numpy and store
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())

        if probabilities is not None:
            self.all_probabilities.extend(probabilities.cpu().numpy())

    def compute_basic_metrics(self) -> Dict[str, float]:
        """
        Compute basic classification metrics.

        Returns:
            Dictionary containing accuracy, precision, recall, and F1 score
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        return metrics

    def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class precision, recall, and F1 score.

        Returns:
            Dictionary mapping class names to their metrics
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)

        # Compute per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        per_class = {}
        for i, class_name in enumerate(self.class_names):
            per_class[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(np.sum(y_true == i))
            }

        return per_class

    def compute_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix.

        Returns:
            Confusion matrix as numpy array
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)

        return confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

    def compute_pr_curve_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute Precision-Recall curve data for each class.

        Returns:
            Dictionary mapping class names to precision/recall arrays
        """
        if not self.all_probabilities:
            return {}

        y_true = np.array(self.all_targets)
        y_prob = np.array(self.all_probabilities)

        pr_data = {}
        for i, class_name in enumerate(self.class_names):
            # One-vs-rest for each class
            y_true_binary = (y_true == i).astype(int)
            y_prob_class = y_prob[:, i]

            precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob_class)

            pr_data[class_name] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }

        return pr_data

    def compute_roc_curve_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute ROC curve data for each class.

        Returns:
            Dictionary mapping class names to FPR/TPR arrays
        """
        if not self.all_probabilities:
            return {}

        y_true = np.array(self.all_targets)
        y_prob = np.array(self.all_probabilities)

        roc_data = {}
        for i, class_name in enumerate(self.class_names):
            # One-vs-rest for each class
            y_true_binary = (y_true == i).astype(int)
            y_prob_class = y_prob[:, i]

            fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_class)

            roc_data[class_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }

        return roc_data

    def compute_average_precision(self) -> Dict[str, float]:
        """
        Compute Average Precision (AP) for each class.

        Returns:
            Dictionary mapping class names to AP scores
        """
        if not self.all_probabilities:
            return {}

        y_true = np.array(self.all_targets)
        y_prob = np.array(self.all_probabilities)

        ap_scores = {}
        for i, class_name in enumerate(self.class_names):
            # One-vs-rest for each class
            y_true_binary = (y_true == i).astype(int)
            y_prob_class = y_prob[:, i]

            ap = average_precision_score(y_true_binary, y_prob_class)
            ap_scores[class_name] = float(ap)

        return ap_scores

    def compute_mean_average_precision(self) -> float:
        """
        Compute mean Average Precision (mAP) across all classes.

        Returns:
            mAP score
        """
        ap_scores = self.compute_average_precision()
        if not ap_scores:
            return 0.0

        return float(np.mean(list(ap_scores.values())))

    def compute_roc_auc(self) -> Dict[str, float]:
        """
        Compute ROC AUC score for each class.

        Returns:
            Dictionary mapping class names to AUC scores
        """
        if not self.all_probabilities:
            return {}

        y_true = np.array(self.all_targets)
        y_prob = np.array(self.all_probabilities)

        auc_scores = {}
        for i, class_name in enumerate(self.class_names):
            # One-vs-rest for each class
            y_true_binary = (y_true == i).astype(int)
            y_prob_class = y_prob[:, i]

            # Check if we have both classes in the data
            if len(np.unique(y_true_binary)) > 1:
                auc = roc_auc_score(y_true_binary, y_prob_class)
                auc_scores[class_name] = float(auc)
            else:
                auc_scores[class_name] = 0.0

        return auc_scores

    def compute_iou_segmentation(self, predictions: torch.Tensor,
                                 targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute IoU (Intersection over Union) for segmentation tasks.

        Args:
            predictions: Predicted segmentation masks (H x W)
            targets: Ground truth segmentation masks (H x W)

        Returns:
            Dictionary containing IoU scores per class
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        iou_scores = {}
        for i, class_name in enumerate(self.class_names):
            # Binary masks for class i
            pred_mask = (predictions == i)
            target_mask = (targets == i)

            # Compute intersection and union
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            # Compute IoU
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0

            iou_scores[class_name] = float(iou)

        return iou_scores

    def compute_all_metrics(self) -> Dict:
        """
        Compute all available metrics.

        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'basic': self.compute_basic_metrics(),
            'per_class': self.compute_per_class_metrics(),
            'confusion_matrix': self.compute_confusion_matrix(),
            'pr_curve': self.compute_pr_curve_data(),
            'roc_curve': self.compute_roc_curve_data(),
            'average_precision': self.compute_average_precision(),
            'mean_average_precision': self.compute_mean_average_precision(),
            'roc_auc': self.compute_roc_auc(),
        }

        return metrics

    def print_summary(self):
        """Print a summary of key metrics."""
        basic = self.compute_basic_metrics()

        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        print(f"Accuracy:              {basic['accuracy']:.4f}")
        print(f"Precision (macro):     {basic['precision_macro']:.4f}")
        print(f"Recall (macro):        {basic['recall_macro']:.4f}")
        print(f"F1 Score (macro):      {basic['f1_macro']:.4f}")

        if self.all_probabilities:
            map_score = self.compute_mean_average_precision()
            print(f"mAP:                   {map_score:.4f}")

        print("\nPer-Class Metrics:")
        print("-"*60)
        per_class = self.compute_per_class_metrics()
        for class_name, metrics in per_class.items():
            print(f"{class_name:12s} - P: {metrics['precision']:.3f}, "
                  f"R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}, "
                  f"Support: {metrics['support']}")
        print("="*60 + "\n")


def calculate_top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Calculate top-k accuracy.

    Args:
        predictions: Model output probabilities (N x C)
        targets: Ground truth labels (N,)
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = predictions.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return float(correct_k.mul_(100.0 / batch_size))
