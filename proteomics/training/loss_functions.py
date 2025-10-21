"""
Loss Functions Module

Custom loss functions for proteomics spectrum analysis.
Includes spectral angle, cosine similarity, and multi-task losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralAngleLoss(nn.Module):
    """
    Spectral Angle Distance loss.

    Measures the angle between two spectra in vector space.
    Lower angle = more similar spectra.

    Range: [0, π/2] (90 degrees for orthogonal spectra)
    """

    def __init__(self, reduction='mean'):
        """
        Initialize SpectralAngleLoss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(SpectralAngleLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Calculate spectral angle loss.

        Args:
            pred: Predicted spectrum (batch_size, num_bins)
            target: Target spectrum (batch_size, num_bins)

        Returns:
            Spectral angle loss
        """
        # Normalize to unit vectors
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=1)

        # Clamp to avoid numerical issues with acos
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # Spectral angle (in radians)
        angle = torch.acos(cos_sim)

        # Apply reduction
        if self.reduction == 'mean':
            return angle.mean()
        elif self.reduction == 'sum':
            return angle.sum()
        else:  # none
            return angle


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity loss.

    Measures cosine similarity between spectra.
    Loss = 1 - cosine_similarity (so minimizing brings spectra closer)

    Range: [0, 2] (0 = identical, 2 = opposite)
    """

    def __init__(self, reduction='mean'):
        """
        Initialize CosineSimilarityLoss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Calculate cosine similarity loss.

        Args:
            pred: Predicted spectrum
            target: Target spectrum

        Returns:
            Cosine similarity loss
        """
        # Cosine similarity
        cos_sim = F.cosine_similarity(pred, target, dim=1)

        # Convert to loss (1 - similarity)
        loss = 1.0 - cos_sim

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error loss.

    Weights higher intensity peaks more heavily.
    Useful for spectrum prediction tasks.
    """

    def __init__(self, weight_power=1.0, reduction='mean'):
        """
        Initialize WeightedMSELoss.

        Args:
            weight_power: Exponent for weighting by target intensity
            reduction: Reduction method
        """
        super(WeightedMSELoss, self).__init__()
        self.weight_power = weight_power
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Calculate weighted MSE loss.

        Args:
            pred: Predicted spectrum
            target: Target spectrum

        Returns:
            Weighted MSE loss
        """
        # Calculate squared error
        sq_error = (pred - target) ** 2

        # Weight by target intensity
        weights = target ** self.weight_power
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize weights

        # Weighted error
        weighted_error = sq_error * weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_error.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return weighted_error.sum()
        else:
            return weighted_error.sum(dim=1)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification.

    Focuses training on hard examples by down-weighting easy examples.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize FocalLoss.

        Args:
            alpha: Class weights (tensor of shape [num_classes])
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Calculate focal loss.

        Args:
            pred: Predicted logits (batch_size, num_classes)
            target: Target labels (batch_size,)

        Returns:
            Focal loss
        """
        # Get probabilities
        probs = F.softmax(pred, dim=1)

        # Get probability of correct class
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).float()
        pt = (probs * target_one_hot).sum(dim=1)

        # Focal term
        focal_term = (1 - pt) ** self.gamma

        # Cross entropy
        ce = F.cross_entropy(pred, target, reduction='none')

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            loss = alpha_t * focal_term * ce
        else:
            loss = focal_term * ce

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for joint classification and regression.

    Combines classification loss (cross entropy) and regression loss (MSE)
    with learnable task weights.
    """

    def __init__(
        self,
        task_weights=None,  # [weight_cls, weight_reg]
        use_uncertainty=False,  # Use uncertainty-based weighting
    ):
        """
        Initialize MultiTaskLoss.

        Args:
            task_weights: Manual task weights [classification, regression]
            use_uncertainty: Use learnable uncertainty-based weights
        """
        super(MultiTaskLoss, self).__init__()

        if task_weights is not None:
            self.task_weights = torch.tensor(task_weights)
        else:
            self.task_weights = torch.tensor([1.0, 1.0])

        self.use_uncertainty = use_uncertainty

        # Learnable uncertainty parameters (log variance)
        if use_uncertainty:
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(self, class_pred, class_target, reg_pred, reg_target):
        """
        Calculate multi-task loss.

        Args:
            class_pred: Classification logits
            class_target: Classification labels
            reg_pred: Regression predictions
            reg_target: Regression targets

        Returns:
            Combined loss, classification loss, regression loss
        """
        # Classification loss
        cls_loss = F.cross_entropy(class_pred, class_target)

        # Regression loss
        reg_loss = F.mse_loss(reg_pred, reg_target)

        if self.use_uncertainty:
            # Uncertainty-based weighting (Kendall et al., 2018)
            # L = (1/2σ²₁)L₁ + (1/2σ²₂)L₂ + log(σ₁σ₂)
            precision_cls = torch.exp(-self.log_var_cls)
            precision_reg = torch.exp(-self.log_var_reg)

            total_loss = (
                precision_cls * cls_loss + self.log_var_cls +
                precision_reg * reg_loss + self.log_var_reg
            )
        else:
            # Manual weighting
            total_loss = self.task_weights[0] * cls_loss + self.task_weights[1] * reg_loss

        return total_loss, cls_loss, reg_loss


class TripletLoss(nn.Module):
    """
    Triplet loss for learning spectrum embeddings.

    Used for similarity learning tasks.
    """

    def __init__(self, margin=1.0, reduction='mean'):
        """
        Initialize TripletLoss.

        Args:
            margin: Margin for triplet loss
            reduction: Reduction method
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor, positive, negative):
        """
        Calculate triplet loss.

        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same class)
            negative: Negative embeddings (different class)

        Returns:
            Triplet loss
        """
        # Distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Utility function to get loss function by name
def get_loss_function(name, **kwargs):
    """
    Get loss function by name.

    Args:
        name: Loss function name
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance
    """
    loss_functions = {
        'cross_entropy': nn.CrossEntropyLoss,
        'mse': nn.MSELoss,
        'spectral_angle': SpectralAngleLoss,
        'cosine': CosineSimilarityLoss,
        'weighted_mse': WeightedMSELoss,
        'focal': FocalLoss,
        'multi_task': MultiTaskLoss,
        'triplet': TripletLoss,
    }

    if name not in loss_functions:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(loss_functions.keys())}")

    return loss_functions[name](**kwargs)


# Example usage
if __name__ == "__main__":
    # Test Spectral Angle Loss
    print("Testing SpectralAngleLoss...")
    loss_fn = SpectralAngleLoss()
    pred = torch.randn(4, 100)
    target = torch.randn(4, 100)
    loss = loss_fn(pred, target)
    print(f"Spectral Angle Loss: {loss.item():.4f}")

    # Test Cosine Similarity Loss
    print("\nTesting CosineSimilarityLoss...")
    loss_fn = CosineSimilarityLoss()
    loss = loss_fn(pred, target)
    print(f"Cosine Similarity Loss: {loss.item():.4f}")

    # Test Multi-Task Loss
    print("\nTesting MultiTaskLoss...")
    loss_fn = MultiTaskLoss(use_uncertainty=True)
    class_pred = torch.randn(4, 10)
    class_target = torch.randint(0, 10, (4,))
    reg_pred = torch.randn(4, 1)
    reg_target = torch.randn(4, 1)
    total_loss, cls_loss, reg_loss = loss_fn(class_pred, class_target, reg_pred, reg_target)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Classification Loss: {cls_loss.item():.4f}")
    print(f"Regression Loss: {reg_loss.item():.4f}")
