"""Training Module"""

from .trainer import ProteomicsTrainer
from .loss_functions import (
    SpectralAngleLoss,
    CosineSimilarityLoss,
    WeightedMSELoss,
    FocalLoss,
    MultiTaskLoss,
    TripletLoss,
    get_loss_function
)

__all__ = [
    'ProteomicsTrainer',
    'SpectralAngleLoss',
    'CosineSimilarityLoss',
    'WeightedMSELoss',
    'FocalLoss',
    'MultiTaskLoss',
    'TripletLoss',
    'get_loss_function'
]
