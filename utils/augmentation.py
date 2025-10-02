"""
Advanced data augmentation techniques for modern deep learning.

This module implements:
- RandAugment: Automated augmentation with random policies
- MixUp: Linear interpolation between samples and labels
- CutMix: Cut-and-paste augmentation with label mixing
- CutOut / Random Erasing: Random patch occlusion
- AutoAugment-style policies
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class RandAugment:
    """
    RandAugment: Practical automated data augmentation.

    Paper: "RandAugment: Practical automated data augmentation with a reduced search space"
    https://arxiv.org/abs/1909.13719

    Randomly selects N augmentation operations from a pool and applies them
    with magnitude M.

    Args:
        n: Number of augmentation operations to apply (typically 1-3)
        m: Magnitude of augmentation (0-30, typically 9-15)
    """

    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augment_list = [
            ('AutoContrast', 0, 1),
            ('Equalize', 0, 1),
            ('Invert', 0, 1),
            ('Rotate', 0, 30),
            ('Posterize', 0, 4),
            ('Solarize', 0, 256),
            ('Color', 0.1, 1.9),
            ('Contrast', 0.1, 1.9),
            ('Brightness', 0.1, 1.9),
            ('Sharpness', 0.1, 1.9),
            ('ShearX', 0, 0.3),
            ('ShearY', 0, 0.3),
            ('TranslateX', 0, 10),
            ('TranslateY', 0, 10),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op_name, min_val, max_val in ops:
            val = (self.m / 30) * (max_val - min_val) + min_val
            img = self._apply_op(img, op_name, val)

        return img

    def _apply_op(self, img, op_name, magnitude):
        """Apply a single augmentation operation."""
        if op_name == 'AutoContrast':
            return ImageOps.autocontrast(img)
        elif op_name == 'Equalize':
            return ImageOps.equalize(img)
        elif op_name == 'Invert':
            return ImageOps.invert(img)
        elif op_name == 'Rotate':
            return img.rotate(magnitude)
        elif op_name == 'Posterize':
            return ImageOps.posterize(img, int(magnitude))
        elif op_name == 'Solarize':
            return ImageOps.solarize(img, int(magnitude))
        elif op_name == 'Color':
            return ImageEnhance.Color(img).enhance(magnitude)
        elif op_name == 'Contrast':
            return ImageEnhance.Contrast(img).enhance(magnitude)
        elif op_name == 'Brightness':
            return ImageEnhance.Brightness(img).enhance(magnitude)
        elif op_name == 'Sharpness':
            return ImageEnhance.Sharpness(img).enhance(magnitude)
        elif op_name == 'ShearX':
            return img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
        elif op_name == 'ShearY':
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
        elif op_name == 'TranslateX':
            return img.transform(img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0))
        elif op_name == 'TranslateY':
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude))
        else:
            return img


class Cutout:
    """
    Cutout: Randomly mask out square regions of input.

    Paper: "Improved Regularization of Convolutional Neural Networks with Cutout"
    https://arxiv.org/abs/1708.04552

    Randomly masks out one or more square regions in the input image.
    This forces the model to use more diverse features.

    Args:
        n_holes: Number of patches to cut out (default 1)
        length: Length of the square patch (default 16 for CIFAR-10)
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img: Tensor image of size (C, H, W)

        Returns:
            Tensor: Image with n_holes of dimension length x length cut out
        """
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones((h, w), dtype=torch.float32)

        for _ in range(self.n_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1:y2, x1:x2] = 0.0

        mask = mask.unsqueeze(0).expand_as(img)
        return img * mask


class RandomErasing:
    """
    Random Erasing: Randomly erase rectangular regions.

    Paper: "Random Erasing Data Augmentation"
    https://arxiv.org/abs/1708.04896

    Similar to Cutout but with variable size and aspect ratio.

    Args:
        p: Probability of performing random erasing (default 0.5)
        scale: Range of proportion of erased area (default (0.02, 0.33))
        ratio: Range of aspect ratio of erased area (default (0.3, 3.3))
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_h, img_w = img.shape[1], img.shape[2]
        img_area = img_h * img_w

        for _ in range(10):
            erase_area = random.uniform(self.scale[0], self.scale[1]) * img_area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h = int(round((erase_area * aspect_ratio) ** 0.5))
            w = int(round((erase_area / aspect_ratio) ** 0.5))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                img[:, i:i+h, j:j+w] = torch.randn_like(img[:, i:i+h, j:j+w])
                return img

        return img


class MixUp:
    """
    MixUp: Beyond Empirical Risk Minimization.

    Paper: "mixup: Beyond Empirical Risk Minimization"
    https://arxiv.org/abs/1710.09412

    Creates virtual training examples by linearly interpolating between
    random pairs of training examples and their labels.

    mixed_x = lambda * x_i + (1 - lambda) * x_j
    mixed_y = lambda * y_i + (1 - lambda) * y_j

    where lambda ~ Beta(alpha, alpha)

    Usage:
        Apply during training inside the training loop, not in data loader.

    Args:
        alpha: Beta distribution parameter (typical values: 0.2, 0.4, 1.0)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y):
        """
        Args:
            x: Input batch [B, C, H, W]
            y: Target batch [B]

        Returns:
            Mixed inputs, targets_a, targets_b, lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


class CutMix:
    """
    CutMix: Regularization Strategy to Train Strong Classifiers.

    Paper: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    https://arxiv.org/abs/1905.04899

    Cuts and pastes patches among training images.
    Mixes both images and labels proportionally to the area of patches.

    Usage:
        Apply during training inside the training loop.

    Args:
        alpha: Beta distribution parameter (typical value: 1.0)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y):
        """
        Args:
            x: Input batch [B, C, H, W]
            y: Target batch [B]

        Returns:
            Mixed inputs, targets_a, targets_b, lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # Get random box
        _, _, H, W = x.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform sampling of center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        y_a, y_b = y, y[index]

        return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for MixUp/CutMix.

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First target
        y_b: Second target
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CIFAR10Augmentation:
    """
    Comprehensive augmentation pipeline for CIFAR-10.

    Args:
        mode: 'basic', 'standard', 'autoaugment', 'randaugment'
        cutout: Whether to apply cutout (default False)
        random_erasing: Whether to apply random erasing (default False)
    """

    def __init__(self, mode='standard', cutout=False, random_erasing=False):
        self.mode = mode

        # Normalization (CIFAR-10 statistics)
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )

        if mode == 'basic':
            # Basic augmentation (horizontal flip only)
            aug_list = [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]

        elif mode == 'standard':
            # Standard augmentation (flip + crop + color jitter)
            aug_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize
            ]

        elif mode == 'autoaugment':
            # AutoAugment CIFAR-10 policy
            aug_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                normalize
            ]

        elif mode == 'randaugment':
            # RandAugment
            aug_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugment(n=2, m=10),
                transforms.ToTensor(),
                normalize
            ]

        else:
            raise ValueError(f"Unknown augmentation mode: {mode}")

        # Add Cutout or Random Erasing
        if cutout:
            aug_list.append(Cutout(n_holes=1, length=16))
        if random_erasing:
            aug_list.append(RandomErasing(p=0.5))

        self.transform = transforms.Compose(aug_list)

    def __call__(self, img):
        return self.transform(img)


class TestAugmentation:
    """
    Test-time augmentation (TTA).

    Applies multiple augmentations at test time and averages predictions.

    Args:
        n_augmentations: Number of augmented versions (default 5)
    """

    def __init__(self, n_augmentations=5):
        self.n_augmentations = n_augmentations

        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.tta_transforms = [
            # Original
            transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                normalize
            ]),
            # Small rotations
            transforms.Compose([
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                normalize
            ]),
            # Color jitter
            transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                normalize
            ]),
            # Crop
            transforms.Compose([
                transforms.RandomCrop(32, padding=2),
                transforms.ToTensor(),
                normalize
            ]),
        ]

    def __call__(self, img):
        """Returns list of augmented images."""
        return [transform(img) for transform in self.tta_transforms[:self.n_augmentations]]


def get_cifar10_transforms(train=True, augmentation_mode='standard',
                          cutout=False, random_erasing=False):
    """
    Get CIFAR-10 data transforms.

    Args:
        train: Whether for training (True) or testing (False)
        augmentation_mode: 'basic', 'standard', 'autoaugment', 'randaugment'
        cutout: Apply cutout
        random_erasing: Apply random erasing

    Returns:
        Transform pipeline
    """
    if train:
        return CIFAR10Augmentation(mode=augmentation_mode,
                                  cutout=cutout,
                                  random_erasing=random_erasing)
    else:
        # Test transform (no augmentation)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
