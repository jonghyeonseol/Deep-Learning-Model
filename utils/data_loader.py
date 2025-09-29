import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np


class CIFAR10DataLoader:
    def __init__(self, batch_size=32, validation_split=0.1, data_dir='./data',
                 normalize=True, augment_train=True):
        """
        CIFAR-10 data loader with train/validation/test splits.

        Args:
            batch_size (int): Batch size for data loading
            validation_split (float): Fraction of training data to use for validation
            data_dir (str): Directory to store/load CIFAR-10 data
            normalize (bool): Whether to normalize the data
            augment_train (bool): Whether to apply data augmentation to training data
        """
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.data_dir = data_dir
        self.normalize = normalize
        self.augment_train = augment_train

        # CIFAR-10 class names
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        # Data transformations
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

        # Load datasets
        self._load_datasets()
        self._create_data_loaders()

    def _get_train_transform(self):
        """Get training data transformations."""
        transforms_list = []

        if self.augment_train:
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])

        transforms_list.append(transforms.ToTensor())

        if self.normalize:
            # CIFAR-10 dataset statistics
            transforms_list.append(
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                   std=[0.2023, 0.1994, 0.2010])
            )

        return transforms.Compose(transforms_list)

    def _get_test_transform(self):
        """Get test data transformations."""
        transforms_list = [transforms.ToTensor()]

        if self.normalize:
            transforms_list.append(
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                   std=[0.2023, 0.1994, 0.2010])
            )

        return transforms.Compose(transforms_list)

    def _load_datasets(self):
        """Load CIFAR-10 datasets."""
        # Load training data
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.train_transform
        )

        # Load test data
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.test_transform
        )

        # Split training data into train and validation
        if self.validation_split > 0:
            train_size = int((1 - self.validation_split) * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
        else:
            self.train_dataset = full_train_dataset
            self.val_dataset = None

    def _create_data_loaders(self):
        """Create data loaders for train, validation, and test sets."""
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
            )
        else:
            self.val_loader = None

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def get_data_loaders(self):
        """
        Get all data loaders.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader

    def get_dataset_info(self):
        """Get information about the datasets."""
        info = {
            'num_classes': len(self.classes),
            'class_names': self.classes,
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'input_shape': (3, 32, 32),
            'batch_size': self.batch_size
        }

        if self.val_dataset is not None:
            info['val_size'] = len(self.val_dataset)

        return info

    def get_sample_batch(self, dataset='train'):
        """
        Get a sample batch from the specified dataset.

        Args:
            dataset (str): Which dataset to sample from ('train', 'val', 'test')

        Returns:
            tuple: (images, labels) batch
        """
        if dataset == 'train':
            loader = self.train_loader
        elif dataset == 'val' and self.val_loader is not None:
            loader = self.val_loader
        elif dataset == 'test':
            loader = self.test_loader
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

        return next(iter(loader))

    def denormalize(self, tensor):
        """
        Denormalize a tensor for visualization.

        Args:
            tensor (torch.Tensor): Normalized tensor

        Returns:
            torch.Tensor: Denormalized tensor
        """
        if not self.normalize:
            return tensor

        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

        return tensor * std + mean

    def get_class_distribution(self, dataset='train'):
        """
        Get class distribution for the specified dataset.

        Args:
            dataset (str): Which dataset to analyze ('train', 'val', 'test')

        Returns:
            dict: Class distribution
        """
        if dataset == 'train':
            loader = self.train_loader
        elif dataset == 'val' and self.val_loader is not None:
            loader = self.val_loader
        elif dataset == 'test':
            loader = self.test_loader
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

        class_counts = torch.zeros(len(self.classes))

        for _, labels in loader:
            for label in labels:
                class_counts[label] += 1

        distribution = {}
        for i, class_name in enumerate(self.classes):
            distribution[class_name] = int(class_counts[i])

        return distribution


def create_cifar10_loader(batch_size=32, validation_split=0.1, data_dir='./data'):
    """
    Convenience function to create CIFAR-10 data loader.

    Args:
        batch_size (int): Batch size
        validation_split (float): Validation split ratio
        data_dir (str): Data directory

    Returns:
        CIFAR10DataLoader: Configured data loader
    """
    return CIFAR10DataLoader(
        batch_size=batch_size,
        validation_split=validation_split,
        data_dir=data_dir
    )