"""
Spectrum Dataset Module

PyTorch Dataset for loading and preprocessing mass spectrometry data.
Supports .raw, mzML, and CSV formats with caching and augmentation.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Callable
import numpy as np
import torch
from torch.utils.data import Dataset

from .raw_converter import RawConverter
from .mzml_reader import MzMLReader
from .csv_reader import CSVReader
from .preprocessing import SpectrumPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectrumDataset(Dataset):
    """
    PyTorch Dataset for mass spectrometry data.

    Supports multiple input formats and automatic preprocessing.
    """

    def __init__(
        self,
        data_path: Union[str, Path, List[str]],
        format: str = "auto",  # 'auto', 'raw', 'mzml', 'csv'
        task: str = "classification",  # 'classification', 'regression', 'similarity'
        labels: Optional[Union[List, np.ndarray, Dict]] = None,
        mz_range: Tuple[float, float] = (50.0, 2000.0),
        bin_size: float = 0.1,
        normalization: str = "tic",
        ms_level: int = 2,
        min_peaks: int = 5,
        augment: bool = False,
        cache_dir: Optional[str] = "./data/proteomics/processed",
        use_cache: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize SpectrumDataset.

        Args:
            data_path: Path to data file(s) or directory
            format: Input format ('auto', 'raw', 'mzml', 'csv')
            task: Task type ('classification', 'regression', 'similarity')
            labels: Labels for classification/regression (list, array, or dict mapping scan_id -> label)
            mz_range: m/z range for binning
            bin_size: Bin size in Da
            normalization: Normalization method
            ms_level: MS level to extract (1 or 2)
            min_peaks: Minimum number of peaks required
            augment: Enable data augmentation
            cache_dir: Directory for caching preprocessed spectra
            use_cache: Use cached data if available
            transform: Additional custom transform function
        """
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.format = format
        self.task = task
        self.labels = labels
        self.ms_level = ms_level
        self.min_peaks = min_peaks
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self.transform = transform

        # Initialize preprocessor
        self.preprocessor = SpectrumPreprocessor(
            mz_range=mz_range,
            bin_size=bin_size,
            normalization=normalization
        )

        # Load and preprocess spectra
        self.spectra = []
        self.processed_labels = []
        self._load_data()

        logger.info(f"SpectrumDataset initialized: {len(self.spectra)} spectra")

    def _load_data(self):
        """Load and preprocess spectral data."""
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache key
        cache_file = None
        if self.use_cache and self.cache_dir:
            cache_key = f"{self.data_path.stem}_{self.format}_{self.ms_level}_{self.preprocessor.bin_size}.pkl"
            cache_file = self.cache_dir / cache_key

            # Try to load from cache
            if cache_file.exists():
                logger.info(f"Loading from cache: {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.spectra = cached_data['spectra']
                        self._process_labels()
                        return
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")

        # Detect format if auto
        if self.format == "auto":
            self.format = self._detect_format()

        # Load spectra based on format
        logger.info(f"Loading spectra from {self.format} file...")
        raw_spectra = self._load_raw_spectra()

        # Preprocess and bin spectra
        logger.info("Preprocessing and binning spectra...")
        for spec in raw_spectra:
            try:
                # Preprocess
                mz, intensity = self.preprocessor.preprocess(
                    spec['mz'],
                    spec['intensity'],
                    remove_precursor=spec.get('precursor_mz', None)
                )

                # Bin spectrum
                binned = self.preprocessor.bin_spectrum(mz, intensity)

                # Store processed spectrum with metadata
                self.spectra.append({
                    'spectrum': binned,
                    'scan_id': spec['scan_id'],
                    'rt': spec.get('rt', None),
                    'precursor_mz': spec.get('precursor_mz', None),
                    'ms_level': spec['ms_level'],
                })
            except Exception as e:
                logger.warning(f"Failed to process spectrum {spec.get('scan_id', 'unknown')}: {e}")

        # Cache processed spectra
        if cache_file:
            logger.info(f"Saving to cache: {cache_file}")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'spectra': self.spectra}, f)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

        # Process labels
        self._process_labels()

    def _detect_format(self) -> str:
        """Auto-detect input format."""
        suffix = self.data_path.suffix.lower()

        if suffix == '.raw':
            return 'raw'
        elif suffix in ['.mzml', '.mzxml']:
            return 'mzml'
        elif suffix == '.csv':
            return 'csv'
        else:
            logger.warning(f"Unknown format for {suffix}, defaulting to mzml")
            return 'mzml'

    def _load_raw_spectra(self) -> List[Dict]:
        """Load raw spectra from file."""
        if self.format == 'raw':
            # Convert .raw to mzML first
            converter = RawConverter()
            mzml_file = converter.convert(self.data_path)
            if mzml_file is None:
                raise ValueError(f"Failed to convert .raw file: {self.data_path}")
            # Read mzML
            reader = MzMLReader(mzml_file)
            return reader.read_spectra(ms_level=self.ms_level, min_peaks=self.min_peaks)

        elif self.format == 'mzml':
            reader = MzMLReader(self.data_path)
            return reader.read_spectra(ms_level=self.ms_level, min_peaks=self.min_peaks)

        elif self.format == 'csv':
            reader = CSVReader(self.data_path)
            return reader.read_spectra(min_peaks=self.min_peaks)

        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _process_labels(self):
        """Process labels for the task."""
        if self.labels is None:
            # No labels provided (e.g., for unsupervised learning)
            self.processed_labels = [None] * len(self.spectra)
            return

        # Handle different label formats
        if isinstance(self.labels, dict):
            # Dict mapping scan_id -> label
            self.processed_labels = []
            for spec in self.spectra:
                scan_id = spec['scan_id']
                label = self.labels.get(scan_id, None)
                self.processed_labels.append(label)
        elif isinstance(self.labels, (list, np.ndarray)):
            # List/array of labels (must match spectrum order)
            if len(self.labels) != len(self.spectra):
                logger.warning(
                    f"Label count mismatch: {len(self.labels)} labels for {len(self.spectra)} spectra"
                )
            self.processed_labels = list(self.labels)
        else:
            raise ValueError(f"Unsupported label type: {type(self.labels)}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.spectra)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[Union[int, float]]]:
        """
        Get a single spectrum and label.

        Returns:
            (spectrum_tensor, label) or (spectrum_tensor, None) if no labels
        """
        spec_data = self.spectra[idx]
        spectrum = spec_data['spectrum'].copy()

        # Apply augmentation if enabled (only for training)
        if self.augment:
            # Add noise
            noise = np.random.normal(0, 0.02 * np.mean(spectrum), size=spectrum.shape)
            spectrum = np.maximum(spectrum + noise, 0)

            # Random intensity scaling
            scale = np.random.uniform(0.8, 1.2)
            spectrum = spectrum * scale

        # Convert to tensor
        spectrum_tensor = torch.from_numpy(spectrum).float()

        # Apply custom transform if provided
        if self.transform:
            spectrum_tensor = self.transform(spectrum_tensor)

        # Get label
        label = self.processed_labels[idx]

        # Convert label to tensor if not None
        if label is not None:
            if self.task == 'classification':
                label = torch.tensor(label, dtype=torch.long)
            else:  # regression
                label = torch.tensor(label, dtype=torch.float)

        return spectrum_tensor, label

    def get_metadata(self, idx: int) -> Dict:
        """
        Get spectrum metadata.

        Args:
            idx: Spectrum index

        Returns:
            Metadata dictionary
        """
        spec_data = self.spectra[idx]
        return {
            'scan_id': spec_data['scan_id'],
            'rt': spec_data['rt'],
            'precursor_mz': spec_data['precursor_mz'],
            'ms_level': spec_data['ms_level'],
        }

    def get_spectrum_shape(self) -> Tuple[int]:
        """Get shape of binned spectrum."""
        return (self.preprocessor.num_bins,)


def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        batch_size: Batch size
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for SpectrumDataset

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    from torch.utils.data import DataLoader

    dataloaders = {}

    # Training dataloader (with augmentation)
    train_dataset = SpectrumDataset(train_path, augment=True, **dataset_kwargs)
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Validation dataloader (no augmentation)
    if val_path:
        val_dataset = SpectrumDataset(val_path, augment=False, **dataset_kwargs)
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # Test dataloader (no augmentation)
    if test_path:
        test_dataset = SpectrumDataset(test_path, augment=False, **dataset_kwargs)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return dataloaders


# Example usage
if __name__ == "__main__":
    # Create dataset
    # dataset = SpectrumDataset(
    #     data_path="./data/proteomics/mzml/sample.mzML",
    #     format="mzml",
    #     task="classification",
    #     labels=[0, 1, 0, 1, 1],  # Example labels
    # )

    # print(f"Dataset size: {len(dataset)}")
    # spectrum, label = dataset[0]
    # print(f"Spectrum shape: {spectrum.shape}, Label: {label}")

    print("SpectrumDataset initialized.")
