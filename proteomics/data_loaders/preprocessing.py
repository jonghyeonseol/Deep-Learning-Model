"""
Spectrum Preprocessing Module

Provides functions for spectrum preprocessing, normalization, binning, and augmentation.
"""

import logging
from typing import Tuple, Optional, Dict, List
import numpy as np
from scipy import sparse
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectrumPreprocessor:
    """
    Preprocessor for mass spectra.

    Handles binning, normalization, peak picking, and data augmentation.
    """

    def __init__(
        self,
        mz_range: Tuple[float, float] = (50.0, 2000.0),
        bin_size: float = 0.1,
        normalization: str = "tic",  # 'tic', 'max', 'sqrt', 'none'
        min_intensity_ratio: float = 0.01,  # Filter peaks below 1% of base peak
    ):
        """
        Initialize SpectrumPreprocessor.

        Args:
            mz_range: m/z range (min, max) to keep
            bin_size: Bin size in Da for spectrum binning
            normalization: Normalization method ('tic', 'max', 'sqrt', 'none')
            min_intensity_ratio: Minimum intensity ratio to base peak
        """
        self.mz_range = mz_range
        self.bin_size = bin_size
        self.normalization = normalization
        self.min_intensity_ratio = min_intensity_ratio

        # Calculate number of bins
        self.num_bins = int((mz_range[1] - mz_range[0]) / bin_size)
        self.mz_bins = np.linspace(mz_range[0], mz_range[1], self.num_bins + 1)

        logger.info(
            f"SpectrumPreprocessor: {self.num_bins} bins "
            f"({mz_range[0]:.1f}-{mz_range[1]:.1f} Da, {bin_size} Da bins)"
        )

    def preprocess(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        remove_precursor: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a single spectrum.

        Args:
            mz: m/z array
            intensity: Intensity array
            remove_precursor: Precursor m/z to remove (±17 Da window)

        Returns:
            Preprocessed (mz, intensity) arrays
        """
        # Filter by m/z range
        mz, intensity = self._filter_mz_range(mz, intensity)

        # Remove precursor peak (for MS2 spectra)
        if remove_precursor is not None:
            mz, intensity = self._remove_precursor(mz, intensity, remove_precursor)

        # Filter low-intensity peaks
        mz, intensity = self._filter_low_intensity(mz, intensity)

        # Normalize intensities
        intensity = self._normalize(intensity)

        return mz, intensity

    def bin_spectrum(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        sparse_output: bool = False
    ) -> np.ndarray:
        """
        Bin spectrum into fixed-size vector.

        Args:
            mz: m/z array
            intensity: Intensity array
            sparse_output: Return scipy sparse array (for memory efficiency)

        Returns:
            Binned spectrum vector (num_bins,)
        """
        # Initialize binned spectrum
        if sparse_output:
            binned = sparse.lil_matrix((1, self.num_bins), dtype=np.float32)
        else:
            binned = np.zeros(self.num_bins, dtype=np.float32)

        # Assign peaks to bins (sum intensities in same bin)
        for mz_val, intensity_val in zip(mz, intensity):
            if self.mz_range[0] <= mz_val <= self.mz_range[1]:
                bin_idx = int((mz_val - self.mz_range[0]) / self.bin_size)
                bin_idx = min(bin_idx, self.num_bins - 1)  # Clamp to last bin

                if sparse_output:
                    binned[0, bin_idx] += intensity_val
                else:
                    binned[bin_idx] += intensity_val

        if sparse_output:
            return binned.tocsr()
        else:
            return binned

    def _filter_mz_range(
        self,
        mz: np.ndarray,
        intensity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter peaks by m/z range."""
        mask = (mz >= self.mz_range[0]) & (mz <= self.mz_range[1])
        return mz[mask], intensity[mask]

    def _remove_precursor(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        precursor_mz: float,
        window: float = 17.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove precursor peak and isotopes (±17 Da window)."""
        mask = ~((mz >= precursor_mz - window) & (mz <= precursor_mz + window))
        return mz[mask], intensity[mask]

    def _filter_low_intensity(
        self,
        mz: np.ndarray,
        intensity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter peaks below minimum intensity ratio."""
        if len(intensity) == 0:
            return mz, intensity

        base_peak_intensity = np.max(intensity)
        min_intensity = base_peak_intensity * self.min_intensity_ratio

        mask = intensity >= min_intensity
        return mz[mask], intensity[mask]

    def _normalize(self, intensity: np.ndarray) -> np.ndarray:
        """
        Normalize intensity array.

        Methods:
        - 'tic': Total ion current (sum) normalization
        - 'max': Max intensity normalization
        - 'sqrt': Square root scaling
        - 'none': No normalization
        """
        if len(intensity) == 0 or self.normalization == 'none':
            return intensity

        if self.normalization == 'tic':
            total = np.sum(intensity)
            return intensity / total if total > 0 else intensity

        elif self.normalization == 'max':
            max_val = np.max(intensity)
            return intensity / max_val if max_val > 0 else intensity

        elif self.normalization == 'sqrt':
            return np.sqrt(intensity)

        else:
            logger.warning(f"Unknown normalization: {self.normalization}")
            return intensity

    def peak_picking(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        height: Optional[float] = None,
        distance: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform peak picking to reduce number of peaks.

        Args:
            mz: m/z array
            intensity: Intensity array
            height: Minimum peak height (default: 1% of max)
            distance: Minimum distance between peaks (in data points)

        Returns:
            Filtered (mz, intensity) arrays containing only peaks
        """
        if len(intensity) == 0:
            return mz, intensity

        # Set default height threshold
        if height is None:
            height = np.max(intensity) * 0.01

        # Find peaks
        peaks, _ = find_peaks(intensity, height=height, distance=distance)

        return mz[peaks], intensity[peaks]

    def augment_spectrum(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        noise_level: float = 0.02,
        intensity_scale_range: Tuple[float, float] = (0.8, 1.2),
        mz_shift_range: Tuple[float, float] = (-0.05, 0.05)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment spectrum with noise and random transformations.

        Args:
            mz: m/z array
            intensity: Intensity array
            noise_level: Gaussian noise level (std as fraction of mean intensity)
            intensity_scale_range: Random intensity scaling range
            mz_shift_range: Random m/z shift range (Da)

        Returns:
            Augmented (mz, intensity) arrays
        """
        if len(intensity) == 0:
            return mz, intensity

        # Add Gaussian noise to intensities
        noise = np.random.normal(0, noise_level * np.mean(intensity), size=intensity.shape)
        intensity_aug = np.maximum(intensity + noise, 0)  # Clip to non-negative

        # Random intensity scaling
        scale = np.random.uniform(*intensity_scale_range)
        intensity_aug = intensity_aug * scale

        # Random m/z shift
        mz_shift = np.random.uniform(*mz_shift_range)
        mz_aug = mz + mz_shift

        return mz_aug, intensity_aug


class SpectrumBatcher:
    """
    Batch spectrum processing utilities.
    """

    @staticmethod
    def batch_bin_spectra(
        spectra: List[Dict],
        preprocessor: SpectrumPreprocessor,
        sparse_output: bool = False
    ) -> np.ndarray:
        """
        Bin multiple spectra into matrix.

        Args:
            spectra: List of spectrum dictionaries (from readers)
            preprocessor: SpectrumPreprocessor instance
            sparse_output: Return scipy sparse matrix

        Returns:
            Binned spectrum matrix (num_spectra, num_bins)
        """
        binned_spectra = []

        for spec in spectra:
            # Preprocess
            mz, intensity = preprocessor.preprocess(
                spec['mz'],
                spec['intensity'],
                remove_precursor=spec.get('precursor_mz', None)
            )

            # Bin
            binned = preprocessor.bin_spectrum(mz, intensity, sparse_output=False)
            binned_spectra.append(binned)

        # Stack into matrix
        if sparse_output:
            return sparse.vstack(binned_spectra, format='csr')
        else:
            return np.vstack(binned_spectra)

    @staticmethod
    def calculate_spectral_similarity(
        spec1: np.ndarray,
        spec2: np.ndarray,
        method: str = "cosine"
    ) -> float:
        """
        Calculate spectral similarity between two binned spectra.

        Args:
            spec1: Binned spectrum 1
            spec2: Binned spectrum 2
            method: Similarity method ('cosine', 'dot', 'euclidean')

        Returns:
            Similarity score
        """
        if method == "cosine":
            # Cosine similarity
            dot = np.dot(spec1, spec2)
            norm1 = np.linalg.norm(spec1)
            norm2 = np.linalg.norm(spec2)
            return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        elif method == "dot":
            # Dot product
            return np.dot(spec1, spec2)

        elif method == "euclidean":
            # Euclidean distance (convert to similarity)
            dist = np.linalg.norm(spec1 - spec2)
            return 1.0 / (1.0 + dist)

        else:
            raise ValueError(f"Unknown method: {method}")


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = SpectrumPreprocessor(
        mz_range=(50, 2000),
        bin_size=0.1,
        normalization='tic'
    )

    # Example spectrum
    mz = np.array([100.5, 150.3, 200.1, 500.7, 1000.2])
    intensity = np.array([1000, 5000, 2000, 8000, 3000])

    # Preprocess
    mz_proc, intensity_proc = preprocessor.preprocess(mz, intensity)
    print(f"Preprocessed: {len(mz_proc)} peaks")

    # Bin spectrum
    binned = preprocessor.bin_spectrum(mz_proc, intensity_proc)
    print(f"Binned shape: {binned.shape}")

    # Augment
    mz_aug, intensity_aug = preprocessor.augment_spectrum(mz, intensity)
    print(f"Augmented: {len(mz_aug)} peaks")
