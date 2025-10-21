"""
CSV Reader Module

Reads pre-processed proteomics data from CSV files.
Supports various CSV formats with RT, m/z, intensity data.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVReader:
    """
    Reader for pre-processed proteomics data in CSV format.

    Supports multiple CSV schemas:
    1. Spectrum-level: Each row is a spectrum with aggregated features
    2. Peak-level: Each row is a peak (m/z, intensity) with scan metadata
    3. Feature matrix: Rows are samples, columns are m/z bins
    """

    def __init__(self, file_path: Union[str, Path], format: str = "auto"):
        """
        Initialize CSVReader.

        Args:
            file_path: Path to CSV file
            format: CSV format ('auto', 'spectrum', 'peak', 'matrix')
        """
        self.file_path = Path(file_path)
        self.format = format

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Load CSV
        self.df = pd.read_csv(self.file_path)
        logger.info(f"Loaded CSV: {self.file_path.name} ({len(self.df)} rows)")

        # Auto-detect format if requested
        if self.format == "auto":
            self.format = self._detect_format()
            logger.info(f"Detected format: {self.format}")

    def _detect_format(self) -> str:
        """
        Auto-detect CSV format based on column names.

        Returns:
            Detected format: 'spectrum', 'peak', or 'matrix'
        """
        columns = set(self.df.columns.str.lower())

        # Check for peak-level format (has both scan_id and mz/intensity per row)
        if 'mz' in columns and 'intensity' in columns and 'scan_id' in columns:
            return 'peak'

        # Check for spectrum-level format (has scan_id and aggregated features)
        if 'scan_id' in columns and 'num_peaks' in columns:
            return 'spectrum'

        # Check for matrix format (numeric columns are m/z bins)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) / len(self.df.columns) > 0.8:  # Mostly numeric
            return 'matrix'

        # Default to spectrum format
        logger.warning("Could not auto-detect format. Defaulting to 'spectrum'")
        return 'spectrum'

    def read_spectra(
        self,
        min_peaks: int = 5,
        rt_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict]:
        """
        Read spectra from CSV.

        Args:
            min_peaks: Minimum number of peaks required
            rt_range: Retention time range (min_rt, max_rt)

        Returns:
            List of spectrum dictionaries (same format as MzMLReader)
        """
        if self.format == 'peak':
            return self._read_peak_format(min_peaks, rt_range)
        elif self.format == 'spectrum':
            return self._read_spectrum_format(min_peaks, rt_range)
        elif self.format == 'matrix':
            return self._read_matrix_format(min_peaks, rt_range)
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _read_peak_format(
        self,
        min_peaks: int,
        rt_range: Optional[Tuple[float, float]]
    ) -> List[Dict]:
        """
        Read peak-level CSV format.

        Expected columns: scan_id, rt, mz, intensity, [precursor_mz], [precursor_charge]
        """
        spectra = []
        df = self.df.copy()

        # Normalize column names
        df.columns = df.columns.str.lower()

        # Group by scan_id
        for scan_id, group in df.groupby('scan_id'):
            # Extract RT (take first value, should be same for all peaks in scan)
            rt = group['rt'].iloc[0] if 'rt' in group.columns else None

            # Filter by RT range
            if rt_range and rt is not None and not (rt_range[0] <= rt <= rt_range[1]):
                continue

            # Extract m/z and intensity arrays
            mz_array = group['mz'].values
            intensity_array = group['intensity'].values

            # Filter by minimum peaks
            if len(mz_array) < min_peaks:
                continue

            # Extract precursor info
            precursor_mz = group['precursor_mz'].iloc[0] if 'precursor_mz' in group.columns else None
            precursor_charge = group['precursor_charge'].iloc[0] if 'precursor_charge' in group.columns else None

            # Infer MS level (MS2 if has precursor, MS1 otherwise)
            ms_level = 2 if precursor_mz is not None else 1

            spectra.append({
                'scan_id': scan_id,
                'ms_level': ms_level,
                'rt': rt,
                'mz': mz_array,
                'intensity': intensity_array,
                'precursor_mz': precursor_mz,
                'precursor_charge': precursor_charge,
            })

        logger.info(f"Extracted {len(spectra)} spectra from peak-level CSV")
        return spectra

    def _read_spectrum_format(
        self,
        min_peaks: int,
        rt_range: Optional[Tuple[float, float]]
    ) -> List[Dict]:
        """
        Read spectrum-level CSV format.

        Expected columns: scan_id, rt, num_peaks, [precursor_mz], [mz_array], [intensity_array]

        Note: This format may not have full peak arrays, only aggregated features.
        """
        spectra = []
        df = self.df.copy()

        # Normalize column names
        df.columns = df.columns.str.lower()

        for idx, row in df.iterrows():
            scan_id = row.get('scan_id', f'scan_{idx}')
            rt = row.get('rt', None)
            num_peaks = row.get('num_peaks', 0)

            # Filter by RT range
            if rt_range and rt is not None and not (rt_range[0] <= rt <= rt_range[1]):
                continue

            # Filter by minimum peaks
            if num_peaks < min_peaks:
                continue

            # Try to extract m/z and intensity arrays (if available)
            mz_array = self._parse_array(row.get('mz_array', None))
            intensity_array = self._parse_array(row.get('intensity_array', None))

            # If arrays not available, create empty arrays
            if mz_array is None or intensity_array is None:
                mz_array = np.array([])
                intensity_array = np.array([])

            precursor_mz = row.get('precursor_mz', None)
            precursor_charge = row.get('precursor_charge', None)
            ms_level = row.get('ms_level', 2 if precursor_mz is not None else 1)

            spectra.append({
                'scan_id': scan_id,
                'ms_level': ms_level,
                'rt': rt,
                'mz': mz_array,
                'intensity': intensity_array,
                'precursor_mz': precursor_mz,
                'precursor_charge': precursor_charge,
            })

        logger.info(f"Extracted {len(spectra)} spectra from spectrum-level CSV")
        return spectra

    def _read_matrix_format(
        self,
        min_peaks: int,
        rt_range: Optional[Tuple[float, float]]
    ) -> List[Dict]:
        """
        Read feature matrix CSV format.

        Rows are samples/scans, columns are m/z bins (features).
        First column may be sample ID, subsequent columns are numeric features.
        """
        spectra = []
        df = self.df.copy()

        # Identify ID column (usually first non-numeric column)
        id_col = None
        for col in df.columns:
            if df[col].dtype == 'object' or col.lower() in ['scan_id', 'id', 'sample']:
                id_col = col
                break

        # Extract feature columns (numeric)
        if id_col:
            feature_cols = [col for col in df.columns if col != id_col]
        else:
            feature_cols = df.columns.tolist()
            id_col = None  # No ID column

        # Try to parse m/z bins from column names (e.g., "mz_100.5", "100.5", etc.)
        mz_bins = []
        for col in feature_cols:
            try:
                # Try to extract numeric value from column name
                mz_value = float(col.replace('mz_', '').replace('mz', ''))
                mz_bins.append(mz_value)
            except (ValueError, AttributeError):
                # Use column index as m/z value
                mz_bins.append(float(feature_cols.index(col)))

        mz_bins = np.array(mz_bins)

        for idx, row in df.iterrows():
            scan_id = row[id_col] if id_col else f'scan_{idx}'
            intensity_array = row[feature_cols].values.astype(float)

            # Filter zero/low intensity peaks
            non_zero_mask = intensity_array > 0
            mz_array = mz_bins[non_zero_mask]
            intensity_array = intensity_array[non_zero_mask]

            # Filter by minimum peaks
            if len(mz_array) < min_peaks:
                continue

            spectra.append({
                'scan_id': scan_id,
                'ms_level': 1,  # Assume MS1 for matrix format
                'rt': None,  # RT not available in matrix format
                'mz': mz_array,
                'intensity': intensity_array,
                'precursor_mz': None,
                'precursor_charge': None,
            })

        logger.info(f"Extracted {len(spectra)} spectra from matrix CSV")
        return spectra

    @staticmethod
    def _parse_array(value: Union[str, np.ndarray, None]) -> Optional[np.ndarray]:
        """
        Parse array from string representation.

        Supports: "[1, 2, 3]", "1,2,3", "1 2 3"
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None

        if isinstance(value, np.ndarray):
            return value

        if isinstance(value, str):
            # Remove brackets and split
            value = value.strip('[]()').replace(',', ' ')
            try:
                return np.array([float(x) for x in value.split()])
            except ValueError:
                return None

        return None

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the loaded DataFrame.

        Returns:
            Original pandas DataFrame
        """
        return self.df


# Example usage
if __name__ == "__main__":
    # Read CSV file
    # reader = CSVReader("./data/proteomics/processed/spectra.csv", format="auto")

    # Extract spectra
    # spectra = reader.read_spectra(min_peaks=10)
    # print(f"Found {len(spectra)} spectra")

    print("CSVReader initialized. Use read_spectra() to extract spectra.")
