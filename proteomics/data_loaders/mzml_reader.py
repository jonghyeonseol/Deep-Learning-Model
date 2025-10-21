"""
mzML Reader Module

Reads and extracts mass spectra from mzML/mzXML files using pyteomics and pymzml.
Supports MS1, MS2 (MS/MS) spectrum extraction with metadata.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from pyteomics import mzml, mzxml
    PYTEOMICS_AVAILABLE = True
except ImportError:
    PYTEOMICS_AVAILABLE = False
    logging.warning("pyteomics not available. Install with: pip install pyteomics")

try:
    import pymzml
    PYMZML_AVAILABLE = True
except ImportError:
    PYMZML_AVAILABLE = False
    logging.warning("pymzml not available. Install with: pip install pymzml")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MzMLReader:
    """
    Reader for mzML and mzXML mass spectrometry data files.

    Extracts spectra with retention time, precursor m/z, and intensity arrays.
    """

    def __init__(self, file_path: Union[str, Path], backend: str = "pyteomics"):
        """
        Initialize MzMLReader.

        Args:
            file_path: Path to mzML or mzXML file
            backend: Parser backend ('pyteomics' or 'pymzml')
        """
        self.file_path = Path(file_path)
        self.backend = backend

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Determine file format
        self.file_format = self.file_path.suffix.lower()
        if self.file_format not in ['.mzml', '.mzxml']:
            raise ValueError(f"Unsupported format: {self.file_format}. Use .mzml or .mzxml")

        # Check backend availability
        if backend == "pyteomics" and not PYTEOMICS_AVAILABLE:
            raise ImportError("pyteomics not installed. Install with: pip install pyteomics")
        if backend == "pymzml" and not PYMZML_AVAILABLE:
            raise ImportError("pymzml not installed. Install with: pip install pymzml")

        logger.info(f"MzMLReader initialized: {self.file_path.name} (backend: {backend})")

    def read_spectra(
        self,
        ms_level: Optional[int] = None,
        min_peaks: int = 5,
        rt_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict]:
        """
        Read all spectra from the file.

        Args:
            ms_level: MS level to extract (None = all, 1 = MS1, 2 = MS2/MS/MS)
            min_peaks: Minimum number of peaks required
            rt_range: Retention time range (min_rt, max_rt) in minutes

        Returns:
            List of spectrum dictionaries with keys:
                - scan_id: Scan/spectrum ID
                - ms_level: MS level (1 or 2)
                - rt: Retention time (minutes)
                - mz: m/z array (numpy array)
                - intensity: Intensity array (numpy array)
                - precursor_mz: Precursor m/z (for MS2, None for MS1)
                - precursor_charge: Precursor charge (for MS2, None for MS1)
        """
        if self.backend == "pyteomics":
            return self._read_pyteomics(ms_level, min_peaks, rt_range)
        elif self.backend == "pymzml":
            return self._read_pymzml(ms_level, min_peaks, rt_range)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _read_pyteomics(
        self,
        ms_level: Optional[int],
        min_peaks: int,
        rt_range: Optional[Tuple[float, float]]
    ) -> List[Dict]:
        """Read spectra using pyteomics backend."""
        spectra = []

        # Choose reader based on format
        if self.file_format == '.mzml':
            reader = mzml.read(str(self.file_path))
        else:  # .mzxml
            reader = mzxml.read(str(self.file_path))

        for spectrum in reader:
            # Extract MS level
            spec_ms_level = spectrum.get('ms level', 1)
            if ms_level is not None and spec_ms_level != ms_level:
                continue

            # Extract retention time (convert to minutes)
            rt = self._extract_rt_pyteomics(spectrum)
            if rt is None:
                continue

            # Filter by RT range
            if rt_range and not (rt_range[0] <= rt <= rt_range[1]):
                continue

            # Extract m/z and intensity arrays
            mz_array = spectrum.get('m/z array', np.array([]))
            intensity_array = spectrum.get('intensity array', np.array([]))

            # Filter by minimum peaks
            if len(mz_array) < min_peaks:
                continue

            # Extract scan ID
            scan_id = spectrum.get('id', spectrum.get('index', 'unknown'))

            # Extract precursor information (for MS2)
            precursor_mz = None
            precursor_charge = None
            if spec_ms_level == 2:
                precursor_mz, precursor_charge = self._extract_precursor_pyteomics(spectrum)

            spectra.append({
                'scan_id': scan_id,
                'ms_level': spec_ms_level,
                'rt': rt,
                'mz': mz_array,
                'intensity': intensity_array,
                'precursor_mz': precursor_mz,
                'precursor_charge': precursor_charge,
            })

        logger.info(f"Extracted {len(spectra)} spectra (MS level: {ms_level or 'all'})")
        return spectra

    def _read_pymzml(
        self,
        ms_level: Optional[int],
        min_peaks: int,
        rt_range: Optional[Tuple[float, float]]
    ) -> List[Dict]:
        """Read spectra using pymzml backend."""
        spectra = []

        run = pymzml.run.Reader(str(self.file_path))

        for spectrum in run:
            # Extract MS level
            spec_ms_level = spectrum.ms_level
            if ms_level is not None and spec_ms_level != ms_level:
                continue

            # Extract retention time (already in minutes)
            rt = spectrum.scan_time_in_minutes()
            if rt is None:
                continue

            # Filter by RT range
            if rt_range and not (rt_range[0] <= rt <= rt_range[1]):
                continue

            # Extract m/z and intensity arrays
            mz_array = spectrum.mz
            intensity_array = spectrum.i

            # Filter by minimum peaks
            if len(mz_array) < min_peaks:
                continue

            # Extract scan ID
            scan_id = spectrum.ID

            # Extract precursor information (for MS2)
            precursor_mz = None
            precursor_charge = None
            if spec_ms_level == 2:
                precursor = spectrum.selected_precursors
                if precursor and len(precursor) > 0:
                    precursor_mz = precursor[0].get('mz', None)
                    precursor_charge = precursor[0].get('charge', None)

            spectra.append({
                'scan_id': scan_id,
                'ms_level': spec_ms_level,
                'rt': rt,
                'mz': mz_array,
                'intensity': intensity_array,
                'precursor_mz': precursor_mz,
                'precursor_charge': precursor_charge,
            })

        logger.info(f"Extracted {len(spectra)} spectra (MS level: {ms_level or 'all'})")
        return spectra

    @staticmethod
    def _extract_rt_pyteomics(spectrum: Dict) -> Optional[float]:
        """Extract retention time from pyteomics spectrum (convert to minutes)."""
        try:
            # Try scanList first (mzML)
            if 'scanList' in spectrum:
                scan_list = spectrum['scanList']
                if 'scan' in scan_list and len(scan_list['scan']) > 0:
                    scan = scan_list['scan'][0]
                    rt = scan.get('scan start time', None)
                    if rt is not None:
                        return rt  # Already in minutes

            # Try retentionTime (mzXML)
            if 'retentionTime' in spectrum:
                rt = spectrum['retentionTime']
                # Convert from "PT{value}S" format if necessary
                if isinstance(rt, str) and rt.startswith('PT'):
                    rt_seconds = float(rt[2:-1])  # Remove "PT" and "S"
                    return rt_seconds / 60.0
                return rt

            return None
        except Exception as e:
            logger.warning(f"Failed to extract RT: {e}")
            return None

    @staticmethod
    def _extract_precursor_pyteomics(spectrum: Dict) -> Tuple[Optional[float], Optional[int]]:
        """Extract precursor m/z and charge from pyteomics spectrum."""
        try:
            if 'precursorList' in spectrum:
                precursor_list = spectrum['precursorList']
                if 'precursor' in precursor_list and len(precursor_list['precursor']) > 0:
                    precursor = precursor_list['precursor'][0]

                    # Extract selected ion
                    if 'selectedIonList' in precursor:
                        selected_ion_list = precursor['selectedIonList']
                        if 'selectedIon' in selected_ion_list and len(selected_ion_list['selectedIon']) > 0:
                            selected_ion = selected_ion_list['selectedIon'][0]
                            precursor_mz = selected_ion.get('selected ion m/z', None)
                            precursor_charge = selected_ion.get('charge state', None)
                            return precursor_mz, precursor_charge

            return None, None
        except Exception as e:
            logger.warning(f"Failed to extract precursor info: {e}")
            return None, None

    def to_dataframe(
        self,
        ms_level: Optional[int] = None,
        aggregate_peaks: bool = False
    ) -> pd.DataFrame:
        """
        Convert spectra to pandas DataFrame.

        Args:
            ms_level: MS level to extract
            aggregate_peaks: If True, aggregate all peaks into single rows (loses peak-level info)

        Returns:
            DataFrame with spectrum metadata (and aggregated peaks if requested)
        """
        spectra = self.read_spectra(ms_level=ms_level)

        if aggregate_peaks:
            # Aggregate: one row per spectrum
            df_data = []
            for spec in spectra:
                df_data.append({
                    'scan_id': spec['scan_id'],
                    'ms_level': spec['ms_level'],
                    'rt': spec['rt'],
                    'num_peaks': len(spec['mz']),
                    'precursor_mz': spec['precursor_mz'],
                    'precursor_charge': spec['precursor_charge'],
                    'total_intensity': np.sum(spec['intensity']),
                    'base_peak_mz': spec['mz'][np.argmax(spec['intensity'])] if len(spec['mz']) > 0 else None,
                    'base_peak_intensity': np.max(spec['intensity']) if len(spec['intensity']) > 0 else None,
                })
            return pd.DataFrame(df_data)
        else:
            # Non-aggregated: one row per peak (long format)
            df_data = []
            for spec in spectra:
                for mz, intensity in zip(spec['mz'], spec['intensity']):
                    df_data.append({
                        'scan_id': spec['scan_id'],
                        'ms_level': spec['ms_level'],
                        'rt': spec['rt'],
                        'precursor_mz': spec['precursor_mz'],
                        'precursor_charge': spec['precursor_charge'],
                        'mz': mz,
                        'intensity': intensity,
                    })
            return pd.DataFrame(df_data)

    def get_spectrum_by_scan_id(self, scan_id: str) -> Optional[Dict]:
        """
        Retrieve a single spectrum by scan ID.

        Args:
            scan_id: Scan/spectrum ID

        Returns:
            Spectrum dictionary or None if not found
        """
        spectra = self.read_spectra()
        for spec in spectra:
            if spec['scan_id'] == scan_id:
                return spec
        return None


# Example usage
if __name__ == "__main__":
    # Read mzML file
    # reader = MzMLReader("./data/proteomics/mzml/sample.mzML")

    # Extract MS2 spectra
    # ms2_spectra = reader.read_spectra(ms_level=2, min_peaks=10)
    # print(f"Found {len(ms2_spectra)} MS2 spectra")

    # Convert to DataFrame
    # df = reader.to_dataframe(ms_level=2, aggregate_peaks=True)
    # print(df.head())

    print("MzMLReader initialized. Use read_spectra() to extract spectra.")
