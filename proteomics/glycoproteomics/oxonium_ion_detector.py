"""
Oxonium Ion Detector

Detects diagnostic oxonium ions in MS/MS spectra for glycopeptide identification.
Oxonium ions are small fragment ions characteristic of glycans (HexNAc, NeuAc, etc.)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Diagnostic oxonium ions for glycopeptides (m/z values)
OXONIUM_IONS = {
    # HexNAc fragments
    'HexNAc': 204.087,
    'HexNAc-H2O': 186.076,
    'HexNAc-2H2O': 168.066,
    'HexNAc-CH3COOH': 144.065,
    'HexNAc-ring': 138.055,
    'HexNAc-CH3': 126.055,

    # Hex fragments
    'Hex': 163.060,
    'Hex-H2O': 145.050,

    # Fucose
    'Fuc': 147.065,
    'Fuc-H2O': 129.055,

    # Sialic acid (NeuAc)
    'NeuAc': 292.103,
    'NeuAc-H2O': 274.092,
    'NeuAc-2H2O': 256.082,

    # NeuGc
    'NeuGc': 308.098,
    'NeuGc-H2O': 290.087,

    # Disaccharides
    'HexNAc-Hex': 366.139,
    'HexNAc2': 407.166,
    'Hex2': 325.113,
    'HexNAc-Fuc': 350.144,

    # Trisaccharides
    'HexNAc-Hex2': 528.192,
    'HexNAc2-Hex': 569.219,
    'HexNAc-Hex-Fuc': 512.197,
    'HexNAc-Hex-NeuAc': 657.235,

    # Complex fragments
    'HexNAc2-Hex-Fuc': 715.272,
    'HexNAc2-Hex-NeuAc': 860.309,
}

# Priority oxonium ions for glycopeptide identification
PRIORITY_OXONIUM_IONS = {
    'HexNAc': 204.087,
    'HexNAc-H2O': 186.076,
    'HexNAc-ring': 138.055,
    'NeuAc': 292.103,
    'NeuAc-H2O': 274.092,
    'HexNAc-Hex': 366.139,
}


@dataclass
class OxoniumIonHit:
    """Represents a detected oxonium ion."""
    name: str
    theoretical_mz: float
    observed_mz: float
    intensity: float
    ppm_error: float


class OxoniumIonDetector:
    """
    Detector for oxonium ions in MS/MS spectra.

    Identifies glycopeptides based on presence of diagnostic oxonium ions.
    """

    def __init__(
        self,
        tolerance: float = 0.02,  # Da
        min_intensity_ratio: float = 0.01,  # 1% of base peak
        priority_only: bool = False
    ):
        """
        Initialize detector.

        Args:
            tolerance: Mass tolerance in Da for ion matching
            min_intensity_ratio: Minimum intensity ratio to base peak
            priority_only: Only check priority oxonium ions (faster)
        """
        self.tolerance = tolerance
        self.min_intensity_ratio = min_intensity_ratio

        if priority_only:
            self.oxonium_ions = PRIORITY_OXONIUM_IONS
        else:
            self.oxonium_ions = OXONIUM_IONS

    def detect_oxonium_ions(
        self,
        mz: np.ndarray,
        intensity: np.ndarray
    ) -> List[OxoniumIonHit]:
        """
        Detect oxonium ions in a spectrum.

        Args:
            mz: m/z array
            intensity: Intensity array

        Returns:
            List of detected oxonium ions
        """
        if len(mz) == 0 or len(intensity) == 0:
            return []

        # Filter by minimum intensity
        base_peak_intensity = np.max(intensity)
        min_intensity = base_peak_intensity * self.min_intensity_ratio

        hits = []

        for ion_name, theoretical_mz in self.oxonium_ions.items():
            # Find peaks within tolerance
            mass_diff = np.abs(mz - theoretical_mz)
            within_tolerance = mass_diff <= self.tolerance

            if np.any(within_tolerance):
                # Get closest match
                idx = np.argmin(mass_diff)
                observed_mz = mz[idx]
                observed_intensity = intensity[idx]

                # Check intensity threshold
                if observed_intensity >= min_intensity:
                    ppm_error = ((observed_mz - theoretical_mz) / theoretical_mz) * 1e6

                    hits.append(OxoniumIonHit(
                        name=ion_name,
                        theoretical_mz=theoretical_mz,
                        observed_mz=observed_mz,
                        intensity=observed_intensity,
                        ppm_error=ppm_error
                    ))

        return hits

    def is_glycopeptide(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        min_oxonium_ions: int = 2
    ) -> Tuple[bool, List[OxoniumIonHit]]:
        """
        Determine if spectrum likely contains a glycopeptide.

        Args:
            mz: m/z array
            intensity: Intensity array
            min_oxonium_ions: Minimum number of oxonium ions required

        Returns:
            (is_glycopeptide, list of oxonium ion hits)
        """
        hits = self.detect_oxonium_ions(mz, intensity)
        is_glyco = len(hits) >= min_oxonium_ions

        return is_glyco, hits

    def calculate_glycan_score(
        self,
        mz: np.ndarray,
        intensity: np.ndarray
    ) -> float:
        """
        Calculate glycopeptide confidence score based on oxonium ions.

        Args:
            mz: m/z array
            intensity: Intensity array

        Returns:
            Score between 0 (no glycan) and 1 (strong glycan signal)
        """
        hits = self.detect_oxonium_ions(mz, intensity)

        if not hits:
            return 0.0

        # Calculate score based on:
        # 1. Number of oxonium ions detected
        # 2. Total intensity of oxonium ions
        # 3. Presence of high-priority ions

        base_peak = np.max(intensity)
        total_oxonium_intensity = sum(hit.intensity for hit in hits)

        # Number score (more ions = higher confidence)
        num_score = min(len(hits) / 5.0, 1.0)  # Normalize to 0-1

        # Intensity score
        intensity_score = min(total_oxonium_intensity / base_peak, 1.0)

        # Priority ion bonus
        priority_ions = {'HexNAc', 'NeuAc', 'HexNAc-Hex'}
        detected_priority = sum(1 for hit in hits if hit.name in priority_ions)
        priority_score = detected_priority / len(priority_ions)

        # Weighted combination
        score = (0.4 * num_score + 0.4 * intensity_score + 0.2 * priority_score)

        return min(score, 1.0)

    def get_glycan_type(
        self,
        oxonium_hits: List[OxoniumIonHit]
    ) -> str:
        """
        Infer glycan type from oxonium ion pattern.

        Args:
            oxonium_hits: List of detected oxonium ions

        Returns:
            Glycan type ('N-glycan', 'O-glycan', 'sialylated', 'fucosylated', etc.)
        """
        hit_names = set(hit.name for hit in oxonium_hits)

        types = []

        # Check for sialylation
        if 'NeuAc' in hit_names or 'NeuAc-H2O' in hit_names or 'NeuGc' in hit_names:
            types.append('sialylated')

        # Check for fucosylation
        if 'Fuc' in hit_names or 'Fuc-H2O' in hit_names or 'HexNAc-Fuc' in hit_names:
            types.append('fucosylated')

        # Check for N-glycan (presence of complex fragments)
        if 'HexNAc-Hex' in hit_names or 'HexNAc-Hex2' in hit_names:
            types.append('N-glycan')

        # Check for O-glycan (simpler fragments)
        if 'HexNAc' in hit_names and not types:
            types.append('O-glycan')

        return ', '.join(types) if types else 'glycopeptide'


def batch_detect_glycopeptides(
    spectra: List[Dict],
    detector: OxoniumIonDetector
) -> List[Dict]:
    """
    Batch detect glycopeptides from list of spectra.

    Args:
        spectra: List of spectrum dicts with 'mz' and 'intensity' keys
        detector: OxoniumIonDetector instance

    Returns:
        List of spectra identified as glycopeptides with annotations
    """
    glycopeptides = []

    for spec in spectra:
        mz = spec['mz']
        intensity = spec['intensity']

        is_glyco, hits = detector.is_glycopeptide(mz, intensity)

        if is_glyco:
            glyco_score = detector.calculate_glycan_score(mz, intensity)
            glycan_type = detector.get_glycan_type(hits)

            glycopeptides.append({
                **spec,
                'oxonium_ions': hits,
                'glycan_score': glyco_score,
                'glycan_type': glycan_type,
                'num_oxonium_ions': len(hits)
            })

    return glycopeptides


# Example usage
if __name__ == "__main__":
    # Example spectrum (simulated glycopeptide)
    mz = np.array([100.0, 138.055, 186.076, 204.087, 274.092, 292.103, 366.139, 500.0, 800.0])
    intensity = np.array([1000, 5000, 8000, 10000, 3000, 4000, 6000, 2000, 15000])

    # Create detector
    detector = OxoniumIonDetector(tolerance=0.02, min_intensity_ratio=0.01)

    # Detect oxonium ions
    hits = detector.detect_oxonium_ions(mz, intensity)
    print(f"Detected {len(hits)} oxonium ions:")
    for hit in hits:
        print(f"  {hit.name}: {hit.observed_mz:.3f} m/z ({hit.ppm_error:.1f} ppm, {hit.intensity:.0f} int)")

    # Check if glycopeptide
    is_glyco, hits = detector.is_glycopeptide(mz, intensity, min_oxonium_ions=2)
    print(f"\nIs glycopeptide: {is_glyco}")

    # Calculate score
    score = detector.calculate_glycan_score(mz, intensity)
    print(f"Glycan score: {score:.3f}")

    # Infer type
    glycan_type = detector.get_glycan_type(hits)
    print(f"Glycan type: {glycan_type}")
