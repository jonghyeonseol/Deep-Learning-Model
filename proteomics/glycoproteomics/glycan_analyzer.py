"""
Glycan Composition Analyzer

Analyzes glycan compositions from MS/MS spectra including Y-ion detection
and glycan structure inference.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .glycan_mass import GlycanMassCalculator, GlycanComposition, OXONIUM_IONS
from .oxonium_ion_detector import OxoniumIonDetector


@dataclass
class YIonMatch:
    """Represents a Y-ion match (peptide + glycan fragment)."""
    mz: float
    intensity: float
    glycan_composition: Dict[str, int]
    composition_str: str
    mass_error: float


class GlycanAnalyzer:
    """
    Comprehensive analyzer for glycan composition from MS/MS spectra.

    Combines oxonium ion detection, Y-ion analysis, and composition inference.
    """

    def __init__(
        self,
        mass_tolerance: float = 0.02,
        ppm_tolerance: float = 20.0
    ):
        """
        Initialize analyzer.

        Args:
            mass_tolerance: Mass tolerance in Da
            ppm_tolerance: PPM tolerance for matching
        """
        self.mass_tolerance = mass_tolerance
        self.ppm_tolerance = ppm_tolerance
        self.mass_calculator = GlycanMassCalculator()
        self.oxonium_detector = OxoniumIonDetector(tolerance=mass_tolerance)

    def detect_y_ions(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        peptide_mass: float,
        max_glycan_mass: float = 3000.0
    ) -> List[YIonMatch]:
        """
        Detect Y-ions (peptide + glycan fragments).

        Args:
            mz: m/z array
            intensity: Intensity array
            peptide_mass: Mass of peptide backbone (without glycan)
            max_glycan_mass: Maximum expected glycan mass

        Returns:
            List of Y-ion matches
        """
        y_ions = []

        # Generate possible glycan compositions
        common_glycans = self.mass_calculator.generate_common_n_glycans()

        for glycan_comp in common_glycans:
            # Calculate expected Y-ion mass (peptide + glycan fragment)
            glycan_mass = self.mass_calculator.calculate_mass(
                glycan_comp.composition,
                adduct='',
                reducing_end='free'
            )

            if glycan_mass > max_glycan_mass:
                continue

            expected_mz = peptide_mass + glycan_mass + 1.0078  # [M+H]+

            # Check if this m/z exists in spectrum
            mass_diff = np.abs(mz - expected_mz)
            ppm_diff = (mass_diff / expected_mz) * 1e6

            within_tolerance = (mass_diff <= self.mass_tolerance) | (ppm_diff <= self.ppm_tolerance)

            if np.any(within_tolerance):
                idx = np.argmin(mass_diff)
                observed_mz = mz[idx]
                observed_intensity = intensity[idx]

                y_ions.append(YIonMatch(
                    mz=observed_mz,
                    intensity=observed_intensity,
                    glycan_composition=glycan_comp.composition,
                    composition_str=str(glycan_comp),
                    mass_error=mass_diff[idx]
                ))

        return y_ions

    def infer_glycan_composition(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        precursor_mz: float,
        peptide_mass: Optional[float] = None
    ) -> List[Tuple[GlycanComposition, float]]:
        """
        Infer glycan composition from precursor m/z.

        Args:
            mz: m/z array (for validation)
            intensity: Intensity array
            precursor_mz: Precursor m/z value
            peptide_mass: Known peptide mass (if available)

        Returns:
            List of (GlycanComposition, score) tuples
        """
        # Calculate expected glycan mass
        if peptide_mass:
            glycan_mass = precursor_mz - peptide_mass - 1.0078  # Remove peptide and proton
        else:
            # Use only precursor for matching
            glycan_mass = precursor_mz

        # Find matching compositions
        matches = self.mass_calculator.match_mass_to_composition(
            glycan_mass,
            tolerance=self.mass_tolerance * 5  # Wider tolerance for precursor
        )

        # Score matches based on oxonium ion support
        scored_matches = []
        oxonium_hits = self.oxonium_detector.detect_oxonium_ions(mz, intensity)
        oxonium_names = set(hit.name for hit in oxonium_hits)

        for glycan_comp, mass_error in matches:
            # Calculate support score
            score = self._calculate_composition_score(
                glycan_comp.composition,
                oxonium_names,
                mass_error
            )
            scored_matches.append((glycan_comp, score))

        # Sort by score
        scored_matches.sort(key=lambda x: x[1], reverse=True)

        return scored_matches

    def _calculate_composition_score(
        self,
        composition: Dict[str, int],
        oxonium_ions: set,
        mass_error: float
    ) -> float:
        """
        Calculate score for glycan composition based on supporting evidence.

        Args:
            composition: Glycan composition
            oxonium_ions: Set of detected oxonium ion names
            mass_error: Mass error in Da

        Returns:
            Score (higher = better match)
        """
        score = 0.0

        # Mass accuracy score (inverse of error)
        mass_score = 1.0 / (1.0 + mass_error * 10)
        score += mass_score * 0.3

        # Oxonium ion support score
        expected_oxonium = set()

        if composition.get('HexNAc', 0) > 0:
            expected_oxonium.update(['HexNAc', 'HexNAc-H2O', 'HexNAc-ring'])

        if composition.get('NeuAc', 0) > 0:
            expected_oxonium.update(['NeuAc', 'NeuAc-H2O'])

        if composition.get('Fuc', 0) > 0:
            expected_oxonium.add('Fuc')

        if composition.get('Hex', 0) > 0 and composition.get('HexNAc', 0) > 0:
            expected_oxonium.add('HexNAc-Hex')

        # Calculate overlap
        if expected_oxonium:
            overlap = len(expected_oxonium & oxonium_ions) / len(expected_oxonium)
            score += overlap * 0.7

        return score

    def analyze_glycopeptide_spectrum(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        precursor_mz: float,
        peptide_mass: Optional[float] = None,
        scan_id: Optional[str] = None
    ) -> Dict:
        """
        Comprehensive glycopeptide spectrum analysis.

        Args:
            mz: m/z array
            intensity: Intensity array
            precursor_mz: Precursor m/z
            peptide_mass: Peptide mass (if known)
            scan_id: Scan identifier

        Returns:
            Analysis results dictionary
        """
        results = {
            'scan_id': scan_id,
            'precursor_mz': precursor_mz,
            'peptide_mass': peptide_mass,
        }

        # 1. Detect oxonium ions
        oxonium_hits = self.oxonium_detector.detect_oxonium_ions(mz, intensity)
        is_glyco, _ = self.oxonium_detector.is_glycopeptide(mz, intensity)
        glycan_score = self.oxonium_detector.calculate_glycan_score(mz, intensity)

        results['is_glycopeptide'] = is_glyco
        results['glycan_score'] = glycan_score
        results['oxonium_ions'] = [
            {'name': hit.name, 'mz': hit.observed_mz, 'intensity': hit.intensity}
            for hit in oxonium_hits
        ]

        # 2. Infer glycan composition
        compositions = self.infer_glycan_composition(mz, intensity, precursor_mz, peptide_mass)
        results['glycan_compositions'] = [
            {'composition': str(comp), 'score': score}
            for comp, score in compositions[:5]  # Top 5
        ]

        # 3. Detect Y-ions (if peptide mass known)
        if peptide_mass:
            y_ions = self.detect_y_ions(mz, intensity, peptide_mass)
            results['y_ions'] = [
                {
                    'mz': y.mz,
                    'intensity': y.intensity,
                    'glycan': y.composition_str,
                    'mass_error': y.mass_error
                }
                for y in y_ions[:10]  # Top 10
            ]
        else:
            results['y_ions'] = []

        return results


# Example usage
if __name__ == "__main__":
    # Simulated glycopeptide spectrum
    mz = np.array([
        138.055, 186.076, 204.087,  # HexNAc oxonium ions
        274.092, 292.103,            # NeuAc oxonium ions
        366.139,                      # HexNAc-Hex
        500.0, 650.0, 800.0,         # Peptide fragments
        1200.5, 1500.8,              # Y-ions (peptide + partial glycan)
        1850.5                        # Precursor
    ])

    intensity = np.array([
        5000, 8000, 10000,
        3000, 4000,
        6000,
        2000, 1500, 1800,
        5000, 4000,
        20000
    ])

    # Create analyzer
    analyzer = GlycanAnalyzer(mass_tolerance=0.02)

    # Analyze spectrum
    results = analyzer.analyze_glycopeptide_spectrum(
        mz=mz,
        intensity=intensity,
        precursor_mz=1850.5,
        peptide_mass=1200.0,  # Example peptide mass
        scan_id='scan_12345'
    )

    print("Glycopeptide Analysis Results")
    print("=" * 60)
    print(f"Scan: {results['scan_id']}")
    print(f"Is glycopeptide: {results['is_glycopeptide']}")
    print(f"Glycan score: {results['glycan_score']:.3f}")
    print(f"\nDetected oxonium ions: {len(results['oxonium_ions'])}")
    for ox in results['oxonium_ions']:
        print(f"  {ox['name']}: {ox['mz']:.3f} m/z")

    print(f"\nTop glycan compositions:")
    for comp in results['glycan_compositions']:
        print(f"  {comp['composition']} (score: {comp['score']:.3f})")

    if results['y_ions']:
        print(f"\nDetected Y-ions: {len(results['y_ions'])}")
        for y in results['y_ions'][:3]:
            print(f"  {y['mz']:.2f} m/z - {y['glycan']}")
