"""
Glycan Mass Calculator

Calculate theoretical masses of glycan compositions for glycoproteomics analysis.
Based on monosaccharide building blocks (Hex, HexNAc, Fuc, NeuAc, etc.)
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


# Monoisotopic masses of common monosaccharides (dehydrated form)
MONOSACCHARIDE_MASSES = {
    # Basic monosaccharides
    'Hex': 162.0528,        # Hexose (Glucose, Galactose, Mannose)
    'HexNAc': 203.0794,     # N-Acetylhexosamine (GlcNAc, GalNAc)
    'dHex': 146.0579,       # Deoxyhexose (Fucose)
    'Fuc': 146.0579,        # Fucose
    'Pent': 132.0423,       # Pentose (Xylose, Arabinose)
    'Xyl': 132.0423,        # Xylose

    # Sialic acids
    'NeuAc': 291.0954,      # N-Acetylneuraminic acid
    'NeuGc': 307.0903,      # N-Glycolylneuraminic acid
    'Sia': 291.0954,        # Sialic acid (general)

    # Phosphate/sulfate modifications
    'Phospho': 79.9663,     # Phosphate
    'Sulfo': 79.9568,       # Sulfate

    # Other
    'HexA': 176.0321,       # Hexuronic acid
    'KDN': 250.0689,        # 3-deoxy-D-glycero-D-galacto-nonulosonic acid
}

# Common oxonium ion masses (for glycopeptide identification)
OXONIUM_IONS = {
    'HexNAc-H2O': 138.055,
    'HexNAc-2H2O': 126.055,
    'HexNAc': 204.087,
    'HexNAc-CH3': 186.076,
    'Hex': 163.060,
    'HexNAc-Hex': 366.139,
    'HexNAc2': 407.166,
    'NeuAc': 292.103,
    'NeuAc-H2O': 274.092,
    'HexNAc-Hex-NeuAc': 657.235,
    'Fuc': 147.065,
}


@dataclass
class GlycanComposition:
    """Represents a glycan composition."""
    composition: Dict[str, int]

    def __str__(self) -> str:
        """String representation (e.g., Hex3HexNAc4Fuc1)."""
        parts = []
        for mono in ['Hex', 'HexNAc', 'Fuc', 'NeuAc', 'NeuGc', 'Xyl', 'Phospho', 'Sulfo']:
            if mono in self.composition and self.composition[mono] > 0:
                parts.append(f"{mono}{self.composition[mono]}")
        return ''.join(parts) if parts else "Empty"

    def to_short_notation(self) -> str:
        """Short notation (e.g., H3N4F1)."""
        mapping = {
            'Hex': 'H',
            'HexNAc': 'N',
            'Fuc': 'F',
            'NeuAc': 'S',
            'NeuGc': 'G',
            'Xyl': 'X',
        }
        parts = []
        for mono, abbr in mapping.items():
            if mono in self.composition and self.composition[mono] > 0:
                parts.append(f"{abbr}{self.composition[mono]}")
        return ''.join(parts) if parts else "H0N0"


class GlycanMassCalculator:
    """
    Calculator for glycan masses based on monosaccharide composition.
    """

    def __init__(self, custom_masses: Optional[Dict[str, float]] = None):
        """
        Initialize calculator.

        Args:
            custom_masses: Optional custom monosaccharide masses to override defaults
        """
        self.masses = MONOSACCHARIDE_MASSES.copy()
        if custom_masses:
            self.masses.update(custom_masses)

    def calculate_mass(
        self,
        composition: Dict[str, int],
        adduct: str = 'H',
        reducing_end: str = 'free'
    ) -> float:
        """
        Calculate mass of glycan composition.

        Args:
            composition: Dict mapping monosaccharide name to count
            adduct: Adduct ion ('H', 'Na', 'K', 'NH4')
            reducing_end: Reducing end modification ('free', 'reduced', '2AB', 'perm')

        Returns:
            Monoisotopic mass in Da
        """
        # Calculate base glycan mass (dehydrated monosaccharides)
        mass = sum(self.masses.get(mono, 0) * count for mono, count in composition.items())

        # Add reducing end mass
        reducing_end_masses = {
            'free': 18.0106,        # H2O (free reducing end)
            'reduced': 2.0157,      # 2H (reduced, alditol)
            '2AB': 120.0687,        # 2-aminobenzamide
            'perm': 0.0,            # Permethylated (no additional mass)
        }
        mass += reducing_end_masses.get(reducing_end, 18.0106)

        # Add adduct mass
        adduct_masses = {
            'H': 1.0078,           # [M+H]+
            'Na': 22.9898,         # [M+Na]+
            'K': 38.9637,          # [M+K]+
            'NH4': 18.0338,        # [M+NH4]+
            '': 0.0,               # Neutral mass
        }
        mass += adduct_masses.get(adduct, 1.0078)

        return mass

    def parse_composition(self, composition_str: str) -> Dict[str, int]:
        """
        Parse glycan composition string.

        Args:
            composition_str: String like "Hex3HexNAc4Fuc1" or "H3N4F1"

        Returns:
            Dict mapping monosaccharide to count
        """
        composition = {}

        # Try long form first (HexNAc4Hex3)
        pattern = r'([A-Za-z]+)(\d+)'
        matches = re.findall(pattern, composition_str)

        if matches:
            for mono, count in matches:
                if mono in self.masses:
                    composition[mono] = int(count)
                else:
                    # Try abbreviation mapping
                    abbrev_map = {
                        'H': 'Hex',
                        'N': 'HexNAc',
                        'F': 'Fuc',
                        'S': 'NeuAc',
                        'G': 'NeuGc',
                        'X': 'Xyl',
                    }
                    if mono in abbrev_map:
                        composition[abbrev_map[mono]] = int(count)

        return composition

    def generate_common_n_glycans(self) -> List[GlycanComposition]:
        """
        Generate list of common N-glycan compositions.

        Returns:
            List of GlycanComposition objects
        """
        glycans = []

        # High-mannose type (Man5-9)
        for man_count in range(5, 10):
            glycans.append(GlycanComposition({
                'Hex': man_count,
                'HexNAc': 2
            }))

        # Complex type (biantennary to tetraantennary)
        for antenna in range(2, 5):  # bi-, tri-, tetra-
            for fuc in [0, 1]:  # +/- core fucose
                for sia in range(0, antenna + 1):  # 0 to fully sialylated
                    glycans.append(GlycanComposition({
                        'Hex': 3 + antenna,
                        'HexNAc': 2 + antenna,
                        'Fuc': fuc,
                        'NeuAc': sia
                    }))

        return glycans

    def match_mass_to_composition(
        self,
        observed_mass: float,
        tolerance: float = 0.05,
        max_hex: int = 12,
        max_hexnac: int = 12,
        max_fuc: int = 4,
        max_neuac: int = 4
    ) -> List[Tuple[GlycanComposition, float]]:
        """
        Find glycan compositions matching observed mass.

        Args:
            observed_mass: Observed m/z value
            tolerance: Mass tolerance in Da
            max_hex: Maximum number of Hex
            max_hexnac: Maximum number of HexNAc
            max_fuc: Maximum number of Fuc
            max_neuac: Maximum number of NeuAc

        Returns:
            List of (GlycanComposition, mass_error) tuples
        """
        matches = []

        # Brute-force search over composition space
        for hex_count in range(0, max_hex + 1):
            for hexnac_count in range(0, max_hexnac + 1):
                for fuc_count in range(0, max_fuc + 1):
                    for neuac_count in range(0, max_neuac + 1):
                        composition = {
                            'Hex': hex_count,
                            'HexNAc': hexnac_count,
                            'Fuc': fuc_count,
                            'NeuAc': neuac_count
                        }

                        # Filter out empty compositions
                        composition = {k: v for k, v in composition.items() if v > 0}
                        if not composition:
                            continue

                        # Calculate theoretical mass
                        theoretical_mass = self.calculate_mass(composition)
                        mass_error = abs(theoretical_mass - observed_mass)

                        if mass_error <= tolerance:
                            matches.append((GlycanComposition(composition), mass_error))

        # Sort by mass error
        matches.sort(key=lambda x: x[1])
        return matches


def calculate_y_ion_mass(peptide_mass: float, glycan_composition: Dict[str, int]) -> float:
    """
    Calculate Y-ion mass (peptide + GlcNAc or partial glycan).

    Args:
        peptide_mass: Mass of peptide backbone
        glycan_composition: Glycan fragment composition

    Returns:
        Y-ion m/z
    """
    calculator = GlycanMassCalculator()
    glycan_mass = calculator.calculate_mass(glycan_composition, adduct='', reducing_end='free')
    return peptide_mass + glycan_mass


# Example usage
if __name__ == "__main__":
    calculator = GlycanMassCalculator()

    # Example 1: Calculate mass of specific composition
    composition = {'Hex': 5, 'HexNAc': 4, 'Fuc': 1, 'NeuAc': 2}
    mass = calculator.calculate_mass(composition, adduct='H')
    print(f"Glycan composition: {GlycanComposition(composition)}")
    print(f"[M+H]+ mass: {mass:.4f} Da")

    # Example 2: Parse composition string
    comp_str = "Hex3HexNAc4Fuc1"
    parsed = calculator.parse_composition(comp_str)
    print(f"\nParsed '{comp_str}': {parsed}")

    # Example 3: Match mass to compositions
    observed_mass = 1809.6
    matches = calculator.match_mass_to_composition(observed_mass, tolerance=0.1)
    print(f"\nMatches for m/z {observed_mass} (±0.1 Da):")
    for glycan, error in matches[:5]:  # Top 5 matches
        print(f"  {glycan} (error: {error:.4f} Da)")

    # Example 4: Common N-glycans
    common_glycans = calculator.generate_common_n_glycans()
    print(f"\nGenerated {len(common_glycans)} common N-glycan compositions")
    print("Examples:")
    for glycan in common_glycans[:5]:
        mass = calculator.calculate_mass(glycan.composition)
        print(f"  {glycan} - {mass:.2f} Da")
