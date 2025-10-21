"""
Glycoproteomics Module

Tools for glycoproteomics analysis including:
- Glycan mass calculation
- Oxonium ion detection
- Y-ion analysis
- Glycan composition inference
"""

from .glycan_mass import (
    GlycanMassCalculator,
    GlycanComposition,
    MONOSACCHARIDE_MASSES,
    OXONIUM_IONS,
    calculate_y_ion_mass
)

from .oxonium_ion_detector import (
    OxoniumIonDetector,
    OxoniumIonHit,
    batch_detect_glycopeptides,
    PRIORITY_OXONIUM_IONS
)

from .glycan_analyzer import (
    GlycanAnalyzer,
    YIonMatch
)

__all__ = [
    # Glycan mass
    'GlycanMassCalculator',
    'GlycanComposition',
    'MONOSACCHARIDE_MASSES',
    'OXONIUM_IONS',
    'calculate_y_ion_mass',

    # Oxonium detection
    'OxoniumIonDetector',
    'OxoniumIonHit',
    'batch_detect_glycopeptides',
    'PRIORITY_OXONIUM_IONS',

    # Glycan analysis
    'GlycanAnalyzer',
    'YIonMatch',
]
