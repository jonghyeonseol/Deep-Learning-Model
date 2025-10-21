# Glycoproteomics Analysis Module

Comprehensive tools for analyzing glycoproteomics mass spectrometry data.

## Overview

This module provides specialized tools for identifying and characterizing glycopeptides from LC-MS/MS data, including:

- **Oxonium ion detection** - Diagnostic fragments for glycopeptide identification
- **Glycan mass calculation** - Theoretical mass calculation from composition
- **Y-ion analysis** - Peptide + glycan fragment detection
- **Glycan composition inference** - Automated composition assignment

## Quick Start

### 1. Standalone .raw to mzML Conversion

```bash
# Convert single .raw file
python3 convert_raw_to_mzml.py sample.raw

# Batch convert directory
python3 convert_raw_to_mzml.py data/proteomics/raw/ --batch

# Custom output directory
python3 convert_raw_to_mzml.py sample.raw --output data/proteomics/mzml/
```

### 2. Glycoproteomics Analysis

```bash
# Analyze mzML file
python3 analyze_glycoproteomics.py data.mzML --output results.csv

# Convert and analyze .raw file
python3 analyze_glycoproteomics.py data.raw --convert --output results.csv

# Batch analysis
python3 analyze_glycoproteomics.py data_dir/ --batch --output results_dir/
```

### 3. Python API

```python
from proteomics.data_loaders import MzMLReader
from proteomics.glycoproteomics import (
    GlycanAnalyzer,
    OxoniumIonDetector,
    GlycanMassCalculator
)

# Read spectrum
reader = MzMLReader('data.mzML')
spectra = reader.read_spectra(ms_level=2, min_peaks=10)

# Detect oxonium ions
detector = OxoniumIonDetector(tolerance=0.02)
is_glyco, hits = detector.is_glycopeptide(
    spectra[0]['mz'],
    spectra[0]['intensity']
)

# Analyze glycan composition
analyzer = GlycanAnalyzer()
results = analyzer.analyze_glycopeptide_spectrum(
    mz=spectra[0]['mz'],
    intensity=spectra[0]['intensity'],
    precursor_mz=spectra[0]['precursor_mz']
)

print(f"Glycan score: {results['glycan_score']:.3f}")
print(f"Top composition: {results['glycan_compositions'][0]}")
```

## Key Concepts

### Oxonium Ions

Small diagnostic fragment ions characteristic of glycans:

| Ion | m/z | Composition |
|-----|-----|-------------|
| HexNAc | 204.087 | N-Acetylhexosamine |
| HexNAc-H2O | 186.076 | HexNAc minus water |
| HexNAc-ring | 138.055 | HexNAc ring fragment |
| NeuAc | 292.103 | Sialic acid |
| NeuAc-H2O | 274.092 | Sialic acid minus water |
| HexNAc-Hex | 366.139 | HexNAc-Hexose disaccharide |
| Fuc | 147.065 | Fucose |

**Detection Strategy**: Presence of ≥2 oxonium ions indicates likely glycopeptide.

### Glycan Notation

Glycan compositions are represented as:

- **Long form**: `Hex5HexNAc4Fuc1NeuAc2`
- **Short form**: `H5N4F1S2`

Where:
- Hex/H = Hexose (Glucose, Mannose, Galactose)
- HexNAc/N = N-Acetylhexosamine (GlcNAc, GalNAc)
- Fuc/F = Fucose
- NeuAc/S = N-Acetylneuraminic acid (sialic acid)

### Y-Ions

Fragment ions consisting of intact peptide + glycan fragment:

```
Glycopeptide:  Peptide-[Glycan]
                   ↓ HCD fragmentation
Y-ion:         Peptide-[GlcNAc] (Y1 ion)
               Peptide-[GlcNAc-Hex] (Y2 ion)
               Peptide-[Full Glycan] (Y0 ion)
```

## Module Components

### 1. Glycan Mass Calculator (`glycan_mass.py`)

```python
from proteomics.glycoproteomics import GlycanMassCalculator

calculator = GlycanMassCalculator()

# Calculate mass from composition
composition = {'Hex': 5, 'HexNAc': 4, 'Fuc': 1}
mass = calculator.calculate_mass(composition, adduct='H')
print(f"[M+H]+ = {mass:.4f} Da")

# Parse composition string
comp = calculator.parse_composition("Hex3HexNAc4Fuc1")

# Match observed mass to compositions
matches = calculator.match_mass_to_composition(
    observed_mass=1809.6,
    tolerance=0.1
)

# Generate common N-glycans
common_glycans = calculator.generate_common_n_glycans()
```

### 2. Oxonium Ion Detector (`oxonium_ion_detector.py`)

```python
from proteomics.glycoproteomics import OxoniumIonDetector

detector = OxoniumIonDetector(
    tolerance=0.02,  # Da
    min_intensity_ratio=0.01  # 1% of base peak
)

# Detect oxonium ions
hits = detector.detect_oxonium_ions(mz_array, intensity_array)

# Check if glycopeptide
is_glyco, hits = detector.is_glycopeptide(
    mz_array,
    intensity_array,
    min_oxonium_ions=2
)

# Calculate confidence score (0-1)
score = detector.calculate_glycan_score(mz_array, intensity_array)

# Infer glycan type
glycan_type = detector.get_glycan_type(hits)
# Returns: 'N-glycan', 'sialylated', 'fucosylated', etc.
```

### 3. Glycan Analyzer (`glycan_analyzer.py`)

Comprehensive analysis combining all tools:

```python
from proteomics.glycoproteomics import GlycanAnalyzer

analyzer = GlycanAnalyzer(mass_tolerance=0.02)

# Full spectrum analysis
results = analyzer.analyze_glycopeptide_spectrum(
    mz=spectrum['mz'],
    intensity=spectrum['intensity'],
    precursor_mz=1850.5,
    peptide_mass=1200.0,  # Optional
    scan_id='scan_12345'
)

# Results include:
# - Oxonium ion hits
# - Glycan composition matches
# - Y-ion detections
# - Confidence scores
```

## Output Format

The `analyze_glycoproteomics.py` script generates CSV files with:

| Column | Description |
|--------|-------------|
| scan_id | Spectrum scan number |
| rt | Retention time (minutes) |
| precursor_mz | Precursor m/z value |
| glycan_score | Confidence score (0-1) |
| num_oxonium_ions | Number of oxonium ions detected |
| top_glycan_composition | Most likely glycan composition |
| composition_score | Composition match score |
| glycan_type | Inferred type (N-glycan, sialylated, etc.) |
| oxonium_ions_detected | List of detected oxonium ions |

## Workflow Examples

### Example 1: High-Throughput Screening

```bash
# 1. Batch convert .raw files
python3 convert_raw_to_mzml.py raw_data/ --batch --output mzml_data/

# 2. Batch analyze glycopeptides
python3 analyze_glycoproteomics.py mzml_data/ --batch --output results/

# 3. Filter high-confidence glycopeptides
python3 -c "
import pandas as pd
df = pd.read_csv('results/combined_glycoproteomics_results.csv')
high_conf = df[df['glycan_score'] > 0.5]
high_conf.to_csv('high_confidence_glycopeptides.csv', index=False)
print(f'Found {len(high_conf)} high-confidence glycopeptides')
"
```

### Example 2: Targeted Glycan Analysis

```python
from proteomics.data_loaders import MzMLReader
from proteomics.glycoproteomics import GlycanMassCalculator, GlycanAnalyzer

# Define target glycans
target_glycans = [
    {'Hex': 5, 'HexNAc': 4, 'Fuc': 1, 'NeuAc': 2},  # Complex sialylated
    {'Hex': 3, 'HexNAc': 4, 'Fuc': 1},              # Core fucosylated
    {'Hex': 9, 'HexNAc': 2},                         # High mannose
]

calculator = GlycanMassCalculator()

# Calculate target masses
target_masses = [calculator.calculate_mass(comp) for comp in target_glycans]
print("Target glycan masses:", target_masses)

# Read data and search for targets
reader = MzMLReader('sample.mzML')
spectra = reader.read_spectra(ms_level=2)

analyzer = GlycanAnalyzer()
matches = []

for spec in spectra:
    results = analyzer.analyze_glycopeptide_spectrum(
        spec['mz'], spec['intensity'], spec['precursor_mz']
    )

    # Check if any target composition found
    for comp_result in results['glycan_compositions']:
        if any(str(GlycanComposition(target)) in comp_result['composition']
               for target in target_glycans):
            matches.append((spec['scan_id'], comp_result))

print(f"Found {len(matches)} spectra with target glycans")
```

### Example 3: Custom Oxonium Ion Threshold

```python
from proteomics.glycoproteomics import OxoniumIonDetector, batch_detect_glycopeptides

# Strict criteria: require 3+ oxonium ions
detector = OxoniumIonDetector(
    tolerance=0.01,  # Tighter tolerance
    min_intensity_ratio=0.02  # Higher intensity threshold
)

# Filter glycopeptides
glycopeptides = []
for spec in spectra:
    is_glyco, hits = detector.is_glycopeptide(
        spec['mz'], spec['intensity'],
        min_oxonium_ions=3  # Stricter criterion
    )
    if is_glyco:
        glycopeptides.append(spec)

print(f"High-confidence glycopeptides: {len(glycopeptides)}")
```

## Performance Tips

1. **Mass Tolerance**: Use 0.02 Da for high-resolution MS (Orbitrap, FTICR), 0.05-0.1 Da for lower resolution
2. **Batch Processing**: Process multiple files in parallel for large datasets
3. **Filtering**: Pre-filter spectra by presence of at least one HexNAc oxonium ion before detailed analysis
4. **Memory**: Process large mzML files in chunks to avoid memory issues

## References

- Oxonium Ion Scanning: Nature Biomedical Engineering (2023)
- N-Glycan Structures: GlycoWorkbench database
- Fragmentation Patterns: Molecular & Cellular Proteomics guidelines

## Support

For issues or questions, see the main proteomics module documentation or open an issue on GitHub.
