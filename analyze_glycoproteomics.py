#!/usr/bin/env python3
"""
Glycoproteomics Analysis Workflow

Complete workflow for analyzing glycoproteomics data from .raw files or mzML.

Features:
- .raw to mzML conversion
- Oxonium ion detection for glycopeptide identification
- Glycan composition inference
- Y-ion analysis
- Export results to CSV

Usage:
    python3 analyze_glycoproteomics.py data.mzML --output results.csv
    python3 analyze_glycoproteomics.py data.raw --convert --output results.csv
    python3 analyze_glycoproteomics.py data_directory/ --batch --output results/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict

# Import proteomics modules
from proteomics.data_loaders import MzMLReader, RawConverter
from proteomics.glycoproteomics import (
    GlycanAnalyzer,
    OxoniumIonDetector,
    GlycanMassCalculator,
    batch_detect_glycopeptides
)


def analyze_single_file(
    mzml_path: Path,
    ms_level: int = 2,
    min_oxonium_ions: int = 2,
    output_csv: Path = None
) -> pd.DataFrame:
    """
    Analyze a single mzML file for glycopeptides.

    Args:
        mzml_path: Path to mzML file
        ms_level: MS level to analyze (default: 2 for MS/MS)
        min_oxonium_ions: Minimum oxonium ions for glycopeptide ID
        output_csv: Path to save results CSV

    Returns:
        DataFrame with glycopeptide analysis results
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {mzml_path.name}")
    print(f"{'='*70}\n")

    # Read spectra
    print(f"Reading MS{ms_level} spectra...")
    reader = MzMLReader(mzml_path)
    spectra = reader.read_spectra(ms_level=ms_level, min_peaks=10)
    print(f"Loaded {len(spectra)} spectra")

    # Initialize analyzers
    oxonium_detector = OxoniumIonDetector(tolerance=0.02, min_intensity_ratio=0.01)
    glycan_analyzer = GlycanAnalyzer(mass_tolerance=0.02)

    # Detect glycopeptides
    print("\nDetecting glycopeptides via oxonium ions...")
    glycopeptides = batch_detect_glycopeptides(spectra, oxonium_detector)
    print(f"Identified {len(glycopeptides)} potential glycopeptides")

    # Detailed analysis
    print("\nPerforming detailed glycan composition analysis...")
    results = []

    for i, spec in enumerate(glycopeptides, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(glycopeptides)} spectra...")

        # Analyze spectrum
        analysis = glycan_analyzer.analyze_glycopeptide_spectrum(
            mz=spec['mz'],
            intensity=spec['intensity'],
            precursor_mz=spec.get('precursor_mz', None),
            peptide_mass=None,  # Could be from database search
            scan_id=spec['scan_id']
        )

        # Extract top composition
        top_composition = analysis['glycan_compositions'][0] if analysis['glycan_compositions'] else {'composition': 'Unknown', 'score': 0.0}

        # Compile result
        results.append({
            'scan_id': spec['scan_id'],
            'rt': spec.get('rt', None),
            'precursor_mz': spec.get('precursor_mz', None),
            'glycan_score': analysis['glycan_score'],
            'num_oxonium_ions': len(analysis['oxonium_ions']),
            'top_glycan_composition': top_composition['composition'],
            'composition_score': top_composition['score'],
            'glycan_type': spec.get('glycan_type', ''),
            'oxonium_ions_detected': ', '.join([ox['name'] for ox in analysis['oxonium_ions']])
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by glycan score
    df = df.sort_values('glycan_score', ascending=False)

    # Print summary
    print(f"\n{'='*70}")
    print("Analysis Summary")
    print(f"{'='*70}")
    print(f"Total glycopeptides identified: {len(df)}")
    print(f"High-confidence (score > 0.5): {len(df[df['glycan_score'] > 0.5])}")
    print(f"\nTop 5 glycopeptides by score:")
    print(df[['scan_id', 'rt', 'glycan_score', 'top_glycan_composition']].head())

    # Save to CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to: {output_csv}")

    return df


def analyze_batch(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.mzML",
    **kwargs
):
    """Batch analyze multiple mzML files."""
    # Find all mzML files
    mzml_files = list(input_dir.glob(pattern))

    if not mzml_files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"\n{'='*70}")
    print(f"Batch Analysis: {len(mzml_files)} files")
    print(f"{'='*70}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each file
    all_results = []

    for mzml_file in mzml_files:
        output_csv = output_dir / f"{mzml_file.stem}_glycoproteomics.csv"

        try:
            df = analyze_single_file(mzml_file, output_csv=output_csv, **kwargs)
            all_results.append(df)
        except Exception as e:
            print(f"✗ Error processing {mzml_file.name}: {e}")
            continue

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_csv = output_dir / "combined_glycoproteomics_results.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n✓ Combined results saved to: {combined_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Glycoproteomics Analysis Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single mzML file
  python3 analyze_glycoproteomics.py data.mzML --output results.csv

  # Convert .raw and analyze
  python3 analyze_glycoproteomics.py data.raw --convert --output results.csv

  # Batch analyze directory
  python3 analyze_glycoproteomics.py data_dir/ --batch --output results_dir/

  # Custom oxonium ion threshold
  python3 analyze_glycoproteomics.py data.mzML --min-oxonium 3 --output results.csv
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input mzML file, .raw file, or directory'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output CSV file or directory (for batch mode)'
    )

    parser.add_argument(
        '--convert', '-c',
        action='store_true',
        help='Convert .raw file to mzML before analysis'
    )

    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Batch mode: analyze all mzML files in directory'
    )

    parser.add_argument(
        '--ms-level',
        type=int,
        default=2,
        choices=[1, 2],
        help='MS level to analyze (default: 2)'
    )

    parser.add_argument(
        '--min-oxonium',
        type=int,
        default=2,
        help='Minimum number of oxonium ions for glycopeptide ID (default: 2)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mzML',
        help='File pattern for batch mode (default: *.mzML)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Check input exists
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    # Handle .raw conversion
    if args.convert or input_path.suffix.lower() == '.raw':
        print("Converting .raw to mzML...")
        converter = RawConverter()

        if input_path.is_file():
            mzml_file = converter.convert(input_path)
            if not mzml_file:
                print("✗ Conversion failed")
                sys.exit(1)
            input_path = mzml_file
        else:
            mzml_files = converter.convert_directory(input_path)
            if not mzml_files:
                print("✗ No files converted")
                sys.exit(1)

    # Analyze
    if args.batch or input_path.is_dir():
        analyze_batch(
            input_path,
            output_path,
            pattern=args.pattern,
            ms_level=args.ms_level,
            min_oxonium_ions=args.min_oxonium
        )
    else:
        analyze_single_file(
            input_path,
            ms_level=args.ms_level,
            min_oxonium_ions=args.min_oxonium,
            output_csv=output_path
        )

    print(f"\n{'='*70}")
    print("✓ Glycoproteomics analysis complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
