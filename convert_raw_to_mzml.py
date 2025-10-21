#!/usr/bin/env python3
"""
Standalone .raw to mzML Converter Script

Converts Thermo Fisher .raw files to open mzML format for proteomics/glycoproteomics analysis.

Usage:
    python3 convert_raw_to_mzml.py input.raw
    python3 convert_raw_to_mzml.py input_directory/ --batch
    python3 convert_raw_to_mzml.py input.raw --output output_dir/
"""

import argparse
import sys
from pathlib import Path
from proteomics.data_loaders import RawConverter


def convert_single_file(raw_file: Path, output_dir: Path, force: bool = False):
    """Convert a single .raw file."""
    print(f"\n{'='*60}")
    print(f"Converting: {raw_file.name}")
    print(f"{'='*60}")

    converter = RawConverter(output_dir=str(output_dir))
    mzml_file = converter.convert(raw_file, force=force)

    if mzml_file:
        print(f"✓ Success: {mzml_file}")
        return True
    else:
        print(f"✗ Failed: {raw_file}")
        return False


def convert_batch(input_dir: Path, output_dir: Path, pattern: str = "*.raw", force: bool = False):
    """Convert all .raw files in a directory."""
    print(f"\n{'='*60}")
    print(f"Batch Conversion")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Pattern: {pattern}")
    print(f"{'='*60}\n")

    converter = RawConverter(output_dir=str(output_dir))
    mzml_files = converter.convert_directory(input_dir, pattern=pattern, force=force)

    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"Successfully converted: {len(mzml_files)} files")

    if mzml_files:
        print("\nOutput files:")
        for mzml_file in mzml_files:
            print(f"  - {mzml_file.name}")

    return mzml_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert Thermo Fisher .raw files to mzML format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python3 convert_raw_to_mzml.py sample.raw

  # Convert single file to specific output directory
  python3 convert_raw_to_mzml.py sample.raw --output data/proteomics/mzml/

  # Batch convert all .raw files in directory
  python3 convert_raw_to_mzml.py data/proteomics/raw/ --batch

  # Force overwrite existing mzML files
  python3 convert_raw_to_mzml.py sample.raw --force

  # Convert with custom pattern
  python3 convert_raw_to_mzml.py data/proteomics/raw/ --batch --pattern "glyco*.raw"

Requirements:
  - ThermoRawFileParser must be installed
  - Install with: dotnet tool install ThermoRawFileParser -g
  - Or download from: https://github.com/compomics/ThermoRawFileParser
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input .raw file or directory containing .raw files'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data/proteomics/mzml',
        help='Output directory for mzML files (default: ./data/proteomics/mzml)'
    )

    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Batch mode: convert all .raw files in input directory'
    )

    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.raw',
        help='File pattern for batch mode (default: *.raw)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force overwrite existing mzML files'
    )

    parser.add_argument(
        '--thermorawfileparser',
        type=str,
        help='Path to ThermoRawFileParser executable (auto-detected if not specified)'
    )

    args = parser.parse_args()

    # Parse input path
    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if input exists
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    # Determine conversion mode
    if args.batch or input_path.is_dir():
        # Batch conversion
        if not input_path.is_dir():
            print(f"Error: --batch mode requires input to be a directory")
            sys.exit(1)

        mzml_files = convert_batch(input_path, output_dir, args.pattern, args.force)

        if not mzml_files:
            print("\n✗ No files were converted")
            sys.exit(1)
        else:
            print(f"\n✓ Successfully converted {len(mzml_files)} files")
            sys.exit(0)

    else:
        # Single file conversion
        if not input_path.suffix.lower() == '.raw':
            print(f"Warning: File does not have .raw extension: {input_path}")

        success = convert_single_file(input_path, output_dir, args.force)

        if success:
            print("\n✓ Conversion completed successfully")
            sys.exit(0)
        else:
            print("\n✗ Conversion failed")
            sys.exit(1)


if __name__ == '__main__':
    main()
