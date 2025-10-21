"""
RAW File Converter Module

Converts Thermo .raw files to open mzML format using ThermoRawFileParser.
Supports batch conversion and caching.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RawConverter:
    """
    Converter for Thermo .raw files to mzML format.

    Requires ThermoRawFileParser to be installed:
    - Download from: https://github.com/compomics/ThermoRawFileParser
    - Or install via: dotnet tool install ThermoRawFileParser -g
    """

    def __init__(
        self,
        output_dir: str = "./data/proteomics/mzml",
        thermorawfileparser_path: Optional[str] = None,
        format: str = "mzML"
    ):
        """
        Initialize RawConverter.

        Args:
            output_dir: Directory to save converted mzML files
            thermorawfileparser_path: Path to ThermoRawFileParser executable
            format: Output format (default: mzML, options: mzML, mgf)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.format = format
        self.format_code = {"mzML": 2, "mgf": 1}.get(format, 2)

        # Try to find ThermoRawFileParser
        if thermorawfileparser_path:
            self.parser_path = thermorawfileparser_path
        else:
            self.parser_path = self._find_thermorawfileparser()

        logger.info(f"RawConverter initialized. Output: {self.output_dir}")

    def _find_thermorawfileparser(self) -> Optional[str]:
        """
        Attempt to locate ThermoRawFileParser executable.

        Returns:
            Path to executable or None if not found
        """
        # Common installation locations
        possible_paths = [
            "ThermoRawFileParser",  # Global dotnet tool
            "ThermoRawFileParser.exe",
            "/usr/local/bin/ThermoRawFileParser",
            "~/.dotnet/tools/ThermoRawFileParser",
            "../FragPipe/ThermoRawFileParser",  # Check sibling FragPipe directory
        ]

        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            try:
                result = subprocess.run(
                    [expanded_path, "--help"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"Found ThermoRawFileParser at: {expanded_path}")
                    return expanded_path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        logger.warning(
            "ThermoRawFileParser not found. Please install it:\n"
            "  dotnet tool install ThermoRawFileParser -g\n"
            "Or download from: https://github.com/compomics/ThermoRawFileParser"
        )
        return None

    def convert(
        self,
        raw_file: Union[str, Path],
        output_name: Optional[str] = None,
        force: bool = False
    ) -> Optional[Path]:
        """
        Convert a single .raw file to mzML.

        Args:
            raw_file: Path to .raw file
            output_name: Custom output filename (without extension)
            force: Overwrite existing output file

        Returns:
            Path to converted mzML file, or None if conversion failed
        """
        raw_file = Path(raw_file)

        if not raw_file.exists():
            logger.error(f"RAW file not found: {raw_file}")
            return None

        # Determine output filename
        if output_name:
            output_file = self.output_dir / f"{output_name}.{self.format.lower()}"
        else:
            output_file = self.output_dir / f"{raw_file.stem}.{self.format.lower()}"

        # Skip if already converted
        if output_file.exists() and not force:
            logger.info(f"Output already exists (skipping): {output_file}")
            return output_file

        # Check if ThermoRawFileParser is available
        if not self.parser_path:
            logger.error("ThermoRawFileParser not available. Cannot convert .raw files.")
            return None

        # Build conversion command
        cmd = [
            self.parser_path,
            "-i", str(raw_file.absolute()),
            "-o", str(self.output_dir.absolute()),
            "-f", str(self.format_code),  # 2 = mzML, 1 = MGF
        ]

        if output_name:
            cmd.extend(["-b", output_name])  # Base output name

        logger.info(f"Converting: {raw_file.name} -> {output_file.name}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Conversion successful: {output_file}")
                return output_file
            else:
                logger.error(f"Conversion failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Conversion timeout for: {raw_file}")
            return None
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return None

    def convert_batch(
        self,
        raw_files: List[Union[str, Path]],
        force: bool = False
    ) -> List[Path]:
        """
        Convert multiple .raw files to mzML.

        Args:
            raw_files: List of paths to .raw files
            force: Overwrite existing output files

        Returns:
            List of successfully converted mzML file paths
        """
        converted_files = []

        for raw_file in raw_files:
            output_file = self.convert(raw_file, force=force)
            if output_file:
                converted_files.append(output_file)

        logger.info(f"Batch conversion complete: {len(converted_files)}/{len(raw_files)} files")
        return converted_files

    def convert_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.raw",
        force: bool = False
    ) -> List[Path]:
        """
        Convert all .raw files in a directory.

        Args:
            directory: Directory containing .raw files
            pattern: Glob pattern for .raw files (default: *.raw)
            force: Overwrite existing output files

        Returns:
            List of successfully converted mzML file paths
        """
        directory = Path(directory)
        raw_files = list(directory.glob(pattern))

        if not raw_files:
            logger.warning(f"No .raw files found in: {directory}")
            return []

        logger.info(f"Found {len(raw_files)} .raw files in {directory}")
        return self.convert_batch(raw_files, force=force)


# Example usage
if __name__ == "__main__":
    # Initialize converter
    converter = RawConverter(output_dir="./data/proteomics/mzml")

    # Convert single file
    # mzml_file = converter.convert("./data/proteomics/raw/sample.raw")

    # Convert all files in directory
    # mzml_files = converter.convert_directory("./data/proteomics/raw")

    print("RawConverter initialized. Use convert() or convert_batch() to process files.")
