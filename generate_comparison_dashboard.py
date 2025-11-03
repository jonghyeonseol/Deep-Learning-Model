#!/usr/bin/env python3
"""
Generate Activation Function Comparison Dashboard

This script generates comprehensive comparison dashboards for different activation functions
that have been trained using main.py or main_modern.py.

Usage:
    python generate_comparison_dashboard.py
    python generate_comparison_dashboard.py --checkpoints-dir ./checkpoints
    python generate_comparison_dashboard.py --output-dir ./results/comparison

The script will:
1. Load metrics from all activation function checkpoints
2. Generate comparison plots for each metric (Accuracy, F1, Precision, Recall, mAP)
3. Create a comprehensive heatmap comparing all metrics
4. Generate a radar chart for visual comparison
5. Save all visualizations to the output directory
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.dashboard import ActivationComparisonDashboard
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive comparison dashboard for activation functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate comparison dashboard from default checkpoints directory
  python generate_comparison_dashboard.py

  # Specify custom checkpoints directory
  python generate_comparison_dashboard.py --checkpoints-dir ./my_checkpoints

  # Save to custom output directory
  python generate_comparison_dashboard.py --output-dir ./results/comparison

  # Combine both options
  python generate_comparison_dashboard.py --checkpoints-dir ./checkpoints --output-dir ./results
        """
    )

    parser.add_argument(
        '--checkpoints-dir',
        type=str,
        default='./checkpoints',
        help='Directory containing activation function checkpoints (default: ./checkpoints)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints/comparison',
        help='Directory to save comparison dashboard (default: ./checkpoints/comparison)'
    )

    args = parser.parse_args()

    # Validate checkpoints directory
    checkpoints_path = Path(args.checkpoints_dir)
    if not checkpoints_path.exists():
        logger.error(f"Checkpoints directory not found: {args.checkpoints_dir}")
        logger.info("Please run main.py with different activation functions first.")
        sys.exit(1)

    # Count available activation checkpoints
    activation_dirs = [d for d in checkpoints_path.iterdir()
                      if d.is_dir() and (d / f"{d.name}_metrics.json").exists()]

    if not activation_dirs:
        logger.error("No activation function metrics found in checkpoints directory")
        logger.info("Please run main.py with activation functions to generate metrics first.")
        logger.info("\nExample:")
        logger.info("  python main.py --activation relu --epochs 5")
        logger.info("  python main.py --activation gelu --epochs 5")
        logger.info("  python main.py --activation swish --epochs 5")
        sys.exit(1)

    logger.info("="*80)
    logger.info("ACTIVATION FUNCTION COMPARISON DASHBOARD GENERATOR")
    logger.info("="*80)
    logger.info(f"Checkpoints directory: {args.checkpoints_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Found {len(activation_dirs)} activation functions with metrics:")
    for activation_dir in sorted(activation_dirs):
        logger.info(f"  - {activation_dir.name}")
    logger.info("="*80)

    # Create comparison dashboard
    logger.info("\nGenerating comparison dashboard...")
    dashboard = ActivationComparisonDashboard(save_dir=args.output_dir)

    try:
        dashboard.create_full_comparison_report(checkpoints_dir=args.checkpoints_dir)
        logger.info("\n" + "="*80)
        logger.info("✅ DASHBOARD GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"\nGenerated visualizations:")
        logger.info(f"  1. Individual metric comparisons:")
        logger.info(f"     - activation_comparison_accuracy.png")
        logger.info(f"     - activation_comparison_f1_macro.png")
        logger.info(f"     - activation_comparison_precision_macro.png")
        logger.info(f"     - activation_comparison_recall_macro.png")
        logger.info(f"     - activation_comparison_mAP.png")
        logger.info(f"\n  2. Comprehensive comparisons:")
        logger.info(f"     - activation_comprehensive_comparison.png (heatmap)")
        logger.info(f"     - activation_radar_comparison.png (radar chart)")
        logger.info(f"\nAll visualizations saved to: {args.output_dir}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
