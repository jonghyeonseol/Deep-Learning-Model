"""
Comprehensive metrics dashboard for deep learning model evaluation.

Creates interactive and static visualizations for all metrics:
- Confusion matrix heatmap
- PR curves for all classes
- ROC curves
- Per-class metrics bar charts
- Training history plots
- Activation function comparison dashboard
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime


class MetricsDashboard:
    """Create comprehensive visualizations for model evaluation metrics."""

    def __init__(self, save_dir: str, model_name: str = "model"):
        """
        Initialize metrics dashboard.

        Args:
            save_dir: Directory to save dashboard images
            model_name: Name of the model (for filenames)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                             class_names: List[str],
                             normalize: bool = False,
                             figsize: tuple = (10, 8)):
        """
        Plot confusion matrix as heatmap.

        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize the matrix
            figsize: Figure size
        """
        plt.figure(figsize=figsize)

        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'

        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})

        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        # Save
        filename = f'{self.model_name}_confusion_matrix{"_normalized" if normalize else ""}.png'
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pr_curves(self, pr_data: Dict[str, Dict[str, np.ndarray]],
                      ap_scores: Dict[str, float],
                      figsize: tuple = (12, 8)):
        """
        Plot Precision-Recall curves for all classes.

        Args:
            pr_data: PR curve data for each class
            ap_scores: Average Precision scores for each class
            figsize: Figure size
        """
        if not pr_data:
            return

        plt.figure(figsize=figsize)

        for class_name, data in pr_data.items():
            ap = ap_scores.get(class_name, 0.0)
            plt.plot(data['recall'], data['precision'],
                    label=f'{class_name} (AP={ap:.3f})', linewidth=2)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()

        # Save
        filename = f'{self.model_name}_pr_curves.png'
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, roc_data: Dict[str, Dict[str, np.ndarray]],
                       auc_scores: Dict[str, float],
                       figsize: tuple = (12, 8)):
        """
        Plot ROC curves for all classes.

        Args:
            roc_data: ROC curve data for each class
            auc_scores: AUC scores for each class
            figsize: Figure size
        """
        if not roc_data:
            return

        plt.figure(figsize=figsize)

        for class_name, data in roc_data.items():
            auc = auc_scores.get(class_name, 0.0)
            plt.plot(data['fpr'], data['tpr'],
                    label=f'{class_name} (AUC={auc:.3f})', linewidth=2)

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()

        # Save
        filename = f'{self.model_name}_roc_curves.png'
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_per_class_metrics(self, per_class_metrics: Dict[str, Dict[str, float]],
                               figsize: tuple = (14, 6)):
        """
        Plot per-class precision, recall, and F1 scores.

        Args:
            per_class_metrics: Dictionary of per-class metrics
            figsize: Figure size
        """
        # Prepare data
        classes = list(per_class_metrics.keys())
        precision = [per_class_metrics[c]['precision'] for c in classes]
        recall = [per_class_metrics[c]['recall'] for c in classes]
        f1 = [per_class_metrics[c]['f1'] for c in classes]

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Precision
        axes[0].bar(classes, precision, color='skyblue', edgecolor='black')
        axes[0].set_title('Precision per Class', fontweight='bold')
        axes[0].set_ylabel('Precision', fontsize=11)
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)

        # Recall
        axes[1].bar(classes, recall, color='lightcoral', edgecolor='black')
        axes[1].set_title('Recall per Class', fontweight='bold')
        axes[1].set_ylabel('Recall', fontsize=11)
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)

        # F1 Score
        axes[2].bar(classes, f1, color='lightgreen', edgecolor='black')
        axes[2].set_title('F1 Score per Class', fontweight='bold')
        axes[2].set_ylabel('F1 Score', fontsize=11)
        axes[2].set_ylim([0, 1])
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save
        filename = f'{self.model_name}_per_class_metrics.png'
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metrics_summary(self, basic_metrics: Dict[str, float],
                            map_score: float,
                            figsize: tuple = (12, 6)):
        """
        Plot summary of key metrics.

        Args:
            basic_metrics: Dictionary of basic metrics
            map_score: Mean Average Precision score
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Overall metrics
        metrics_to_plot = {
            'Accuracy': basic_metrics['accuracy'],
            'Precision': basic_metrics['precision_macro'],
            'Recall': basic_metrics['recall_macro'],
            'F1 Score': basic_metrics['f1_macro'],
            'mAP': map_score
        }

        axes[0].barh(list(metrics_to_plot.keys()), list(metrics_to_plot.values()),
                    color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
                    edgecolor='black')
        axes[0].set_xlabel('Score', fontsize=11)
        axes[0].set_title('Overall Metrics Summary', fontweight='bold', fontsize=12)
        axes[0].set_xlim([0, 1])
        axes[0].grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (metric, value) in enumerate(metrics_to_plot.items()):
            axes[0].text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=10)

        # Averaging methods comparison
        avg_methods = {
            'Macro': [basic_metrics['precision_macro'],
                     basic_metrics['recall_macro'],
                     basic_metrics['f1_macro']],
            'Micro': [basic_metrics['precision_micro'],
                     basic_metrics['recall_micro'],
                     basic_metrics['f1_micro']],
            'Weighted': [basic_metrics['precision_weighted'],
                        basic_metrics['recall_weighted'],
                        basic_metrics['f1_weighted']]
        }

        x = np.arange(3)
        width = 0.25

        for i, (method, values) in enumerate(avg_methods.items()):
            axes[1].bar(x + i * width, values, width,
                       label=method, edgecolor='black')

        axes[1].set_xlabel('Metric', fontsize=11)
        axes[1].set_ylabel('Score', fontsize=11)
        axes[1].set_title('Averaging Methods Comparison', fontweight='bold', fontsize=12)
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(['Precision', 'Recall', 'F1'])
        axes[1].legend()
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save
        filename = f'{self.model_name}_metrics_summary.png'
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def create_comprehensive_dashboard(self, metrics: Dict):
        """
        Create comprehensive dashboard with all metrics.

        Args:
            metrics: Dictionary containing all computed metrics
        """
        class_names = list(metrics['per_class'].keys())

        # 1. Confusion Matrix
        self.plot_confusion_matrix(metrics['confusion_matrix'], class_names)
        self.plot_confusion_matrix(metrics['confusion_matrix'], class_names, normalize=True)

        # 2. PR Curves
        if metrics['pr_curve']:
            self.plot_pr_curves(metrics['pr_curve'], metrics['average_precision'])

        # 3. ROC Curves
        if metrics['roc_curve']:
            self.plot_roc_curves(metrics['roc_curve'], metrics['roc_auc'])

        # 4. Per-class metrics
        self.plot_per_class_metrics(metrics['per_class'])

        # 5. Metrics summary
        self.plot_metrics_summary(metrics['basic'], metrics['mean_average_precision'])

        # 6. Save metrics as JSON
        self._save_metrics_json(metrics)

        print(f"\n✅ Dashboard saved to: {self.save_dir}")

    def _save_metrics_json(self, metrics: Dict):
        """Save metrics to JSON file for later analysis."""
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'basic_metrics': metrics['basic'],
            'per_class_metrics': metrics['per_class'],
            'average_precision': metrics['average_precision'],
            'mean_average_precision': metrics['mean_average_precision'],
            'roc_auc': metrics['roc_auc'],
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }

        filename = f'{self.model_name}_metrics.json'
        with open(self.save_dir / filename, 'w') as f:
            json.dump(metrics_json, f, indent=2)


class ActivationComparisonDashboard:
    """Create comparison dashboard for different activation functions."""

    def __init__(self, save_dir: str):
        """
        Initialize comparison dashboard.

        Args:
            save_dir: Directory to save dashboard images
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def load_metrics_from_json(self, checkpoints_dir: str) -> Dict[str, Dict]:
        """
        Load metrics from all activation function checkpoints.

        Args:
            checkpoints_dir: Directory containing activation function checkpoints

        Returns:
            Dictionary mapping activation function names to their metrics
        """
        checkpoints_path = Path(checkpoints_dir)
        all_metrics = {}

        for activation_dir in checkpoints_path.iterdir():
            if activation_dir.is_dir():
                metrics_file = activation_dir / f"{activation_dir.name}_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        all_metrics[activation_dir.name] = json.load(f)

        return all_metrics

    def plot_activation_comparison(self, all_metrics: Dict[str, Dict],
                                   metric_name: str = 'accuracy',
                                   figsize: tuple = (14, 8)):
        """
        Plot comparison of different activation functions.

        Args:
            all_metrics: Dictionary of metrics for each activation function
            metric_name: Metric to compare ('accuracy', 'f1_macro', 'precision_macro', etc.)
            figsize: Figure size
        """
        if not all_metrics:
            return

        # Extract data
        activations = list(all_metrics.keys())

        # Get metric values
        if metric_name in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                          'precision_micro', 'recall_micro', 'f1_micro',
                          'precision_weighted', 'recall_weighted', 'f1_weighted']:
            values = [all_metrics[act]['basic_metrics'][metric_name] for act in activations]
        elif metric_name == 'mAP':
            values = [all_metrics[act]['mean_average_precision'] for act in activations]
        else:
            print(f"Unknown metric: {metric_name}")
            return

        # Sort by values
        sorted_indices = np.argsort(values)[::-1]
        activations = [activations[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        # Plot
        plt.figure(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(activations)))
        bars = plt.barh(activations, values, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for i, (act, val) in enumerate(zip(activations, values)):
            plt.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

        plt.xlabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        plt.ylabel('Activation Function', fontsize=12, fontweight='bold')
        plt.title(f'Activation Function Comparison - {metric_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
        plt.xlim([0, 1])
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        # Save
        filename = f'activation_comparison_{metric_name}.png'
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_comprehensive_comparison(self, all_metrics: Dict[str, Dict],
                                     figsize: tuple = (16, 10)):
        """
        Plot comprehensive comparison across multiple metrics.

        Args:
            all_metrics: Dictionary of metrics for each activation function
            figsize: Figure size
        """
        if not all_metrics:
            return

        activations = list(all_metrics.keys())
        metrics_to_compare = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'mAP']

        # Prepare data
        data = []
        for activation in activations:
            row = [activation]
            for metric in metrics_to_compare:
                if metric == 'mAP':
                    row.append(all_metrics[activation]['mean_average_precision'])
                else:
                    row.append(all_metrics[activation]['basic_metrics'][metric])
            data.append(row)

        df = pd.DataFrame(data, columns=['Activation'] + metrics_to_compare)

        # Plot heatmap
        plt.figure(figsize=figsize)

        # Normalize data for heatmap
        data_matrix = df.iloc[:, 1:].values

        sns.heatmap(data_matrix, annot=True, fmt='.4f', cmap='YlGnBu',
                   xticklabels=[m.replace('_', ' ').title() for m in metrics_to_compare],
                   yticklabels=df['Activation'].values,
                   cbar_kws={'label': 'Score'},
                   vmin=0, vmax=1)

        plt.title('Comprehensive Activation Function Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Metric', fontsize=12, fontweight='bold')
        plt.ylabel('Activation Function', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Save
        filename = 'activation_comprehensive_comparison.png'
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        # Also create radar chart
        self._plot_radar_comparison(df, metrics_to_compare)

    def _plot_radar_comparison(self, df: pd.DataFrame, metrics: List[str],
                               figsize: tuple = (12, 12)):
        """Create radar chart for activation function comparison."""
        from math import pi

        # Number of variables
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        # Plot each activation function
        for idx, row in df.iterrows():
            values = row[metrics].values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Activation'])
            ax.fill(angles, values, alpha=0.15)

        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.title('Activation Function Performance Radar', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        # Save
        filename = 'activation_radar_comparison.png'
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def create_full_comparison_report(self, checkpoints_dir: str):
        """
        Create full comparison report for all activation functions.

        Args:
            checkpoints_dir: Directory containing activation function checkpoints
        """
        # Load all metrics
        all_metrics = self.load_metrics_from_json(checkpoints_dir)

        if not all_metrics:
            print("No metrics found in checkpoint directories")
            return

        print(f"\nFound metrics for {len(all_metrics)} activation functions")

        # Plot individual metric comparisons
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'mAP']:
            self.plot_activation_comparison(all_metrics, metric)

        # Plot comprehensive comparison
        self.plot_comprehensive_comparison(all_metrics)

        print(f"\n✅ Comparison dashboard saved to: {self.save_dir}")
