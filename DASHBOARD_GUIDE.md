# Comprehensive Metrics Dashboard Guide

## Overview

This guide explains how to use the comprehensive metrics dashboard system for evaluating and comparing deep learning models. The dashboard provides detailed insights into model performance using various metrics including Accuracy, Precision, Recall, F1 Score, PR curves, AP (Average Precision), mAP (mean Average Precision), ROC curves, and more.

## Features

### Individual Model Metrics

When you train a model using `main.py` or `main_modern.py`, the system automatically generates:

1. **Confusion Matrix** - Both raw and normalized versions
2. **PR Curves** - Precision-Recall curves for all classes with AP scores
3. **ROC Curves** - ROC curves for all classes with AUC scores
4. **Per-Class Metrics** - Bar charts showing Precision, Recall, and F1 per class
5. **Metrics Summary** - Overall metrics and averaging method comparisons
6. **Metrics JSON** - Machine-readable metrics file for programmatic access

### Activation Function Comparison

Compare different activation functions across all metrics:

1. **Individual Metric Comparisons** - Bar charts for each metric
2. **Comprehensive Heatmap** - All metrics in one visualization
3. **Radar Chart** - Multi-dimensional performance visualization

## Quick Start

### Training a Single Model with Dashboard

```bash
# Train with ReLU activation (dashboard generated automatically)
python main.py --activation relu --epochs 10

# Dashboard files will be saved to: ./checkpoints/relu/
```

**Generated Files:**
```
checkpoints/relu/
├── relu_confusion_matrix.png           # Confusion matrix (raw counts)
├── relu_confusion_matrix_normalized.png # Normalized confusion matrix
├── relu_pr_curves.png                   # Precision-Recall curves
├── relu_roc_curves.png                  # ROC curves
├── relu_per_class_metrics.png           # Per-class P/R/F1 bar charts
├── relu_metrics_summary.png             # Overall metrics summary
└── relu_metrics.json                    # All metrics in JSON format
```

### Comparing Multiple Activation Functions

#### Method 1: Train Multiple Functions at Once

```bash
# Compare modern activation functions (GELU, Swish, Mish, SiLU, Hardswish)
python main.py --activation modern --epochs 5

# Compare classic activation functions (ReLU, Tanh, Sigmoid, LeakyReLU, ELU)
python main.py --activation classic --epochs 5

# Compare all available activation functions
python main.py --activation all --epochs 3
```

The comparison dashboard is automatically generated at the end.

#### Method 2: Generate Comparison from Existing Checkpoints

If you've already trained multiple models:

```bash
# Generate comparison dashboard from existing checkpoints
python generate_comparison_dashboard.py

# Specify custom directories
python generate_comparison_dashboard.py \
    --checkpoints-dir ./checkpoints \
    --output-dir ./results/comparison
```

**Generated Files:**
```
checkpoints/comparison/
├── activation_comparison_accuracy.png
├── activation_comparison_f1_macro.png
├── activation_comparison_precision_macro.png
├── activation_comparison_recall_macro.png
├── activation_comparison_mAP.png
├── activation_comprehensive_comparison.png  # Heatmap
└── activation_radar_comparison.png          # Radar chart
```

## Understanding the Metrics

### Basic Classification Metrics

#### Accuracy
- **Definition**: Percentage of correct predictions
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Use Case**: Overall model performance
- **Note**: Can be misleading with imbalanced datasets

#### Precision
- **Definition**: How many predicted positives are actually positive
- **Formula**: TP / (TP + FP)
- **Use Case**: When false positives are costly
- **Example**: Medical diagnosis (avoid false alarms)

#### Recall (Sensitivity)
- **Definition**: How many actual positives were correctly identified
- **Formula**: TP / (TP + FN)
- **Use Case**: When false negatives are costly
- **Example**: Disease detection (don't miss cases)

#### F1 Score
- **Definition**: Harmonic mean of Precision and Recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Use Case**: Balance between Precision and Recall
- **Interpretation**: Good general-purpose metric

### Advanced Metrics

#### Average Precision (AP)
- **Definition**: Area under the Precision-Recall curve for one class
- **Range**: 0.0 to 1.0 (higher is better)
- **Use Case**: Evaluate model for specific class
- **Interpretation**: Weighted average of precisions at each threshold

#### Mean Average Precision (mAP)
- **Definition**: Average of AP scores across all classes
- **Formula**: (1/N) × Σ(AP_i) for i=1 to N classes
- **Use Case**: Overall multi-class performance
- **Interpretation**: Standard metric for object detection and classification

#### ROC AUC (Area Under ROC Curve)
- **Definition**: Area under the True Positive Rate vs False Positive Rate curve
- **Range**: 0.0 to 1.0 (higher is better)
- **Use Case**: Binary classification performance
- **Interpretation**: 0.5 = random, 1.0 = perfect

#### IoU (Intersection over Union)
- **Definition**: Overlap between predicted and ground truth masks
- **Formula**: Intersection / Union
- **Use Case**: Segmentation tasks (not classification)
- **Note**: Available but more relevant for segmentation

### Averaging Methods

The dashboard shows three averaging methods for Precision, Recall, and F1:

#### Macro Average
- **Method**: Simple average across all classes
- **Formula**: (1/N) × Σ(metric_i)
- **Use Case**: All classes equally important
- **Note**: Gives equal weight to each class regardless of size

#### Micro Average
- **Method**: Aggregate across all classes, then compute
- **Formula**: Metric(Σ TP, Σ FP, Σ FN)
- **Use Case**: Large classes more important
- **Note**: Dominated by frequent classes

#### Weighted Average
- **Method**: Weighted average by class support (number of samples)
- **Formula**: Σ(weight_i × metric_i) where weight_i = support_i / total
- **Use Case**: Balance between macro and micro
- **Note**: Accounts for class imbalance

## Interpreting Dashboard Visualizations

### Confusion Matrix

**What it shows:**
- Diagonal: Correct predictions (darker is better)
- Off-diagonal: Misclassifications (lighter is better)

**How to read:**
- Row i, Column j: Actual class i predicted as class j
- Normalized version shows proportions (easier to compare)

**Example interpretation:**
```
         Predicted
         Dog  Cat
Actual
Dog      90   10   → 90% dogs correctly classified, 10% confused with cats
Cat       5   95   → 95% cats correctly classified, 5% confused with dogs
```

### PR Curves (Precision-Recall)

**What it shows:**
- Trade-off between Precision and Recall at different thresholds
- Higher curve = better performance

**How to read:**
- Top-right corner: Both high Precision and Recall (ideal)
- Horizontal line: High Recall, varying Precision
- Vertical line: High Precision, varying Recall

**AP Score:**
- Area under the PR curve
- Higher is better (max = 1.0)
- Good: >0.8, Fair: 0.5-0.8, Poor: <0.5

### ROC Curves

**What it shows:**
- True Positive Rate (Recall) vs False Positive Rate
- Curve closer to top-left corner is better

**How to read:**
- Diagonal line: Random classifier (AUC = 0.5)
- Above diagonal: Better than random
- Perfect classifier: Hugs top-left (AUC = 1.0)

**AUC Score:**
- Area under ROC curve
- Excellent: >0.9, Good: 0.8-0.9, Fair: 0.7-0.8, Poor: <0.7

### Per-Class Metrics Bar Charts

**What it shows:**
- Precision, Recall, and F1 for each class side-by-side

**How to interpret:**
- Consistent bars: Balanced performance
- High Precision, Low Recall: Model is conservative
- Low Precision, High Recall: Model is aggressive
- Both low: Model struggles with this class

**Use case:**
- Identify problematic classes
- Understand class-specific behavior
- Decide if class imbalance is an issue

### Metrics Summary

**Overall Metrics Bar Chart:**
- Quick overview of 5 key metrics
- All scaled 0-1 for easy comparison
- Look for consistently high values (>0.8)

**Averaging Methods Comparison:**
- Compare Macro, Micro, Weighted for P/R/F1
- Large differences indicate class imbalance
- Micro > Macro: Model better on frequent classes
- Macro > Micro: Model better on rare classes

### Activation Function Comparison

#### Bar Charts
- **Horizontal bars**: Easier to read names
- **Sorted**: Best performers at top
- **Values displayed**: Exact scores for comparison

**What to look for:**
- Clear winner or tight competition?
- Consistent performance across metrics?
- Trade-offs (e.g., high accuracy but low mAP)?

#### Heatmap
- **Color intensity**: Darker = better performance
- **Row-wise comparison**: How one activation does across metrics
- **Column-wise comparison**: Which activation wins for specific metric

**What to look for:**
- Rows with consistent dark colors: Well-rounded activations
- Bright spots: Weak areas needing attention
- Pattern similarity: Activations with similar behavior

#### Radar Chart
- **Multi-dimensional view**: All metrics at once
- **Larger area**: Better overall performance
- **Shape**: Indicates strengths/weaknesses

**What to look for:**
- Circular shape: Balanced performance
- Spiky shape: Strong in some areas, weak in others
- Overlapping areas: Similar overall performance

## Programmatic Access to Metrics

All metrics are saved in JSON format for programmatic access:

```python
import json

# Load metrics for a specific model
with open('./checkpoints/relu/relu_metrics.json', 'r') as f:
    metrics = json.load(f)

# Access basic metrics
print(f"Accuracy: {metrics['basic_metrics']['accuracy']}")
print(f"F1 Score: {metrics['basic_metrics']['f1_macro']}")
print(f"mAP: {metrics['mean_average_precision']}")

# Access per-class metrics
for class_name, class_metrics in metrics['per_class_metrics'].items():
    print(f"{class_name}:")
    print(f"  Precision: {class_metrics['precision']}")
    print(f"  Recall: {class_metrics['recall']}")
    print(f"  F1: {class_metrics['f1']}")

# Access Average Precision scores
for class_name, ap in metrics['average_precision'].items():
    print(f"{class_name} AP: {ap}")
```

## Advanced Usage

### Custom Dashboard Generation

You can generate dashboards programmatically:

```python
from utils.metrics import MetricsCalculator
from utils.dashboard import MetricsDashboard
import torch

# Initialize metrics calculator
metrics_calc = MetricsCalculator(
    num_classes=10,
    class_names=['class0', 'class1', ..., 'class9']
)

# During testing, update metrics
for data, target in test_loader:
    output = model(data)
    probabilities = torch.softmax(output, dim=1)
    _, predicted = output.max(1)

    metrics_calc.update(predicted, target, probabilities)

# Compute all metrics
all_metrics = metrics_calc.compute_all_metrics()

# Generate dashboard
dashboard = MetricsDashboard(
    save_dir='./my_results',
    model_name='my_model'
)
dashboard.create_comprehensive_dashboard(all_metrics)
```

### Custom Comparison Dashboard

```python
from utils.dashboard import ActivationComparisonDashboard

# Create comparison dashboard
dashboard = ActivationComparisonDashboard(save_dir='./comparison')

# Generate specific comparison
dashboard.plot_activation_comparison(
    all_metrics,
    metric_name='f1_macro',
    figsize=(14, 8)
)

# Generate full report
dashboard.create_full_comparison_report(
    checkpoints_dir='./checkpoints'
)
```

## Best Practices

### When Training Models

1. **Always enable dashboard generation** (default behavior)
2. **Use consistent class names** across experiments
3. **Save checkpoints** with meaningful names
4. **Run multiple epochs** for stable metrics (at least 5)
5. **Document experiments** using the generated JSON files

### When Comparing Activations

1. **Use same hyperparameters** (epochs, batch size, learning rate)
2. **Train on same data splits** for fair comparison
3. **Run multiple seeds** if possible and average results
4. **Consider multiple metrics** not just accuracy
5. **Look for trade-offs** (e.g., speed vs accuracy)

### Metrics Selection

**For balanced datasets:**
- Focus on Accuracy, F1 Score, mAP

**For imbalanced datasets:**
- Focus on F1 Score, AP per class, Weighted F1
- Macro averaging gives insight into minority classes
- Confusion matrix reveals misclassification patterns

**For specific use cases:**
- High-risk applications: Prioritize Recall (minimize false negatives)
- Low false alarm tolerance: Prioritize Precision
- Balanced requirements: F1 Score or mAP

## Troubleshooting

### Dashboard Not Generated

**Problem:** No dashboard files after training

**Solutions:**
1. Check if testing was performed: `trainer.test()`
2. Ensure `generate_dashboard=True` (default)
3. Check save directory permissions
4. Look for error messages in console

### Missing Comparison Dashboard

**Problem:** Comparison dashboard not generated

**Solutions:**
1. Verify metrics JSON files exist in checkpoint directories
2. Check naming: `{activation}_metrics.json`
3. Ensure at least 2 activation functions trained
4. Run `generate_comparison_dashboard.py` manually

### Import Errors

**Problem:** `ModuleNotFoundError` or `ImportError`

**Solutions:**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Verify imports
python -c "from utils.metrics import MetricsCalculator"
python -c "from utils.dashboard import MetricsDashboard"
```

### Memory Issues

**Problem:** Out of memory when computing metrics

**Solutions:**
1. Reduce test batch size
2. Process metrics in smaller batches
3. Disable dashboard for quick tests: `generate_dashboard=False`

## Examples and Use Cases

### Example 1: Quick Model Evaluation

```bash
# Train and evaluate a single model
python main.py --activation gelu --epochs 5 --quick

# Check results
ls checkpoints/gelu/
```

**Expected output:** 6 PNG files + 1 JSON file

### Example 2: Comprehensive Activation Study

```bash
# Train all modern activations
python main.py --activation modern --epochs 10

# Results automatically include:
# - Individual dashboards for each activation
# - Comprehensive comparison dashboard
```

### Example 3: Custom Comparison

```bash
# Train specific activations
python main.py --activation relu --epochs 10
python main.py --activation gelu --epochs 10
python main.py --activation swish --epochs 10

# Generate custom comparison
python generate_comparison_dashboard.py \
    --checkpoints-dir ./checkpoints \
    --output-dir ./paper_figures
```

### Example 4: Production Model Selection

```bash
# 1. Train candidate models
for act in relu gelu swish mish; do
    python main.py --activation $act --epochs 50
done

# 2. Generate comparison
python generate_comparison_dashboard.py

# 3. Review metrics
cat checkpoints/*/`*`_metrics.json | grep "mean_average_precision"

# 4. Select best based on mAP and deploy
```

## FAQ

**Q: What's the difference between AP and mAP?**
A: AP is the average precision for a single class, while mAP is the mean of AP scores across all classes.

**Q: Which averaging method should I use?**
A:
- Macro: When all classes are equally important
- Micro: When larger classes are more important
- Weighted: Good balance for most scenarios

**Q: Why is my accuracy high but F1 low?**
A: This often indicates class imbalance. The model performs well on majority class but poorly on minority classes.

**Q: Should I optimize for accuracy or mAP?**
A:
- Accuracy: Simple, balanced datasets
- mAP: Multi-class problems, especially with imbalance
- For CIFAR-10 (balanced), both are good indicators

**Q: How do I disable dashboard generation?**
A: Pass `generate_dashboard=False` to `trainer.test()`

**Q: Can I use this for non-CIFAR-10 datasets?**
A: Yes! Just ensure you pass correct `class_names` to the metrics calculator.

## Additional Resources

- **metrics.py**: Detailed implementation of all metrics
- **dashboard.py**: Visualization code with customization options
- **main.py**: Integration example with basic trainer
- **main_modern.py**: Integration with modern training techniques
- **CLAUDE.md**: Overall project documentation
- **CHEAT_SHEET.md**: Quick reference for common commands

## Performance Benchmarks

**CIFAR-10 Expected Performance:**

| Metric | Good | Excellent |
|--------|------|-----------|
| Accuracy | >90% | >95% |
| F1 (macro) | >0.85 | >0.92 |
| mAP | >0.85 | >0.93 |
| Per-class F1 | >0.80 | >0.90 |

**Dashboard Generation Time:**
- Single model: ~5-10 seconds
- Comparison (5 models): ~15-20 seconds

**File Sizes:**
- Each dashboard image: 100-300 KB
- Metrics JSON: 5-20 KB
- Total per model: ~2-3 MB

## Contributing

To extend the dashboard system:

1. **Add new metrics**: Edit `utils/metrics.py`
2. **Add new visualizations**: Edit `utils/dashboard.py`
3. **Test with**: `pytest tests/test_metrics.py`
4. **Update this guide**: Add documentation for new features

## Version History

- **v1.0** (Current): Initial release with comprehensive metrics and comparison dashboards
  - Accuracy, Precision, Recall, F1
  - PR curves, ROC curves, AP, mAP
  - Confusion matrix (raw and normalized)
  - Per-class metrics visualization
  - Activation function comparison dashboard
  - JSON export for programmatic access

---

**For more information, see:**
- Project README: `README.md`
- User Manual: `USER_MANUAL.md`
- Quick Start: `QUICK_START.md`
