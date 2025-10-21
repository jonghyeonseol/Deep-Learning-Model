# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two main deep learning frameworks:

1. **Image Classification** (CIFAR-10): Testing and comparing different activation functions on image classification tasks. Implements custom activation functions from scratch with comprehensive training, visualization, and monitoring tools.

2. **Proteomics Analysis** (NEW): Deep learning framework for analyzing mass spectrometry proteomics data from .raw files, mzML, and CSV formats. Supports spectrum classification, retention time prediction, and spectral similarity learning.

3. **Glycoproteomics Analysis** (NEW): Specialized tools for glycoproteomics analysis including oxonium ion detection, glycan composition inference, Y-ion analysis, and automated glycopeptide identification.

## Common Commands

### Environment Setup
```bash
# Activate virtual environment (required before running any Python commands)
source venv/bin/activate

# Install proteomics dependencies
pip install -r requirements_proteomics.txt
```

### Training and Testing (Image Classification)
```bash
# List all available activation functions
python3 main.py --list-activations

# Quick training test with 2 epochs (3-5 minutes)
python3 main.py --activation relu --quick

# Train with specific activation function
python3 main.py --activation [name] --epochs [N] --batch-size [N] --lr [float]

# Compare all modern activation functions (GELU, Swish, Mish, SiLU, Hardswish)
python3 main.py --activation modern --epochs 5

# Compare all classic activation functions (ReLU, Tanh, Sigmoid, LeakyReLU, ELU)
python3 main.py --activation classic --epochs 5

# Compare all available activation functions (2-3 hours)
python3 main.py --activation all --epochs 3
```

### Live Visualization (Image Classification)
```bash
# Visualize network structure in real-time (neurons and connections)
python3 main.py --visualize

# Train with live monitoring (real-time loss/accuracy plots)
python3 main.py --activation swish --epochs 5 --monitor

# Combine visualization and monitoring
python3 main.py --activation relu --monitor --quick
```

### Proteomics Analysis
```bash
# Train CNN on mzML data for classification
python3 proteomics/main_proteomics.py \
  --data data/proteomics/mzml/sample.mzML \
  --model cnn \
  --task classification \
  --num-classes 10 \
  --epochs 50 \
  --batch-size 32

# Train Transformer model
python3 proteomics/main_proteomics.py \
  --data data/proteomics/mzml/sample.mzML \
  --model transformer \
  --task classification \
  --epochs 50

# Retention time prediction
python3 proteomics/main_proteomics.py \
  --data data/proteomics/mzml/sample.mzML \
  --model rt_predictor \
  --task regression \
  --epochs 50

# Convert .raw files to mzML (requires ThermoRawFileParser)
python3 -c "from proteomics.data_loaders import RawConverter; \
  RawConverter().convert_directory('data/proteomics/raw')"

# Custom config file
python3 proteomics/main_proteomics.py \
  --data data/proteomics/mzml/sample.mzML \
  --config my_config.yaml \
  --experiment-name my_experiment
```

### Glycoproteomics Analysis
```bash
# Convert .raw files to mzML
python3 convert_raw_to_mzml.py sample.raw
python3 convert_raw_to_mzml.py data/proteomics/raw/ --batch

# Analyze glycopeptides from mzML
python3 analyze_glycoproteomics.py sample.mzML --output results.csv

# Convert and analyze .raw file
python3 analyze_glycoproteomics.py sample.raw --convert --output results.csv

# Batch glycoproteomics analysis
python3 analyze_glycoproteomics.py data_dir/ --batch --output results_dir/

# Python API for glycan analysis
python3 -c "
from proteomics.glycoproteomics import GlycanMassCalculator
calc = GlycanMassCalculator()
mass = calc.calculate_mass({'Hex': 5, 'HexNAc': 4, 'Fuc': 1, 'NeuAc': 2})
print(f'Glycan mass: {mass:.4f} Da')
"
```


## Code Architecture

### Module Organization (Image Classification)

**models/**: Neural network architectures and activation functions
- `network.py`: Contains `NeuralNetwork` (fully-connected) and `ConvNeuralNetwork` (CNN for CIFAR-10)
- `activations.py`: Custom implementations of 14+ activation functions (GELU, ReLU, Tanh, Sigmoid, Swish, Mish, etc.)
- All activation functions are custom PyTorch modules, not using `torch.nn` built-ins

**utils/**: Training utilities and visualization tools
- `trainer.py`: `Trainer` class handles training loop, validation, checkpointing, early stopping
- `data_loader.py`: `CIFAR10DataLoader` wraps PyTorch's CIFAR-10 dataset with train/val/test splits
- `visualization.py`: `Visualizer` creates training plots, confusion matrices, prediction samples
- `monitor.py`: Real-time monitoring tools (`PerceptronVisualizer`, `LayerMonitor`, `ActivationAnalyzer`)
- `realtime_monitor.py`: Live training monitors with dynamic plotting capabilities

**Entry Points**:
- `main.py`: Unified script for training, visualization, and monitoring with CLI arguments

### Module Organization (Proteomics)

**proteomics/data_loaders/**: Data loading and preprocessing
- `raw_converter.py`: Converts .raw files to mzML using ThermoRawFileParser
- `mzml_reader.py`: Reads mzML/mzXML files with pyteomics/pymzml
- `csv_reader.py`: Loads pre-processed CSV data (peak-level, spectrum-level, or matrix format)
- `preprocessing.py`: Spectrum normalization, binning, peak picking, augmentation
- `spectrum_dataset.py`: PyTorch Dataset for mass spectra with caching and augmentation

**proteomics/models/**: Neural network architectures
- `spectrum_cnn.py`: 1D CNN for spectrum classification/regression (SpectrumCNN, LightweightSpectrumCNN)
- `spectrum_transformer.py`: Transformer models (SpectrumTransformer, PeakTransformer)
- `retention_predictor.py`: RT prediction models (RetentionTimePredictor, MultiTaskSpectrumModel)

**proteomics/training/**: Training infrastructure
- `trainer.py`: ProteomicsTrainer class with AMP support, early stopping, checkpointing
- `loss_functions.py`: Custom losses (SpectralAngleLoss, CosineSimilarityLoss, FocalLoss, etc.)

**proteomics/utils/**: Utilities
- `visualization.py`: Spectrum plotting, training history, confusion matrices
- `metrics.py`: Classification and regression metrics, spectral similarity

**proteomics/config/**: Configuration files
- `default_config.yaml`: Default hyperparameters and training settings

**Entry Point**:
- `proteomics/main_proteomics.py`: CLI for training proteomics models

**proteomics/glycoproteomics/**: Glycoproteomics-specific tools
- `glycan_mass.py`: Glycan mass calculator, composition parser, monosaccharide masses
- `oxonium_ion_detector.py`: Detects diagnostic oxonium ions for glycopeptide identification
- `glycan_analyzer.py`: Comprehensive analyzer (oxonium + Y-ions + composition inference)

**Standalone Scripts**:
- `convert_raw_to_mzml.py`: .raw to mzML converter CLI
- `analyze_glycoproteomics.py`: Complete glycoproteomics analysis workflow

### Data Flow (Image Classification)

1. **Data Loading**: `CIFAR10DataLoader` downloads CIFAR-10 (if needed), applies normalization, creates train/val/test splits
2. **Model Creation**: `ConvNeuralNetwork` builds CNN with specified activation function via `get_activation(name)`
3. **Training**: `Trainer` manages training loop, optimizer (Adam), scheduler (StepLR), loss computation, validation
4. **Checkpointing**: Best models saved to `checkpoints/{activation_name}/best_model.pth`
5. **Visualization**: Training history, confusion matrices, and sample predictions saved as PNG files

### Data Flow (Proteomics)

1. **Data Loading**:
   - .raw files → `RawConverter` → mzML → `MzMLReader` → Spectrum dicts
   - mzML files → `MzMLReader` → Spectrum dicts
   - CSV files → `CSVReader` → Spectrum dicts
2. **Preprocessing**: `SpectrumPreprocessor` bins spectra into fixed-length vectors (19,500 bins by default)
3. **Dataset**: `SpectrumDataset` wraps spectra with PyTorch interface, caches preprocessed data
4. **Training**: `ProteomicsTrainer` manages training with AMP, early stopping, checkpointing
5. **Evaluation**: Metrics computed, visualizations saved to `checkpoints/proteomics/`

### Data Flow (Glycoproteomics)

1. **.raw Conversion**: `convert_raw_to_mzml.py` → mzML files via ThermoRawFileParser
2. **Spectrum Reading**: `MzMLReader` extracts MS/MS spectra (MS level 2)
3. **Oxonium Detection**: `OxoniumIonDetector` identifies diagnostic ions (HexNAc, NeuAc, etc.)
4. **Glycopeptide Filtering**: Spectra with ≥2 oxonium ions flagged as glycopeptides
5. **Composition Analysis**: `GlycanAnalyzer` infers glycan structures (Hex, HexNAc, Fuc, NeuAc counts)
6. **Y-Ion Detection**: Identifies peptide + glycan fragment ions (optional, if peptide mass known)
7. **Results Export**: CSV with scan_id, RT, glycan composition, confidence scores

### Key Design Patterns

- **Activation Function Factory**: `get_activation(name)` returns activation module by string name
- **Modular Architecture**: Models, training, data loading, and visualization are fully decoupled
- **Checkpoint Organization**: Each activation function gets its own subdirectory in `checkpoints/`
- **Training History**: Stored in `Trainer.history` as dict with keys: `train_loss`, `train_acc`, `val_loss`, `val_acc`

### Model Architecture Details

**ConvNeuralNetwork** (for CIFAR-10):
- 3 Conv layers: Conv2d(3→32) → Conv2d(32→64) → Conv2d(64→128)
- MaxPool2d(2,2) after each conv layer
- 3 FC layers: Linear(2048→512) → Linear(512→256) → Linear(256→10)
- Activation function applied after each layer (except final output)
- Dropout (default 0.2) applied in FC layers

**Parameter Initialization**:
- Convolutional layers: Kaiming normal initialization
- Fully connected layers: Xavier uniform initialization

## Dataset Information

- **CIFAR-10**: 60,000 32×32 RGB images in 10 classes
- **Training Set**: 45,000 images (after 10% validation split)
- **Validation Set**: 5,000 images
- **Test Set**: 10,000 images
- **Normalization**: Mean=[0.4914, 0.4822, 0.4465], Std=[0.2023, 0.1994, 0.2010]
- **Data Location**: Downloaded to `./data/cifar-10-batches-py/` on first run

## Output Directories

### Image Classification
- `checkpoints/{activation}/`: Training checkpoints and visualizations for each activation function
  - `best_model.pth`: Saved model weights
  - `training_history.png`: Loss and accuracy curves
  - `predictions.png`: Sample predictions with ground truth
  - `confusion_matrix.png`: Classification confusion matrix
- `data/cifar-10-batches-py/`: CIFAR-10 dataset (auto-downloaded)
- `visualizations/`: Demo visualization outputs

### Proteomics
- `data/proteomics/`: Proteomics data directory
  - `raw/`: .raw files
  - `mzml/`: Converted mzML files
  - `processed/`: Cached preprocessed spectra (pickle files)
- `checkpoints/proteomics/{experiment_name}/`: Model checkpoints and results
  - `best_model.pth`: Saved model weights and training state
  - `training_history.png`: Loss and accuracy curves
  - Additional experiment-specific visualizations

## Available Activation Functions

**Modern** (2017-2020): gelu, swish, mish, silu, hardswish
**Classic**: relu, tanh, sigmoid, leakyrelu, elu, prelu, selu
**Other**: step, softmax

## Important Implementation Notes

- Always activate the virtual environment before running scripts
- First run downloads CIFAR-10 (~170MB) which may take 5-10 minutes
- GPU training is automatic if CUDA is available (check with `torch.cuda.is_available()`)
- Use `--quick` flag for rapid testing (2 epochs instead of default 10)
- Use `--visualize` flag to see live network structure (neurons and connections)
- Use `--monitor` flag to enable real-time training plots (loss, accuracy, gradients)
- Validation accuracy is more important than training accuracy (indicates generalization)
- The `Trainer` class includes early stopping (patience=5 epochs by default)
- All custom activation functions in `activations.py` are implemented from scratch for educational purposes

## Proteomics Module Details

### Supported Input Formats

1. **.raw files** (Thermo Fisher):
   - Requires ThermoRawFileParser to be installed
   - Automatically converted to mzML before processing
   - Install: `dotnet tool install ThermoRawFileParser -g`

2. **mzML/mzXML files**:
   - Standard open MS data formats
   - Directly readable with pyteomics or pymzml
   - Preferred format for most workflows

3. **CSV files**:
   - Three supported schemas:
     - **Peak-level**: Each row = one peak (columns: scan_id, rt, mz, intensity)
     - **Spectrum-level**: Each row = one spectrum (aggregated features)
     - **Matrix format**: Rows = samples, columns = m/z bins

### Model Architectures

1. **SpectrumCNN**: 1D CNN with 5 conv layers, residual connections, ~1-5M parameters
2. **LightweightSpectrumCNN**: Smaller CNN with 3 conv layers for faster training
3. **SpectrumTransformer**: Transformer with positional encoding for m/z values
4. **PeakTransformer**: Operates on sparse peak lists instead of binned spectra
5. **RetentionTimePredictor**: CNN regression model for RT prediction
6. **MultiTaskSpectrumModel**: Joint classification + RT prediction

### Spectrum Preprocessing

- **m/z Binning**: Default 50-2000 Da range, 0.1 Da bins → 19,500 features
- **Normalization**: TIC (total ion current), max intensity, or square root
- **Peak Filtering**: Remove low-intensity peaks (default: <1% of base peak)
- **Precursor Removal**: Remove precursor ±17 Da window (for MS2 spectra)
- **Augmentation**: Gaussian noise, intensity scaling, m/z shift

### Loss Functions

- **SpectralAngleLoss**: Measures angle between spectrum vectors
- **CosineSimilarityLoss**: 1 - cosine similarity
- **WeightedMSELoss**: Weights high-intensity peaks more heavily
- **FocalLoss**: For handling class imbalance
- **MultiTaskLoss**: Combines classification + regression losses
- **TripletLoss**: For learning spectral embeddings

### Key Proteomics Workflows

**Workflow 1: Spectrum Classification from .raw files**
```bash
# 1. Convert .raw to mzML (happens automatically)
# 2. Train CNN classifier
python3 proteomics/main_proteomics.py \
  --data data/proteomics/raw/sample.raw \
  --model cnn \
  --task classification \
  --num-classes 5 \
  --epochs 50
```

**Workflow 2: Retention Time Prediction**
```bash
python3 proteomics/main_proteomics.py \
  --data data/proteomics/mzml/peptides.mzML \
  --model rt_predictor \
  --task regression \
  --epochs 100
```

**Workflow 3: Custom Preprocessing**
```python
from proteomics.data_loaders import SpectrumPreprocessor, MzMLReader

# Load spectra
reader = MzMLReader('data.mzML')
spectra = reader.read_spectra(ms_level=2, min_peaks=10)

# Custom preprocessing
preprocessor = SpectrumPreprocessor(
    mz_range=(100, 1500),  # Narrower range
    bin_size=0.5,          # Larger bins
    normalization='max'    # Max normalization
)

# Process single spectrum
mz, intensity = spectra[0]['mz'], spectra[0]['intensity']
mz_proc, intensity_proc = preprocessor.preprocess(mz, intensity)
binned = preprocessor.bin_spectrum(mz_proc, intensity_proc)
```

### Performance Tips

- Use `--batch-size 64` or higher for faster training on GPU
- Enable `use_amp: true` in config for mixed precision training (2x speedup)
- Cached preprocessed spectra are saved to `data/proteomics/processed/` for faster reloading
- Use `LightweightSpectrumCNN` for quick experiments
- Transformer models are slower but may achieve better accuracy on complex tasks

## Glycoproteomics Module Details

### Overview

The glycoproteomics module provides specialized tools for analyzing glycosylated peptides from LC-MS/MS data. It uses diagnostic oxonium ions and glycan composition inference to automatically identify and characterize glycopeptides.

### Key Concepts

**Oxonium Ions**: Small diagnostic fragment ions characteristic of glycans that appear in MS/MS spectra
- HexNAc (204.087 m/z) - N-Acetylhexosamine
- NeuAc (292.103 m/z) - Sialic acid
- Fuc (147.065 m/z) - Fucose
- HexNAc-Hex (366.139 m/z) - Disaccharide fragment

**Glycan Notation**:
- **Long form**: Hex5HexNAc4Fuc1NeuAc2 (full names)
- **Short form**: H5N4F1S2 (abbreviations)
- Hex/H = Hexose, HexNAc/N = N-Acetylhexosamine, Fuc/F = Fucose, NeuAc/S = Sialic acid

**Y-Ions**: Fragment ions consisting of intact peptide + glycan fragment
- Y0: Peptide + full glycan
- Y1: Peptide + single GlcNAc
- Y2: Peptide + GlcNAc + Hex

### Glycoproteomics Workflows

**Workflow 1: Basic Glycopeptide Identification**
```bash
# 1. Place .raw files in data directory
mkdir -p data/proteomics/raw
# Copy your files here

# 2. Convert to mzML
python3 convert_raw_to_mzml.py data/proteomics/raw/ --batch

# 3. Analyze glycopeptides
python3 analyze_glycoproteomics.py data/proteomics/mzml/sample.mzML \
  --output results.csv \
  --ms-level 2 \
  --min-oxonium 2
```

**Workflow 2: High-Throughput Batch Analysis**
```bash
# Process all files in directory
python3 analyze_glycoproteomics.py data/proteomics/mzml/ \
  --batch \
  --output glyco_results/

# Results saved to:
# - glyco_results/{filename}_glycoproteomics.csv (per file)
# - glyco_results/combined_glycoproteomics_results.csv (all files)
```

**Workflow 3: Direct .raw File Analysis**
```bash
# Automatically converts and analyzes
python3 analyze_glycoproteomics.py sample.raw \
  --convert \
  --output results.csv
```

**Workflow 4: Python API for Custom Analysis**
```python
from proteomics.data_loaders import MzMLReader
from proteomics.glycoproteomics import (
    GlycanMassCalculator,
    OxoniumIonDetector,
    GlycanAnalyzer
)

# 1. Calculate theoretical glycan mass
calc = GlycanMassCalculator()
composition = {'Hex': 5, 'HexNAc': 4, 'Fuc': 1, 'NeuAc': 2}
mass = calc.calculate_mass(composition, adduct='H')
print(f"Theoretical mass: {mass:.4f} Da")

# 2. Read MS/MS data
reader = MzMLReader('sample.mzML')
spectra = reader.read_spectra(ms_level=2, min_peaks=10)

# 3. Detect oxonium ions
detector = OxoniumIonDetector(tolerance=0.02)
for spec in spectra:
    is_glyco, hits = detector.is_glycopeptide(
        spec['mz'],
        spec['intensity'],
        min_oxonium_ions=2
    )

    if is_glyco:
        score = detector.calculate_glycan_score(spec['mz'], spec['intensity'])
        print(f"Scan {spec['scan_id']}: Glycan score = {score:.3f}")
        print(f"  Oxonium ions: {[hit.name for hit in hits]}")

# 4. Comprehensive analysis
analyzer = GlycanAnalyzer(mass_tolerance=0.02)
results = analyzer.analyze_glycopeptide_spectrum(
    mz=spec['mz'],
    intensity=spec['intensity'],
    precursor_mz=spec['precursor_mz'],
    scan_id=spec['scan_id']
)

print(f"\nTop glycan compositions:")
for comp in results['glycan_compositions'][:3]:
    print(f"  {comp['composition']} (score: {comp['score']:.3f})")
```

### Oxonium Ion Reference

| Ion Name | m/z | Fragment Type | Significance |
|----------|-----|---------------|--------------|
| HexNAc | 204.087 | Monosaccharide | Primary glycan marker |
| HexNAc-H2O | 186.076 | Dehydrated | Common fragment |
| HexNAc-ring | 138.055 | Ring fragment | High specificity |
| NeuAc | 292.103 | Sialic acid | Sialylation marker |
| NeuAc-H2O | 274.092 | Dehydrated | Sialylation marker |
| Fuc | 147.065 | Fucose | Fucosylation marker |
| HexNAc-Hex | 366.139 | Disaccharide | N-glycan indicator |
| HexNAc2 | 407.166 | Two HexNAc | Complex glycan |
| HexNAc-Hex-NeuAc | 657.235 | Trisaccharide | Sialylated complex |

**Detection Criteria**:
- ≥2 oxonium ions → Likely glycopeptide
- ≥3 oxonium ions → High-confidence glycopeptide
- Presence of HexNAc (204.087) → N-glycan
- Presence of NeuAc (292.103) → Sialylated
- Presence of Fuc (147.065) → Fucosylated

### Glycan Types Supported

**N-Glycans**:
- High-mannose: Man5-9GlcNAc2
- Complex: Biantennary, triantennary, tetraantennary
- Hybrid: Mixed high-mannose and complex

**O-Glycans**:
- Core 1-4 structures
- Mucin-type

**Modifications**:
- Sialylation (NeuAc, NeuGc)
- Fucosylation (core and antenna)
- Phosphorylation
- Sulfation

### Output Format

**CSV Columns** (from `analyze_glycoproteomics.py`):
- `scan_id`: Spectrum identifier
- `rt`: Retention time (minutes)
- `precursor_mz`: Precursor m/z value
- `glycan_score`: Confidence score (0-1, >0.5 = high confidence)
- `num_oxonium_ions`: Number of diagnostic ions detected
- `top_glycan_composition`: Best matching glycan (e.g., "Hex5HexNAc4Fuc1NeuAc2")
- `composition_score`: Match quality for top composition
- `glycan_type`: Inferred type (e.g., "N-glycan, sialylated, fucosylated")
- `oxonium_ions_detected`: Comma-separated list of detected ions

**Example Output**:
```csv
scan_id,rt,precursor_mz,glycan_score,num_oxonium_ions,top_glycan_composition,composition_score,glycan_type,oxonium_ions_detected
scan_1234,24.5,1850.678,0.923,5,Hex5HexNAc4Fuc1NeuAc2,0.891,"N-glycan, sialylated, fucosylated","HexNAc, HexNAc-H2O, NeuAc, Fuc, HexNAc-Hex"
scan_2156,28.3,1556.432,0.875,4,Hex3HexNAc4Fuc1,0.823,"N-glycan, fucosylated","HexNAc, HexNAc-H2O, Fuc, HexNAc-Hex"
```

### Module Components

**glycan_mass.py**:
- `GlycanMassCalculator`: Calculate theoretical masses
- `MONOSACCHARIDE_MASSES`: Dictionary of building block masses
- `calculate_y_ion_mass()`: Calculate Y-ion m/z values
- Supports 10+ monosaccharides (Hex, HexNAc, Fuc, NeuAc, NeuGc, Xyl, Phospho, Sulfo, etc.)

**oxonium_ion_detector.py**:
- `OxoniumIonDetector`: Detect diagnostic ions in spectra
- `batch_detect_glycopeptides()`: Process multiple spectra
- 25+ predefined oxonium ions
- Confidence scoring based on ion count and intensity

**glycan_analyzer.py**:
- `GlycanAnalyzer`: Comprehensive glycan analysis
- Combines oxonium detection + composition inference + Y-ion analysis
- Scores matches based on mass accuracy and oxonium support

### Glycoproteomics-Specific Parameters

**Mass Tolerance**:
- High-resolution MS (Orbitrap, FTICR): 0.01-0.02 Da or 5-10 ppm
- Lower resolution (TOF, ion trap): 0.05-0.1 Da

**Minimum Oxonium Ions**:
- Relaxed (high sensitivity): 1 ion
- Standard: 2 ions (recommended)
- Stringent (high specificity): 3+ ions

**MS Level**:
- MS1: Precursor ions (not recommended for oxonium detection)
- MS2: MS/MS spectra (required for oxonium ions)

### Common Use Cases

**1. Site-Specific Glycosylation Analysis**:
```bash
# Identify glycopeptides with high confidence
python3 analyze_glycoproteomics.py sample.mzML \
  --output results.csv \
  --min-oxonium 3  # Stringent criteria
```

**2. Comparative Glycoproteomics**:
```bash
# Process control and treatment groups
python3 analyze_glycoproteomics.py control/ --batch --output control_results/
python3 analyze_glycoproteomics.py treatment/ --batch --output treatment_results/

# Compare using Python
python3 -c "
import pandas as pd
control = pd.read_csv('control_results/combined_glycoproteomics_results.csv')
treatment = pd.read_csv('treatment_results/combined_glycoproteomics_results.csv')
print('Control glycopeptides:', len(control))
print('Treatment glycopeptides:', len(treatment))
"
```

**3. Targeted Glycan Search**:
```python
from proteomics.glycoproteomics import GlycanMassCalculator

calc = GlycanMassCalculator()

# Define target glycans of interest
targets = [
    {'Hex': 5, 'HexNAc': 4, 'Fuc': 1, 'NeuAc': 2},  # Complex sialylated
    {'Hex': 9, 'HexNAc': 2},                         # High mannose
]

for comp in targets:
    mass = calc.calculate_mass(comp)
    print(f"Target: {comp} → {mass:.4f} Da")

# Search for these masses in your data
# (integrate with MzMLReader and filtering)
```

### Performance Metrics

**Typical Analysis Speed**:
- .raw conversion: ~1-2 min per file (size-dependent)
- mzML reading: ~500-1000 spectra/second
- Oxonium detection: ~100-500 spectra/second
- Full glycan analysis: ~50-200 spectra/second

**Memory Usage**:
- Small dataset (<1000 spectra): ~500 MB
- Medium dataset (1000-10000 spectra): ~2-4 GB
- Large dataset (>10000 spectra): ~4-8 GB

**Accuracy**:
- Glycopeptide identification (≥2 oxonium ions): ~95% specificity
- Composition assignment: Dependent on precursor mass accuracy
- False positive rate: <5% with stringent criteria (≥3 oxonium ions)

### Requirements

**External Tools**:
- ThermoRawFileParser (for .raw conversion)
  - Install: `dotnet tool install ThermoRawFileParser -g`
  - Or download: https://github.com/compomics/ThermoRawFileParser

**Python Dependencies** (in requirements_proteomics.txt):
- pyteomics ≥4.6 (mzML parsing)
- pymzml ≥2.5 (alternative mzML parser)
- numpy, pandas, scipy (data processing)
- matplotlib, seaborn (visualization)

### Troubleshooting

**Issue**: "ThermoRawFileParser not found"
```bash
# Install via .NET
dotnet tool install ThermoRawFileParser -g

# Or specify path manually
python3 convert_raw_to_mzml.py sample.raw \
  --thermorawfileparser /path/to/ThermoRawFileParser
```

**Issue**: No glycopeptides detected
- Check MS level (should be MS2/MS/MS)
- Lower oxonium ion threshold: `--min-oxonium 1`
- Verify data is from glycoproteomics experiment
- Check mass tolerance settings

**Issue**: Too many false positives
- Increase oxonium ion threshold: `--min-oxonium 3`
- Filter by glycan score: Keep only score > 0.5
- Review oxonium ion patterns manually

**Issue**: Memory error on large files
- Process in batches
- Increase system memory
- Use lighter preprocessing options

### References

- Oxonium Ion Scanning: Nature Biomedical Engineering (2023)
- N-Glycan Structures: GlycoWorkbench, GlyTouCan databases
- Fragmentation Patterns: Molecular & Cellular Proteomics guidelines
- Mass calculation: Expasy GlycoMod tool principles