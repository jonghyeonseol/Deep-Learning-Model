# Proteomics Deep Learning Module

Deep learning framework for analyzing mass spectrometry proteomics data.

## Features

- **Multi-format support**: .raw files (Thermo), mzML, mzXML, CSV
- **Multiple architectures**: CNN, Transformer, RT predictors, multi-task models
- **Flexible tasks**: Classification, regression, similarity learning
- **Production-ready**: Automatic mixed precision, caching, early stopping

## Quick Start

### Installation

```bash
# Activate virtual environment
source ../venv/bin/activate

# Install dependencies
pip install -r ../requirements_proteomics.txt

# Optional: Install ThermoRawFileParser for .raw file support
dotnet tool install ThermoRawFileParser -g
```

### Basic Usage

```bash
# Train CNN on mzML data
python3 main_proteomics.py \
  --data ../data/proteomics/mzml/sample.mzML \
  --model cnn \
  --task classification \
  --num-classes 10 \
  --epochs 50

# Retention time prediction
python3 main_proteomics.py \
  --data ../data/proteomics/mzml/peptides.mzML \
  --model rt_predictor \
  --task regression \
  --epochs 100
```

## Module Structure

```
proteomics/
├── data_loaders/          # Data loading and preprocessing
│   ├── raw_converter.py   # .raw → mzML conversion
│   ├── mzml_reader.py     # mzML/mzXML parsing
│   ├── csv_reader.py      # CSV data loading
│   ├── preprocessing.py   # Spectrum preprocessing
│   └── spectrum_dataset.py # PyTorch Dataset
├── models/                # Neural network architectures
│   ├── spectrum_cnn.py    # 1D CNN models
│   ├── spectrum_transformer.py  # Transformer models
│   └── retention_predictor.py   # RT prediction models
├── training/              # Training infrastructure
│   ├── trainer.py         # Training loop
│   └── loss_functions.py  # Custom loss functions
├── utils/                 # Utilities
│   ├── visualization.py   # Plotting functions
│   └── metrics.py         # Evaluation metrics
├── config/                # Configuration files
│   └── default_config.yaml
└── main_proteomics.py     # CLI entry point
```

## Supported Models

| Model | Description | Parameters | Use Case |
|-------|-------------|------------|----------|
| `cnn` | 1D CNN with residual connections | ~1-5M | General classification |
| `lightweight_cnn` | Smaller CNN | ~500K | Fast experiments |
| `transformer` | Transformer with positional encoding | ~2-10M | Complex patterns |
| `rt_predictor` | RT prediction CNN | ~1-3M | Retention time regression |

## Data Formats

### 1. mzML/mzXML (Recommended)
```python
from data_loaders import MzMLReader

reader = MzMLReader('data.mzML')
spectra = reader.read_spectra(ms_level=2, min_peaks=10)
```

### 2. Thermo .raw Files
```python
from data_loaders import RawConverter

converter = RawConverter()
mzml_file = converter.convert('data.raw')
```

### 3. CSV Files
Three supported schemas:
- **Peak-level**: `scan_id, rt, mz, intensity`
- **Spectrum-level**: `scan_id, rt, num_peaks, precursor_mz, ...`
- **Matrix**: Rows = samples, columns = m/z bins

## Custom Training

```python
from data_loaders import SpectrumDataset, create_dataloaders
from models import SpectrumCNN
from training import ProteomicsTrainer
import torch.nn as nn
import torch.optim as optim

# Create dataset
dataset = SpectrumDataset(
    data_path='data.mzML',
    task='classification',
    labels=[0, 1, 0, 1, ...],  # Your labels
    mz_range=(50, 2000),
    bin_size=0.1,
    normalization='tic'
)

# Create dataloaders
dataloaders = create_dataloaders(
    train_path='train.mzML',
    val_path='val.mzML',
    batch_size=32
)

# Create model
model = SpectrumCNN(
    input_dim=19500,
    num_classes=10,
    task='classification'
)

# Train
trainer = ProteomicsTrainer(
    model=model,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001)
)

trainer.train(num_epochs=50)
```

## Configuration

Edit `config/default_config.yaml` or create your own:

```yaml
data:
  mz_range: [50.0, 2000.0]
  bin_size: 0.1
  normalization: "tic"

model:
  architecture: "cnn"
  task: "classification"

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.001
  use_amp: true
```

## Performance Tips

1. **Use GPU**: Automatic if available (CUDA/MPS)
2. **Enable AMP**: `use_amp: true` for 2x speedup
3. **Increase batch size**: `--batch-size 64` or higher
4. **Cache preprocessing**: Automatically saved to `data/proteomics/processed/`
5. **Start with lightweight model**: Use `lightweight_cnn` for quick tests

## Examples

### Example 1: Binary Classification
```bash
python3 main_proteomics.py \
  --data peptides.mzML \
  --model cnn \
  --task classification \
  --num-classes 2 \
  --epochs 30 \
  --experiment-name binary_classifier
```

### Example 2: Multi-class with Transformer
```bash
python3 main_proteomics.py \
  --data compounds.mzML \
  --model transformer \
  --task classification \
  --num-classes 20 \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.0001
```

### Example 3: Retention Time Prediction
```bash
python3 main_proteomics.py \
  --data peptides_with_rt.mzML \
  --model rt_predictor \
  --task regression \
  --epochs 80 \
  --patience 15
```

## Troubleshooting

### ThermoRawFileParser not found
```bash
# Install via .NET
dotnet tool install ThermoRawFileParser -g

# Or specify path manually
python3 -c "from data_loaders import RawConverter; \
  RawConverter(thermorawfileparser_path='/path/to/ThermoRawFileParser')"
```

### Out of memory
- Reduce batch size: `--batch-size 16`
- Use lighter model: `--model lightweight_cnn`
- Reduce m/z range or increase bin size in config

### Slow preprocessing
- Preprocessed data is automatically cached
- Delete cache to reprocess: `rm -rf data/proteomics/processed/*.pkl`

## Citation

If you use this module, please cite:

```
Deep Learning Model for Proteomics Analysis
https://github.com/your-repo/Automated-LC-MS-MS-analaysis_ver2
```

## License

[Your License Here]
