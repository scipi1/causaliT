# causaliT

> ⚠️ **Work in Progress** - This project is under active development.

Causal Process Transformer for sequence prediction.

## Installation

```bash
# Clone and install in editable mode
git clone https://github.com/scipi1/causaliT.git
cd causaliT
pip install -e .
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

## Quick Start

```bash
# Train a model
python -m causaliT.cli train --exp_id <experiment_folder>
```

## Project Structure

```
causaliT/
├── causaliT/          # Main package
│   ├── config/        # Configuration files example
│   ├── core/          # Model architectures
│   ├── training/      # Training logic & forecasters
│   └── evaluation/    # Prediction & evaluation
├── experiments/       # Experiment configs & outputs
├── notebooks/         # Analysis notebooks
└── tests/             # Unit tests
```

## Data

Download industrial data [here](https://polybox.ethz.ch/index.php/s/aNaZXpKF6YZexjF). To get the password, contact fscipion@ethz.ch.

## License

TBD
