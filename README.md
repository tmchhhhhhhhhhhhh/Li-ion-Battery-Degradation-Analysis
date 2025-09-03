# Li‑ion Battery Degradation Analysis

**Short description:** A reproducible project for analyzing lithium‑ion battery degradation: data preprocessing, exploratory data analysis (EDA), feature engineering, model training for predicting capacity fade, and tracking degradation metrics over cycles.

---

## Table of contents

* [Overview](#overview)
* [Repository structure](#repository-structure)
* [Requirements & installation](#requirements--installation)
* [Data](#data)
* [Usage](#usage)
* [Modeling approaches](#modeling-approaches)
* [Reproducibility & results](#reproducibility--results)
* [Visualizations & notebooks](#visualizations--notebooks)
* [Contributing and roadmap](#contributing-and-roadmap)
* [License & contact](#license--contact)

---

## Overview

This repository contains code and experiments for analyzing degradation of lithium‑ion cells. The goal is to provide a full pipeline from raw measurements to predictive models for Remaining Useful Life (RUL) or capacity drop across cycles.

Main goals:

* build a reproducible data pipeline for battery datasets;
* perform EDA to discover degradation patterns;
* train and compare baseline and advanced models (GBDT, linear models, neural nets) for capacity prediction or RUL estimation;
* visualize results, track metrics, and provide guidance for further improvements.

---

## Repository structure

```
Li-ion-Battery-Degradation-Analysis/
├─ data/                  # raw and processed data (data/raw, data/processed)
├─ notebooks/             # Jupyter notebooks: EDA, feature analysis, modelling
├─ src/                   # scripts: preprocess, feature engineering, train, evaluate
├─ results/               # saved models, metrics, figures
├─ configs/               # experiment configs (yaml/json)
├─ requirements.txt       # project dependencies
├─ README.md              # (this file)
└─ LICENSE
```

> Note: adjust paths in commands below if your repository uses different names.

---

## Requirements & installation

Recommended: Python 3.8+.

### Install using pip

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Example `requirements.txt`

```
numpy
pandas
scipy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
jupyterlab
notebook
tqdm
pyyaml
joblib
torch    # if using PyTorch for NN models
tensorflow  # optional, if using TensorFlow/Keras
```

---

## Data

Place raw data files into `data/raw/`.

Typical columns expected (adapt to your dataset):

* `cycle` — cycle number
* `voltage` — voltage readings (time series or summary stats)
* `current` — current
* `temperature` — temperature
* `capacity` — measured capacity per cycle

If you use an external public dataset (e.g., NASA, Oxford, other battery datasets), keep original files in `data/raw/` and run the preprocessing script to create unified `data/processed/` tables.

---

## Usage

### 1) Preprocess raw data

```bash
python src/preprocess.py --input_dir data/raw --output_dir data/processed
```

This script should:

* parse and unify raw files;
* clean and normalize signals;
* export a per‑cycle table with aggregated features and the target variable (capacity or RUL).

### 2) Feature engineering

```bash
python src/feature_engineering.py --input_dir data/processed --output_dir data/features
```

Suggested features: summary statistics (mean, std, min, max) of voltage/current/temperature during charge/discharge; delta‑V/delta‑Q; cycle‑wise slopes; rolling statistics; capacity fade rate; engineered degradation indicators.

### 3) Train a model

```bash
python src/train.py --config configs/experiment.yaml
```

Config options should include:

* model type (xgboost / lightgbm / sklearn / nn)
* hyperparameters
* cross‑validation settings
* target metric (MAE / RMSE / R²)

### 4) Evaluate / Inference

```bash
python src/evaluate.py --model results/model.joblib --test data/features/test.csv
```

Outputs: predictions, evaluation metrics (MAE, RMSE, R²), and diagnostic plots.

---

## Modeling approaches

Recommended baselines and advanced models:

* Gradient boosting trees (LightGBM, XGBoost) — strong, fast baselines for tabular features.
* Random Forests and other ensemble methods — robust baselines.
* Linear models (Ridge, ElasticNet) — interpretable baselines.
* Neural networks (LSTM, 1D‑CNN, Temporal CNN) — for raw time series or sequence modeling when per‑cycle sequences are used.

Metrics:

* MAE (mean absolute error), RMSE, R² for capacity prediction;
* For RUL: absolute error in cycles to EoL, and aggregated error statistics across cells.

---

## Reproducibility & results

* Save experiment metadata (config, seed, git commit hash) alongside results in `results/`.
* Use fixed random seeds for numpy, torch/tensorflow, and any other frameworks.

Example run with logging:

```bash
python src/train.py --config configs/xgb_default.yaml --seed 42 --output_dir results/exp_xgb_default
```

Store: trained model binary, config file, evaluation metrics (JSON/CSV), and example plots.

---

## Visualizations & notebooks

Recommended notebooks:

* `01_EDA.ipynb` — dataset overview, sample charge/discharge curves, correlation heatmaps;
* `02_feature_analysis.ipynb` — feature importance, SHAP or permutation importance;
* `03_modelling.ipynb` — training experiments, quick comparisons and plots.

Start Jupyter Lab / Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

---

## Contributing and roadmap

Ideas for future improvements:

* advanced feature engineering (segment‑based features, spectral features from voltage/current signals);
* robust capacity estimation (outlier detection, measurement uncertainty handling);
* end‑to‑end deep learning on raw time‑series signals;
* CI for experiments (GitHub Actions) and automatic report generation.

Contributions welcome — open an issue or submit a pull request with proposed changes and tests.

---

## License & contact

License: choose your preferred license (MIT / Apache‑2.0 / GPL‑3.0).

Contact: add an email address or GitHub profile link for questions and collaboration.

---

If you want, I can also:

* add an English + Russian bilingual README;
* generate default `configs/` YAML templates and example scripts;
* add CI badges and a minimal GitHub Actions workflow for running tests.
