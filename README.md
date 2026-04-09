# ECE1508 Project — Stock forecasting (LSTM baseline)

PyTorch pipeline for downloading OHLCV data with [yfinance](https://github.com/ranaroussi/yfinance), building sliding windows, and training models with train/validation/test splits and evaluation.

**Python:** 3.12 or newer (see `requires-python` in `pyproject.toml`).

---

## Setup (choose one)

### Option A — `uv` (recommended, matches `pyproject.toml` + `uv.lock`)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the project root:

```bash
uv sync
```

This creates `.venv/`, resolves dependencies from the lockfile, installs the **`ece1508` package in editable mode**, and installs **PyTorch with CUDA 12.8** from the index declared in `pyproject.toml` (`pytorch-cu128`).

Run the **demo** (LSTM + Transformer regressors on the same data):

```bash
uv run python demo.py
```

Run the **full LSTM baseline** (longer intraday run, plots):

```bash
uv run python src/ece1508/lstm/main.py
```

### Option B — `pip` + virtual environment

Create and activate a virtual environment, then install the full dependency tree from the exported lock **and the project itself** (so `import ece1508` works):

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

`requirements.txt` is generated from `uv.lock` (`uv export --no-dev --no-hashes`). It pins `torch` with the `+cu128` build; the extra index is required for that wheel.

**CPU-only PyTorch:** If you do not need CUDA, install a CPU build of PyTorch from the [official instructions](https://pytorch.org/get-started/locally/) and install the rest of the stack from `requirements.txt` after removing or replacing the `torch==...` line, or install dependencies from `pyproject.toml` without the `tool.uv.sources` mapping (for example by installing `torch` separately, then `pip install pandas scikit-learn ...`).

### Option C — Conda base + pip

```bash
conda env create -f environment.yaml
conda activate ece1508-project
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

---

## Demo Script

`demo.py` prints:

1. A few rows of price and detrended features.
2. One training batch tensor shapes `(batch, lookback, features)` and `(batch, 1)`.
3. Training losses for the last epoch and **test metrics** (MSE, MAE, RMSE, directional accuracy) for **both** the LSTM and the `TimeSeriesTransformer` regressors.

Example (values depend on market data at run time):

```text
=== Sample model input / target tensors (one batch) ===
X batch shape: (16, 16, 5)  (batch, lookback, features)
y batch shape: (16, 1)  (batch, 1)
...
LSTM — test (price space) Metrics
...
Transformer — test (price space) Metrics
...
```

---

## Project layout

```text
ECE1508_Project/
├── demo.py                      # LSTM + Transformer quick demo (repo root)
├── pyproject.toml
├── uv.lock / requirements.txt
├── environment.yaml
├── src/ece1508/
│   ├── lstm/
│   │   ├── data_preparation.py
│   │   ├── train.py / evaluate.py
│   │   ├── lstm_forecaster.py   # LSTMForecaster
│   │   ├── baseline.py          # yfinance + detrend helpers
│   │   └── main.py              # full LSTM baseline CLI
│   ├── transformer/
│   │   ├── transformer_model.py # TimeSeriesTransformer
│   │   ├── percentReturn.ipynb
│   │   └── Transformer_Stock_prediction.ipynb
│   └── models/                  # Shared model zoo (compare_all_models)
├── notebooks/
│   ├── compare_all_models.ipynb # primary results (see table above)
│   └── compare_lstm_transformer.ipynb
└── notes/                       # e.g. Optuna log snippets
```

