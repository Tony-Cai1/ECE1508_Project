# ECE1508 Project — Stock forecasting (LSTM baseline)

**Python:** 3.12 or newer (see `requires-python` in `pyproject.toml`).

---

## Setup (choose one)

### Option A — `uv` (recommended, matches `pyproject.toml` + `uv.lock`)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the project root:

```bash
uv sync
```

This creates `.venv/`, resolves dependencies from the lockfile, installs the **`ece1508` package in editable mode**, and installs **PyTorch with CUDA 12.8** from the index declared in `pyproject.toml` (`pytorch-cu128`).

Run code with:

```bash
uv run python scripts/demo.py
uv run python scripts/main.py
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

Create a Conda environment with Python 3.12, then use `pip` for the locked ML stack (PyTorch is not duplicated in `environment.yaml` so it stays consistent with `requirements.txt`):

```bash
conda env create -f environment.yaml
conda activate ece1508-project
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

---

## Usage

| What | Command / location |
|------|-------------------|
| **Quick demo** (short download, 3 epochs, prints sample I/O and metrics) | `uv run python scripts/demo.py` or `python scripts/demo.py` inside the activated env |
| **Notebooks** | `notebooks/` — `compare_all_models.ipynb`, `compare_lstm_transformer.ipynb`, `percentReturn.ipynb`, `Transformer_Stock_prediction.ipynb`. Select the same interpreter where `ece1508` is installed (`uv run python -m ipykernel install --user` if needed). |

---

## Reproducible dependencies

| File | Role |
|------|------|
| `pyproject.toml` | Project metadata and direct dependencies; `tool.uv` pins PyTorch to the CUDA 12.8 wheel index. |
| `uv.lock` | Full locked resolution for `uv sync`. |
| `requirements.txt` | Pip-compatible export of the lockfile (regenerate after lock changes; see below). |
| `environment.yaml` | Conda: Python 3.12 only; install packages with `pip` as in Option C. |

Regenerate `requirements.txt` after you change dependencies and update the lock:

```bash
uv lock
uv export --no-dev --no-hashes --no-annotate --no-emit-project -o requirements.txt
```

---

## Demo sample input / output

`scripts/demo.py` prints:

1. A few rows of price and detrended features.
2. One training batch tensor shapes `(batch, lookback, features)` and `(batch, 1)`.
3. Training loss for the last epoch and test metrics (MSE, MAE, RMSE, directional accuracy).

Example (values depend on market data at run time):

```text
=== Sample model input / target tensors (one batch) ===
X batch shape: (16, 16, 5)  (batch, lookback, features)
y batch shape: (16, 1)  (batch, 1)
...
=== Sample output: test metrics (price space) ===
MSE : ...
MAE : ...
RMSE: ...
Direction Accuracy: ...
```

---

## Project layout

```text
ECE1508_Project/
├── pyproject.toml          # dependencies + hatch build (installable package)
├── uv.lock / requirements.txt
├── environment.yaml
├── src/ece1508/            # importable package
│   ├── data_preparation.py # splits, scaling, DataLoaders
│   ├── train.py / evaluate.py
│   ├── lstm_forecaster.py  # LSTM baseline (`LSTMForecaster`)
│   ├── transformer_model.py # `TimeSeriesTransformer` (notebooks)
│   ├── baseline.py         # yfinance download + detrend helpers for scripts
│   └── models/             # LSTM / Transformer heads (regression, classifier, multitask)
├── scripts/
│   ├── main.py             # full LSTM baseline CLI
│   └── demo.py             # short demo run
├── notebooks/              # experiments and comparisons
└── notes/                  # ad-hoc logs (e.g. Optuna summaries)
```
