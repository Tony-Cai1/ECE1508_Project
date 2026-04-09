# ECE1508 Project — Stock forecasting (LSTM baseline)

PyTorch pipeline for downloading OHLCV data with [yfinance](https://github.com/ranaroussi/yfinance), building sliding windows, and training an LSTM forecaster with train/validation/test splits and evaluation plots/metrics.

**Python:** 3.12 or newer (see `requires-python` in `pyproject.toml`).

---

## Setup (choose one)

### Option A — `uv` (recommended, matches `pyproject.toml` + `uv.lock`)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the project root:

```bash
uv sync
```

This creates `.venv/`, resolves dependencies from the lockfile, and installs **PyTorch with CUDA 12.8** from the index declared in `pyproject.toml` (`pytorch-cu128`).

Run code with:

```bash
uv run python demo.py
uv run python main.py
```

### Option B — `pip` + virtual environment

Create and activate a virtual environment, then install the full dependency tree from the exported lock:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
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
```

---

## Usage

| What | Command / location |
|------|-------------------|
| **Quick demo** (short download, 3 epochs, prints sample I/O and metrics) | `uv run python demo.py` or `python demo.py` inside the activated env |
| **Full LSTM baseline** (longer intraday run, training curves, saved plots) | `uv run python main.py` |
| **Notebooks** | `compare_all_models.ipynb`, `compare_lstm_transformer.ipynb`, `percentReturn.ipynb`, `Transformer_Stock_prediction.ipynb` — open in Jupyter/VS Code; use the same interpreter as above (`uv run python -m ipykernel install --user` if needed) |

`main.py` writes checkpoints and PNG plots (ignored by `.gitignore`). `demo.py` saves `demo_lstm_model.pt` and is meant for a fast sanity check.

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
uv export --no-dev --no-hashes --no-annotate -o requirements.txt
```

---

## Demo sample input / output

`demo.py` prints:

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

## Project layout (core)

- `data_preparation.py` — scaling, chronological splits, `DataLoader` construction.
- `model.py` — LSTM forecaster.
- `train.py` / `evaluate.py` — training loop, metrics, plotting helpers.
- `models/` — additional architectures (e.g. transformer, LSTM module variants) used from notebooks or extensions.
