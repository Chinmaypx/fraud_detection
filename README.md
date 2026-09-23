# Fraud Detection Project

This repository contains two separate binary fraud-detection benchmarks: a generated synthetic banking-transaction pipeline and an MLP evaluated on the ULB/Worldline credit-card dataset. It includes PyTorch training and inference, a FastAPI service, a React/Vite interface, and a Streamlit synthetic-model demo. These are educational benchmark systems, not validated production decision systems.

## Data and models

### Synthetic transaction dataset

`generate_dataset.py` creates 100,000 synthetic transactions with a configured 2% fraud rate. The synthetic schema includes transaction amount and time, customer/account activity, location, merchant category, device ID, and a binary `is_fraud` label. `src/data_pipeline.py` can load `data/transactions.csv`; if that file is absent it can generate the synthetic data in memory. Feature engineering and categorical encoding are specific to this schema.

The synthetic MLP (`FraudDetectorNet`) scores each transaction independently using the engineered and encoded features. The optional synthetic LSTM (`FraudLSTMNet`) uses ordered per-customer sequences when full transaction timestamps are available. It does not apply to ULB data. ULB has no customer or account identifiers, so no LSTM histories are inferred from its `Time` column. Single-transaction LSTM inference uses a padded history and should be treated as a prototype rather than a substitute for a real customer history.

### ULB/Worldline credit-card benchmark

The separate ULB pipeline expects `Time`, `V1` through `V28`, `Amount`, and `Class`. `V1`-`V28` are anonymized PCA features; this project does not assign them semantic meanings. The ULB model is an MLP using 30 features (`Time`, `V1`-`V28`, `Amount`). It is binary fraud classification, not multi-type fraud detection.

The source CSV is not included. Keep it outside the repository and pass its local path explicitly when training. The pipeline sorts/splits chronologically by distinct `Time`, fits its RobustScaler on training rows only, selects the probability threshold on validation data, and evaluates the held-out test partition only after model selection. Exact duplicate rows are handled in memory; the source CSV is not rewritten. The API does not accept dataset paths.

ULB benchmark results for the current saved model:

| Split | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | 0.8750 | 0.7119 | 0.7850 | 0.9334 | 0.6933 |
| Held-out test | 0.9815 | 0.6974 | 0.8154 | 0.9694 | 0.8073 |

The held-out test confusion matrix is TN=59,811, FP=1, FN=23, TP=53. The saved ULB decision threshold is `0.9771430492401123`; inference flags fraud when probability is greater than or equal to that threshold. These values describe the ULB benchmark only and must not be combined with synthetic metrics.

Current synthetic MLP metrics from `models/eval_metrics.json`:

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic held-out test | 0.9915 | 0.7027 | 0.9925 | 0.8228 | 0.9998 | 0.9949 |

These are results from a different dataset and split than the ULB metrics; they are separate benchmark records, not a direct model ranking.

### Model and preprocessing artifacts

Model checkpoints and Python scaler pickles are local generated artifacts and are excluded by `.gitignore`. They must be present locally for the corresponding trained model to load. Metrics, feature-name metadata, and training history are JSON artifacts.

| Model | Checkpoint | Preprocessing artifacts |
|---|---|---|
| Synthetic MLP | `models/fraud_detector.pt` | `scaler_mlp.pkl`, `feature_names.json` |
| Synthetic LSTM (optional) | `models/fraud_detector_lstm.pt` | `scaler_lstm.pkl`, `feature_names_lstm.json` |
| ULB MLP | `models/fraud_detector_ulb.pt` | `scaler_ulb.pkl`, `feature_names_ulb.json` |

`models/scaler.pkl` is a legacy synthetic scaler name and remains available for backward-compatible inference when a model-specific scaler is absent. New MLP, LSTM, and ULB training writes model-specific scaler files; ULB preprocessing is loaded only from its ULB-specific artifacts. The current workspace has no LSTM checkpoint, so LSTM inference is unavailable until that model is trained.

## Environment setup

The project was verified with Python 3.12.6. Create and use a project virtual environment from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python requirements are not a full lock file. Most use minimum versions; scikit-learn is constrained to 1.5.x because the saved scaler artifacts were serialized with 1.5.2. Frontend dependencies are lockfile-managed with `frontend/package-lock.json`.

## Training

Generate and save the synthetic CSV if desired:

```powershell
.\.venv\Scripts\python.exe generate_dataset.py
```

Train the synthetic MLP:

```powershell
.\.venv\Scripts\python.exe -m src.train_model
```

The optional LSTM can be trained from the API's `/train-lstm` endpoint when synthetic rows have full timestamps. It is not trained on the ULB benchmark.

Train/evaluate the ULB MLP by supplying the external CSV path explicitly:

```powershell
.\.venv\Scripts\python.exe -m src.train_ulb --data-path "C:\path\outside\repository\creditcard.csv"
```

`ULB_CSV_PATH` may be used instead of `--data-path`. `--model-dir` can select a local artifact output directory. Training creates or replaces the artifacts for that model; it is not needed to run the API when the checkpoints and preprocessing files already exist.

## Run the applications

Start FastAPI from the repository root (artifact paths are resolved from the project location):

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Start the React/Vite development server in another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

The UI is at `http://localhost:5173`. It offers separate synthetic and ULB prediction modes. The synthetic MLP/LSTM forms and training controls remain synthetic-only. The Streamlit demo is also synthetic-only:

```powershell
cd ..
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

`run_project.ps1` uses the existing `.venv` and installed frontend dependencies. It does not install packages or silently generate data/train models; unavailable optional services are skipped with a warning.

## API endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Loaded model status |
| `GET /model-info`, `GET /model-info-lstm`, `GET /model-info-ulb` | Model details |
| `POST /predict` | Synthetic MLP single prediction |
| `POST /batch-predict` | Synthetic MLP batch (up to 1,000 transactions) |
| `POST /predict-lstm` | Optional synthetic LSTM single prediction |
| `POST /predict-ulb` | ULB MLP single prediction with saved threshold |
| `POST /batch-predict-ulb` | ULB MLP batch (up to 1,000 transactions) |
| `GET /metrics`, `GET /metrics-lstm` | Synthetic evaluation metrics |
| `GET /training-history`, `GET /training-history-lstm` | Synthetic training histories |
| `POST /train`, `POST /train-lstm` | Start synthetic model training |

The ULB request requires exactly the 30 numeric feature values; its response includes fraud probability, predicted class, risk, threshold, and model name. Request validation rejects non-finite values and extra ULB fields; sanitized 422 responses do not echo request values. Prediction batches are limited to 1,000 rows. API training endpoints are intended for local development; this app has no authentication or authorization layer.

## Tests and quality checks

Run the backend suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The frontend currently defines lint and production-build scripts, but no frontend test script:

```powershell
cd frontend
npm.cmd run lint
npm.cmd run build
```

## Repository hygiene

`.gitignore` excludes virtual environments, caches, local environment files, Node modules, generated datasets, archives, and model checkpoint/scaler files. The external ULB CSV and model binaries should remain local. Do not commit API keys or secrets. Before sharing the repository, review `git status` and the complete diff.
