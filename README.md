# Fraud Detection Project

This educational project keeps its original generated banking-transaction demo and adds a separate IEEE-CIS Fraud Detection model. It includes PyTorch MLP/LSTM training, FastAPI inference, a React/Vite frontend, and a Streamlit demo. The generated demo data is programmatically created; it is not a real transaction source. Neither model is represented as a production-validated fraud decision system.

## Models and data

### MLP and LSTM models

The existing MLP and LSTM continue to use the project's generated banking-transaction data and existing architectures. The MLP scores rows independently. The LSTM uses per-customer sequences only where full transaction timestamps are available; its single-transaction API uses a padded sequence and is a prototype input mode.

### IEEE-CIS Fraud Model

The IEEE-CIS adapter joins `train_transaction.csv` to `train_identity.csv` on `TransactionID`, with the transaction table as the base population. A left join preserves transactions without identity records; their `DeviceType` remains missing and is explicitly encoded. Original CSV files are read without being changed.

The model uses these ten user-facing features, in this order:

`TransactionAmt`, `ProductCD`, `hour_of_day`, `card4`, `card6`, `addr1`, `addr2`, `dist1`, `P_emaildomain`, `DeviceType`.

`TransactionDT` is elapsed time, not a wall-clock timestamp. The derived `hour_of_day` is `floor(TransactionDT / 3600) modulo 24`, an elapsed-time cycle bucket. It must not be interpreted as a real local or UTC clock hour. No model-only features are used. `TransactionID`, `C1-C14`, `D1-D15`, `M1-M9`, `V1-V339`, `id_01-id_38`, `dist2`, and `R_emaildomain` are excluded.

Rows are sorted by `TransactionDT` and split chronologically into approximately 70% training, 15% validation, and 15% held-out test partitions. Equal elapsed times stay together. Missing numerical values use training-partition medians; categorical fields use frequency encodings fitted on training rows only. The MLP uses class-weighted loss. Early stopping uses validation data, the probability threshold is selected using validation F1, and the final test partition is evaluated after those choices are complete.

Training creates separate IEEE-CIS artifacts: `fraud_detector_ieee.pt`, `ieee_preprocessor.pkl`, `feature_names_ieee.json`, `ieee_feature_metadata.json`, `eval_metrics_ieee.json`, and `training_history_ieee.json`. Existing MLP and LSTM artifacts are not overwritten. Metrics report validation and held-out test results separately.

Current IEEE-CIS run (10 selected user-facing features; threshold selected by validation F1):

| Split | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.2355 | 0.2827 | 0.2570 | 0.9675 | 0.7539 | 0.1614 |
| Held-out test | 0.2077 | 0.2636 | 0.2323 | 0.9637 | 0.7531 | 0.1537 |

The validation-selected threshold is `0.7428748608`. The held-out test confusion matrix is TN=81,933, FP=3,083, FN=2,257, TP=808. The split contains 88,081 test transactions, including 3,065 fraud cases. These are baseline results for the specified feature set and chronological split; they are not a guarantee of operational performance.

The supplied IEEE-CIS CSV files are local inputs and are not included in the repository. Pass their paths at training time:

```powershell
.\.venv\Scripts\python.exe -m src.train_ieee_cis `
  --transaction-path "C:\path\outside\repository\train_transaction.csv" `
  --identity-path "C:\path\outside\repository\train_identity.csv"
```

You can also set `IEEE_CIS_TRANSACTION_CSV` and `IEEE_CIS_IDENTITY_CSV`. Training does not accept filesystem paths through the API. Do not place the source CSVs in Git.

## Environment setup

The project is configured for Python 3.12. Create and use its virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Python requirements are not a full lock file. scikit-learn is constrained to the 1.5.x series because the existing scaler pickles were serialized with 1.5.2. Frontend dependencies are lockfile-managed in `frontend/package-lock.json`.

## Start the applications

Run the project launcher:

```powershell
.\run_project.ps1
```

It starts FastAPI at `http://127.0.0.1:8000`, React/Vite at `http://localhost:5173`, opens the React app, and can start the optional headless Streamlit demo at `http://localhost:8501`. The launcher does not install packages or start training.

To start services manually, run Uvicorn from the repository root and Vite from `frontend`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
cd frontend
npm.cmd run dev
```

## API

| Method and path | Purpose |
|---|---|
| `GET /health` | Loaded MLP, LSTM, and IEEE-CIS model status |
| `GET /model-info`, `GET /model-info-lstm`, `GET /model-info-ieee` | Model details and IEEE-CIS benchmark metadata |
| `POST /predict`, `POST /batch-predict` | Existing MLP predictions |
| `POST /predict-lstm` | Existing LSTM prediction |
| `POST /predict-ieee`, `POST /batch-predict-ieee` | IEEE-CIS predictions using the ten listed fields |
| `GET /metrics`, `GET /metrics-lstm` | Existing model metrics |
| `GET /training-history`, `GET /training-history-lstm` | Existing MLP/LSTM training histories |
| `POST /train`, `POST /train-lstm` | Existing MLP/LSTM training controls |

IEEE-CIS responses include fraud probability, predicted class, decision, risk level, threshold, and model name. Numeric values must be finite, categorical values must be in the saved training vocabulary (or explicitly missing), and batches are limited to 1,000 transactions. Validation errors do not echo submitted values. API training endpoints are for local development; the API has no authentication layer.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm.cmd run lint
npm.cmd run build
```

The frontend currently has lint and build scripts but no separate frontend test script.

## Repository hygiene

`.gitignore` excludes virtual environments, caches, local environment files, Node modules, datasets, archives, and model checkpoint/preprocessing files. Keep datasets and trained model binaries outside Git. Review `git status` and the complete diff before sharing changes.
