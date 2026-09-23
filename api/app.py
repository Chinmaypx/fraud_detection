"""
FastAPI Application for Fraud Detection (PyTorch)
REST API with CORS for React frontend integration
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
import json
import pickle
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(PROJECT_ROOT))
logger = logging.getLogger(__name__)

from src.predict import FraudDetector, LSTMFraudDetector, ULBFraudDetector

app = FastAPI(
    title="Fraud Detection API (PyTorch)",
    description="Deep Learning API for detecting fraudulent banking transactions",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    """Keep 422 details actionable without echoing submitted values or huge bodies."""
    errors = [
        {key: error[key] for key in ('loc', 'msg', 'type') if key in error}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


# --- Pydantic Models ---

class Transaction(BaseModel):
    transaction_amount: float = Field(..., gt=0, allow_inf_nan=False, description="Transaction amount in INR")
    transaction_time: int = Field(..., ge=0, le=86399, description="Seconds since midnight")
    location: str = Field(..., max_length=100, description="City code")
    device_id: str = Field(..., max_length=256, description="Device identifier")
    merchant_category: str = Field(..., max_length=100, description="Category: retail, grocery, restaurant, gas, online")
    account_age_days: int = Field(..., ge=0, description="Age of account in days")
    transaction_count_24h: int = Field(..., ge=0, description="Transactions in last 24 hours")
    avg_transaction_amount: float = Field(..., gt=0, allow_inf_nan=False, description="Average transaction amount")


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    message: str


class BatchPredictionRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=1000)
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, allow_inf_nan=False)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    flagged_count: int


class ULBTransaction(BaseModel):
    """One ULB transaction: Time, anonymized PCA features, and Amount."""
    model_config = ConfigDict(extra='forbid')
    Time: float = Field(..., ge=0, allow_inf_nan=False)
    V1: float = Field(..., allow_inf_nan=False)
    V2: float = Field(..., allow_inf_nan=False)
    V3: float = Field(..., allow_inf_nan=False)
    V4: float = Field(..., allow_inf_nan=False)
    V5: float = Field(..., allow_inf_nan=False)
    V6: float = Field(..., allow_inf_nan=False)
    V7: float = Field(..., allow_inf_nan=False)
    V8: float = Field(..., allow_inf_nan=False)
    V9: float = Field(..., allow_inf_nan=False)
    V10: float = Field(..., allow_inf_nan=False)
    V11: float = Field(..., allow_inf_nan=False)
    V12: float = Field(..., allow_inf_nan=False)
    V13: float = Field(..., allow_inf_nan=False)
    V14: float = Field(..., allow_inf_nan=False)
    V15: float = Field(..., allow_inf_nan=False)
    V16: float = Field(..., allow_inf_nan=False)
    V17: float = Field(..., allow_inf_nan=False)
    V18: float = Field(..., allow_inf_nan=False)
    V19: float = Field(..., allow_inf_nan=False)
    V20: float = Field(..., allow_inf_nan=False)
    V21: float = Field(..., allow_inf_nan=False)
    V22: float = Field(..., allow_inf_nan=False)
    V23: float = Field(..., allow_inf_nan=False)
    V24: float = Field(..., allow_inf_nan=False)
    V25: float = Field(..., allow_inf_nan=False)
    V26: float = Field(..., allow_inf_nan=False)
    V27: float = Field(..., allow_inf_nan=False)
    V28: float = Field(..., allow_inf_nan=False)
    Amount: float = Field(..., ge=0, allow_inf_nan=False)


class ULBPredictionResponse(BaseModel):
    model_name: str
    is_fraud: bool
    predicted_class: int
    fraud_probability: float
    threshold: float
    risk_level: str
    message: str


class ULBBatchPredictionRequest(BaseModel):
    transactions: List[ULBTransaction] = Field(..., min_length=1, max_length=1000)


class ULBBatchPredictionResponse(BaseModel):
    predictions: List[ULBPredictionResponse]
    total_transactions: int
    flagged_count: int


# --- Global state ---
detector = None
lstm_detector = None
ulb_detector = None
training_history = None
training_history_lstm = None
eval_metrics = None
eval_metrics_lstm = None


def _validate_loaded_preprocessing(detector, model_input_dim):
    """Refuse to serve a synthetic checkpoint with missing/mismatched preprocessing."""
    if detector.scaler is None:
        raise RuntimeError("Model-specific preprocessing scaler is unavailable.")
    if not detector.feature_names:
        raise RuntimeError("Model feature metadata is unavailable.")
    scaler_features = getattr(detector.scaler, 'n_features_in_', None)
    feature_count = len(detector.feature_names)
    if feature_count != model_input_dim or (
        scaler_features is not None and scaler_features != model_input_dim
    ):
        raise RuntimeError("Model, scaler, and feature metadata dimensions do not match.")


@app.on_event("startup")
async def load_model():
    """Load PyTorch models (MLP + LSTM) and scaler on startup"""
    global detector, lstm_detector, ulb_detector, training_history, training_history_lstm, eval_metrics, eval_metrics_lstm
    
    # --- MLP Model ---
    detector = FraudDetector(model_path=MODELS_DIR)
    
    model_path = MODELS_DIR / "fraud_detector.pt"
    if model_path.exists():
        try:
            detector.load_model()
            detector.load_scaler()
            detector.load_feature_names()
            _validate_loaded_preprocessing(
                detector, detector.model.network[0].in_features
            )
            print("PyTorch MLP model loaded successfully")
        except Exception as e:
            logger.exception("Unable to load synthetic MLP artifacts")
            detector.model = None
    else:
        print("Warning: No trained MLP model found. Train first via /train endpoint or CLI.")
    
    # --- LSTM Model ---
    lstm_detector = LSTMFraudDetector(model_path=MODELS_DIR)
    
    lstm_model_path = MODELS_DIR / "fraud_detector_lstm.pt"
    if lstm_model_path.exists():
        try:
            lstm_detector.load_model()
            lstm_detector.load_scaler()
            lstm_detector.load_feature_names()
            _validate_loaded_preprocessing(
                lstm_detector, lstm_detector.model.lstm.input_size
            )
            print("PyTorch LSTM model loaded successfully")
        except Exception as e:
            logger.exception("Unable to load synthetic LSTM artifacts")
            lstm_detector.model = None
    else:
        print("Warning: No trained LSTM model found. Train via /train-lstm endpoint.")

    # ULB model uses its own checkpoint, adapter, and scaler artifacts.
    ulb_detector = ULBFraudDetector(model_path=MODELS_DIR)
    ulb_model_path = MODELS_DIR / "fraud_detector_ulb.pt"
    if ulb_model_path.exists():
        try:
            ulb_detector.load_model()
            ulb_detector.load_preprocessing()
            print("ULB MLP model loaded successfully")
        except Exception as e:
            logger.exception("Unable to load ULB model artifacts")
            ulb_detector.model = None
    else:
        print("Warning: No trained ULB model found.")
    
    # Load MLP training history
    history_path = MODELS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    
    # Load LSTM training history
    lstm_history_path = MODELS_DIR / "training_history_lstm.json"
    if lstm_history_path.exists():
        with open(lstm_history_path, 'r') as f:
            training_history_lstm = json.load(f)
    
    # Load MLP eval metrics
    metrics_path = MODELS_DIR / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            eval_metrics = json.load(f)
    
    # Load LSTM eval metrics
    lstm_metrics_path = MODELS_DIR / "eval_metrics_lstm.json"
    if lstm_metrics_path.exists():
        with open(lstm_metrics_path, 'r') as f:
            eval_metrics_lstm = json.load(f)


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Fraud Detection API (PyTorch)",
        "version": "2.0.0",
        "model_type": "Deep Neural Network",
        "framework": "PyTorch",
        "endpoints": {
            "predict": "/predict",
            "predict_ulb": "/predict-ulb",
            "batch_predict_ulb": "/batch-predict-ulb",
            "predict_lstm": "/predict-lstm",
            "batch_predict": "/batch-predict",
            "health": "/health",
            "model_info": "/model-info",
            "model_info_ulb": "/model-info-ulb",
            "model_info_lstm": "/model-info-lstm",
            "training_history": "/training-history",
            "metrics": "/metrics",
            "train": "/train",
            "train_lstm": "/train-lstm",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None and detector.model is not None,
        "lstm_model_loaded": lstm_detector is not None and lstm_detector.model is not None,
        "ulb_model_loaded": ulb_detector is not None and ulb_detector.model is not None,
        "device": str(detector.device) if detector else "N/A",
        "framework": "PyTorch"
    }


@app.get("/model-info")
async def model_info():
    """Get model architecture and info"""
    if detector is None or detector.model is None:
        return {"message": "No model loaded. Train a model first."}
    
    model = detector.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": "FraudDetectorNet (Deep Neural Network)",
        "framework": "PyTorch",
        "device": str(detector.device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "architecture": str(model),
        "features_used": detector.feature_names or [],
    }


@app.get("/model-info-ulb")
async def ulb_model_info():
    if ulb_detector is None or ulb_detector.model is None:
        return {"message": "No ULB model loaded."}
    return {
        "model_name": "fraud_detector_ulb.pt",
        "dataset": "ULB/Worldline credit-card fraud benchmark",
        "feature_count": len(ulb_detector.feature_names),
        "threshold": ulb_detector.threshold,
        "model_type": "FraudDetectorNet (MLP)",
        "features_used": list(ulb_detector.feature_names),
    }


@app.get("/training-history")
async def get_training_history():
    """Get training metrics history for visualization"""
    global training_history
    
    history_path = MODELS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history = json.load(f)
        return training_history
    
    return {"message": "No training history available. Train a model first."}


@app.get("/metrics")
async def get_metrics():
    """Get MLP evaluation metrics"""
    global eval_metrics
    
    metrics_path = MODELS_DIR / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            eval_metrics = json.load(f)
        return eval_metrics
    
    return {"message": "No evaluation metrics available. Train a model first."}


@app.get("/metrics-lstm")
async def get_lstm_metrics():
    """Get LSTM evaluation metrics"""
    global eval_metrics_lstm
    
    metrics_path = MODELS_DIR / "eval_metrics_lstm.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            eval_metrics_lstm = json.load(f)
        return eval_metrics_lstm
    
    return {"message": "No LSTM evaluation metrics available. Train the LSTM model first."}


@app.get("/training-history-lstm")
async def get_lstm_training_history():
    """Get LSTM training metrics history for visualization"""
    global training_history_lstm
    
    history_path = MODELS_DIR / "training_history_lstm.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history_lstm = json.load(f)
        return training_history_lstm
    
    return {"message": "No LSTM training history available. Train the LSTM model first."}


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """Predict if a transaction is fraudulent"""
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    
    try:
        transaction_dict = transaction.model_dump()
        device_id = transaction_dict.pop('device_id')
        
        result = detector.predict(transaction_dict)
        
        message = ("Transaction flagged as potentially fraudulent" 
                   if result['is_fraud'] 
                   else "Transaction appears legitimate")
        
        return PredictionResponse(
            is_fraud=result['is_fraud'],
            fraud_probability=result['fraud_probability'],
            risk_level=result['risk_level'],
            message=message
        )
        
    except Exception:
        logger.exception("Synthetic MLP prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.") from None


def _ulb_response(result):
    return ULBPredictionResponse(
        model_name="fraud_detector_ulb.pt",
        is_fraud=result['is_fraud'],
        predicted_class=result['predicted_class'],
        fraud_probability=result['fraud_probability'],
        threshold=result['threshold'],
        risk_level=result['risk_level'],
        message=("Transaction flagged as potentially fraudulent" if result['is_fraud']
                 else "Transaction appears legitimate"),
    )


@app.post("/predict-ulb", response_model=ULBPredictionResponse)
async def predict_fraud_ulb(transaction: ULBTransaction):
    """Explicitly select the ULB model for ULB-schema transactions."""
    if ulb_detector is None or ulb_detector.model is None:
        raise HTTPException(status_code=503, detail="ULB model not loaded.")
    try:
        return _ulb_response(ulb_detector.predict(transaction.model_dump()))
    except Exception:
        logger.exception("ULB prediction failed")
        raise HTTPException(status_code=500, detail="ULB prediction failed due to an internal error.") from None


@app.post("/batch-predict-ulb", response_model=ULBBatchPredictionResponse)
async def batch_predict_fraud_ulb(request: ULBBatchPredictionRequest):
    """Batch inference using the ULB model's fixed saved threshold."""
    if ulb_detector is None or ulb_detector.model is None:
        raise HTTPException(status_code=503, detail="ULB model not loaded.")
    try:
        predictions = [_ulb_response(ulb_detector.predict(tx.model_dump()))
                       for tx in request.transactions]
    except Exception:
        logger.exception("ULB batch prediction failed")
        raise HTTPException(status_code=500, detail="ULB batch prediction failed due to an internal error.") from None
    return ULBBatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        flagged_count=sum(item.is_fraud for item in predictions),
    )


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_fraud(request: BatchPredictionRequest):
    """Predict fraud for multiple transactions"""
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    predictions = []
    flagged_count = 0
    try:
        for transaction in request.transactions:
            transaction_dict = transaction.model_dump()
            transaction_dict.pop('device_id')
            result = detector.predict(
                transaction_dict,
                request.threshold if request.threshold is not None else 0.5,
            )
            if result['is_fraud']:
                flagged_count += 1
                message = "Transaction flagged as potentially fraudulent"
            else:
                message = "Transaction appears legitimate"
            predictions.append(PredictionResponse(
                is_fraud=result['is_fraud'],
                fraud_probability=result['fraud_probability'],
                risk_level=result['risk_level'],
                message=message
            ))
    except Exception:
        logger.exception("Synthetic batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed due to an internal error.") from None
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        flagged_count=flagged_count
    )


@app.post("/train")
async def train_model():
    """Train the PyTorch fraud detection model"""
    global detector, training_history, eval_metrics
    
    try:
        from src.data_pipeline import DataPipeline
        from src.preprocessing import Preprocessor
        from src.train_model import PyTorchTrainer, split_train_validation
        from src.pytorch_model import create_data_loaders
        from src.evaluate import ModelEvaluator
        
        # Data pipeline
        pipeline = DataPipeline()
        pipeline.load_data()
        pipeline.handle_missing_values()
        pipeline.feature_engineering()
        pipeline.encode_categorical()
        X_train, X_test, y_train, y_test = pipeline.prepare_data()
        X_train, X_val, y_train, y_val = split_train_validation(X_train, y_train)
        
        # Preprocessing
        preprocessor = Preprocessor()
        X_train_scaled, X_val_scaled = preprocessor.fit_transform(X_train, X_val)
        X_test_scaled = preprocessor.transform(X_test)
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with (MODELS_DIR / 'scaler_mlp.pkl').open('wb') as scaler_file:
            pickle.dump(preprocessor.scaler, scaler_file)
        
        feature_names = list(X_train.columns)
        with (MODELS_DIR / 'feature_names.json').open('w', encoding='utf-8') as f:
            json.dump(feature_names, f)
        
        # Create data loaders
        input_dim = X_train_scaled.shape[1]
        train_loader, val_loader = create_data_loaders(
            X_train_scaled, y_train.values,
            X_val_scaled, y_val.values,
            batch_size=512
        )
        
        # Train
        trainer = PyTorchTrainer(input_dim)
        trainer.build_model()
        training_history = trainer.train(
            train_loader, val_loader, y_train,
            epochs=50, learning_rate=0.001, patience=10
        )
        trainer.save_model(str(MODELS_DIR))
        
        # Evaluate
        predictions, probabilities = trainer.predict(X_test_scaled)
        evaluator = ModelEvaluator(y_test.values, predictions, probabilities, 'PyTorch Neural Network')
        metrics = evaluator.calculate_metrics()
        eval_metrics = {k: float(v) for k, v in metrics.items()}
        
        with (MODELS_DIR / 'eval_metrics.json').open('w', encoding='utf-8') as f:
            json.dump(eval_metrics, f, indent=2)
        
        # Reload model
        detector = FraudDetector(model_path=MODELS_DIR)
        detector.load_model()
        detector.load_scaler()
        detector.load_feature_names()
        
        return {
            "status": "success",
            "message": "PyTorch model trained and saved successfully",
            "metrics": eval_metrics
        }
        
    except Exception:
        logger.exception("Synthetic MLP training failed")
        raise HTTPException(status_code=500, detail="Training failed due to an internal error.") from None


@app.get("/model-info-lstm")
async def lstm_model_info():
    """Get LSTM model architecture and info"""
    if lstm_detector is None or lstm_detector.model is None:
        return {"message": "No LSTM model loaded. Train a model first via /train-lstm."}
    
    model = lstm_detector.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": "FraudLSTMNet (Bidirectional LSTM)",
        "framework": "PyTorch",
        "device": str(lstm_detector.device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "seq_length": lstm_detector.seq_length,
        "architecture": str(model),
        "features_used": lstm_detector.feature_names or [],
    }


@app.post("/predict-lstm", response_model=PredictionResponse)
async def predict_fraud_lstm(transaction: Transaction):
    """Predict if a transaction is fraudulent using the LSTM model"""
    if lstm_detector is None or lstm_detector.model is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded. Train via /train-lstm first.")
    
    try:
        transaction_dict = transaction.model_dump()
        device_id = transaction_dict.pop('device_id')
        
        result = lstm_detector.predict(transaction_dict)
        
        message = ("[LSTM] Transaction flagged as potentially fraudulent" 
                   if result['is_fraud'] 
                   else "[LSTM] Transaction appears legitimate")
        
        return PredictionResponse(
            is_fraud=result['is_fraud'],
            fraud_probability=result['fraud_probability'],
            risk_level=result['risk_level'],
            message=message
        )
        
    except Exception:
        logger.exception("Synthetic LSTM prediction failed")
        raise HTTPException(status_code=500, detail="LSTM prediction failed due to an internal error.") from None


@app.post("/train-lstm")
async def train_lstm_model():
    """Train the LSTM fraud detection model"""
    global lstm_detector
    
    try:
        from src.data_pipeline import DataPipeline
        from src.preprocessing import Preprocessor
        from src.train_model import LSTMTrainer
        from src.pytorch_model import create_chronological_sequence_data_loaders
        from src.evaluate import ModelEvaluator
        import pandas as pd
        
        # Ordered sequences require a full timestamp; time-of-day alone is not chronology.
        pipeline = DataPipeline()
        pipeline.load_data()
        pipeline.handle_missing_values()
        pipeline.feature_engineering()
        pipeline.encode_categorical()
        
        df = pipeline.df.copy()
        if 'transaction_timestamp' not in df.columns:
            raise ValueError(
                'LSTM training requires transaction_timestamp with full date and time. '
                'The loaded synthetic CSV only has time-of-day; regenerate it with '
                'generate_dataset.py to create chronological synthetic data.'
            )
        df['transaction_timestamp'] = pd.to_datetime(
            df['transaction_timestamp'], errors='raise'
        )
        
        # Remove transaction_id & device_id but keep customer_id
        for col in ['transaction_id', 'device_id']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        # A chronological 60/20/20 split; all boundaries are full timestamps.
        ordered_times = df['transaction_timestamp'].sort_values().reset_index(drop=True)
        val_boundary = ordered_times.iloc[int(len(ordered_times) * 0.60)]
        test_boundary = ordered_times.iloc[int(len(ordered_times) * 0.80)]
        train_df = df[df['transaction_timestamp'] < val_boundary].copy()
        val_df = df[(df['transaction_timestamp'] >= val_boundary) &
                    (df['transaction_timestamp'] < test_boundary)].copy()
        test_df = df[df['transaction_timestamp'] >= test_boundary].copy()
        if min(len(train_df), len(val_df), len(test_df)) == 0:
            raise ValueError('Chronological split produced an empty data partition.')
        
        # Determine feature columns (everything except identifiers and target)
        exclude_cols = {
            'customer_id', 'is_fraud', 'transaction_time', 'transaction_timestamp'
        }
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        # Scale features in-place
        preprocessor = Preprocessor()
        train_df[feature_cols] = preprocessor.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = preprocessor.transform(val_df[feature_cols])
        test_df[feature_cols] = preprocessor.transform(test_df[feature_cols])
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with (MODELS_DIR / 'scaler_lstm.pkl').open('wb') as scaler_file:
            pickle.dump(preprocessor.scaler, scaler_file)
        
        with (MODELS_DIR / 'feature_names_lstm.json').open('w', encoding='utf-8') as f:
            json.dump(feature_cols, f)
        
        # Create sequence data loaders
        seq_length = 10
        train_loader, val_loader, test_loader, input_dim = create_chronological_sequence_data_loaders(
            train_df, val_df, test_df, feature_cols,
            seq_length=seq_length, batch_size=512
        )
        
        # Train LSTM
        trainer = LSTMTrainer(input_dim, seq_length=seq_length)
        trainer.build_model()
        y_train = train_df['is_fraud']
        lstm_history = trainer.train(
            train_loader, val_loader, y_train,
            epochs=50, learning_rate=0.001, patience=10
        )
        trainer.save_model(str(MODELS_DIR))
        
        # Evaluate
        predictions, probabilities = trainer.predict(test_loader)
        y_test_labels = test_df['is_fraud'].values
        
        # The test_loader re-builds sequences, so label count matches dataset length
        # Use the labels from the test_loader's dataset directly
        test_labels_from_loader = test_loader.dataset.labels.numpy()
        
        evaluator = ModelEvaluator(
            test_labels_from_loader, predictions, probabilities,
            'LSTM Neural Network'
        )
        metrics = evaluator.calculate_metrics()
        lstm_eval_metrics = {k: float(v) for k, v in metrics.items()}
        
        with (MODELS_DIR / 'eval_metrics_lstm.json').open('w', encoding='utf-8') as f:
            json.dump(lstm_eval_metrics, f, indent=2)
        
        # Reload LSTM model
        lstm_detector = LSTMFraudDetector(model_path=MODELS_DIR)
        lstm_detector.load_model()
        lstm_detector.load_scaler()
        lstm_detector.load_feature_names()
        
        return {
            "status": "success",
            "message": "LSTM model trained and saved successfully",
            "metrics": lstm_eval_metrics
        }
        
    except Exception:
        logger.exception("Synthetic LSTM training failed")
        raise HTTPException(status_code=500, detail="LSTM training failed due to an internal error.") from None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
