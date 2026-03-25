"""
FastAPI Application for Fraud Detection (PyTorch)
REST API with CORS for React frontend integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import json
import pickle
import torch
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import FraudDetector

app = FastAPI(
    title="Fraud Detection API (PyTorch)",
    description="Deep Learning API for detecting fraudulent banking transactions",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class Transaction(BaseModel):
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in INR")
    transaction_time: int = Field(..., ge=0, le=86399, description="Seconds since midnight")
    location: str = Field(..., description="City code")
    device_id: str = Field(..., description="Device identifier")
    merchant_category: str = Field(..., description="Category: retail, grocery, restaurant, gas, online")
    account_age_days: int = Field(..., ge=0, description="Age of account in days")
    transaction_count_24h: int = Field(..., ge=0, description="Transactions in last 24 hours")
    avg_transaction_amount: float = Field(..., gt=0, description="Average transaction amount")


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    message: str


class BatchPredictionRequest(BaseModel):
    transactions: List[Transaction]
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    flagged_count: int


# --- Global state ---
detector = None
training_history = None
eval_metrics = None


@app.on_event("startup")
async def load_model():
    """Load PyTorch model and scaler on startup"""
    global detector, training_history, eval_metrics
    
    detector = FraudDetector()
    
    model_path = Path("models/fraud_detector.pt")
    if model_path.exists():
        try:
            detector.load_model()
            detector.load_scaler()
            detector.load_feature_names()
            print("PyTorch model loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading model: {e}")
            detector.model = None
    else:
        print("Warning: No trained model found. Train first via /train endpoint or CLI.")
    
    # Load training history
    history_path = Path("models/training_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    
    # Load eval metrics
    metrics_path = Path("models/eval_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            eval_metrics = json.load(f)


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
            "batch_predict": "/batch-predict",
            "health": "/health",
            "model_info": "/model-info",
            "training_history": "/training-history",
            "metrics": "/metrics",
            "train": "/train",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None and detector.model is not None,
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


@app.get("/training-history")
async def get_training_history():
    """Get training metrics history for visualization"""
    global training_history
    
    history_path = Path("models/training_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history = json.load(f)
        return training_history
    
    return {"message": "No training history available. Train a model first."}


@app.get("/metrics")
async def get_metrics():
    """Get evaluation metrics"""
    global eval_metrics
    
    metrics_path = Path("models/eval_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            eval_metrics = json.load(f)
        return eval_metrics
    
    return {"message": "No evaluation metrics available. Train a model first."}


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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_fraud(request: BatchPredictionRequest):
    """Predict fraud for multiple transactions"""
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    predictions = []
    flagged_count = 0
    
    for transaction in request.transactions:
        transaction_dict = transaction.model_dump()
        transaction_dict.pop('device_id')
        
        result = detector.predict(transaction_dict, request.threshold)
        
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
        from src.train_model import PyTorchTrainer
        from src.pytorch_model import create_data_loaders
        from src.evaluate import ModelEvaluator
        
        # Data pipeline
        pipeline = DataPipeline()
        pipeline.load_data()
        pipeline.handle_missing_values()
        pipeline.feature_engineering()
        pipeline.encode_categorical()
        X_train, X_test, y_train, y_test = pipeline.prepare_data()
        
        # Preprocessing
        preprocessor = Preprocessor()
        X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_test)
        
        os.makedirs('models', exist_ok=True)
        pickle.dump(preprocessor.scaler, open('models/scaler.pkl', 'wb'))
        
        feature_names = list(X_train.columns)
        with open('models/feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        
        # Create data loaders
        input_dim = X_train_scaled.shape[1]
        train_loader, test_loader = create_data_loaders(
            X_train_scaled, y_train.values,
            X_test_scaled, y_test.values,
            batch_size=512
        )
        
        # Train
        trainer = PyTorchTrainer(input_dim)
        trainer.build_model()
        training_history = trainer.train(
            train_loader, test_loader, y_train,
            epochs=50, learning_rate=0.001, patience=10
        )
        trainer.save_model('models/')
        
        # Evaluate
        predictions, probabilities = trainer.predict(X_test_scaled)
        evaluator = ModelEvaluator(y_test.values, predictions, probabilities, 'PyTorch Neural Network')
        metrics = evaluator.calculate_metrics()
        eval_metrics = {k: float(v) for k, v in metrics.items()}
        
        with open('models/eval_metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        # Reload model
        detector = FraudDetector()
        detector.load_model()
        detector.load_scaler()
        detector.load_feature_names()
        
        return {
            "status": "success",
            "message": "PyTorch model trained and saved successfully",
            "metrics": eval_metrics
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
