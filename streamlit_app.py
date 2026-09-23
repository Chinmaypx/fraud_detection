"""
Streamlit Demo Application for Fraud Detection
Interactive web interface for testing fraud predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DIR = PROJECT_ROOT / "models"

from src.data_pipeline import DataPipeline
from src.preprocessing import Preprocessor
from src.train_model import PyTorchTrainer
from src.evaluate import ModelEvaluator
from src.predict import FraudDetector

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide"
)

st.title("🔒 Banking Fraud Detection System")
st.markdown("""
This application demonstrates a complete machine learning pipeline for detecting
fraudulent banking transactions in real-time.
""")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Make Prediction", "Model Evaluation", "About"])

@st.cache_resource
def load_detector():
    """
    Load trained FraudDetector (PyTorch model, scaler, feature names)
    """
    detector = FraudDetector(model_path=MODEL_DIR)
    model_path = MODEL_DIR / "fraud_detector.pt"

    if not model_path.exists():
        return None

    try:
        detector.load_model()
        detector.load_scaler()
        detector.load_feature_names()
    except Exception:
        return None

    return detector


def create_transaction_form():
    """
    Create input form for transaction details
    """
    st.subheader("Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.01,
            value=250.0,
            step=10.0
        )
        
        transaction_time = st.slider(
            "Time of Day (Hour)",
            min_value=0,
            max_value=23,
            value=12,
            help="Hour of the day (0-23)"
        )
        
        location = st.selectbox(
            "Location",
            ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
        )
        
        merchant_category = st.selectbox(
            "Merchant Category",
            ["retail", "grocery", "restaurant", "gas", "online"]
        )
    
    with col2:
        account_age_days = st.number_input(
            "Account Age (days)",
            min_value=0,
            value=365,
            step=30
        )
        
        transaction_count_24h = st.number_input(
            "Transactions in Last 24h",
            min_value=0,
            value=3,
            step=1
        )
        
        avg_transaction_amount = st.number_input(
            "Average Transaction Amount (₹)",
            min_value=0.01,
            value=150.0,
            step=10.0
        )
        
        device_id = st.text_input("Device ID", value="DEV001")
    
    transaction_data = {
        'transaction_amount': transaction_amount,
        'transaction_time': transaction_time * 3600,
        'location': location,
        'device_id': device_id,
        'merchant_category': merchant_category,
        'account_age_days': account_age_days,
        'transaction_count_24h': transaction_count_24h,
        'avg_transaction_amount': avg_transaction_amount,
    }
    
    threshold = st.slider(
        "Fraud Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Lower threshold = more fraud alerts (more false positives)"
    )
    
    return transaction_data, threshold


if page == "Home":
    st.header("Welcome to the Fraud Detection System")
    
    st.markdown("""
    ### How It Works
    
    This system uses machine learning to identify potentially fraudulent
    banking transactions in real-time.
    
    ### Key Features
    
    - **Real-time Prediction**: Analyze transactions instantly
    - **Risk Scoring**: Get probability scores and risk levels
    - **Multiple Models**: Compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
    - **Business Metrics**: Evaluate financial impact of fraud detection
    
    ### Why Fraud Detection Matters
    
    1. **Financial Loss Prevention**: Fraud causes billions in losses annually
    2. **Customer Protection**: Protect customers from unauthorized transactions
    3. **Regulatory Compliance**: Meet anti-fraud regulations
    4. **Reputation Management**: Maintain trust in the banking system
    
    ---
    
    **Get Started**: Go to "Make Prediction" in the sidebar to test the system!
    """)
    
    st.info("💡 Tip: Train the model first using the model training script!")


elif page == "Make Prediction":
    st.header("Make a Fraud Prediction")

    detector = load_detector()

    if detector is None:
        st.warning("⚠️ No trained model found. Please train the model first.")
    else:
        transaction_data, threshold = create_transaction_form()
        
        if st.button("Predict Fraud", type="primary"):
            with st.spinner("Analyzing transaction..."):
                result = detector.predict(transaction_data, threshold)
                
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Fraud Probability", f"{result['fraud_probability']*100:.1f}%")
                
                with col2:
                    risk_color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
                    st.markdown(f"### Risk Level: :{risk_color[result['risk_level']]}[{result['risk_level']}]")
                
                with col3:
                    if result['is_fraud']:
                        st.error("🚨 FLAGGED AS FRAUD")
                    else:
                        st.success("✅ Legitimate Transaction")
                
                if result['is_fraud']:
                    st.warning("⚠️ This transaction has been flagged as potentially fraudulent. Additional verification may be required.")
                else:
                    st.info("✓ This transaction appears to be legitimate.")


elif page == "Model Evaluation":
    st.header("Model Evaluation Metrics")

    detector = load_detector()

    if detector is None:
        st.warning("⚠️ No trained model found. Please train the model first.")
    else:
        import json
        metrics_path = MODEL_DIR / "eval_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                saved_metrics = json.load(f)
            st.subheader("Model Performance Metrics")
            metrics_data = {
                "Metric": list(saved_metrics.keys()),
                "Value": [f"{v:.4f}" for v in saved_metrics.values()],
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)
        else:
            st.info("No evaluation metrics found. Train the model to generate metrics.")

        st.markdown("""
        ### Understanding the Metrics

        | Metric | What It Measures | Why It Matters |
        |--------|-----------------|----------------|
        | **Precision** | Accuracy of fraud predictions | Low precision = too many false alarms |
        | **Recall** | Fraud detection rate | Low recall = missed fraud (financial loss!) |
        | **F1 Score** | Balance between precision and recall | Best single metric for imbalanced data |
        | **ROC-AUC** | Overall discrimination ability | Higher = better at distinguishing fraud |
        | **PR-AUC** | Performance on rare class | Best for highly imbalanced datasets |

        ### Business Impact

        - **False Negatives (Missed Fraud)**: Direct financial loss to bank and customer
        - **False Positives (False Alarms)**: Customer inconvenience, potential churn
        """)


elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### Banking Fraud Detection System
    
    This is a complete end-to-end machine learning project designed to detect
    fraudulent banking transactions.
    
    ### Project Structure
    
    ```
    fraud_detection_project/
    ├── data/                    # Dataset files
    ├── notebooks/               # Jupyter notebooks
    ├── src/
    │   ├── data_pipeline.py    # Data loading and preprocessing
    │   ├── preprocessing.py     # Feature scaling and SMOTE
    │   ├── train_model.py       # Model training
    │   ├── evaluate.py          # Model evaluation
    │   └── predict.py           # Prediction module
    ├── models/                  # Saved models
    ├── api/
    │   └── app.py              # FastAPI application
    ├── streamlit_app.py         # Streamlit web app
    └── requirements.txt         # Dependencies
    ```
    
    ### Technologies Used
    
    - **Python**: Programming language
    - **Pandas & NumPy**: Data manipulation
    - **Scikit-learn**: Machine learning models
    - **Imbalanced-learn**: SMOTE for handling class imbalance
    - **FastAPI**: REST API framework
    - **Streamlit**: Web UI framework
    
    ### Key Techniques
    
    1. **SMOTE**: Synthetic Minority Over-sampling Technique
    2. **Cost-Sensitive Learning**: Class weights for imbalanced data
    3. **Feature Engineering**: Creating meaningful features from raw data
    4. **Ensemble Methods**: Random Forest and Gradient Boosting
    
    ---
    
    ### Real-World Impact
    
    This fraud detection system can help banks:
    - Reduce financial losses from fraud
    - Improve customer experience
    - Meet regulatory requirements
    - Scale fraud detection to millions of transactions
    """)
