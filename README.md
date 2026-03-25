# Banking System Fraud Detection using Machine Learning

A complete end-to-end production-style machine learning project for detecting fraudulent banking transactions.

## Project Overview

This project implements a comprehensive fraud detection system that addresses the challenge of detecting rare fraudulent transactions in highly imbalanced datasets (<1% fraud rate).

---

## 1. Problem Understanding

### The Challenge
- **Imbalanced Data**: Fraudulent transactions represent less than 1% of all transactions
- **Evolving Patterns**: Fraudsters constantly adapt their methods
- **False Positives vs False Negatives**: Balancing customer experience with fraud prevention

### Business Context
- Digital banking processes millions of transactions daily
- Traditional rule-based systems generate many false positives
- Financial losses from fraud are significant
- Customer experience matters - too many declined transactions leads to churn

---

## 2. Data Pipeline

### Files: `src/data_pipeline.py`

Complete data processing pipeline:

```
DataPipeline
├── load_data()           # Load from CSV or generate synthetic data
├── explore_data()        # EDA and data understanding
├── handle_missing_values() # Imputation
├── detect_outliers()     # IQR-based outlier detection
├── feature_engineering() # Create new features
├── encode_categorical()  # One-hot encoding
└── prepare_data()        # Train/test split
```

### Feature Engineering
- `hour_of_day`: Time of transaction
- `is_night_transaction`: Late night transactions are higher risk
- `amount_to_avg_ratio`: Unusually high amounts
- `high_amount`: Flag for high-value transactions
- `unusual_merchant`: Online purchases are riskier
- `new_customer`: New accounts have higher fraud risk
- `high_frequency`: Unusual transaction frequency

---

## 3. Handling Imbalanced Data

### Why Imbalanced Data is Problematic
- Model learns to predict majority class
- 98% accuracy can be achieved by predicting all legitimate!
- Misses the important minority class (fraud)

### Techniques Implemented

#### A. SMOTE (Synthetic Minority Over-sampling Technique)
**File: `src/preprocessing.py`**

```
SMOTE Process:
1. Select random minority sample
2. Find k-nearest neighbors (k=5)
3. Randomly select one neighbor
4. Generate synthetic sample between them
```

**Pros:**
- Creates new synthetic samples (not duplicates)
- Helps model learn better decision boundaries
- Doesn't discard valuable majority class data

**Cons:**
- Can create noise when applied incorrectly
- May not work well with high-dimensional data

#### B. Random Under Sampling
```
Process: Randomly remove majority class samples to match minority class
```

**Pros:**
- Simple to implement
- Faster training

**Cons:**
- Loses valuable information
- May create biased sample

#### C. Cost-Sensitive Learning
Instead of changing data, we modify the learning algorithm:

```python
class_weight='balanced'  # In sklearn models
```

This assigns higher penalty to misclassification of minority class.

---

## 4. Model Training

### File: `src/train_model.py`

### Models Compared

| Model | Why Used |
|-------|----------|
| **Logistic Regression** | Interpretable baseline, provides probabilities, fast |
| **Decision Tree** | Interpretable, captures non-linear patterns |
| **Random Forest** | Ensemble of trees, reduces overfitting, handles high-dimensional data |
| **Gradient Boosting** | Sequential correction of errors, often best performance |

### Training with SMOTE
```python
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## 5. Evaluation Metrics

### File: `src/evaluate.py`

### Why NOT Accuracy?
A model predicting ALL transactions as legitimate achieves 98% accuracy with 2% fraud rate!

### Metrics Used

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Precision** | TP/(TP+FP) | Of flagged fraud, how many are actual fraud |
| **Recall** | TP/(TP+FN) | Of actual fraud, how many did we catch |
| **F1 Score** | 2×(P×R)/(P+R) | Balance between precision and recall |
| **ROC-AUC** | Area under ROC | Discrimination ability |
| **PR-AUC** | Area under PR curve | Best for imbalanced data |

### Confusion Matrix Analysis

```
                    Predicted
                  Legit    Fraud
Actual Legit     [TN]     [FP]
Actual Fraud    [FN]     [TP]
```

### Business Impact

**FALSE NEGATIVES (Missed Fraud) - MOST DANGEROUS:**
- Direct financial loss to bank and customer
- Reputational damage
- Regulatory penalties
- Customer loses trust
- Example: $1000 fraud not caught = $1000 loss

**FALSE POSITIVES (False Alarms):**
- Customer frustration
- Transaction declined at checkout
- Customer switches to competitor
- Customer support costs increase
- Example: Legit purchase declined = poor experience

---

## 6. Model Comparison

### Example Results Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----------|---------|--------|
| Logistic Regression | 0.92 | 0.75 | 0.68 | 0.71 | 0.88 | 0.62 |
| Decision Tree | 0.94 | 0.82 | 0.72 | 0.77 | 0.89 | 0.68 |
| Random Forest | 0.96 | 0.85 | 0.78 | 0.81 | 0.93 | 0.74 |
| Gradient Boosting | 0.97 | 0.88 | 0.82 | 0.85 | 0.95 | 0.79 |

**Best Model: Gradient Boosting** (highest F1 Score and PR-AUC)

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Transaction  │───▶│    Feature   │───▶│     ML       │     │
│  │    Stream    │    │  Processor  │    │    Model     │     │
│  └──────────────┘    └──────────────┘    └──────┬───────┘     │
│                                                  │              │
│                                                  ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Fraud      │◀───│   Decision   │◀───│  Probability │     │
│  │    Alert     │    │   Threshold  │    │   Calculator │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Components:
1. Transaction Stream: Real-time transaction data from banking systems
2. Feature Processor: Extracts and transforms features
3. ML Model: Gradient Boosting classifier
4. Probability Calculator: Outputs fraud probability
5. Decision Threshold: Adjustable threshold (0.5 default)
6. Fraud Alert System: Notifies fraud team, blocks transaction
```

---

## 8. Deployment API

### FastAPI Application: `api/app.py`

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single transaction prediction |
| `/batch-predict` | POST | Batch prediction |
| `/train` | POST | Train model |

#### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 1500.00,
    "transaction_time": 36000,
    "location": "NYC",
    "device_id": "DEV001",
    "merchant_category": "retail",
    "account_age_days": 365,
    "transaction_count_24h": 3,
    "avg_transaction_amount": 150.00
  }'
```

#### Example Response
```json
{
  "is_fraud": false,
  "fraud_probability": 0.15,
  "risk_level": "LOW",
  "message": "Transaction appears legitimate"
}
```

---

## 9. Streamlit Demo

### File: `streamlit_app.py`

Interactive web interface with:
- Make predictions
- Model evaluation metrics
- Model training
- Documentation

Run with:
```bash
streamlit run streamlit_app.py
```

---

## 10. Real-World Banking Impact

### How Banks Use These Systems

1. **Real-Time Decisioning**: Every transaction is scored in milliseconds
2. **Layered Defense**: ML + Rules + Human review
3. **Continuous Learning**: Models retrained with new fraud patterns
4. **Customer Communication**: Alerts for suspicious activity

### How Companies Reduce Fraud Risk

| Strategy | Implementation |
|----------|---------------|
| **Velocity Checks** | Block if too many transactions in short time |
| **Geolocation** | Flag transactions from unusual locations |
| **Device Fingerprinting** | Identify compromised devices |
| **Behavioral Analytics** | Learn spending patterns |
| **Network Analysis** | Graph-based fraud ring detection |
| **2FA/3D Secure** | Additional authentication |
| **Dynamic CVV** | Generate one-time CVV codes |

### Industry Statistics
- Banks lose $25-30 billion annually to card fraud globally
- Machine learning reduces fraud losses by 50-70%
- False positive reduction improves customer experience significantly

---

## 11. Future Improvements

### Deep Learning Approaches
- **Autoencoders**: Learn normal transaction patterns, flag anomalies
- **LSTM/GRU**: Capture temporal patterns in transaction sequences
- **Transformer Models**: Handle sequential transaction data
- **Neural Networks**: Deep feature learning

### Graph Fraud Detection
- **Graph Neural Networks (GNN)**: Detect fraud rings
- **Entity Resolution**: Link related entities
- **Social Network Analysis**: Identify suspicious networks

### Production Considerations
- Model monitoring and drift detection
- A/B testing for model updates
- Feature store for consistent features
- Model registry for version control
- Real-time feature computation
- Low-latency inference requirements

---

## Project Structure

```
fraud_detection_project/
│
├── data/                      # Data files
│   └── (your dataset.csv)
│
├── notebooks/                 # Jupyter notebooks
│   └── exploration.ipynb
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_pipeline.py      # Data loading & preprocessing
│   ├── preprocessing.py       # Scaling & SMOTE
│   ├── train_model.py         # Model training
│   ├── evaluate.py            # Evaluation metrics
│   └── predict.py             # Prediction module
│
├── models/                    # Saved models
│   ├── gradient_boosting_model.pkl
│   └── scaler.pkl
│
├── api/                       # FastAPI app
│   └── app.py
│
├── streamlit_app.py           # Streamlit demo
│
├── requirements.txt           # Dependencies
│
└── README.md                  # This file
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
cd fraud_detection_project
python -m src.train_model
```

### 3. Run API
```bash
uvicorn api.app:app --reload
```

### 4. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## License

This project is for educational purposes.
