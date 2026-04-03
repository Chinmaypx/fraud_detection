"""
Evaluation Module for Fraud Detection Models
Comprehensive model evaluation with business metrics
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive evaluation for fraud detection models
    """
    
    def __init__(self, y_test, y_pred, y_pred_proba, model_name):
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.model_name = model_name
        self.metrics = {}
        
    def calculate_metrics(self):
        """
        Calculate all evaluation metrics
        
        Why NOT to rely on accuracy:
        - With 2% fraud rate, a model predicting all transactions
          as legitimate achieves 98% accuracy!
        - This hides the real problem: not detecting fraud
        """
        self.metrics = {
            'Accuracy': accuracy_score(self.y_test, self.y_pred),
            'Precision': precision_score(self.y_test, self.y_pred, zero_division=0),
            'Recall (Sensitivity)': recall_score(self.y_test, self.y_pred, zero_division=0),
            'Specificity': self._calculate_specificity(),
            'F1 Score': f1_score(self.y_test, self.y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(self.y_test, self.y_pred_proba),
            'PR-AUC': average_precision_score(self.y_test, self.y_pred_proba),
        }
        
        return self.metrics
    
    def _calculate_specificity(self):
        """
        Calculate specificity (true negative rate)
        """
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def get_confusion_matrix(self):
        """
        Get confusion matrix
        
        In fraud detection:
        - True Positive (TP): Correctly detected fraud ✓
        - True Negative (TN): Correctly identified legitimate ✓
        - False Positive (FP): Legitimate flagged as fraud (customer inconvenience)
        - False Negative (FN): Missed fraud (financial loss) - MOST DANGEROUS!
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        print(f"\n--- Confusion Matrix: {self.model_name} ---")
        print(f"                    Predicted")
        print(f"                  Legit    Fraud")
        print(f"Actual Legit     {cm[0][0]:5d}   {cm[0][1]:5d}")
        print(f"Actual Fraud     {cm[1][0]:5d}   {cm[1][1]:5d}")
        
        return cm
    
    def analyze_errors(self):
        """
        Analyze false positives and false negatives
        
        Business Impact Analysis:
        
        FALSE NEGATIVES (Missed Fraud) - MOST DANGEROUS:
        - Financial loss for the bank
        - Customer loses money
        - Reputational damage
        - Regulatory penalties
        - Fraudulent transactions go through
        - Example: $1000 fraud not detected = $1000 loss
        
        FALSE POSITIVES (Blocked Legitimate Transactions):
        - Customer frustration
        - Transaction declined at checkout
        - Customer switches to competitor
        - Customer support costs increase
        - Potential loss of lifetime value
        - Example: Legit purchase declined = poor customer experience
        """
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        
        analysis = {
            'True Negatives (TN)': tn,
            'True Positives (TP)': tp,
            'False Positives (FP)': fp,
            'False Negatives (FN)': fn,
        }
        
        total = tn + fp + fn + tp
        
        print(f"\n--- Error Analysis: {self.model_name} ---")
        print(f"False Negatives (Missed Fraud): {fn} ({fn/total*100:.2f}%)")
        print(f"  → Financial Loss Risk: HIGH")
        print(f"False Positives (False Alarms): {fp} ({fp/total*100:.2f}%)")
        print(f"  → Customer Experience Impact: MEDIUM")
        
        return analysis
    
    def get_classification_report(self):
        """
        Get detailed classification report
        """
        report = classification_report(
            self.y_test, self.y_pred, 
            target_names=['Legitimate', 'Fraud'],
            zero_division=0
        )
        
        print(f"\n--- Classification Report: {self.model_name} ---")
        print(report)
        
        return report
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix heatmap
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
        
    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC Curve
        
        ROC (Receiver Operating Characteristic) shows:
        - TPR (True Positive Rate) vs FPR (False Positive Rate)
        - AUC = Area Under Curve (1.0 = perfect, 0.5 = random)
        
        Higher AUC = better model at distinguishing fraud from legit
        """
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.close()
        
    def plot_precision_recall_curve(self, save_path=None):
        """
        Plot Precision-Recall Curve
        
        Important for imbalanced datasets:
        - PR-AUC is more informative than ROC-AUC when classes are imbalanced
        - Shows trade-off between precision and recall
        """
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_proba
        )
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        
        plt.close()
    
    def print_summary(self):
        """
        Print evaluation summary
        """
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY: {self.model_name}")
        print("="*60)
        
        self.calculate_metrics()
        
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        
        self.get_confusion_matrix()
        self.analyze_errors()


def compare_models(model_results, model_names):
    """
    Create comparison table for all models
    
    Returns a DataFrame comparing all models on different metrics
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_data = []
    
    for model_name, (y_test, y_pred, y_pred_proba) in model_results.items():
        evaluator = ModelEvaluator(y_test, y_pred, y_pred_proba, model_name)
        metrics = evaluator.calculate_metrics()
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['Accuracy']:.4f}",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall (Sensitivity)']:.4f}",
            'F1 Score': f"{metrics['F1 Score']:.4f}",
            'ROC-AUC': f"{metrics['ROC-AUC']:.4f}",
            'PR-AUC': f"{metrics['PR-AUC']:.4f}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    return comparison_df


def select_best_model(model_results, metric='F1 Score'):
    """
    Select the best model based on specified metric
    
    For fraud detection, recommended metrics:
    - F1 Score: Balance between precision and recall
    - PR-AUC: Best for imbalanced datasets
    - ROC-AUC: General model performance
    """
    best_model = None
    best_score = -1
    
    for model_name, (y_test, y_pred, y_pred_proba) in model_results.items():
        evaluator = ModelEvaluator(y_test, y_pred, y_pred_proba, model_name)
        metrics = evaluator.calculate_metrics()
        
        if metric == 'F1 Score':
            score = metrics['F1 Score']
        elif metric == 'PR-AUC':
            score = metrics['PR-AUC']
        elif metric == 'ROC-AUC':
            score = metrics['ROC-AUC']
        elif metric == 'Recall':
            score = metrics['Recall (Sensitivity)']
        else:
            score = metrics.get(metric, 0)
        
        if score > best_score:
            best_score = score
            best_model = model_name
    
    print(f"\nBest model based on {metric}: {best_model} ({best_score:.4f})")
    
    return best_model, best_score


def calculate_business_metrics(y_test, y_pred, avg_fraud_amount=500):
    """
    Calculate business-specific metrics
    
    Financial Impact Analysis:
    - Amount saved by detecting fraud
    - Customer inconvenience cost
    - Net benefit
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    fraud_detected = tp
    fraud_missed = fn
    legitimate_blocked = fp
    
    amount_saved = fraud_detected * avg_fraud_amount
    amount_lost = fraud_missed * avg_fraud_amount
    
    business_metrics = {
        'Fraud Detected': fraud_detected,
        'Fraud Missed': fraud_missed,
        'Legitimate Blocked': legitimate_blocked,
        'Amount Saved ($)': amount_saved,
        'Amount Lost ($)': amount_lost,
        'Net Benefit ($)': amount_saved - amount_lost,
    }
    
    print("\n--- Business Metrics ---")
    for key, value in business_metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: ${value:,.2f}")
    
    return business_metrics


if __name__ == "__main__":
    print("Run src.train_model.main() to train and evaluate the model.")
