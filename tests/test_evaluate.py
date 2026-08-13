"""
Tests for evaluate.py
"""

import pytest
import numpy as np
from src.evaluate import (
    ModelEvaluator,
    calculate_business_metrics,
    compare_models,
    select_best_model
)


class TestModelEvaluator:
    """Tests for ModelEvaluator class"""

    def test_init(self, sample_predictions):
        """Test ModelEvaluator initialization"""
        evaluator = ModelEvaluator(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_proba'],
            'Test Model'
        )
        assert evaluator.y_test is not None
        assert evaluator.y_pred is not None
        assert evaluator.model_name == 'Test Model'

    def test_calculate_metrics(self, sample_predictions):
        """Test calculate_metrics returns all metrics"""
        evaluator = ModelEvaluator(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_proba'],
            'Test Model'
        )
        metrics = evaluator.calculate_metrics()
        
        expected_keys = [
            'Accuracy', 'Precision', 'Recall (Sensitivity)',
            'Specificity', 'F1 Score', 'ROC-AUC', 'PR-AUC'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (float, np.floating))

    def test_accuracy_perfect_predictions(self):
        """Test accuracy with perfect predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'Perfect')
        metrics = evaluator.calculate_metrics()
        
        assert metrics['Accuracy'] == 1.0

    def test_accuracy_all_wrong(self):
        """Test accuracy with all wrong predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.1, 0.2])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'Wrong')
        metrics = evaluator.calculate_metrics()
        
        assert metrics['Accuracy'] == 0.0

    def test_precision_perfect(self):
        """Test precision with perfect predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'Perfect')
        metrics = evaluator.calculate_metrics()
        
        assert metrics['Precision'] == 1.0

    def test_precision_no_positives(self):
        """Test precision when no positives predicted"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.1, 0.2])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'No Positives')
        metrics = evaluator.calculate_metrics()
        
        assert metrics['Precision'] == 0.0

    def test_recall(self):
        """Test recall calculation"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([0.1, 0.6, 0.9, 0.8])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'Test')
        metrics = evaluator.calculate_metrics()
        
        assert metrics['Recall (Sensitivity)'] == 1.0

    def test_recall_no_fraud_detected(self):
        """Test recall when no fraud detected"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.1, 0.2])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'No Fraud')
        metrics = evaluator.calculate_metrics()
        
        assert metrics['Recall (Sensitivity)'] == 0.0

    def test_f1_score(self):
        """Test F1 score calculation"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([0.1, 0.6, 0.9, 0.8])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'Test')
        metrics = evaluator.calculate_metrics()
        
        expected_f1 = 2 * metrics['Precision'] * metrics['Recall (Sensitivity)']
        expected_f1 /= metrics['Precision'] + metrics['Recall (Sensitivity)']
        
        assert abs(metrics['F1 Score'] - expected_f1) < 0.0001

    def test_roc_auc_perfect(self):
        """Test ROC-AUC with perfect predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])
        
        evaluator = ModelEvaluator(y_true, y_true, y_proba, 'Perfect')
        metrics = evaluator.calculate_metrics()
        
        assert metrics['ROC-AUC'] == 1.0

    def test_roc_auc_random(self):
        """Test ROC-AUC with random predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])
        
        evaluator = ModelEvaluator(y_true, y_true, y_proba, 'Random')
        metrics = evaluator.calculate_metrics()
        
        assert metrics['ROC-AUC'] == 0.5

    def test_pr_auc(self):
        """Test PR-AUC calculation"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'Test')
        metrics = evaluator.calculate_metrics()
        
        assert 'PR-AUC' in metrics
        assert 0 <= metrics['PR-AUC'] <= 1

    def test_specificity(self):
        """Test specificity calculation"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'Test')
        metrics = evaluator.calculate_metrics()
        
        assert 'Specificity' in metrics
        assert 0 <= metrics['Specificity'] <= 1

    def test_get_confusion_matrix(self, sample_predictions):
        """Test confusion matrix generation"""
        evaluator = ModelEvaluator(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_proba'],
            'Test'
        )
        cm = evaluator.get_confusion_matrix()
        
        assert cm.shape == (2, 2)
        assert cm[0][0] >= 0
        assert cm[1][1] >= 0

    def test_analyze_errors(self, sample_predictions):
        """Test error analysis"""
        evaluator = ModelEvaluator(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_proba'],
            'Test'
        )
        analysis = evaluator.analyze_errors()
        
        assert 'True Negatives (TN)' in analysis
        assert 'True Positives (TP)' in analysis
        assert 'False Positives (FP)' in analysis
        assert 'False Negatives (FN)' in analysis

    def test_get_classification_report(self, sample_predictions):
        """Test classification report generation"""
        evaluator = ModelEvaluator(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_proba'],
            'Test'
        )
        report = evaluator.get_classification_report()
        
        assert 'Legitimate' in report
        assert 'Fraud' in report

    def test_print_summary(self, sample_predictions, capsys):
        """Test print_summary runs without error"""
        evaluator = ModelEvaluator(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            sample_predictions['y_proba'],
            'Test'
        )
        evaluator.print_summary()
        
        captured = capsys.readouterr()
        assert 'Test' in captured.out

    def test_calculate_metrics_empty(self):
        """Test calculate_metrics handles edge case"""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_proba = np.array([0.1, 0.9])
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba, 'Edge')
        metrics = evaluator.calculate_metrics()
        
        assert 'Accuracy' in metrics


class TestCalculateBusinessMetrics:
    """Tests for business metrics calculation"""

    def test_calculate_business_metrics_basic(self):
        """Test basic business metrics calculation"""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1])
        
        metrics = calculate_business_metrics(y_true, y_pred)
        
        assert 'Fraud Detected' in metrics
        assert 'Fraud Missed' in metrics
        assert 'Legitimate Blocked' in metrics
        assert 'Net Benefit ($)' in metrics

    def test_fraud_detected_count(self):
        """Test fraud detected count"""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1])
        
        metrics = calculate_business_metrics(y_true, y_pred)
        
        assert metrics['Fraud Detected'] == 2
        assert metrics['Fraud Missed'] == 1

    def test_net_benefit_calculation(self):
        """Test net benefit calculation"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        metrics = calculate_business_metrics(y_true, y_pred, avg_fraud_amount=100)
        
        assert metrics['Amount Saved ($)'] == 200
        assert metrics['Net Benefit ($)'] == 200

    def test_net_loss_when_fraud_missed(self):
        """Test net loss when fraud is missed"""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 0])
        
        metrics = calculate_business_metrics(y_true, y_pred, avg_fraud_amount=100)
        
        assert metrics['Amount Lost ($)'] == 100
        assert metrics['Net Benefit ($)'] == -100


class TestCompareModels:
    """Tests for model comparison functions"""

    def test_compare_models(self):
        """Test compare_models function"""
        model_results = {
            'Model A': (
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 1, 1]),
                np.array([0.1, 0.2, 0.9, 0.8])
            ),
            'Model B': (
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 0, 1]),
                np.array([0.1, 0.2, 0.3, 0.9])
            ),
        }
        
        df = compare_models(model_results, ['Model A', 'Model B'])
        
        assert len(df) == 2
        assert 'Model' in df.columns
        assert 'F1 Score' in df.columns
        assert 'Model A' in df['Model'].values
        assert 'Model B' in df['Model'].values

    def test_select_best_model(self):
        """Test select_best_model function"""
        model_results = {
            'Model A': (
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 1, 1]),
                np.array([0.1, 0.2, 0.9, 0.8])
            ),
            'Model B': (
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 0, 1]),
                np.array([0.1, 0.2, 0.3, 0.9])
            ),
        }
        
        best_model, score = select_best_model(model_results, metric='F1 Score')
        
        assert best_model == 'Model A'
        assert score > 0

    def test_select_best_model_by_recall(self):
        """Test select_best_model by recall"""
        model_results = {
            'Model A': (
                np.array([0, 0, 1, 1]),
                np.array([0, 1, 1, 1]),
                np.array([0.1, 0.6, 0.9, 0.8])
            ),
            'Model B': (
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 1, 1]),
                np.array([0.1, 0.2, 0.9, 0.8])
            ),
        }
        
        best_model, score = select_best_model(model_results, metric='Recall')
        
        assert best_model == 'Model A'
        assert score == 1.0