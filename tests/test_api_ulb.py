import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sklearn.preprocessing import RobustScaler

from api import app as api
from src.predict import ULBFraudDetector
from src.ulb_dataset import ULB_FEATURES


def sample_ulb():
    return {name: float(index + 1) for index, name in enumerate(ULB_FEATURES)}


class FixedProbabilityModel:
    def __init__(self, probability):
        self.probability = probability
        self.seen = None

    def eval(self):
        return self

    def __call__(self, tensor):
        self.seen = tensor.detach().cpu().numpy()
        return torch.tensor([self.probability])


def make_detector(tmp_path, probability=0.98):
    scaler = RobustScaler().fit(np.arange(90, dtype=float).reshape(3, 30))
    detector = ULBFraudDetector(model_path=str(tmp_path), device=torch.device('cpu'))
    detector.adapter.scaler = scaler
    detector.adapter.save_preprocessing(tmp_path)
    detector.load_preprocessing()
    detector.model = FixedProbabilityModel(probability)
    detector.threshold = 0.9771430492401123
    return detector


def test_ulb_prediction_uses_saved_preprocessing_order_threshold_and_probability(tmp_path):
    detector = make_detector(tmp_path, probability=0.98)
    row = sample_ulb()
    reversed_row = dict(reversed(list(row.items())))

    result = detector.predict(reversed_row)

    expected = detector.adapter.transform(__import__('pandas').DataFrame([row])).to_numpy()
    np.testing.assert_allclose(detector.model.seen, expected.astype(np.float32))
    assert result['fraud_probability'] == pytest.approx(0.98)
    assert result['threshold'] == 0.9771430492401123
    assert result['is_fraud'] is True
    assert result['predicted_class'] == 1


def test_ulb_threshold_is_applied_at_saved_cutoff(tmp_path):
    below = make_detector(tmp_path / 'below', probability=0.97714)
    above = make_detector(tmp_path / 'above', probability=0.97715)
    below_result = below.predict(sample_ulb())
    above_result = above.predict(sample_ulb())
    assert below_result['predicted_class'] == 0
    assert below_result['is_fraud'] is False
    assert below_result['risk_level'] == 'LOW'
    assert above_result['predicted_class'] == 1
    assert above_result['is_fraud'] is True
    assert above_result['risk_level'] == 'HIGH'


def test_ulb_schema_rejects_missing_invalid_and_extra_features():
    row = sample_ulb()
    del row['V4']
    with pytest.raises(ValidationError):
        api.ULBTransaction(**row)
    row = sample_ulb()
    row['V4'] = float('nan')
    with pytest.raises(ValidationError):
        api.ULBTransaction(**row)
    row = sample_ulb()
    row['customer_id'] = 'synthetic-only'
    with pytest.raises(ValidationError):
        api.ULBTransaction(**row)


def test_api_request_schemas_reject_nonfinite_values_and_oversized_batches():
    synthetic = {
        'transaction_amount': float('inf'), 'transaction_time': 1,
        'location': 'X', 'device_id': 'D', 'merchant_category': 'retail',
        'account_age_days': 1, 'transaction_count_24h': 1,
        'avg_transaction_amount': 1.0,
    }
    with pytest.raises(ValidationError):
        api.Transaction(**synthetic)

    ulb_row = api.ULBTransaction(**sample_ulb())
    with pytest.raises(ValidationError):
        api.ULBBatchPredictionRequest(transactions=[])
    with pytest.raises(ValidationError):
        api.ULBBatchPredictionRequest(transactions=[ulb_row] * 1001)

    with pytest.raises(ValidationError):
        api.BatchPredictionRequest(transactions=[])
    with pytest.raises(ValidationError):
        api.BatchPredictionRequest(transactions=[api.Transaction(
            transaction_amount=1, transaction_time=1, location='X', device_id='D',
            merchant_category='retail', account_age_days=1, transaction_count_24h=1,
            avg_transaction_amount=1,
        )] * 1001)
def test_ulb_api_prediction_and_batch_and_model_selection(monkeypatch):
    class StubULB:
        model = object()
        threshold = 0.9771430492401123
        feature_names = list(ULB_FEATURES)

        def predict(self, data):
            return {
                'is_fraud': data['V1'] > 0,
                'predicted_class': int(data['V1'] > 0),
                'fraud_probability': 0.99 if data['V1'] > 0 else 0.1,
                'threshold': self.threshold,
                'risk_level': 'HIGH' if data['V1'] > 0 else 'LOW',
            }

    class StubSynthetic:
        model = object()

        def predict(self, data):
            return {'is_fraud': False, 'fraud_probability': 0.1, 'risk_level': 'LOW'}

    monkeypatch.setattr(api, 'ulb_detector', StubULB())
    monkeypatch.setattr(api, 'detector', StubSynthetic())
    request = api.ULBTransaction(**sample_ulb())
    single = asyncio.run(api.predict_fraud_ulb(request))
    assert single.model_name == 'fraud_detector_ulb.pt'
    assert single.fraud_probability == 0.99
    assert single.predicted_class == 1

    batch = asyncio.run(api.batch_predict_fraud_ulb(
        api.ULBBatchPredictionRequest(transactions=[request, request])
    ))
    assert batch.total_transactions == 2
    assert batch.flagged_count == 2

    info = asyncio.run(api.ulb_model_info())
    assert info['dataset'].startswith('ULB/')
    assert info['feature_count'] == 30
    assert info['threshold'] == 0.9771430492401123

    synthetic_request = api.Transaction(**{
        'transaction_amount': 10, 'transaction_time': 10, 'location': 'X',
        'device_id': 'D', 'merchant_category': 'retail', 'account_age_days': 1,
        'transaction_count_24h': 1, 'avg_transaction_amount': 10,
    })
    old_route_result = asyncio.run(api.predict_fraud(synthetic_request))
    assert old_route_result.is_fraud is False


def test_api_failures_are_clear_without_returning_internal_exception(monkeypatch):
    class BrokenDetector:
        model = object()

        def predict(self, *_args, **_kwargs):
            raise RuntimeError(r'local artifact path D:\private\model.pkl')

    monkeypatch.setattr(api, 'ulb_detector', BrokenDetector())
    with pytest.raises(HTTPException) as ulb_error:
        asyncio.run(api.predict_fraud_ulb(api.ULBTransaction(**sample_ulb())))
    assert ulb_error.value.status_code == 500
    assert 'D:\\private' not in ulb_error.value.detail

    monkeypatch.setattr(api, 'detector', BrokenDetector())
    synthetic_request = api.Transaction(
        transaction_amount=1, transaction_time=1, location='X', device_id='D',
        merchant_category='retail', account_age_days=1, transaction_count_24h=1,
        avg_transaction_amount=1,
    )
    with pytest.raises(HTTPException) as synthetic_error:
        asyncio.run(api.predict_fraud(synthetic_request))
    assert synthetic_error.value.status_code == 500
    assert 'D:\\private' not in synthetic_error.value.detail

    monkeypatch.setattr(api, 'ulb_detector', None)
    with pytest.raises(HTTPException) as unavailable:
        asyncio.run(api.predict_fraud_ulb(api.ULBTransaction(**sample_ulb())))
    assert unavailable.value.status_code == 503


def test_validation_error_response_does_not_echo_input_or_context():
    error = RequestValidationError([{
        'loc': ('body', 'V2'), 'msg': 'Input should be a valid number',
        'type': 'float_parsing', 'input': 'submitted-sensitive-value',
        'ctx': {'internal': 'context'},
    }])
    response = asyncio.run(api.validation_error_handler(None, error))
    body = response.body.decode()
    assert response.status_code == 422
    assert 'submitted-sensitive-value' not in body
    assert 'internal' not in body
    assert 'V2' in body


def test_synthetic_artifact_loader_rejects_missing_or_mismatched_preprocessing():
    valid = SimpleNamespace(
        scaler=SimpleNamespace(n_features_in_=2), feature_names=['a', 'b']
    )
    api._validate_loaded_preprocessing(valid, model_input_dim=2)

    with pytest.raises(RuntimeError, match='scaler'):
        api._validate_loaded_preprocessing(
            SimpleNamespace(scaler=None, feature_names=['a', 'b']), model_input_dim=2
        )
    with pytest.raises(RuntimeError, match='dimensions'):
        api._validate_loaded_preprocessing(
            SimpleNamespace(
                scaler=SimpleNamespace(n_features_in_=1), feature_names=['a', 'b']
            ),
            model_input_dim=2,
        )
