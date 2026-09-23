import asyncio

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from pydantic import ValidationError

import api.app as api
from src.ieee_cis_dataset import IEEE_FEATURES, IEECISDataset
from src.predict import IEECISFraudDetector
from src.pytorch_model import FraudDetectorNet


def sample_ieee():
    return {
        "TransactionAmt": 125.50,
        "ProductCD": "W",
        "hour_of_day": 13,
        "card4": "visa",
        "card6": "debit",
        "addr1": "315",
        "addr2": "87",
        "dist1": 4.0,
        "P_emaildomain": "gmail.com",
        "DeviceType": "desktop",
    }


class FixedProbabilityModel(torch.nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.seen = None

    def forward(self, tensor):
        self.seen = tensor.detach().cpu().numpy()
        return torch.full((tensor.shape[0], 1), self.probability, device=tensor.device)


def make_fitted_detector(tmp_path, probability=0.9, threshold=0.8):
    adapter = IEECISDataset()
    adapter.feature_names = list(IEEE_FEATURES)
    adapter.numeric_medians = {"TransactionAmt": 100, "hour_of_day": 12, "dist1": 2}
    adapter.numeric_center = {"TransactionAmt": 5.0, "hour_of_day": 11.5, "dist1": 5.0}
    adapter.numeric_scale = {"TransactionAmt": 5.0, "hour_of_day": 11.5, "dist1": 5.0}
    adapter.category_frequencies = {
        name: {value: 1.0, "__MISSING__": 0.0}
        for name, value in {
            "ProductCD": "W", "card4": "visa", "card6": "debit",
            "addr1": "315", "addr2": "87", "P_emaildomain": "gmail.com",
            "DeviceType": "desktop",
        }.items()
    }
    adapter.category_values = {
        name: [value for value in values if value != "__MISSING__"]
        for name, values in adapter.category_frequencies.items()
    }
    adapter.save_preprocessing(tmp_path)

    detector = IEECISFraudDetector(model_path=str(tmp_path), device=torch.device("cpu"))
    detector.load_preprocessing()
    detector.model = FixedProbabilityModel(probability)
    detector.threshold = threshold
    return detector


def test_ieee_prediction_probability_threshold_and_feature_order(tmp_path):
    detector = make_fitted_detector(tmp_path, probability=0.81, threshold=0.8)
    row = sample_ieee()
    shuffled = dict(reversed(list(row.items())))
    result = detector.predict(shuffled)
    expected = detector.adapter.transform(__import__("pandas").DataFrame([row])).to_numpy()
    np.testing.assert_allclose(detector.model.seen, expected)
    assert result["model_name"] == "fraud_detector_ieee.pt"
    assert result["fraud_probability"] == pytest.approx(0.81)
    assert result["threshold"] == pytest.approx(0.8)
    assert result["predicted_class"] == 1
    assert result["is_fraud"] is True
    assert result["risk_level"] == "HIGH"


def test_ieee_threshold_boundary_is_consistent(tmp_path):
    below = make_fitted_detector(tmp_path / "below", probability=0.79, threshold=0.8)
    at = make_fitted_detector(tmp_path / "at", probability=0.8, threshold=0.8)
    assert below.predict(sample_ieee())["risk_level"] == "LOW"
    at_result = at.predict(sample_ieee())
    assert at_result["predicted_class"] == 1
    assert at_result["is_fraud"] is True
    assert at_result["risk_level"] == "HIGH"


def test_ieee_artifact_loader_checks_checkpoint_and_preprocessor(tmp_path):
    adapter = IEECISDataset()
    adapter.feature_names = list(IEEE_FEATURES)
    adapter.numeric_medians = {"TransactionAmt": 0, "hour_of_day": 0, "dist1": 0}
    adapter.numeric_center = {name: 0.0 for name in ("TransactionAmt", "hour_of_day", "dist1")}
    adapter.numeric_scale = {name: 1.0 for name in ("TransactionAmt", "hour_of_day", "dist1")}
    adapter.category_frequencies = {
        name: {"__MISSING__": 1.0} for name in (
            "ProductCD", "card4", "card6", "addr1", "addr2", "P_emaildomain", "DeviceType"
        )
    }
    adapter.category_values = {name: [] for name in adapter.category_frequencies}
    adapter.save_preprocessing(tmp_path)
    model = FraudDetectorNet(len(IEEE_FEATURES))
    torch.save({
        "input_dim": len(IEEE_FEATURES),
        "feature_names": IEEE_FEATURES,
        "threshold": 0.7,
        "model_state_dict": model.state_dict(),
    }, tmp_path / "fraud_detector_ieee.pt")
    detector = IEECISFraudDetector(model_path=str(tmp_path), device=torch.device("cpu"))
    detector.load_model()
    detector.load_preprocessing()
    assert detector.model is not None
    assert detector.threshold == pytest.approx(0.7)


def test_ieee_api_single_batch_and_model_info(monkeypatch, tmp_path):
    detector = make_fitted_detector(tmp_path)
    monkeypatch.setattr(api, "ieee_detector", detector)
    monkeypatch.setattr(api, "eval_metrics_ieee", {"test": {"metrics": {"F1 Score": 0.7}}})
    request = api.IEEETransaction(**sample_ieee())

    single = asyncio.run(api.predict_fraud_ieee(request))
    assert single.model_name == "fraud_detector_ieee.pt"
    assert single.predicted_class == 1
    assert single.is_fraud is True

    batch = asyncio.run(api.batch_predict_fraud_ieee(
        api.IEEEBatchPredictionRequest(transactions=[request, request])
    ))
    assert batch.total_transactions == 2
    assert batch.flagged_count == 2

    info = asyncio.run(api.ieee_model_info())
    assert info["dataset"] == "IEEE-CIS Fraud Detection"
    assert info["features_used"] == IEEE_FEATURES
    assert info["metrics"]["test"]["metrics"]["F1 Score"] == 0.7


def test_ieee_api_rejects_unknown_categories_and_unavailable_model(monkeypatch, tmp_path):
    detector = make_fitted_detector(tmp_path)
    monkeypatch.setattr(api, "ieee_detector", detector)
    row = sample_ieee()
    row["ProductCD"] = "unknown"
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.predict_fraud_ieee(api.IEEETransaction(**row)))
    assert error.value.status_code == 422
    assert "ProductCD" in error.value.detail

    monkeypatch.setattr(api, "ieee_detector", None)
    with pytest.raises(HTTPException) as missing:
        asyncio.run(api.predict_fraud_ieee(api.IEEETransaction(**sample_ieee())))
    assert missing.value.status_code == 503


def test_ieee_api_schema_rejects_nonfinite_long_and_extra_fields():
    row = sample_ieee()
    row["dist1"] = float("inf")
    with pytest.raises(ValidationError):
        api.IEEETransaction(**row)
    row = sample_ieee()
    row["P_emaildomain"] = "x" * 129
    with pytest.raises(ValidationError):
        api.IEEETransaction(**row)
    row = sample_ieee()
    row["V1"] = 0
    with pytest.raises(ValidationError):
        api.IEEETransaction(**row)
    row = sample_ieee()
    row["hour_of_day"] = 24
    with pytest.raises(ValidationError):
        api.IEEETransaction(**row)


def test_ieee_batch_size_remains_capped():
    request = api.IEEETransaction(**sample_ieee())
    with pytest.raises(ValidationError):
        api.IEEEBatchPredictionRequest(transactions=[request] * 1001)
