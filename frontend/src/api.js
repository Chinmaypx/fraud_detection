const API_BASE = 'http://localhost:8000';

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error('Could not reach the fraud detection API. Check that the backend is running.');
  }
  let data;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    throw new Error(data?.detail || data?.message || `API request failed (${response.status})`);
  }
  return data;
}

function postJson(path, payload) {
  return requestJson(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function checkHealth() {
  return requestJson('/health');
}

export async function getModelInfo() {
  return requestJson('/model-info');
}

export async function getIEEEModelInfo() {
  return requestJson('/model-info-ieee');
}

export async function getLSTMModelInfo() {
  return requestJson('/model-info-lstm');
}

export async function getTrainingHistory() {
  return requestJson('/training-history');
}

export async function getLSTMTrainingHistory() {
  return requestJson('/training-history-lstm');
}

export async function getMetrics() {
  return requestJson('/metrics');
}

export async function getLSTMMetrics() {
  return requestJson('/metrics-lstm');
}

export async function predictFraud(transaction) {
  return postJson('/predict', transaction);
}

export async function predictFraudIEEE(transaction) {
  return postJson('/predict-ieee', transaction);
}

export async function predictFraudBatchIEEE(transactions) {
  return postJson('/batch-predict-ieee', { transactions });
}

export async function predictFraudLSTM(transaction) {
  return postJson('/predict-lstm', transaction);
}

export async function trainModel() {
  return requestJson('/train', { method: 'POST' });
}

export async function trainLSTMModel() {
  return requestJson('/train-lstm', { method: 'POST' });
}

