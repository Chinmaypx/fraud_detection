const API_BASE = 'http://localhost:8000';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`);
  return res.json();
}

export async function getTrainingHistory() {
  const res = await fetch(`${API_BASE}/training-history`);
  return res.json();
}

export async function getMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);
  return res.json();
}

export async function predictFraud(transaction) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Prediction failed');
  }
  return res.json();
}

export async function trainModel() {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Training failed');
  }
  return res.json();
}
