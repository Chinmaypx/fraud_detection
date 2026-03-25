import { useEffect, useState } from 'react';
import { getMetrics, getTrainingHistory, getModelInfo } from '../api';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getMetrics().catch(() => null),
      getTrainingHistory().catch(() => null),
      getModelInfo().catch(() => null),
    ]).then(([m, h, mi]) => {
      setMetrics(m?.message ? null : m);
      setHistory(h?.message ? null : h);
      setModelInfo(mi?.message ? null : mi);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner"></div>
        <span>Loading dashboard...</span>
      </div>
    );
  }

  const statCards = metrics
    ? [
        { label: 'Accuracy', value: (metrics['Accuracy'] * 100).toFixed(1) + '%', type: 'neutral', delta: 'Overall correct predictions' },
        { label: 'F1 Score', value: (metrics['F1 Score'] * 100).toFixed(1) + '%', type: 'positive', delta: 'Harmonic mean of P & R' },
        { label: 'Recall', value: (metrics['Recall (Sensitivity)'] * 100).toFixed(1) + '%', type: 'warning', delta: 'Fraud catch rate' },
        { label: 'ROC-AUC', value: (metrics['ROC-AUC'] * 100).toFixed(1) + '%', type: 'neutral', delta: 'Discrimination ability' },
      ]
    : [
        { label: 'Accuracy', value: '—', type: 'neutral', delta: 'Train model first' },
        { label: 'F1 Score', value: '—', type: 'neutral', delta: 'Train model first' },
        { label: 'Recall', value: '—', type: 'neutral', delta: 'Train model first' },
        { label: 'ROC-AUC', value: '—', type: 'neutral', delta: 'Train model first' },
      ];

  return (
    <div>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Overview of your PyTorch fraud detection model performance</p>
      </div>

      {/* Stat Cards */}
      <div className="stats-grid">
        {statCards.map((stat, i) => (
          <div key={stat.label} className={`stat-card animate-in animate-in-delay-${i + 1}`}>
            <div className="stat-label">{stat.label}</div>
            <div className={`stat-value ${stat.type}`}>{stat.value}</div>
            <div className="stat-delta">{stat.delta}</div>
          </div>
        ))}
      </div>

      <div className="grid-2">
        {/* Training Loss Chart */}
        <div className="card animate-in animate-in-delay-2">
          <div className="card-header">
            <span className="card-title">📉 Training Progress</span>
            {history && <span className="card-subtitle">{history.train_loss?.length || 0} epochs</span>}
          </div>
          {history ? (
            <LossChart trainLoss={history.train_loss} valLoss={history.val_loss} />
          ) : (
            <div className="chart-placeholder">
              <span style={{ fontSize: '2rem', marginBottom: '8px' }}>📊</span>
              <span>Train a model to see loss curves</span>
            </div>
          )}
        </div>

        {/* Model Architecture */}
        <div className="card animate-in animate-in-delay-3">
          <div className="card-header">
            <span className="card-title">🧠 Model Architecture</span>
            <span className="card-subtitle">
              {modelInfo ? `${modelInfo.total_parameters?.toLocaleString()} params` : 'N/A'}
            </span>
          </div>
          {modelInfo && !modelInfo.message ? (
            <div className="layer-list">
              {[
                'Input → Linear(128) + BatchNorm + ReLU + Dropout',
                'Linear(128 → 256) + BatchNorm + ReLU + Dropout',
                'Linear(256 → 128) + BatchNorm + ReLU + Dropout',
                'Linear(128 → 64) + BatchNorm + ReLU + Dropout',
                'Linear(64 → 32) + ReLU + Dropout',
                'Linear(32 → 1) + Sigmoid → Output',
              ].map((layer, i) => (
                <div key={i} className="layer-item">
                  <span className="layer-dot"></span>
                  {layer}
                </div>
              ))}
            </div>
          ) : (
            <div className="chart-placeholder">
              <span style={{ fontSize: '2rem', marginBottom: '8px' }}>🧠</span>
              <span>No model loaded yet</span>
            </div>
          )}
        </div>
      </div>

      {/* System Architecture */}
      <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title">⚡ System Architecture</span>
        </div>
        <div className="architecture-flow">
          <div className="arch-node">
            <div className="node-icon">💳</div>
            <div>Transaction</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node">
            <div className="node-icon">⚙️</div>
            <div>Feature Engine</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node">
            <div className="node-icon">📐</div>
            <div>Scaler</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node">
            <div className="node-icon">🧠</div>
            <div>PyTorch DNN</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node">
            <div className="node-icon">📊</div>
            <div>Probability</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node">
            <div className="node-icon">🚨</div>
            <div>Decision</div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* Simple SVG line chart for training loss */
function LossChart({ trainLoss, valLoss }) {
  if (!trainLoss || trainLoss.length === 0) return null;

  const width = 500;
  const height = 220;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const allValues = [...trainLoss, ...valLoss];
  const maxVal = Math.max(...allValues) * 1.1;
  const minVal = Math.min(...allValues) * 0.9;

  const toX = (i) => pad.left + (i / (trainLoss.length - 1)) * chartW;
  const toY = (v) => pad.top + ((maxVal - v) / (maxVal - minVal)) * chartH;

  const makePath = (data) =>
    data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="svg-chart" style={{ width: '100%', height: '220px' }}>
      {/* Grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = pad.top + f * chartH;
        const val = maxVal - f * (maxVal - minVal);
        return (
          <g key={f}>
            <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} className="chart-grid-line" />
            <text x={pad.left - 8} y={y + 3} textAnchor="end" className="chart-label">{val.toFixed(3)}</text>
          </g>
        );
      })}

      {/* Train loss line */}
      <path d={makePath(trainLoss)} className="chart-line" stroke="#3b82f6" />
      {/* Val loss line */}
      <path d={makePath(valLoss)} className="chart-line" stroke="#f59e0b" />

      {/* Legend */}
      <circle cx={pad.left + 10} cy={height - 8} r={4} fill="#3b82f6" />
      <text x={pad.left + 20} y={height - 4} className="chart-label">Train Loss</text>
      <circle cx={pad.left + 100} cy={height - 8} r={4} fill="#f59e0b" />
      <text x={pad.left + 110} y={height - 4} className="chart-label">Val Loss</text>
    </svg>
  );
}
