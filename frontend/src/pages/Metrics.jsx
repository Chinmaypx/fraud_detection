import { useEffect, useState } from 'react';
import { getMetrics, getTrainingHistory } from '../api';

export default function Metrics() {
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getMetrics().catch(() => null),
      getTrainingHistory().catch(() => null),
    ]).then(([m, h]) => {
      setMetrics(m?.message ? null : m);
      setHistory(h?.message ? null : h);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner"></div>
        <span>Loading metrics...</span>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div>
        <div className="page-header">
          <h2>📈 Model Metrics</h2>
          <p>Detailed evaluation of the trained model</p>
        </div>
        <div className="card">
          <div className="empty-state">
            <div className="icon">📊</div>
            <p>No metrics available</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              Train a model first to see evaluation metrics
            </p>
          </div>
        </div>
      </div>
    );
  }

  const metricsList = [
    { key: 'Accuracy', desc: 'Overall correct predictions', color: '#3b82f6' },
    { key: 'Precision', desc: 'Of flagged fraud, how many were real', color: '#8b5cf6' },
    { key: 'Recall (Sensitivity)', desc: 'Of real fraud, how many did we catch', color: '#f59e0b' },
    { key: 'Specificity', desc: 'True negative rate', color: '#06b6d4' },
    { key: 'F1 Score', desc: 'Balance of precision & recall', color: '#10b981' },
    { key: 'ROC-AUC', desc: 'Discrimination ability', color: '#ec4899' },
    { key: 'PR-AUC', desc: 'Best for imbalanced data', color: '#f97316' },
  ];

  return (
    <div>
      <div className="page-header">
        <h2>📈 Model Metrics</h2>
        <p>Detailed evaluation of the PyTorch neural network</p>
      </div>

      {/* Metrics Cards */}
      <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
        {metricsList.map((m, i) => {
          const val = metrics[m.key];
          if (val === undefined) return null;
          return (
            <div key={m.key} className={`stat-card animate-in animate-in-delay-${Math.min(i + 1, 4)}`}>
              <div className="stat-label">{m.key}</div>
              <div className="stat-value" style={{ color: m.color }}>
                {(val * 100).toFixed(1)}%
              </div>
              <div className="stat-delta">{m.desc}</div>
              <div className="progress-bar-container" style={{ marginTop: '12px' }}>
                <div
                  className="progress-bar-fill"
                  style={{ width: `${val * 100}%`, background: m.color }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid-2" style={{ marginTop: 'var(--space-xl)' }}>
        {/* Metrics Table */}
        <div className="card animate-in">
          <div className="card-header">
            <span className="card-title">📋 Detailed Metrics</span>
          </div>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Score</th>
                <th style={{ width: '40%' }}>Performance</th>
              </tr>
            </thead>
            <tbody>
              {metricsList.map((m) => {
                const val = metrics[m.key];
                if (val === undefined) return null;
                return (
                  <tr key={m.key}>
                    <td>
                      <div style={{ fontWeight: 500 }}>{m.key}</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{m.desc}</div>
                    </td>
                    <td>
                      <span className="metric-value" style={{ color: m.color }}>
                        {(val * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td>
                      <div className="metric-bar">
                        <div className="metric-bar-track">
                          <div
                            className="metric-bar-fill"
                            style={{ width: `${val * 100}%`, background: m.color }}
                          ></div>
                        </div>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* F1 Chart */}
        <div className="card animate-in animate-in-delay-1">
          <div className="card-header">
            <span className="card-title">📈 F1 Score Over Training</span>
            {history && <span className="card-subtitle">{history.train_f1?.length} epochs</span>}
          </div>
          {history?.train_f1 ? (
            <F1Chart trainF1={history.train_f1} valF1={history.val_f1} />
          ) : (
            <div className="chart-placeholder">
              <span style={{ fontSize: '2rem', marginBottom: '8px' }}>📈</span>
              <span>No training history available</span>
            </div>
          )}
        </div>
      </div>

      {/* Training Performance Graphs */}
      {history && (
        <>
          <div className="grid-2" style={{ marginTop: 'var(--space-xl)' }}>
            <div className="card animate-in">
              <div className="card-header">
                <span className="card-title">📉 Training & Validation Loss</span>
                <span className="card-subtitle">Lower is better</span>
              </div>
              {history?.train_loss ? (
                <LossChart trainLoss={history.train_loss} valLoss={history.val_loss} />
              ) : (
                <div className="chart-placeholder"><span>No data</span></div>
              )}
            </div>

            <div className="card animate-in animate-in-delay-1">
              <div className="card-header">
                <span className="card-title">🎯 Training & Validation Accuracy</span>
                <span className="card-subtitle">Higher is better</span>
              </div>
              {history?.train_acc ? (
                <AccuracyChart trainAcc={history.train_acc} valAcc={history.val_acc} />
              ) : (
                <div className="chart-placeholder"><span>No data</span></div>
              )}
            </div>
          </div>

          <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
            <div className="card-header">
              <span className="card-title">📊 Precision & Recall Over Training</span>
              <span className="card-subtitle">Recall = fraud detection rate | Precision = accuracy of fraud flags</span>
            </div>
            {history?.train_precision ? (
              <PrecisionRecallChart 
                trainPrec={history.train_precision} 
                valPrec={history.val_precision}
                trainRec={history.train_recall}
                valRec={history.val_recall}
              />
            ) : (
              <div className="chart-placeholder"><span>No data</span></div>
            )}
          </div>
        </>
      )}

      {/* Why These Metrics Matter */}
      <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title">💡 Understanding the Metrics</span>
        </div>
        <div className="grid-2" style={{ gap: 'var(--space-lg)' }}>
          <div>
            <h4 style={{ color: 'var(--danger)', marginBottom: '8px', fontSize: '0.9rem' }}>
              🚨 False Negatives (Missed Fraud) — MOST DANGEROUS
            </h4>
            <ul style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', lineHeight: 1.8, paddingLeft: '16px' }}>
              <li>Direct financial loss to the bank & customer</li>
              <li>Reputational damage</li>
              <li>Regulatory penalties</li>
              <li>Measured by <strong style={{ color: 'var(--text-primary)' }}>Recall</strong></li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: 'var(--warning)', marginBottom: '8px', fontSize: '0.9rem' }}>
              ⚠️ False Positives (False Alarms)
            </h4>
            <ul style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', lineHeight: 1.8, paddingLeft: '16px' }}>
              <li>Customer frustration</li>
              <li>Transaction declined at checkout</li>
              <li>Customer support costs</li>
              <li>Measured by <strong style={{ color: 'var(--text-primary)' }}>Precision</strong></li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function F1Chart({ trainF1, valF1 }) {
  if (!trainF1 || trainF1.length === 0) return null;

  const width = 500;
  const height = 220;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const allValues = [...trainF1, ...valF1];
  const maxVal = Math.min(Math.max(...allValues) * 1.1, 1);
  const minVal = Math.max(Math.min(...allValues) * 0.9, 0);

  const toX = (i) => pad.left + (i / (trainF1.length - 1)) * chartW;
  const toY = (v) => pad.top + ((maxVal - v) / (maxVal - minVal)) * chartH;

  const makePath = (data) =>
    data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="svg-chart" style={{ width: '100%', height: '220px' }}>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = pad.top + f * chartH;
        const val = maxVal - f * (maxVal - minVal);
        return (
          <g key={f}>
            <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} className="chart-grid-line" />
            <text x={pad.left - 8} y={y + 3} textAnchor="end" className="chart-label">{val.toFixed(2)}</text>
          </g>
        );
      })}
      <path d={makePath(trainF1)} className="chart-line" stroke="#10b981" />
      <path d={makePath(valF1)} className="chart-line" stroke="#f59e0b" />
      <circle cx={pad.left + 10} cy={height - 8} r={4} fill="#10b981" />
      <text x={pad.left + 20} y={height - 4} className="chart-label">Train F1</text>
      <circle cx={pad.left + 90} cy={height - 8} r={4} fill="#f59e0b" />
      <text x={pad.left + 100} y={height - 4} className="chart-label">Val F1</text>
    </svg>
  );
}

function LossChart({ trainLoss, valLoss }) {
  if (!trainLoss || trainLoss.length === 0) return null;

  const width = 500;
  const height = 220;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const allValues = [...trainLoss, ...valLoss];
  const maxVal = Math.min(Math.max(...allValues) * 1.1, 1);
  const minVal = Math.max(Math.min(...allValues) * 0.9, 0);

  const toX = (i) => pad.left + (i / (trainLoss.length - 1)) * chartW;
  const toY = (v) => pad.top + ((maxVal - v) / (maxVal - minVal)) * chartH;

  const makePath = (data) =>
    data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="svg-chart" style={{ width: '100%', height: '220px' }}>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = pad.top + f * chartH;
        const val = maxVal - f * (maxVal - minVal);
        return (
          <g key={f}>
            <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} className="chart-grid-line" />
            <text x={pad.left - 8} y={y + 3} textAnchor="end" className="chart-label">{val.toFixed(2)}</text>
          </g>
        );
      })}
      <path d={makePath(trainLoss)} className="chart-line" stroke="#3b82f6" />
      <path d={makePath(valLoss)} className="chart-line" stroke="#ef4444" />
      <circle cx={pad.left + 10} cy={height - 8} r={4} fill="#3b82f6" />
      <text x={pad.left + 20} y={height - 4} className="chart-label">Train Loss</text>
      <circle cx={pad.left + 100} cy={height - 8} r={4} fill="#ef4444" />
      <text x={pad.left + 110} y={height - 4} className="chart-label">Val Loss</text>
    </svg>
  );
}

function AccuracyChart({ trainAcc, valAcc }) {
  if (!trainAcc || trainAcc.length === 0) return null;

  const width = 500;
  const height = 220;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const allValues = [...trainAcc, ...valAcc];
  const maxVal = Math.min(Math.max(...allValues) * 1.05, 1);
  const minVal = Math.max(Math.min(...allValues) * 0.95, 0);

  const toX = (i) => pad.left + (i / (trainAcc.length - 1)) * chartW;
  const toY = (v) => pad.top + ((maxVal - v) / (maxVal - minVal)) * chartH;

  const makePath = (data) =>
    data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="svg-chart" style={{ width: '100%', height: '220px' }}>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = pad.top + f * chartH;
        const val = maxVal - f * (maxVal - minVal);
        return (
          <g key={f}>
            <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} className="chart-grid-line" />
            <text x={pad.left - 8} y={y + 3} textAnchor="end" className="chart-label">{val.toFixed(2)}</text>
          </g>
        );
      })}
      <path d={makePath(trainAcc)} className="chart-line" stroke="#8b5cf6" />
      <path d={makePath(valAcc)} className="chart-line" stroke="#06b6d4" />
      <circle cx={pad.left + 10} cy={height - 8} r={4} fill="#8b5cf6" />
      <text x={pad.left + 20} y={height - 4} className="chart-label">Train Acc</text>
      <circle cx={pad.left + 90} cy={height - 8} r={4} fill="#06b6d4" />
      <text x={pad.left + 100} y={height - 4} className="chart-label">Val Acc</text>
    </svg>
  );
}

function PrecisionRecallChart({ trainPrec, valPrec, trainRec, valRec }) {
  if (!trainPrec || trainPrec.length === 0) return null;

  const width = 500;
  const height = 220;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const allValues = [...trainPrec, ...valPrec, ...trainRec, ...valRec];
  const maxVal = Math.min(Math.max(...allValues) * 1.1, 1);
  const minVal = Math.max(Math.min(...allValues) * 0.9, 0);

  const toX = (i) => pad.left + (i / (trainPrec.length - 1)) * chartW;
  const toY = (v) => pad.top + ((maxVal - v) / (maxVal - minVal)) * chartH;

  const makePath = (data) =>
    data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="svg-chart" style={{ width: '100%', height: '220px' }}>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = pad.top + f * chartH;
        const val = maxVal - f * (maxVal - minVal);
        return (
          <g key={f}>
            <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} className="chart-grid-line" />
            <text x={pad.left - 8} y={y + 3} textAnchor="end" className="chart-label">{val.toFixed(2)}</text>
          </g>
        );
      })}
      <path d={makePath(trainPrec)} className="chart-line" stroke="#f97316" />
      <path d={makePath(valPrec)} className="chart-line" stroke="#ec4899" />
      <path d={makePath(trainRec)} className="chart-line" stroke="#22c55e" strokeDasharray="4" />
      <path d={makePath(valRec)} className="chart-line" stroke="#14b8a6" strokeDasharray="4" />
      <circle cx={pad.left + 10} cy={height - 18} r={3} fill="#f97316" />
      <text x={pad.left + 18} y={height - 14} className="chart-label" style={{ fontSize: '10px' }}>Train Precision</text>
      <circle cx={pad.left + 110} cy={height - 18} r={3} fill="#ec4899" />
      <text x={pad.left + 118} y={height - 14} className="chart-label" style={{ fontSize: '10px' }}>Val Precision</text>
      <circle cx={pad.left + 10} cy={height - 8} r={3} fill="#22c55e" />
      <text x={pad.left + 18} y={height - 4} className="chart-label" style={{ fontSize: '10px' }}>Train Recall</text>
      <circle cx={pad.left + 90} cy={height - 8} r={3} fill="#14b8a6" />
      <text x={pad.left + 98} y={height - 4} className="chart-label" style={{ fontSize: '10px' }}>Val Recall</text>
    </svg>
  );
}
