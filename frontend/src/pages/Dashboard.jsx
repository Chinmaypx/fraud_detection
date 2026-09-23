import { useEffect, useState } from 'react';
import { getMetrics, getLSTMMetrics, getTrainingHistory, getLSTMTrainingHistory, getModelInfo, getLSTMModelInfo, getULBModelInfo } from '../api';
import BenchmarkRecords, { ULBMetricDetails } from '../components/BenchmarkRecords';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [lstmMetrics, setLstmMetrics] = useState(null);
  const [history, setHistory] = useState(null);
  const [lstmHistory, setLstmHistory] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [lstmModelInfo, setLstmModelInfo] = useState(null);
  const [ulbModelInfo, setUlbModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getMetrics().catch(() => null),
      getLSTMMetrics().catch(() => null),
      getTrainingHistory().catch(() => null),
      getLSTMTrainingHistory().catch(() => null),
      getModelInfo().catch(() => null),
      getLSTMModelInfo().catch(() => null),
      getULBModelInfo().catch(() => null),
    ]).then(([m, lm, h, lh, mi, lmi, umi]) => {
      setMetrics(m?.message ? null : m);
      setLstmMetrics(lm?.message ? null : lm);
      setHistory(h?.message ? null : h);
      setLstmHistory(lh?.message ? null : lh);
      setModelInfo(mi?.message ? null : mi);
      setLstmModelInfo(lmi?.message ? null : lmi);
      setUlbModelInfo(umi?.message ? null : umi);
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

  const makeStatCard = (label, mlpVal, lstmVal, delta) => ({
    label,
    mlp: mlpVal != null ? (mlpVal * 100).toFixed(1) + '%' : '—',
    lstm: lstmVal != null ? (lstmVal * 100).toFixed(1) + '%' : '—',
    mlpRaw: mlpVal,
    lstmRaw: lstmVal,
    delta,
  });

  const comparisonCards = [
    makeStatCard('Accuracy', metrics?.['Accuracy'], lstmMetrics?.['Accuracy'], 'Overall correct predictions'),
    makeStatCard('F1 Score', metrics?.['F1 Score'], lstmMetrics?.['F1 Score'], 'Harmonic mean of P & R'),
    makeStatCard('Recall', metrics?.['Recall (Sensitivity)'], lstmMetrics?.['Recall (Sensitivity)'], 'Fraud catch rate'),
    makeStatCard('Precision', metrics?.['Precision'], lstmMetrics?.['Precision'], 'Accuracy of fraud flags'),
    makeStatCard('ROC-AUC', metrics?.['ROC-AUC'], lstmMetrics?.['ROC-AUC'], 'Discrimination ability'),
    makeStatCard('PR-AUC', metrics?.['PR-AUC'], lstmMetrics?.['PR-AUC'], 'Best for imbalanced data'),
  ];

  const mlpAccuracy = metrics?.['Accuracy'];
  const lstmAccuracy = lstmMetrics?.['Accuracy'];

  return (
    <div>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Separate views of the synthetic models and the ULB real-world benchmark</p>
      </div>

      {/* Accuracy Score Hero Cards */}
      <div className="grid-2" style={{ marginBottom: 'var(--space-xl)' }}>
        {/* MLP Accuracy Card */}
        <div className="card animate-in" style={{ position: 'relative', overflow: 'hidden' }}>
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, height: '4px',
            background: 'linear-gradient(90deg, #3b82f6, #60a5fa)',
          }}></div>
          <div className="card-header">
            <span className="card-title">Synthetic MLP Accuracy</span>
            <span className="model-tag mlp">MLP</span>
          </div>
          <div style={{ textAlign: 'center', padding: 'var(--space-lg) 0' }}>
            <div style={{
              fontSize: '3.5rem', fontWeight: 800,
              background: 'linear-gradient(135deg, #3b82f6, #60a5fa)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              lineHeight: 1.1,
            }}>
              {mlpAccuracy != null ? (mlpAccuracy * 100).toFixed(2) + '%' : '—'}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              Overall Correct Predictions
            </div>
            {mlpAccuracy != null && (
              <div className="progress-bar-container" style={{ marginTop: 'var(--space-md)', height: '10px' }}>
                <div className="progress-bar-fill" style={{
                  width: `${mlpAccuracy * 100}%`,
                  background: 'linear-gradient(90deg, #3b82f6, #60a5fa)',
                  transition: 'width 1.2s cubic-bezier(0.4, 0, 0.2, 1)',
                }}></div>
              </div>
            )}
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '6px',
              marginTop: 'var(--space-md)', padding: '4px 12px',
              background: 'rgba(59, 130, 246, 0.1)', borderRadius: '99px',
              fontSize: '0.75rem', color: '#3b82f6',
            }}>
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#3b82f6' }}></span>
              {mlpAccuracy != null ? 'Model Active' : 'Not Trained'}
            </div>
          </div>
        </div>

        {/* LSTM Accuracy Card */}
        <div className="card animate-in animate-in-delay-1" style={{ position: 'relative', overflow: 'hidden' }}>
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, height: '4px',
            background: 'linear-gradient(90deg, #8b5cf6, #a78bfa)',
          }}></div>
          <div className="card-header">
            <span className="card-title">Synthetic LSTM Accuracy</span>
            <span className="model-tag lstm">LSTM</span>
          </div>
          <div style={{ textAlign: 'center', padding: 'var(--space-lg) 0' }}>
            <div style={{
              fontSize: '3.5rem', fontWeight: 800,
              background: 'linear-gradient(135deg, #8b5cf6, #a78bfa)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              lineHeight: 1.1,
            }}>
              {lstmAccuracy != null ? (lstmAccuracy * 100).toFixed(2) + '%' : '—'}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              Overall Correct Predictions
            </div>
            {lstmAccuracy != null && (
              <div className="progress-bar-container" style={{ marginTop: 'var(--space-md)', height: '10px' }}>
                <div className="progress-bar-fill" style={{
                  width: `${lstmAccuracy * 100}%`,
                  background: 'linear-gradient(90deg, #8b5cf6, #a78bfa)',
                  transition: 'width 1.2s cubic-bezier(0.4, 0, 0.2, 1)',
                }}></div>
              </div>
            )}
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '6px',
              marginTop: 'var(--space-md)', padding: '4px 12px',
              background: 'rgba(139, 92, 246, 0.1)', borderRadius: '99px',
              fontSize: '0.75rem', color: '#8b5cf6',
            }}>
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#8b5cf6' }}></span>
              {lstmAccuracy != null ? 'Model Active' : 'Not Trained'}
            </div>
          </div>
        </div>
      </div>

      {/* Synthetic MLP/LSTM metrics remain separate from the ULB benchmarks below. */}
      <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
        {comparisonCards.map((stat, i) => (
          <div key={stat.label} className={`stat-card animate-in animate-in-delay-${Math.min(i + 1, 4)}`}>
            <div className="stat-label">{stat.label}</div>
            <div className="model-comparison-values">
              <div className="model-comparison-item">
                <span className="model-comparison-tag mlp">MLP</span>
                <span className="stat-value" style={{ fontSize: '1.3rem', color: stat.mlpRaw != null ? '#3b82f6' : 'var(--text-muted)' }}>{stat.mlp}</span>
              </div>
              <div className="model-comparison-item">
                <span className="model-comparison-tag lstm">LSTM</span>
                <span className="stat-value" style={{ fontSize: '1.3rem', color: stat.lstmRaw != null ? '#8b5cf6' : 'var(--text-muted)' }}>{stat.lstm}</span>
              </div>
            </div>
            <div className="stat-delta">{stat.delta}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 'var(--space-xl)' }}>
        <BenchmarkRecords syntheticMetrics={metrics} syntheticFeatures={modelInfo?.features_used} />
      </div>
      <div style={{ marginTop: 'var(--space-xl)' }}>
        <ULBMetricDetails />
      </div>
      <div className="card" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title">ULB Model Status</span>
          <span className={`model-tag ${ulbModelInfo ? 'mlp' : ''}`}>{ulbModelInfo ? 'Loaded' : 'Unavailable'}</span>
        </div>
        <p className="benchmark-note">
          {ulbModelInfo
            ? `${ulbModelInfo.model_name} · ${ulbModelInfo.dataset} · ${ulbModelInfo.feature_count} features · threshold ${Number(ulbModelInfo.threshold).toFixed(6)}`
            : 'The API did not report a loaded ULB model.'}
        </p>
      </div>

      <div className="grid-2">
        {/* MLP Training Progress */}
        <div className="card animate-in animate-in-delay-2">
          <div className="card-header">
            <span className="card-title">MLP Training Progress</span>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span className="model-tag mlp">MLP</span>
              {history && <span className="card-subtitle">{history.train_loss?.length || 0} epochs</span>}
            </div>
          </div>
          {history ? (
            <LossChart trainLoss={history.train_loss} valLoss={history.val_loss} />
          ) : (
            <div className="chart-placeholder">
              <span>Train MLP model to see loss curves</span>
            </div>
          )}
        </div>

        {/* LSTM Training Progress */}
        <div className="card animate-in animate-in-delay-3">
          <div className="card-header">
            <span className="card-title">LSTM Training Progress</span>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span className="model-tag lstm">LSTM</span>
              {lstmHistory && <span className="card-subtitle">{lstmHistory.train_loss?.length || 0} epochs</span>}
            </div>
          </div>
          {lstmHistory ? (
            <LossChart trainLoss={lstmHistory.train_loss} valLoss={lstmHistory.val_loss} />
          ) : (
            <div className="chart-placeholder">
              <span>Train LSTM model to see loss curves</span>
            </div>
          )}
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 'var(--space-xl)' }}>
        {/* MLP Architecture */}
        <div className="card animate-in animate-in-delay-2">
          <div className="card-header">
            <span className="card-title">MLP Architecture</span>
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
              <span>No MLP model loaded yet</span>
            </div>
          )}
        </div>

        {/* LSTM Architecture */}
        <div className="card animate-in animate-in-delay-3">
          <div className="card-header">
            <span className="card-title">LSTM Architecture</span>
            <span className="card-subtitle">
              {lstmModelInfo ? `${lstmModelInfo.total_parameters?.toLocaleString()} params` : 'N/A'}
            </span>
          </div>
          {lstmModelInfo && !lstmModelInfo.message ? (
            <div className="layer-list">
              {[
                'Input (batch, seq=10, features)',
                'LSTM(2 layers, hidden=64, bidir)',
                'Last step → Linear(128, 64)',
                'BatchNorm + ReLU + Dropout(0.3)',
                'Linear(64, 32) + ReLU',
                'Dropout(0.15)',
                'Linear(32, 1) + Sigmoid → Output',
              ].map((layer, i) => (
                <div key={i} className="layer-item">
                  <span className="layer-dot" style={{ background: '#8b5cf6' }}></span>
                  {layer}
                </div>
              ))}
            </div>
          ) : (
            <div className="chart-placeholder">
              <span>No LSTM model loaded yet</span>
            </div>
          )}
        </div>
      </div>

      {/* System Architecture */}
      <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title">System Architecture</span>
        </div>
        <div className="architecture-flow">
          <div className="arch-node">
            <div>Transaction</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node">
            <div>Feature Engine</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node">
            <div>Scaler</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ fontSize: '0.7rem', color: '#3b82f6' }}>MLP DNN</div>
            <div style={{ fontSize: '0.7rem', color: '#8b5cf6' }}>LSTM Seq</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node">
            <div>Probability</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-node">
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
