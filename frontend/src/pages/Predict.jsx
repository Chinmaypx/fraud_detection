import { useState } from 'react';
import { predictFraud } from '../api';

export default function Predict() {
  const [form, setForm] = useState({
    transaction_amount: 250,
    transaction_time: 43200,
    location: 'Mumbai',
    device_id: 'DEV001',
    merchant_category: 'retail',
    account_age_days: 365,
    transaction_count_24h: 3,
    avg_transaction_amount: 150,
  });
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictFraud(form);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const riskClass = result
    ? result.risk_level === 'HIGH'
      ? 'danger'
      : result.risk_level === 'MEDIUM'
      ? 'warning'
      : 'safe'
    : '';

  const hourOfDay = Math.floor(form.transaction_time / 3600);

  return (
    <div>
      <div className="page-header">
        <h2>🔍 Fraud Prediction</h2>
        <p>Analyze a transaction for fraud risk using the PyTorch neural network</p>
      </div>

      <div className="grid-2">
        {/* Input Form */}
        <div className="card animate-in">
          <div className="card-header">
            <span className="card-title">Transaction Details</span>
          </div>

          <div className="grid-2" style={{ gap: 'var(--space-md)' }}>
            <div className="form-group">
              <label className="form-label">Amount (₹)</label>
              <input
                type="number"
                className="form-input"
                value={form.transaction_amount}
                onChange={(e) => handleChange('transaction_amount', parseFloat(e.target.value) || 0)}
                min="0.01"
                step="10"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Time of Day (Hour: {hourOfDay}:00)</label>
              <input
                type="range"
                min="0"
                max="86399"
                value={form.transaction_time}
                onChange={(e) => handleChange('transaction_time', parseInt(e.target.value))}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Location</label>
              <select
                className="form-select"
                value={form.location}
                onChange={(e) => handleChange('location', e.target.value)}
              >
                {['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad'].map((loc) => (
                  <option key={loc} value={loc}>{loc}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Merchant Category</label>
              <select
                className="form-select"
                value={form.merchant_category}
                onChange={(e) => handleChange('merchant_category', e.target.value)}
              >
                {['retail', 'grocery', 'restaurant', 'gas', 'online'].map((cat) => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Account Age (days)</label>
              <input
                type="number"
                className="form-input"
                value={form.account_age_days}
                onChange={(e) => handleChange('account_age_days', parseInt(e.target.value) || 0)}
                min="0"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Transactions (24h)</label>
              <input
                type="number"
                className="form-input"
                value={form.transaction_count_24h}
                onChange={(e) => handleChange('transaction_count_24h', parseInt(e.target.value) || 0)}
                min="0"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Avg Transaction (₹)</label>
              <input
                type="number"
                className="form-input"
                value={form.avg_transaction_amount}
                onChange={(e) => handleChange('avg_transaction_amount', parseFloat(e.target.value) || 0)}
                min="0.01"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Device ID</label>
              <input
                type="text"
                className="form-input"
                value={form.device_id}
                onChange={(e) => handleChange('device_id', e.target.value)}
              />
            </div>
          </div>

          <div className="slider-container" style={{ marginTop: 'var(--space-md)' }}>
            <div className="slider-header">
              <label className="form-label" style={{ margin: 0 }}>Detection Threshold</label>
              <span className="slider-value">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              <span>More Alerts</span>
              <span>Fewer Alerts</span>
            </div>
          </div>

          <button
            className="btn btn-primary btn-lg"
            style={{ width: '100%', marginTop: 'var(--space-md)' }}
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }}></div>
                Analyzing...
              </>
            ) : (
              <>🔍 Analyze Transaction</>
            )}
          </button>
        </div>

        {/* Result */}
        <div className="card animate-in animate-in-delay-1">
          <div className="card-header">
            <span className="card-title">Analysis Result</span>
          </div>

          {error && (
            <div className="prediction-result danger">
              <div className="result-icon">⚠️</div>
              <div className="result-label">Error</div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{error}</p>
            </div>
          )}

          {result && (
            <div className={`prediction-result ${riskClass}`}>
              <div className="result-icon">
                {result.is_fraud ? '🚨' : '✅'}
              </div>
              <div className="result-label">
                {result.is_fraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}
              </div>
              <div className="result-probability" style={{ color: result.is_fraud ? 'var(--danger)' : 'var(--success)' }}>
                {(result.fraud_probability * 100).toFixed(1)}%
              </div>
              <span className={`risk-badge ${result.risk_level.toLowerCase()}`}>
                {result.risk_level} RISK
              </span>

              <div className="result-details">
                <div className="result-detail">
                  <div className="result-detail-label">Probability</div>
                  <div className="result-detail-value">{(result.fraud_probability * 100).toFixed(2)}%</div>
                </div>
                <div className="result-detail">
                  <div className="result-detail-label">Risk Level</div>
                  <div className="result-detail-value">{result.risk_level}</div>
                </div>
                <div className="result-detail">
                  <div className="result-detail-label">Verdict</div>
                  <div className="result-detail-value">{result.is_fraud ? 'Flagged' : 'Clear'}</div>
                </div>
              </div>

              {/* Probability gauge */}
              <div style={{ marginTop: 'var(--space-xl)' }}>
                <div className="progress-bar-container" style={{ height: '12px' }}>
                  <div
                    className="progress-bar-fill"
                    style={{
                      width: `${result.fraud_probability * 100}%`,
                      background: result.is_fraud
                        ? 'var(--gradient-danger)'
                        : 'var(--gradient-success)',
                    }}
                  ></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  <span>Safe</span>
                  <span>Fraudulent</span>
                </div>
              </div>

              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: 'var(--space-lg)' }}>
                {result.message}
              </p>
            </div>
          )}

          {!result && !error && (
            <div className="empty-state">
              <div className="icon">🔮</div>
              <p>Submit a transaction to see the analysis result</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                The PyTorch neural network will analyze the transaction and return a fraud probability score
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Quick Test Scenarios */}
      <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title">⚡ Quick Test Scenarios</span>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
          <button
            className="btn btn-outline"
            onClick={() => {
              setForm({
                transaction_amount: 50,
                transaction_time: 43200,
                location: 'Mumbai',
                device_id: 'DEV001',
                merchant_category: 'grocery',
                account_age_days: 730,
                transaction_count_24h: 2,
                avg_transaction_amount: 75,
              });
              setResult(null);
            }}
          >
            ✅ Normal Transaction
          </button>
          <button
            className="btn btn-outline"
            onClick={() => {
              setForm({
                transaction_amount: 2500,
                transaction_time: 10800,
                location: 'Delhi',
                device_id: 'DEV099',
                merchant_category: 'online',
                account_age_days: 15,
                transaction_count_24h: 25,
                avg_transaction_amount: 50,
              });
              setResult(null);
            }}
          >
            🚨 Suspicious Transaction
          </button>
          <button
            className="btn btn-outline"
            onClick={() => {
              setForm({
                transaction_amount: 800,
                transaction_time: 82800,
                location: 'Bengaluru',
                device_id: 'DEV050',
                merchant_category: 'online',
                account_age_days: 60,
                transaction_count_24h: 12,
                avg_transaction_amount: 100,
              });
              setResult(null);
            }}
          >
            ⚠️ Night Transaction
          </button>
        </div>
      </div>
    </div>
  );
}
