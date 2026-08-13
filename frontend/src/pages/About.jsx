export default function About() {
  return (
    <div>
      <div className="page-header">
        <h2>About FraudShield AI</h2>
        <p>End-to-end deep learning fraud detection system</p>
      </div>

      <div className="grid-2">
        <div className="card animate-in">
          <div className="card-header">
            <span className="card-title">Technology Stack</span>
          </div>
          <div className="layer-list">
            {[
              ['Deep Learning', 'PyTorch — MLP (DNN) + LSTM'],
              ['Backend', 'FastAPI (Python) — REST API'],
              ['Frontend', 'React + Vite — Modern SPA'],
              ['Data Processing', 'Pandas, NumPy, Scikit-learn'],
              ['Imbalance Handling', 'Class-weighted BCE Loss'],
              ['Preprocessing', 'RobustScaler, Feature Engineering'],
              ['Model Format', 'PyTorch .pt checkpoint'],
            ].map(([key, val]) => (
              <div key={key} className="layer-item">
                <span className="layer-dot"></span>
                <span style={{ minWidth: '150px' }}>{key}</span>
                <span style={{ color: 'var(--text-muted)' }}>{val}</span>
              </div>
            ))}
          </div>
        </div>

        {/* MLP + LSTM Architecture Side-by-Side */}
        <div className="card animate-in animate-in-delay-1">
          <div className="card-header">
            <span className="card-title">Model Architectures</span>
          </div>
          <div className="arch-comparison">
            {/* MLP Architecture */}
            <div className="arch-card">
              <div className="arch-card-title">
                MLP (FraudDetectorNet)
              </div>
              <div className="arch-card-subtitle">6-layer fully-connected DNN — treats each transaction independently</div>
              <div className="layer-list">
                {[
                  'Input → Linear(n, 128)',
                  'BatchNorm + ReLU + Dropout(0.3)',
                  'Linear(128, 256) → expand',
                  'Linear(256, 128) → contract',
                  'Linear(128, 64) + Dropout(0.15)',
                  'Linear(64, 32) + ReLU',
                  'Linear(32, 1) + Sigmoid',
                ].map((layer, i) => (
                  <div key={i} className="layer-item">
                    <span className="layer-dot"></span>
                    {layer}
                  </div>
                ))}
              </div>
            </div>

            {/* LSTM Architecture */}
            <div className="arch-card">
              <div className="arch-card-title">
                LSTM (FraudLSTMNet)
              </div>
              <div className="arch-card-subtitle">Bidirectional 2-layer LSTM — learns temporal patterns across transaction sequences</div>
              <div className="layer-list">
                {[
                  'Input (batch, seq=10, features)',
                  'LSTM(2 layers, hidden=64, bidir)',
                  'Last step → Linear(128, 64)',
                  'BatchNorm + ReLU + Dropout(0.3)',
                  'Linear(64, 32) + ReLU',
                  'Dropout(0.15)',
                  'Linear(32, 1) + Sigmoid',
                ].map((layer, i) => (
                  <div key={i} className="layer-item">
                    <span className="layer-dot"></span>
                    {layer}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card animate-in animate-in-delay-2">
          <div className="card-header">
            <span className="card-title">Feature Engineering</span>
          </div>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['hour_of_day', 'Hour extracted from transaction time'],
                ['is_night_transaction', 'Flag for 10PM-5AM transactions'],
                ['amount_to_avg_ratio', 'Amount / average spending ratio'],
                ['high_amount', 'Flag for transactions > $500'],
                ['unusual_merchant', 'Online merchant category flag'],
                ['new_customer', 'Account younger than 90 days'],
                ['high_frequency', 'More than 10 transactions in 24h'],
              ].map(([feat, desc]) => (
                <tr key={feat}>
                  <td><code style={{ color: 'var(--accent-cyan)', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.8rem' }}>{feat}</code></td>
                  <td style={{ color: 'var(--text-secondary)' }}>{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card animate-in animate-in-delay-3">
          <div className="card-header">
            <span className="card-title">Project Structure</span>
          </div>
          <pre style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            lineHeight: 1.8,
            padding: 'var(--space-md)',
            background: 'var(--bg-glass)',
            borderRadius: 'var(--radius-md)',
            overflow: 'auto',
          }}>
{`fraud_detection_project/
├── src/
│   ├── pytorch_model.py    # MLP + LSTM model definitions
│   ├── train_model.py      # Training pipelines (both models)
│   ├── predict.py          # Inference (MLP + LSTM)
│   ├── evaluate.py         # Evaluation metrics
│   ├── data_pipeline.py    # Data processing
│   └── preprocessing.py    # Feature scaling
├── api/
│   └── app.py              # FastAPI (MLP + LSTM endpoints)
├── frontend/               # React + Vite
│   └── src/
│       ├── pages/          # Dashboard, Predict, etc.
│       ├── components/     # Sidebar, shared UI
│       └── api.js          # API client
├── models/                 # Saved .pt models (MLP + LSTM)
└── requirements.txt`}
          </pre>
        </div>
      </div>

      <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title">Real-World Impact</span>
        </div>
        <div className="stats-grid" style={{ marginBottom: 0 }}>
          {[
            { label: 'Global Card Fraud', value: '$25-30B', type: 'negative', delta: 'Annual losses worldwide' },
            { label: 'ML Fraud Reduction', value: '50-70%', type: 'positive', delta: 'Loss reduction with ML' },
            { label: 'Transactions/Day', value: 'Millions', type: 'neutral', delta: 'Processed by major banks' },
            { label: 'Detection Speed', value: '<100ms', type: 'neutral', delta: 'Real-time scoring needed' },
          ].map((stat) => (
            <div key={stat.label} className="stat-card">
              <div className="stat-label">{stat.label}</div>
              <div className={`stat-value ${stat.type}`}>{stat.value}</div>
              <div className="stat-delta">{stat.delta}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
