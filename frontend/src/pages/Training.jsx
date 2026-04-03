import { useRef } from 'react';
import { trainModel } from '../api';

export default function Training({ trainingState, onTrainingUpdate }) {
  const { training, result, error } = trainingState;
  const abortRef = useRef(null);

  const handleTrain = async () => {
    onTrainingUpdate({ training: true, error: null, result: null });
    try {
      const res = await trainModel();
      onTrainingUpdate({ training: false, result: res });
    } catch (err) {
      onTrainingUpdate({ training: false, error: err.message });
    }
  };

  return (
    <div>
      <div className="page-header">
        <h2>⚡ Model Training</h2>
        <p>Train the PyTorch neural network on synthetic fraud data</p>
      </div>

      <div className="grid-2">
        {/* Training Config */}
        <div className="card animate-in">
          <div className="card-header">
            <span className="card-title">Training Configuration</span>
          </div>

          <div className="layer-list" style={{ marginBottom: 'var(--space-xl)' }}>
            {[
              ['Framework', 'PyTorch'],
              ['Architecture', 'FraudDetectorNet (6-layer DNN)'],
              ['Optimizer', 'Adam (lr=0.001, weight_decay=1e-5)'],
              ['Loss', 'BCE with class weights'],
              ['Scheduler', 'ReduceLROnPlateau'],
              ['Epochs', '50 (early stopping, patience=10)'],
              ['Batch Size', '512'],
              ['Dataset', '100,000 synthetic transactions (2% fraud)'],
              ['Imbalance Handling', 'Class-weighted loss function'],
            ].map(([key, val]) => (
              <div key={key} className="layer-item">
                <span className="layer-dot"></span>
                <span style={{ color: 'var(--text-muted)', minWidth: '120px' }}>{key}:</span>
                <span style={{ color: 'var(--text-primary)' }}>{val}</span>
              </div>
            ))}
          </div>

          <button
            className="btn btn-primary btn-lg"
            style={{ width: '100%' }}
            onClick={handleTrain}
            disabled={training}
          >
            {training ? (
              <>
                <div className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }}></div>
                Training in progress... (this may take 1-2 min)
              </>
            ) : (
              <>⚡ Start Training</>
            )}
          </button>

          {training && (
            <div className="training-persist-notice">
              <span className="training-persist-icon">🔒</span>
              <span>Training continues even if you switch tabs</span>
            </div>
          )}
        </div>

        {/* Training Result */}
        <div className="card animate-in animate-in-delay-1">
          <div className="card-header">
            <span className="card-title">Training Result</span>
          </div>

          {training && (
            <div className="loading-overlay" style={{ minHeight: '300px' }}>
              <div className="spinner"></div>
              <span>Training PyTorch model...</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Generating data → Preprocessing → Training 50 epochs → Evaluating
              </span>
              <div className="training-steps-anim">
                <div className="training-step active">
                  <div className="training-step-dot"></div>
                  <span>Data Pipeline</span>
                </div>
                <div className="training-step-connector"></div>
                <div className="training-step">
                  <div className="training-step-dot"></div>
                  <span>Preprocessing</span>
                </div>
                <div className="training-step-connector"></div>
                <div className="training-step">
                  <div className="training-step-dot"></div>
                  <span>Training</span>
                </div>
                <div className="training-step-connector"></div>
                <div className="training-step">
                  <div className="training-step-dot"></div>
                  <span>Evaluation</span>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="prediction-result danger">
              <div className="result-icon">❌</div>
              <div className="result-label">Training Failed</div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{error}</p>
            </div>
          )}

          {result && (
            <div>
              <div className="prediction-result safe" style={{ marginBottom: 'var(--space-lg)' }}>
                <div className="result-icon">✅</div>
                <div className="result-label">Training Complete!</div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{result.message}</p>
              </div>

              {result.metrics && (
                <table className="metrics-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Score</th>
                      <th>Visual</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.metrics).map(([key, val]) => (
                      <tr key={key}>
                        <td>{key}</td>
                        <td>
                          <span className="metric-value">{(val * 100).toFixed(1)}%</span>
                        </td>
                        <td>
                          <div className="metric-bar">
                            <div className="metric-bar-track">
                              <div className="metric-bar-fill" style={{ width: `${val * 100}%` }}></div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {!training && !result && !error && (
            <div className="empty-state" style={{ minHeight: '300px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <div className="icon">🚀</div>
              <p>Click "Start Training" to train the model</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                The model will train on 100K synthetic bank transactions
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
