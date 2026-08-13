import { useState } from 'react';
import { trainModel, trainLSTMModel } from '../api';

export default function Training({ trainingState, onTrainingUpdate }) {
  const { training, result, error } = trainingState;
  const [selectedModel, setSelectedModel] = useState('mlp');

  const handleTrain = async () => {
    onTrainingUpdate({ training: true, error: null, result: null });
    try {
      const trainFn = selectedModel === 'lstm' ? trainLSTMModel : trainModel;
      const res = await trainFn();
      onTrainingUpdate({ training: false, result: { ...res, model_trained: selectedModel } });
    } catch (err) {
      onTrainingUpdate({ training: false, error: err.message });
    }
  };

  const mlpConfig = [
    ['Framework', 'PyTorch'],
    ['Architecture', 'FraudDetectorNet (6-layer DNN)'],
    ['Optimizer', 'Adam (lr=0.001, weight_decay=1e-5)'],
    ['Loss', 'BCE with class weights'],
    ['Scheduler', 'ReduceLROnPlateau'],
    ['Epochs', '50 (early stopping, patience=10)'],
    ['Batch Size', '512'],
    ['Dataset', '100,000 synthetic transactions (2% fraud)'],
    ['Imbalance Handling', 'Class-weighted loss function'],
  ];

  const lstmConfig = [
    ['Framework', 'PyTorch'],
    ['Architecture', 'FraudLSTMNet (Bidirectional 2-layer LSTM)'],
    ['Sequence Length', '10 transactions per customer'],
    ['Hidden Dim', '64 (bidirectional → 128)'],
    ['Optimizer', 'Adam (lr=0.001, weight_decay=1e-5)'],
    ['Loss', 'BCE with class weights'],
    ['Scheduler', 'ReduceLROnPlateau'],
    ['Epochs', '50 (early stopping, patience=10)'],
    ['Batch Size', '512'],
    ['Dataset', '100,000 transactions → per-customer sequences'],
    ['Imbalance Handling', 'Class-weighted loss function'],
  ];

  const config = selectedModel === 'lstm' ? lstmConfig : mlpConfig;

  const trainingStepsMLP = ['Data Pipeline', 'Preprocessing', 'Training', 'Evaluation'];
  const trainingStepsLSTM = ['Data Pipeline', 'Sequencing', 'Training', 'Evaluation'];
  const trainingSteps = selectedModel === 'lstm' ? trainingStepsLSTM : trainingStepsMLP;

  return (
    <div>
      <div className="page-header">
        <h2>Model Training</h2>
        <p>Train deep learning models on synthetic fraud data</p>
      </div>

      {/* Model Selector */}
      <div className="model-selector">
        <span className="model-selector-label">Train</span>
        <div className="model-selector-pills">
          <button
            className={`model-pill ${selectedModel === 'mlp' ? 'active' : ''}`}
            onClick={() => !training && setSelectedModel('mlp')}
            disabled={training}
          >
            MLP
            <span className="model-pill-badge">6-Layer DNN</span>
          </button>
          <button
            className={`model-pill ${selectedModel === 'lstm' ? 'active' : ''}`}
            onClick={() => !training && setSelectedModel('lstm')}
            disabled={training}
          >
            LSTM
            <span className="model-pill-badge">Sequence</span>
          </button>
        </div>
      </div>

      <div className="grid-2">
        {/* Training Config */}
        <div className="card animate-in">
          <div className="card-header">
            <span className="card-title">Training Configuration</span>
            <span className={`model-tag ${selectedModel}`}>
              {selectedModel === 'lstm' ? 'LSTM' : 'MLP'}
            </span>
          </div>

          <div className="layer-list" style={{ marginBottom: 'var(--space-xl)' }}>
            {config.map(([key, val]) => (
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
                Training {selectedModel === 'lstm' ? 'LSTM' : 'MLP'} in progress... (this may take {selectedModel === 'lstm' ? '3-5' : '1-2'} min)
              </>
            ) : (
              <>Start Training ({selectedModel === 'lstm' ? 'LSTM' : 'MLP'})</>
            )}
          </button>

          {training && (
            <div className="training-persist-notice">
              <span>Training continues even if you switch tabs</span>
            </div>
          )}
        </div>

        {/* Training Result */}
        <div className="card animate-in animate-in-delay-1">
          <div className="card-header">
            <span className="card-title">Training Result</span>
            {result && result.model_trained && (
              <span className={`model-tag ${result.model_trained}`}>
                {result.model_trained === 'lstm' ? 'LSTM' : 'MLP'}
              </span>
            )}
          </div>

          {training && (
            <div className="loading-overlay" style={{ minHeight: '300px' }}>
              <div className="spinner"></div>
              <span>Training {selectedModel === 'lstm' ? 'LSTM' : 'PyTorch'} model...</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {selectedModel === 'lstm'
                  ? 'Generating data → Building sequences → Training 50 epochs → Evaluating'
                  : 'Generating data → Preprocessing → Training 50 epochs → Evaluating'}
              </span>
              <div className="training-steps-anim">
                {trainingSteps.map((step, idx) => (
                  <div key={step}>
                    {idx > 0 && <div className="training-step-connector"></div>}
                    <div className={`training-step ${idx === 0 ? 'active' : ''}`}>
                      <div className="training-step-dot"></div>
                      <span>{step}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div className="prediction-result danger">
              <div className="result-label">Training Failed</div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{error}</p>
            </div>
          )}

          {result && (
            <div>
              <div className="prediction-result safe" style={{ marginBottom: 'var(--space-lg)' }}>
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
              <p>Select a model and click &quot;Start Training&quot;</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                {selectedModel === 'lstm'
                  ? 'The LSTM model will train on per-customer transaction sequences'
                  : 'The MLP model will train on 100K synthetic bank transactions'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
