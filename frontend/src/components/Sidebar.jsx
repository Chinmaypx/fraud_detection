import { useEffect, useState } from 'react';
import { checkHealth } from '../api';

export default function Sidebar({ activePage, onNavigate, isTraining }) {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
    
    const interval = setInterval(() => {
      checkHealth()
        .then(setHealth)
        .catch(() => setHealth(null));
    }, 10000);
    
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'predict', label: 'Predict', icon: '🔍' },
    { id: 'training', label: 'Training', icon: '⚡' },
    { id: 'metrics', label: 'Model Metrics', icon: '📈' },
    { id: 'about', label: 'About', icon: '💡' },
  ];

  const isOnline = health !== null;
  const modelLoaded = health?.model_loaded;

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <h1>🛡️ FraudShield AI</h1>
        <div className="subtitle">PyTorch Deep Learning</div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <div
            key={item.id}
            className={`nav-item ${activePage === item.id ? 'active' : ''} ${item.id === 'training' && isTraining ? 'training-active' : ''}`}
            onClick={() => onNavigate(item.id)}
          >
            <span className="icon">{item.icon}</span>
            <span>{item.label}</span>
            {item.id === 'training' && isTraining && (
              <span className="nav-training-badge">
                <span className="nav-training-dot"></span>
                Running
              </span>
            )}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-badge">
          <span className={`status-dot ${isOnline ? '' : 'offline'}`}></span>
          <span>{isOnline ? 'API Connected' : 'API Offline'}</span>
        </div>
        {isOnline && (
          <div className="status-badge" style={{ marginTop: '6px' }}>
            <span className={`status-dot ${modelLoaded ? '' : 'offline'}`}></span>
            <span>{modelLoaded ? 'Model Loaded' : 'No Model'}</span>
          </div>
        )}
      </div>
    </aside>
  );
}
