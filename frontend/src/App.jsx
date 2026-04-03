import { useState, useRef, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import Training from './pages/Training';
import Metrics from './pages/Metrics';
import About from './pages/About';

function App() {
  const [activePage, setActivePage] = useState('dashboard');

  // Shared training state — lifted to App so it persists across tab switches
  const [trainingState, setTrainingState] = useState({
    training: false,
    result: null,
    error: null,
  });

  const handleTrainingUpdate = useCallback((update) => {
    setTrainingState((prev) => ({ ...prev, ...update }));
  }, []);

  return (
    <div className="app-layout">
      <Sidebar
        activePage={activePage}
        onNavigate={setActivePage}
        isTraining={trainingState.training}
      />
      <main className="main-content">
        {/* All pages rendered but only the active one is visible.
            This prevents Training from unmounting mid-request. */}
        <div style={{ display: activePage === 'dashboard' ? 'block' : 'none' }}>
          <Dashboard />
        </div>
        <div style={{ display: activePage === 'predict' ? 'block' : 'none' }}>
          <Predict />
        </div>
        <div style={{ display: activePage === 'training' ? 'block' : 'none' }}>
          <Training
            trainingState={trainingState}
            onTrainingUpdate={handleTrainingUpdate}
          />
        </div>
        <div style={{ display: activePage === 'metrics' ? 'block' : 'none' }}>
          <Metrics />
        </div>
        <div style={{ display: activePage === 'about' ? 'block' : 'none' }}>
          <About />
        </div>
      </main>
    </div>
  );
}

export default App;
