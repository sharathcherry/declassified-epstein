import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import ChatInterface from './components/ChatInterface';
import MetricsDashboard from './components/MetricsDashboard';
import './App.css';

function App() {
  const [currentTab, setTab] = useState('chat');

  return (
    <div className="app-container">
      <Sidebar currentTab={currentTab} setTab={setTab} />

      <div className="main-area">
        {currentTab === 'chat' && <ChatInterface />}

        {currentTab === 'dashboard' && (
          <div style={{ padding: '24px', overflowY: 'auto', flex: 1 }}>
            <Dashboard />
          </div>
        )}

        {currentTab === 'metrics' && (
          <div style={{ padding: '24px', overflowY: 'auto', flex: 1 }}>
            <MetricsDashboard />
          </div>
        )}

        {currentTab === 'search' && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ textAlign: 'center', color: 'var(--text-dim)' }}>
              <p style={{ fontSize: '2rem', marginBottom: '8px' }}>🔍</p>
              <p style={{ fontSize: '1rem' }}>Document Search — Coming Soon</p>
            </div>
          </div>
        )}

        {currentTab === 'graph' && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ textAlign: 'center', color: 'var(--text-dim)' }}>
              <p style={{ fontSize: '2rem', marginBottom: '8px' }}>🕸️</p>
              <p style={{ fontSize: '1rem' }}>Knowledge Graph — Coming Soon</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
