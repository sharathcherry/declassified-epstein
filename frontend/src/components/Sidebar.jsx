import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';

const Sidebar = ({ currentTab, setTab }) => {
    const [status, setStatus] = useState(null);

    useEffect(() => {
        const getStatus = async () => {
            try {
                const data = await apiService.getStatus();
                setStatus(data);
            } catch (e) {
                console.error("Status fetch failed", e);
            }
        };
        getStatus();
        const interval = setInterval(getStatus, 15000);
        return () => clearInterval(interval);
    }, []);

    const navItems = [
        { id: 'chat', label: 'Intelligence', icon: '💬' },
        { id: 'dashboard', label: 'System Overview', icon: '📊' },
        { id: 'search', label: 'Document Search', icon: '🔍' },
        { id: 'graph', label: 'Knowledge Graph', icon: '🕸️' },
        { id: 'metrics', label: 'RAG Metrics', icon: '📈' },
    ];

    return (
        <div className="sidebar">
            {/* Header with New Chat */}
            <div className="sidebar-header">
                <button
                    className="new-chat-btn"
                    onClick={() => setTab('chat')}
                >
                    <span style={{ fontSize: '1rem' }}>✏️</span>
                    New investigation
                </button>
            </div>

            {/* Navigation */}
            <nav className="sidebar-nav">
                <div className="sidebar-section-label">Navigation</div>
                {navItems.map(item => (
                    <button
                        key={item.id}
                        className={`sidebar-item ${currentTab === item.id ? 'active' : ''}`}
                        onClick={() => setTab(item.id)}
                    >
                        <span>{item.icon}</span>
                        {item.label}
                    </button>
                ))}

                {/* System Info */}
                {status && (
                    <>
                        <div className="sidebar-section-label" style={{ marginTop: '16px' }}>System</div>
                        <div className="sidebar-item" style={{ cursor: 'default', opacity: 0.7 }}>
                            <span className="status-dot online"></span>
                            {status.total_chunks?.toLocaleString() || '0'} chunks indexed
                        </div>
                        <div className="sidebar-item" style={{ cursor: 'default', opacity: 0.7 }}>
                            <span>🔑</span>
                            {status.features?.api_keys || 0} API keys active
                        </div>
                    </>
                )}
            </nav>

            {/* Footer */}
            <div className="sidebar-footer">
                <div className="sidebar-footer-item">
                    <span>⚡</span>
                    <span style={{ fontSize: '0.8rem' }}>NVIDIA NIM + FAISS</span>
                </div>
                <div className="sidebar-footer-item" style={{ opacity: 0.5 }}>
                    <span style={{
                        width: '24px',
                        height: '24px',
                        borderRadius: '50%',
                        background: '#5436DA',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.65rem',
                        fontWeight: 700,
                        color: 'white',
                    }}>
                        SC
                    </span>
                    <span style={{ fontSize: '0.8rem' }}>VIBE RAG v2.0</span>
                </div>
            </div>
        </div>
    );
};

export default Sidebar;
