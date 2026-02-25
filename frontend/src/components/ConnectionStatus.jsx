import React from 'react';

const ConnectionStatus = ({ apiKeys, embedMode }) => {
    return (
        <div className="glass-panel" style={{ padding: '1rem', marginTop: '1rem' }}>
            <h4 style={{ fontSize: '0.8rem', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '0.8rem' }}>Infrastructure</h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Embedding Engine</span>
                    <span style={{ color: 'var(--accent)', fontWeight: 600 }}>{embedMode === 'local' ? 'RTX 3050 Ti' : 'NVIDIA cloud'}</span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>LLM Cluster</span>
                    <span>Llama 3.1 70B</span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>API Pool</span>
                    <span style={{ color: apiKeys > 1 ? 'var(--accent)' : 'var(--primary)' }}>{apiKeys} Active Keys</span>
                </div>

                <div style={{ height: '1px', background: 'var(--border)', margin: '0.4rem 0' }}></div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--accent)' }}>
                    <div className="status-dot online" style={{ width: '6px', height: '6px' }}></div>
                    Ready for Discovery
                </div>
            </div>
        </div>
    );
};

export default ConnectionStatus;
