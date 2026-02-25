import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';

const Dashboard = () => {
    const [status, setStatus] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const data = await apiService.getStatus();
                setStatus(data);
            } catch (err) {
                console.error('Status fetch failed', err);
            } finally {
                setLoading(false);
            }
        };

        fetchStatus();
        const interval = setInterval(fetchStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return null;

    const isReady = status?.status === 'ready';

    return (
        <div className="main-content fade-in" style={{ paddingBottom: '2rem' }}>
            <div className="glass-panel" style={{ padding: '2rem' }}>
                <h2 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    System Intelligence Status
                    <span className={`status-pill`}>
                        <span className={`status-dot ${isReady ? 'online' : ''}`}></span>
                        {status?.status?.toUpperCase()}
                    </span>
                </h2>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
                    <StatCard
                        label="Total Doc Chunks"
                        value={status?.total_chunks?.toLocaleString()}
                        sub="Indexed & Embedded"
                    />
                    <StatCard
                        label="Knowledge Graph"
                        value={`${status?.graph_nodes?.toLocaleString()} Nodes`}
                        sub={`${status?.graph_edges?.toLocaleString()} Connections`}
                    />
                    <StatCard
                        label="Community Clusters"
                        value={status?.graph_communities}
                        sub="Detected Themes"
                    />
                    <StatCard
                        label="GPU Embed Mode"
                        value={status?.features?.embed_mode === 'local' ? 'Local RTX GPU' : 'NVIDIA Cloud'}
                        sub={status?.features?.api_keys ? `${status.features.api_keys} API Keys Loaded` : ''}
                    />
                </div>

                {!isReady && (
                    <div style={{ marginTop: '2.5rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                                {status?.message}
                            </span>
                            <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                                {status?.progress}%
                            </span>
                        </div>
                        <div style={{
                            height: '8px',
                            background: 'rgba(255,255,255,0.05)',
                            borderRadius: '4px',
                            overflow: 'hidden'
                        }}>
                            <div style={{
                                height: '100%',
                                width: `${status?.progress}%`,
                                background: 'linear-gradient(90deg, var(--primary), var(--secondary))',
                                transition: 'width 0.5s ease-out'
                            }}></div>
                        </div>
                    </div>
                )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                    <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>Feature Matrix</h3>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
                        <FeatureBadge label="Query Rewriting" enabled={status?.features?.query_rewriting} />
                        <FeatureBadge label="GraphRAG" enabled={status?.features?.graphrag} />
                        <FeatureBadge label="Context Compression" enabled={status?.features?.context_compression} />
                        <FeatureBadge label="Multi-Vector RRF" enabled={status?.features?.multi_vector} />
                        <FeatureBadge label="TRF Entity Extraction" enabled={status?.features?.spacy_model?.includes('trf')} />
                    </div>
                </div>

                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                    <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>Model Configuration</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--text-muted)' }}>SPAcy Model:</span>
                            <span>{status?.features?.spacy_model}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--text-muted)' }}>RAG Resolution:</span>
                            <span>{status?.features?.graphrag_resolution || '1.0'}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const StatCard = ({ label, value, sub }) => (
    <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.4rem' }}>{label}</p>
        <p style={{ fontSize: '1.5rem', fontWeight: 700, margin: 0 }}>{value || '0'}</p>
        <p style={{ color: 'var(--text-dim)', fontSize: '0.75rem', marginTop: '0.2rem' }}>{sub}</p>
    </div>
);

const FeatureBadge = ({ label, enabled }) => (
    <div className="status-pill" style={{ opacity: enabled ? 1 : 0.4, borderColor: enabled ? 'var(--primary)' : 'var(--border)' }}>
        <span className="status-dot" style={{ background: enabled ? 'var(--accent)' : 'var(--text-dim)' }}></span>
        {label}
    </div>
);

export default Dashboard;
