import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';

const MetricsDashboard = () => {
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const data = await apiService.getMetrics();
                setMetrics(data);
            } catch (e) {
                console.error('Metrics fetch failed', e);
            } finally {
                setLoading(false);
            }
        };
        fetchMetrics();
        const interval = setInterval(fetchMetrics, 5000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <div className="glass-panel fade-in" style={{ padding: '3rem', textAlign: 'center' }}>
                <h2 style={{ color: 'var(--text-dim)' }}>Loading Metrics...</h2>
            </div>
        );
    }

    if (!metrics || metrics.total_queries === 0) {
        return (
            <div className="glass-panel fade-in" style={{ padding: '3rem', textAlign: 'center' }}>
                <h2 style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>📈 No Queries Yet</h2>
                <p style={{ color: 'var(--text-dim)' }}>
                    Send some chat queries to start tracking metrics.
                    <br />Recall@k, MRR, NDCG, Latency, Cost, and Failures will appear here.
                </p>
            </div>
        );
    }

    const lb = metrics.latency_breakdown || {};
    const latencyStages = [
        { label: 'Rewrite', ms: lb.rewrite_ms || 0, color: '#6366f1' },
        { label: 'Embed', ms: lb.embed_ms || 0, color: '#8b5cf6' },
        { label: 'Retrieve', ms: lb.retrieval_ms || 0, color: '#06b6d4' },
        { label: 'Rerank', ms: lb.rerank_ms || 0, color: '#f59e0b' },
        { label: 'Compress', ms: lb.compress_ms || 0, color: '#10b981' },
        { label: 'Graph', ms: lb.graph_ms || 0, color: '#ec4899' },
        { label: 'Generate', ms: lb.generation_ms || 0, color: '#ef4444' },
    ];
    const totalLatency = latencyStages.reduce((s, st) => s + st.ms, 0) || 1;

    const statCards = [
        { label: 'Total Queries', value: metrics.total_queries, icon: '🔢', color: '#3b82f6' },
        { label: 'Avg Latency', value: `${Math.round(metrics.avg_latency_ms)}ms`, icon: '⚡', color: '#f59e0b' },
        { label: 'P95 Latency', value: `${Math.round(metrics.p95_latency_ms)}ms`, icon: '📊', color: '#ef4444' },
        { label: 'NDCG@10', value: metrics.avg_ndcg_10?.toFixed(3) || '0', icon: '🎯', color: '#10b981' },
        { label: 'MRR', value: metrics.avg_mrr?.toFixed(3) || '0', icon: '🏆', color: '#8b5cf6' },
        { label: 'Recall@10', value: metrics.avg_recall_10?.toFixed(3) || '0', icon: '📋', color: '#06b6d4' },
        { label: 'Total Cost', value: `$${metrics.total_cost_usd?.toFixed(4) || '0'}`, icon: '💰', color: '#22c55e' },
        { label: 'Failure Rate', value: `${(metrics.failure_rate * 100).toFixed(1)}%`, icon: '⚠️', color: metrics.failure_rate > 0.05 ? '#ef4444' : '#10b981' },
    ];

    return (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', paddingBottom: '2rem' }}>

            {/* Stat Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                {statCards.map((card, i) => (
                    <div key={i} className="glass-panel" style={{
                        padding: '1.2rem',
                        borderRadius: '16px',
                        borderLeft: `4px solid ${card.color}`,
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontSize: '1.5rem' }}>{card.icon}</span>
                            <span style={{
                                fontSize: '1.4rem',
                                fontWeight: 700,
                                color: card.color,
                                fontFamily: 'var(--font-mono, monospace)',
                            }}>{card.value}</span>
                        </div>
                        <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            {card.label}
                        </p>
                    </div>
                ))}
            </div>

            {/* Latency Breakdown */}
            <div className="glass-panel" style={{ padding: '1.5rem', borderRadius: '16px' }}>
                <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>
                    ⚡ Pipeline Latency Breakdown (avg per query)
                </h3>
                <div style={{
                    display: 'flex',
                    height: '36px',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    marginBottom: '1rem',
                }}>
                    {latencyStages.map((stage, i) => {
                        const pct = (stage.ms / totalLatency) * 100;
                        if (pct < 1) return null;
                        return (
                            <div key={i} style={{
                                width: `${pct}%`,
                                backgroundColor: stage.color,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '0.65rem',
                                color: '#fff',
                                fontWeight: 600,
                                minWidth: pct > 5 ? 'auto' : '0',
                                transition: 'width 0.3s ease',
                            }}>
                                {pct > 8 ? `${stage.label} ${Math.round(stage.ms)}ms` : ''}
                            </div>
                        );
                    })}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
                    {latencyStages.map((stage, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem' }}>
                            <span style={{
                                width: '10px', height: '10px', borderRadius: '50%',
                                backgroundColor: stage.color, display: 'inline-block',
                            }} />
                            <span style={{ color: 'var(--text-muted)' }}>{stage.label}</span>
                            <span style={{ color: 'var(--text-dim)', fontFamily: 'monospace' }}>{Math.round(stage.ms)}ms</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Recent Queries Table */}
            <div className="glass-panel" style={{ padding: '1.5rem', borderRadius: '16px' }}>
                <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>
                    📋 Recent Queries
                </h3>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                                {['Query', 'Latency', 'NDCG@10', 'MRR', 'Recall@10', 'Confidence', 'Cost', 'Status'].map(h => (
                                    <th key={h} style={{
                                        padding: '0.6rem 0.8rem',
                                        textAlign: 'left',
                                        color: 'var(--text-dim)',
                                        fontWeight: 500,
                                        textTransform: 'uppercase',
                                        fontSize: '0.7rem',
                                        letterSpacing: '0.05em',
                                    }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {(metrics.recent_queries || []).slice().reverse().map((q, i) => (
                                <tr key={i} style={{
                                    borderBottom: '1px solid rgba(255,255,255,0.03)',
                                    background: q.is_failure ? 'rgba(239,68,68,0.05)' : 'transparent',
                                }}>
                                    <td style={{ padding: '0.6rem 0.8rem', color: 'var(--text-muted)', maxWidth: '250px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {q.query}
                                    </td>
                                    <td style={{ padding: '0.6rem 0.8rem', fontFamily: 'monospace', color: q.latency_ms > 5000 ? '#ef4444' : q.latency_ms > 3000 ? '#f59e0b' : '#10b981' }}>
                                        {q.latency_ms}ms
                                    </td>
                                    <td style={{ padding: '0.6rem 0.8rem', fontFamily: 'monospace', color: q.ndcg_10 > 0.8 ? '#10b981' : q.ndcg_10 > 0.5 ? '#f59e0b' : '#ef4444' }}>
                                        {q.ndcg_10?.toFixed(3)}
                                    </td>
                                    <td style={{ padding: '0.6rem 0.8rem', fontFamily: 'monospace', color: q.mrr > 0.8 ? '#10b981' : '#f59e0b' }}>
                                        {q.mrr?.toFixed(3)}
                                    </td>
                                    <td style={{ padding: '0.6rem 0.8rem', fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                                        {q.recall_10?.toFixed(3)}
                                    </td>
                                    <td style={{ padding: '0.6rem 0.8rem' }}>
                                        <span style={{
                                            padding: '0.15rem 0.5rem',
                                            borderRadius: '999px',
                                            fontSize: '0.7rem',
                                            fontWeight: 600,
                                            background: q.confidence === 'HIGH' ? 'rgba(16,185,129,0.15)' :
                                                q.confidence === 'MEDIUM' ? 'rgba(245,158,11,0.15)' : 'rgba(239,68,68,0.15)',
                                            color: q.confidence === 'HIGH' ? '#10b981' :
                                                q.confidence === 'MEDIUM' ? '#f59e0b' : '#ef4444',
                                        }}>
                                            {q.confidence || 'N/A'}
                                        </span>
                                    </td>
                                    <td style={{ padding: '0.6rem 0.8rem', fontFamily: 'monospace', color: 'var(--text-dim)' }}>
                                        ${q.cost_usd?.toFixed(4)}
                                    </td>
                                    <td style={{ padding: '0.6rem 0.8rem' }}>
                                        {q.is_failure ? (
                                            <span style={{ color: '#ef4444', fontWeight: 600 }}>FAIL</span>
                                        ) : (
                                            <span style={{ color: '#10b981' }}>OK</span>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Failure Log */}
            {metrics.failures && metrics.failures.length > 0 && (
                <div className="glass-panel" style={{ padding: '1.5rem', borderRadius: '16px', borderLeft: '4px solid #ef4444' }}>
                    <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: '#ef4444' }}>
                        ⚠️ Recent Failures ({metrics.failures.length})
                    </h3>
                    {metrics.failures.slice().reverse().map((f, i) => (
                        <div key={i} style={{
                            padding: '0.8rem',
                            marginBottom: '0.5rem',
                            background: 'rgba(239,68,68,0.05)',
                            borderRadius: '8px',
                            fontSize: '0.82rem',
                        }}>
                            <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>{f.query}</span>
                            <p style={{ color: '#ef4444', fontFamily: 'monospace', fontSize: '0.75rem', marginTop: '0.3rem' }}>
                                {f.error}
                            </p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default MetricsDashboard;
