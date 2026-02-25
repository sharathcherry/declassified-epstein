import React, { useState, useRef, useEffect } from 'react';
import { apiService } from '../services/api';

const ChatInterface = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [rawMode, setRawMode] = useState(false);
    const scrollRef = useRef(null);

    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const submitQuestion = async (question) => {
        if (!question.trim() || loading) return;

        setMessages(prev => [...prev, { role: 'user', content: question }]);
        setInput('');
        setLoading(true);

        try {
            const data = await apiService.chat({ question, raw_mode: rawMode });
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.answer,
                sources: data.sources,
                isRaw: rawMode,
                follow_ups: data.follow_up_questions || [],
                latency: data.latency
            }]);
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: '⚠️ Failed to connect to the backend. Make sure the server is running on port 8000.'
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        submitQuestion(input);
    };

    // Empty state
    if (messages.length === 0 && !loading) {
        return (
            <>
                <div className="main-header">
                    <button className="model-selector">
                        VIBE RAG 2.0 ▾
                    </button>
                    <button
                        className={`mode-toggle ${rawMode ? 'active' : ''}`}
                        onClick={() => setRawMode(!rawMode)}
                        style={{ position: 'absolute', right: '20px' }}
                    >
                        {rawMode ? '📋 RAW' : '🤖 AI'}
                    </button>
                </div>

                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '24px', padding: '20px' }}>
                    <div style={{ fontSize: '2.5rem' }}>🔍</div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: 600, color: 'var(--text-main)' }}>
                        Investigate the Epstein Files
                    </h2>
                    <p style={{ color: 'var(--text-dim)', textAlign: 'center', maxWidth: '500px', lineHeight: '1.6' }}>
                        Ask about people, events, locations, or relationships across 25K+ documents.
                        {rawMode && <span style={{ color: '#ef4444', display: 'block', marginTop: '8px' }}>📋 Raw mode: showing actual documents, no AI summary</span>}
                    </p>

                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center', maxWidth: '600px', marginTop: '8px' }}>
                        {[
                            "Who are the key people connected to Jeffrey Epstein?",
                            "What do flight logs reveal about island visitors?",
                            "What was Ghislaine Maxwell's role?",
                            "Tell me about the financial connections"
                        ].map((q, i) => (
                            <button
                                key={i}
                                onClick={() => submitQuestion(q)}
                                style={{
                                    padding: '10px 16px',
                                    borderRadius: '12px',
                                    border: '1px solid var(--border)',
                                    background: 'transparent',
                                    color: 'var(--text-muted)',
                                    cursor: 'pointer',
                                    fontSize: '0.85rem',
                                    transition: 'var(--transition)',
                                    textAlign: 'left',
                                }}
                                onMouseEnter={e => { e.target.style.background = 'var(--bg-hover)'; e.target.style.color = 'var(--text-main)'; }}
                                onMouseLeave={e => { e.target.style.background = 'transparent'; e.target.style.color = 'var(--text-muted)'; }}
                            >
                                {q}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="input-area">
                    <div className="input-container">
                        <form onSubmit={handleSubmit}>
                            <input
                                className="input-box"
                                type="text"
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                placeholder="Ask anything about the Epstein Files..."
                            />
                            <button type="submit" className="send-btn" disabled={!input.trim()}>
                                ↑
                            </button>
                        </form>
                    </div>
                    <div className="input-footer">
                        VIBE RAG searches 397K chunks across 25K documents using hybrid retrieval
                    </div>
                </div>
            </>
        );
    }

    // Chat state
    return (
        <>
            <div className="main-header">
                <button className="model-selector">
                    VIBE RAG 2.0 ▾
                </button>
                <button
                    className={`mode-toggle ${rawMode ? 'active' : ''}`}
                    onClick={() => setRawMode(!rawMode)}
                    style={{ position: 'absolute', right: '20px' }}
                >
                    {rawMode ? '📋 RAW' : '🤖 AI'}
                </button>
            </div>

            <div className="chat-messages">
                {messages.map((msg, i) => (
                    <div key={i} className={`message-row ${msg.role} fade-in`}>
                        <div className="message-content">
                            <div className={`message-avatar ${msg.role === 'user' ? 'user-avatar' : 'assistant-avatar'}`}>
                                {msg.role === 'user' ? 'U' : '✦'}
                            </div>
                            <div style={{ flex: 1 }}>
                                <div className="message-text">
                                    {msg.content}
                                </div>

                                {/* Sources */}
                                {msg.sources && msg.sources.length > 0 && !msg.isRaw && (
                                    <div className="sources-section">
                                        <div className="sources-label">Sources</div>
                                        <div className="source-pills">
                                            {msg.sources.map((s, si) => (
                                                <div key={si} className="source-pill" title={s.text || 'No preview'}>
                                                    📄 [{si + 1}] {s.filename || s.doc_filename || 'Document'}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Latency */}
                                {msg.latency && (
                                    <div className="latency-badge">
                                        ⚡ {(Object.values(msg.latency).reduce((a, b) => a + b, 0) / 1000).toFixed(1)}s
                                        {msg.isRaw ? ' • Raw documents' : ' • AI analysis'}
                                    </div>
                                )}

                                {/* Follow-ups */}
                                {msg.follow_ups && msg.follow_ups.length > 0 && (
                                    <div className="followup-section">
                                        <div className="sources-label">Follow-up questions</div>
                                        {msg.follow_ups.map((fq, fi) => (
                                            <button
                                                key={fi}
                                                className="followup-btn"
                                                onClick={() => submitQuestion(fq)}
                                                disabled={loading}
                                            >
                                                → {fq}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}

                {/* Loading indicator */}
                {loading && (
                    <div className="message-row assistant fade-in">
                        <div className="message-content">
                            <div className="message-avatar assistant-avatar">✦</div>
                            <div style={{ flex: 1 }}>
                                <div className="typing-indicator">
                                    <div className="typing-dot"></div>
                                    <div className="typing-dot"></div>
                                    <div className="typing-dot"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={scrollRef}></div>
            </div>

            <div className="input-area">
                <div className="input-container">
                    <form onSubmit={handleSubmit}>
                        <input
                            className="input-box"
                            type="text"
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            placeholder="Ask anything..."
                            autoFocus
                        />
                        <button type="submit" className="send-btn" disabled={loading || !input.trim()}>
                            ↑
                        </button>
                    </form>
                </div>
                <div className="input-footer">
                    VIBE RAG can make mistakes. Verify with source documents.
                </div>
            </div>
        </>
    );
};

export default ChatInterface;
