// In dev: Vite proxy handles /api → localhost:8000
// In production: set VITE_API_URL to your backend URL (e.g. https://username-space.hf.space)
const API_BASE_URL = import.meta.env.VITE_API_URL
    ? `${import.meta.env.VITE_API_URL}/api`
    : '/api';

export const apiService = {
    async getStatus() {
        const response = await fetch(`${API_BASE_URL}/status`);
        if (!response.ok) throw new Error('Failed to fetch status');
        return response.json();
    },

    async search(params) {
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!response.ok) throw new Error('Search failed');
        return response.json();
    },

    async chat(params) {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!response.ok) throw new Error('Chat failed');
        return response.json();
    },

    async getMetrics() {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        if (!response.ok) throw new Error('Failed to fetch metrics');
        return response.json();
    }
};
