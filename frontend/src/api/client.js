/**
 * api/client.js — Axios wrapper for all Aravalli Intelligence backend endpoints.
 * The Vite dev proxy routes /api/* to http://localhost:8000 — see vite.config.js.
 */
import axios from 'axios';

const api = axios.create({ baseURL: '/api', timeout: 60000 });

// ── Endpoints ────────────────────────────────────────────────────────────────

export const fetchHealth = () => api.get('/health').then(r => r.data);
export const fetchSummary = () => api.get('/summary').then(r => r.data);
export const fetchConfig = () => api.get('/config').then(r => r.data);
export const fetchAccuracy = () => api.get('/accuracy').then(r => r.data);
export const fetchGeoJSON = () => api.get('/zones/geojson').then(r => r.data);

export const fetchZones = (page = 1, size = 100, detectedOnly = false) =>
    api.get('/zones', { params: { page, size, detected_only: detectedOnly } }).then(r => r.data);

export const fetchZone = (id) => api.get(`/zones/${id}`).then(r => r.data);
export const fetchTimeseries = (id) => api.get(`/zones/${id}/timeseries`).then(r => r.data);
export const fetchImportance = (id) => api.get(`/zones/${id}/importance`).then(r => r.data);
export const fetchNeighbors = (id) => api.get(`/zones/${id}/neighbors`).then(r => r.data);

export const runAnalysis = (sensitivity = 'medium', overrides = {}) =>
    api.post('/analyze', { sensitivity, overrides }).then(r => r.data);

export default api;
