/**
 * hooks/useStore.js — Zustand global state store.
 * Manages: selected zone, config, summary data, pipeline status, panel mode.
 */
import { create } from 'zustand';
import {
    fetchConfig,
    fetchSummary,
    fetchGeoJSON,
    fetchHealth,
    runAnalysis,
} from '../api/client';

const useStore = create((set, get) => ({
    // ── Zone selection ─────────────────────────────────────────────────────
    selectedZoneId: null,
    selectZone: (id) => set({ selectedZoneId: id }),
    clearZone: () => set({ selectedZoneId: null }),

    // ── Panel mode ─────────────────────────────────────────────────────────
    panelMode: 'normal', // 'normal' | 'scientific'
    setPanelMode: (mode) => set({ panelMode: mode }),

    // ── Health / status ────────────────────────────────────────────────────
    health: null,
    loadHealth: async () => {
        try { set({ health: await fetchHealth() }); }
        catch { set({ health: null }); }
    },

    // ── Summary (AccuracyBar data) ─────────────────────────────────────────
    summary: null,
    loadSummary: async () => {
        try { set({ summary: await fetchSummary() }); }
        catch { set({ summary: null }); }
    },

    // ── GeoJSON (map data) ─────────────────────────────────────────────────
    geojson: null,
    loadGeoJSON: async () => {
        try { set({ geojson: await fetchGeoJSON() }); }
        catch { set({ geojson: null }); }
    },

    // ── Config (for ParameterPanel init) ───────────────────────────────────
    config: null,
    loadConfig: async () => {
        try { set({ config: await fetchConfig() }); }
        catch { set({ config: null }); }
    },

    // ── Pipeline run ───────────────────────────────────────────────────────
    isRunning: false,
    runResult: null,
    runError: null,
    triggerAnalysis: async (sensitivity, overrides) => {
        set({ isRunning: true, runError: null });
        try {
            const result = await runAnalysis(sensitivity, overrides);
            set({ runResult: result, isRunning: false });
            // Refresh data after successful run
            get().loadSummary();
            get().loadGeoJSON();
            get().loadHealth();
        } catch (err) {
            set({
                runError: err?.response?.data?.detail || 'Pipeline failed',
                isRunning: false,
            });
        }
    },

    // ── Boot: load everything on mount ─────────────────────────────────────
    boot: async () => {
        const s = get();
        await Promise.allSettled([
            s.loadHealth(),
            s.loadSummary(),
            s.loadGeoJSON(),
            s.loadConfig(),
        ]);
    },
}));

export default useStore;
