import React, { useEffect, useState } from 'react';
import useStore from '../../hooks/useStore';
import { fetchZone, fetchTimeseries, fetchImportance } from '../../api/client';
import TimeseriesChart from '../charts/TimeseriesChart';
import MultiIndexChart from '../charts/MultiIndexChart';
import DriversChart from '../charts/DriversChart';

/**
 * ZonePopup — PRD §18
 * Right drawer that opens when a zone is selected on the map.
 * Shows: summary bar, seasonal analysis, spectral indices, detection methods,
 * primary factors chart, trend analysis charts, AI report, and action buttons.
 */

const THREAT_COLORS = {
    deforestation: 'var(--threat-deforestation)',
    mining: 'var(--threat-mining)',
    encroachment: 'var(--threat-encroachment)',
    localized_disturbance: 'var(--threat-localized)',
    unclassified: 'var(--threat-unclassified)',
};

const METHODS = [
    { key: 'isolation_forest', label: 'IsolationForest', weight: 0.35 },
    { key: 'dbscan', label: 'DBSCAN', weight: 0.30 },
    { key: 'lof', label: 'LOF', weight: 0.25 },
    { key: 'kmeans', label: 'KMeans', weight: 0.10 },
];

export default function ZonePopup() {
    const zoneId = useStore(s => s.selectedZoneId);
    const clearZone = useStore(s => s.clearZone);

    const [zone, setZone] = useState(null);
    const [ts, setTs] = useState(null);
    const [drivers, setDrivers] = useState(null);
    const [loading, setLoading] = useState(true);
    const [reportExpanded, setReportExpanded] = useState(false);

    useEffect(() => {
        if (!zoneId) return;
        setLoading(true);
        Promise.allSettled([
            fetchZone(zoneId).then(setZone),
            fetchTimeseries(zoneId).then(setTs),
            fetchImportance(zoneId).then(d => setDrivers(d?.top_features)),
        ]).finally(() => setLoading(false));
    }, [zoneId]);

    if (!zoneId) return null;
    if (loading) return <div className="zone-popup"><div className="loading-state"><span className="spinner" /> Loading zone data...</div></div>;
    if (!zone) return <div className="zone-popup"><div className="loading-state">Zone data unavailable</div></div>;

    const threatColor = THREAT_COLORS[zone.threat_type] || 'var(--text-muted)';
    const severity = zone.drift_severity || zone.severity || 'unknown';
    const ensembleVotes = zone.ensemble_votes ?? 0;
    const report = zone.ai_report || '';
    const reportSource = zone.report_source || 'unknown';

    return (
        <div className="zone-popup" style={{ position: 'relative' }}>
            <button className="popup-close" onClick={clearZone} title="Close">✕</button>

            {/* ── HEADER ────────────────────────────────────────────────── */}
            <div className="zone-popup__header">
                <div className="zone-popup__title-row">
                    <span className="zone-popup__threat-dot" style={{ background: threatColor }} />
                    <span className="zone-popup__type" style={{ color: threatColor }}>
                        {(zone.threat_type || 'unknown').replace(/_/g, ' ')}
                    </span>
                    <span className="zone-popup__zone-id">{zoneId}</span>
                </div>

                {/* Threat score bar */}
                <div className="score-bar">
                    <div className="score-bar__fill" style={{
                        width: `${Math.min(100, zone.threat_score || 0)}%`,
                        background: threatColor,
                    }} />
                </div>

                {/* Badges */}
                <div className="zone-popup__badges">
                    <span className={`badge ${(zone.confidence || 0) >= 60 ? 'badge--success' : 'badge--warning'}`}>
                        {(zone.confidence || 0).toFixed(0)}% confidence
                    </span>
                    <span className={`badge ${(zone.drift_score || 0) >= 7 ? 'badge--error' : 'badge--warning'}`}>
                        {(zone.drift_score || 0).toFixed(1)} / 10 drift
                    </span>
                    <span className="badge badge--warning" style={{ textTransform: 'capitalize' }}>
                        {severity}
                    </span>
                </div>
            </div>

            {/* ── SEASONAL ANALYSIS ─────────────────────────────────────── */}
            <div className="popup-section">
                <div className="popup-section__title">Seasonal Analysis</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
                    <div>
                        <span style={{ color: 'var(--text-muted)' }}>DSR</span>
                        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 600 }}>
                            {(zone.dsr || 0).toFixed(2)}
                        </div>
                    </div>
                    <div>
                        <span style={{ color: 'var(--text-muted)' }}>Classification</span>
                        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-primary)' }}>
                            {(zone.dsr_classification || '--').replace(/_/g, ' ')}
                        </div>
                    </div>
                </div>
            </div>

            {/* ── SPECTRAL INDICES ──────────────────────────────────────── */}
            <div className="popup-section">
                <div className="popup-section__title">Spectral Indices</div>
                <div className="index-table">
                    {[
                        { name: 'NDVI', delta: zone.ndvi_delta },
                        { name: 'NDBI', delta: zone.ndbi_delta },
                        { name: 'BSI', delta: zone.bsi_delta },
                        { name: 'Nightlight', delta: zone.nightlight_delta },
                    ].map(idx => (
                        <div key={idx.name} className="index-row">
                            <span className="index-row__name">{idx.name}</span>
                            <span className={`index-row__delta ${(idx.delta || 0) >= 0 ? 'index-row__delta--positive' : 'index-row__delta--negative'}`}>
                                {(idx.delta || 0) >= 0 ? '▲' : '▼'} {Math.abs(idx.delta || 0).toFixed(3)}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* ── DETECTION METHODS ─────────────────────────────────────── */}
            <div className="popup-section">
                <div className="popup-section__title">Detection Methods — Flagged by {ensembleVotes} of 4</div>
                <div className="methods-grid">
                    {METHODS.map((m, i) => {
                        const flagged = i < ensembleVotes;
                        return (
                            <div key={m.key} className={`method-card ${flagged ? 'method-card--flagged' : 'method-card--clear'}`}>
                                <span className="method-card__icon">{flagged ? '✓' : '—'}</span>
                                <span>{m.label}</span>
                                <span className="method-card__weight">{m.weight}</span>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* ── PRIMARY FACTORS (Drivers Chart) ───────────────────────── */}
            <div className="popup-section">
                <div className="popup-section__title">Primary Factors</div>
                <DriversChart drivers={drivers} />
            </div>

            {/* ── TREND ANALYSIS (Timeseries Charts) ────────────────────── */}
            <div className="popup-section">
                <div className="popup-section__title">Trend Analysis — NDVI</div>
                <TimeseriesChart data={ts?.data} metadata={ts?.metadata} />
                {ts?.data && (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                        {zone.consecutive_declines > 0 && `Declining for ${zone.consecutive_declines} consecutive months`}
                        {zone.slope_short != null && ` · Short-term: ${zone.slope_short > 0 ? '+' : ''}${zone.slope_short.toFixed(3)}`}
                    </div>
                )}
            </div>

            <div className="popup-section">
                <div className="popup-section__title">All Indices — 84-Month Overlay</div>
                <MultiIndexChart data={ts?.data} />
            </div>

            {/* ── AI REPORT ─────────────────────────────────────────────── */}
            {report && (
                <div className="popup-section">
                    <div className="popup-section__title">Analysis Report</div>
                    <div className="ai-report">
                        {reportExpanded ? report : report.slice(0, 160) + (report.length > 160 ? '...' : '')}
                        {report.length > 160 && (
                            <button
                                onClick={() => setReportExpanded(!reportExpanded)}
                                style={{
                                    display: 'block', marginTop: 6, background: 'none', border: 'none',
                                    color: 'var(--accent)', cursor: 'pointer', fontSize: 12, padding: 0,
                                }}
                            >
                                {reportExpanded ? 'Show less' : 'Show full report'}
                            </button>
                        )}
                        <span className={`badge ai-report__source ${reportSource.includes('live') ? 'badge--success' :
                            reportSource === 'cached' ? 'badge--warning' : 'badge--error'
                            }`}>
                            {reportSource.replace(/_/g, ' ')}
                        </span>
                    </div>
                </div>
            )}

            {/* ── ACTIONS ───────────────────────────────────────────────── */}
            <div className="action-row">
                <button
                    className="btn btn--secondary"
                    style={{ flex: 1 }}
                    onClick={() => window.open(`https://maps.google.com/?q=${zone.lat},${zone.lon}`, '_blank')}
                >
                    Open in Google Maps
                </button>
                <button
                    className="btn btn--secondary"
                    style={{ flex: 1 }}
                    onClick={() => {
                        navigator.clipboard.writeText(`${zone.lat}, ${zone.lon}`);
                    }}
                >
                    Copy Coordinates
                </button>
            </div>
        </div>
    );
}
