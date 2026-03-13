import React, { useEffect } from 'react';
import useStore from '../../hooks/useStore';

/**
 * AccuracyBar — PRD §20
 * Bottom 72px strip with 6 metric cards powered by GET /api/summary.
 */

function borderColor(type, value) {
    if (type === 'precision' || type === 'recall') {
        if (value >= 85) return 'var(--status-success)';
        if (value >= 70) return 'var(--status-warning)';
        return 'var(--status-error)';
    }
    if (type === 'f1') {
        if (value >= 0.80) return 'var(--status-success)';
        if (value >= 0.65) return 'var(--status-warning)';
        return 'var(--status-error)';
    }
    if (type === 'drift') {
        if (value < 3) return 'var(--status-success)';
        if (value < 5) return 'var(--status-warning)';
        if (value < 7) return 'var(--threat-mining)';
        return 'var(--status-error)';
    }
    return 'var(--text-secondary)';
}

function threatColor(total, threats) {
    const pct = total > 0 ? (threats / total) * 100 : 0;
    if (pct < 5) return 'var(--status-success)';
    if (pct < 15) return 'var(--status-warning)';
    return 'var(--status-error)';
}

function MetricCard({ label, value, borderClr }) {
    return (
        <div className="metric-card">
            <span className="metric-card__label">{label}</span>
            <span className="metric-card__value">{value}</span>
            <div className="metric-card__border" style={{ background: borderClr }} />
        </div>
    );
}

export default function AccuracyBar() {
    const summary = useStore(s => s.summary);

    if (!summary) {
        return (
            <footer className="accuracy-bar">
                {['Zones Analyzed', 'Threats', 'Precision', 'Recall', 'F1 Score', 'Avg Drift'].map(l => (
                    <MetricCard key={l} label={l} value="--" borderClr="var(--text-muted)" />
                ))}
            </footer>
        );
    }

    const prec = summary.pipeline_accuracy?.precision;
    const rec = summary.pipeline_accuracy?.recall;
    const f1 = summary.pipeline_accuracy?.f1 ?? summary.pipeline_accuracy?.f1_score;
    const drift = summary.avg_drift_score;

    return (
        <footer className="accuracy-bar">
            <MetricCard
                label="Zones Analyzed"
                value={summary.total_zones?.toLocaleString() ?? '--'}
                borderClr="var(--text-secondary)"
            />
            <MetricCard
                label="Threats Detected"
                value={summary.total_threats?.toString() ?? '--'}
                borderClr={threatColor(summary.total_zones, summary.total_threats)}
            />
            <MetricCard
                label="Precision"
                value={prec != null ? `${(prec * 100).toFixed(1)}%` : '--'}
                borderClr={borderColor('precision', prec != null ? prec * 100 : 0)}
            />
            <MetricCard
                label="Recall"
                value={rec != null ? `${(rec * 100).toFixed(1)}%` : '--'}
                borderClr={borderColor('recall', rec != null ? rec * 100 : 0)}
            />
            <MetricCard
                label="F1 Score"
                value={f1 != null ? f1.toFixed(3) : '--'}
                borderClr={borderColor('f1', f1 ?? 0)}
            />
            <MetricCard
                label="Avg Drift"
                value={drift != null ? drift.toFixed(1) : '--'}
                borderClr={borderColor('drift', drift ?? 0)}
            />
        </footer>
    );
}
