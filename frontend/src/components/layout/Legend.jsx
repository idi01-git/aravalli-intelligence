import React from 'react';

/**
 * Legend — PRD §17
 * Glass panel on bottom-left of map showing threat type colors.
 */
const TYPES = [
    { label: 'Deforestation', color: 'var(--threat-deforestation)' },
    { label: 'Mining', color: 'var(--threat-mining)' },
    { label: 'Encroachment', color: 'var(--threat-encroachment)' },
    { label: 'Localized', color: 'var(--threat-localized)' },
    { label: 'Unclassified', color: 'var(--threat-unclassified)' },
];

export default function Legend() {
    return (
        <div className="legend glass-panel">
            <div className="legend__title">Threat Types</div>
            {TYPES.map(t => (
                <div key={t.label} className="legend__item">
                    <span className="legend__dot" style={{ background: t.color }} />
                    {t.label}
                </div>
            ))}
            <div className="legend__hint">
                Size = Threat Score<br />
                Opacity = Confidence
            </div>
        </div>
    );
}
