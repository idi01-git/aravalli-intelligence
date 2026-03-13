import React from 'react';
import useStore from '../../hooks/useStore';

/**
 * Header — PRD §15
 * Left: title + subtitle. Right: pipeline status dot + data mode pill.
 */
export default function Header() {
    const health = useStore(s => s.health);
    const isRunning = useStore(s => s.isRunning);

    const dataMode = health?.data_mode || 'synthetic';
    const isSynthetic = dataMode !== 'real_file' && dataMode !== 'gee_live';

    return (
        <header className="header">
            <div className="header-left">
                <h1>Aravalli Intelligence</h1>
                <span className="subtitle">Ecological Monitoring System</span>
            </div>
            <div className="header-right">
                {/* Pipeline status */}
                <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                    <span className={`status-dot ${isRunning ? 'status-dot--running' : 'status-dot--healthy'}`} />
                    {isRunning ? 'Running...' : 'Ready'}
                </span>

                {/* Data mode badge */}
                <span className={`badge ${isSynthetic ? 'badge--warning' : 'badge--success'}`}>
                    {isSynthetic ? 'SYNTHETIC' : 'REAL DATA'}
                </span>
            </div>
        </header>
    );
}
