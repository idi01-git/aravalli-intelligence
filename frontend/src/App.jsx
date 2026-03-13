import React, { useEffect } from 'react';
import useStore from './hooks/useStore';
import Header from './components/layout/Header';
import MapView from './components/layout/MapView';
import AccuracyBar from './components/layout/AccuracyBar';
import ZonePopup from './components/zone-popup/ZonePopup';
import ParameterPanel from './components/parameter-panel/ParameterPanel';

/**
 * App — PRD §15 Layout
 *
 * ┌─────────────── HEADER ────────────────┐
 * │ Left side panel  │  MapView (70%)     │
 * │ (ZonePopup or    │                    │
 * │  ParameterPanel) │                    │
 * └─────────── ACCURACY BAR ──────────────┘
 *
 * When a zone is selected → right panel shows ZonePopup.
 * Otherwise → ParameterPanel is shown.
 */
function App() {
    const boot = useStore(s => s.boot);
    const selectedZoneId = useStore(s => s.selectedZoneId);

    // Hydrate all data on initial mount
    useEffect(() => { boot(); }, [boot]);

    return (
        <div className="app">
            <Header />
            <main className="main-content">
                <MapView />
                <div className="right-panel">
                    {selectedZoneId ? <ZonePopup /> : <ParameterPanel />}
                </div>
            </main>
            <AccuracyBar />
        </div>
    );
}

export default App;
