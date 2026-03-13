import React, { useState, useEffect, useMemo, useCallback } from 'react';
import useStore from '../../hooks/useStore';

/**
 * ParameterPanel — PRD §19 + ui_parameters_plan.md
 *
 * Two modes toggled at top:
 *   - Normal:     3 simple sections (Data Source, Sensitivity, AI Reporter)
 *   - Scientific: All 84 parameters in 8 collapsible accordion sections with
 *                 weight normalization guards, validation banners, and run button lock.
 */

// ── Reusable micro-components ────────────────────────────────────────────────

function SliderRow({ label, value, min, max, step, onChange, disabled }) {
    return (
        <div className="param-row">
            <div className="param-row__label">
                <span className="param-row__name">{label}</span>
                <span className="param-row__value">{typeof value === 'number' ? value.toFixed(step < 1 ? 2 : 0) : value}</span>
            </div>
            <input
                type="range" min={min} max={max} step={step}
                value={value} onChange={e => onChange(parseFloat(e.target.value))}
                disabled={disabled}
            />
        </div>
    );
}

function ToggleRow({ label, checked, onChange }) {
    return (
        <div className="toggle-row">
            <span className="toggle-row__label">{label}</span>
            <label className="toggle">
                <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
                <span className="toggle__track" />
                <span className="toggle__thumb" />
            </label>
        </div>
    );
}

function NumberRow({ label, value, min, max, onChange }) {
    const clamp = (v) => Math.min(max, Math.max(min, v));
    return (
        <div className="param-row">
            <div className="param-row__label">
                <span className="param-row__name">{label}</span>
            </div>
            <input
                type="number" min={min} max={max} value={value}
                onChange={e => onChange(clamp(parseInt(e.target.value) || min))}
            />
        </div>
    );
}

function Section({ title, defaultOpen, children, badge }) {
    const [open, setOpen] = useState(defaultOpen || false);
    return (
        <div className="param-section">
            <div className="param-section__header" onClick={() => setOpen(!open)}>
                <span className="param-section__label">{title}{badge ? ` (${badge})` : ''}</span>
                <span className={`param-section__chevron ${open ? 'param-section__chevron--open' : ''}`}>▾</span>
            </div>
            {open && <div className="param-section__body">{children}</div>}
        </div>
    );
}

function WeightBar({ sum, onNormalize }) {
    const valid = Math.abs(sum - 1.0) < 0.02;
    return (
        <div className={`weight-bar ${valid ? 'weight-bar--valid' : 'weight-bar--invalid'}`}>
            <span>Weight Sum: {sum.toFixed(2)} / 1.00</span>
            {!valid && (
                <button className="btn btn--normalize btn--secondary" onClick={onNormalize}>Normalize</button>
            )}
        </div>
    );
}

// ── Default parameter state ──────────────────────────────────────────────────

function getDefaults() {
    return {
        // Data
        data_mode: 'synthetic',
        zone_count: 1000,
        noise_level: 0.03,
        deforestation_pct: 0.10,
        encroachment_pct: 0.05,
        mining_pct: 0.03,
        seed: 42,

        // Ensemble
        iso_enabled: true, iso_weight: 0.35, iso_contamination: 0.10, iso_trees: 200,
        dbscan_enabled: true, dbscan_weight: 0.30, dbscan_eps: 0.50, dbscan_min_samples: 3,
        lof_enabled: true, lof_weight: 0.25, lof_neighbors: 20,
        kmeans_enabled: true, kmeans_weight: 0.10, kmeans_clusters: 8,
        min_weighted_score: 0.50,
        feature_scaler: 'robust',

        // Classification signatures
        defor_ndvi_min: -0.10, defor_bsi_max: 0.08, defor_ndbi_max: 0.10,
        mining_ndvi_min: -0.08, mining_bsi_rise: 0.08, mining_ndbi_max: 0.10,
        encr_ndvi_min: -0.08, encr_ndbi_rise: 0.05, encr_nightlight: 5,

        // Score weights
        sw_vegetation: 0.30, sw_urban: 0.20, sw_soil: 0.20, sw_nightlight: 0.15, sw_seasonal: 0.15,

        // Drift & temporal
        dsr_threshold: 1.50,
        smoothing_window: 3,
        consecutive_required: 6,
        drift_ndvi_w: 0.35, drift_temporal_w: 0.25, drift_spatial_w: 0.20, drift_dsr_w: 0.20,
        sev_moderate: 3.0, sev_high: 5.0, sev_severe: 7.0, sev_critical: 8.5,

        // Filters
        monsoon_enabled: true,
        recovery_enabled: true,
        regional_enabled: true,
        duration_enabled: true,
        duration_min: 3,
        confidence_floor: 30,

        // Spatial
        k_neighbors: 8,
        morans_threshold: 0.30,
        geojson_radius: 1.5,

        // AI reporting
        primary_model: 'openai/gpt-oss-120b',
        primary_temp: 0.30,
        primary_max_tokens: 300,
        fallback_model: 'llama-3.3-70b-versatile',
        fallback_temp: 0.30,

        // Normal mode macros
        sensitivity: 'medium',
        report_style: 'balanced',
        ai_strictness: 0.30,
    };
}

// ── Build overrides for POST /api/analyze ─────────────────────────────────────

function buildOverrides(p) {
    return {
        data: { mode: p.data_mode, synthetic: { zone_count: p.zone_count, noise_level: p.noise_level, deforestation_pct: p.deforestation_pct, encroachment_pct: p.encroachment_pct, mining_pct: p.mining_pct, seed: p.seed } },
        ensemble: {
            methods: {
                isolation_forest: { enabled: p.iso_enabled, weight: p.iso_weight, contamination: p.iso_contamination, n_estimators: p.iso_trees },
                dbscan: { enabled: p.dbscan_enabled, weight: p.dbscan_weight, eps: p.dbscan_eps, min_samples: p.dbscan_min_samples },
                lof: { enabled: p.lof_enabled, weight: p.lof_weight, n_neighbors: p.lof_neighbors },
                kmeans: { enabled: p.kmeans_enabled, weight: p.kmeans_weight, n_clusters: p.kmeans_clusters },
            },
            min_weighted_score: p.min_weighted_score,
            feature_scaler: p.feature_scaler,
        },
        classification: {
            deforestation: { ndvi_drop_min: p.defor_ndvi_min, bsi_max: p.defor_bsi_max, ndbi_max: p.defor_ndbi_max },
            mining: { ndvi_drop_min: p.mining_ndvi_min, bsi_rise_min: p.mining_bsi_rise, ndbi_max: p.mining_ndbi_max },
            encroachment: { ndvi_drop_min: p.encr_ndvi_min, ndbi_rise_min: p.encr_ndbi_rise, nightlight_min: p.encr_nightlight },
            score_weights: { vegetation: p.sw_vegetation, urban: p.sw_urban, soil: p.sw_soil, nightlight: p.sw_nightlight, seasonal_proof: p.sw_seasonal },
        },
        dsr: { threshold_normal: p.dsr_threshold },
        baseline: { smoothing_window: p.smoothing_window },
        drift: {
            weights: { ndvi: p.drift_ndvi_w, temporal: p.drift_temporal_w, spatial: p.drift_spatial_w, dsr: p.drift_dsr_w },
            severity: { moderate: p.sev_moderate, high: p.sev_high, severe: p.sev_severe, critical: p.sev_critical },
        },
        filters: {
            monsoon: { enabled: p.monsoon_enabled },
            recovery: { enabled: p.recovery_enabled },
            regional_context: { enabled: p.regional_enabled },
            duration_check: { enabled: p.duration_enabled, min_consecutive: p.duration_min },
            confidence_floor: { value: p.confidence_floor },
        },
        spatial: { k_neighbors: p.k_neighbors, morans_i_threshold: p.morans_threshold, geojson: { buffer_radius_km: p.geojson_radius } },
        llm: {
            primary: { model: p.primary_model, temperature: p.primary_temp, max_tokens: p.primary_max_tokens },
            fallback: { model: p.fallback_model, temperature: p.fallback_temp },
        },
    };
}

// ── Main component ───────────────────────────────────────────────────────────

export default function ParameterPanel() {
    const panelMode = useStore(s => s.panelMode);
    const setMode = useStore(s => s.setPanelMode);
    const isRunning = useStore(s => s.isRunning);
    const runError = useStore(s => s.runError);
    const trigger = useStore(s => s.triggerAnalysis);

    const [p, setP] = useState(getDefaults);
    const set = useCallback((key, val) => setP(prev => ({ ...prev, [key]: val })), []);

    // ── Validation ──────────────────────────────────────────────────────────
    const enabledMethods = [
        p.iso_enabled && p.iso_weight,
        p.dbscan_enabled && p.dbscan_weight,
        p.lof_enabled && p.lof_weight,
        p.kmeans_enabled && p.kmeans_weight,
    ].filter(Boolean);
    const ensembleSum = (p.iso_enabled ? p.iso_weight : 0) + (p.dbscan_enabled ? p.dbscan_weight : 0) +
        (p.lof_enabled ? p.lof_weight : 0) + (p.kmeans_enabled ? p.kmeans_weight : 0);
    const ensembleValid = Math.abs(ensembleSum - 1.0) < 0.02;
    const anyMethodOn = enabledMethods.length > 0;

    const scoreSum = p.sw_vegetation + p.sw_urban + p.sw_soil + p.sw_nightlight + p.sw_seasonal;
    const scoreValid = Math.abs(scoreSum - 1.0) < 0.02;

    const driftSum = p.drift_ndvi_w + p.drift_temporal_w + p.drift_spatial_w + p.drift_dsr_w;
    const driftValid = Math.abs(driftSum - 1.0) < 0.02;

    const eventSum = p.deforestation_pct + p.encroachment_pct + p.mining_pct;
    const eventValid = eventSum <= 0.60;

    const sevValid = p.sev_moderate < p.sev_high && p.sev_high < p.sev_severe && p.sev_severe < p.sev_critical;

    const allFiltersOff = !p.monsoon_enabled && !p.recovery_enabled && !p.regional_enabled && !p.duration_enabled;

    const canRun = panelMode === 'normal' || (anyMethodOn && ensembleValid && scoreValid && driftValid && eventValid && sevValid);

    // ── Normalize helpers ──────────────────────────────────────────────────
    const normalizeEnsemble = () => {
        const keys = [
            ['iso_enabled', 'iso_weight'], ['dbscan_enabled', 'dbscan_weight'],
            ['lof_enabled', 'lof_weight'], ['kmeans_enabled', 'kmeans_weight'],
        ];
        const active = keys.filter(([ek]) => p[ek]);
        const total = active.reduce((s, [, wk]) => s + p[wk], 0);
        if (total === 0) return;
        const updates = {};
        active.forEach(([, wk]) => { updates[wk] = parseFloat((p[wk] / total).toFixed(2)); });
        setP(prev => ({ ...prev, ...updates }));
    };

    const normalizeScoreWeights = () => {
        const keys = ['sw_vegetation', 'sw_urban', 'sw_soil', 'sw_nightlight', 'sw_seasonal'];
        const total = keys.reduce((s, k) => s + p[k], 0);
        if (total === 0) return;
        const updates = {};
        keys.forEach(k => { updates[k] = parseFloat((p[k] / total).toFixed(2)); });
        setP(prev => ({ ...prev, ...updates }));
    };

    const normalizeDriftWeights = () => {
        const keys = ['drift_ndvi_w', 'drift_temporal_w', 'drift_spatial_w', 'drift_dsr_w'];
        const total = keys.reduce((s, k) => s + p[k], 0);
        if (total === 0) return;
        const updates = {};
        keys.forEach(k => { updates[k] = parseFloat((p[k] / total).toFixed(2)); });
        setP(prev => ({ ...prev, ...updates }));
    };

    // ── Apply sensitivity preset in Normal mode ────────────────────────────
    useEffect(() => {
        if (panelMode !== 'normal') return;
        // High sensitivity = lower threshold = more detections
        // Low sensitivity = higher threshold = fewer detections
        if (p.sensitivity === 'low') {
            setP(prev => ({ ...prev, min_weighted_score: 0.70, confidence_floor: 50 }));
        } else if (p.sensitivity === 'high') {
            setP(prev => ({ ...prev, min_weighted_score: 0.30, confidence_floor: 20 }));
        } else {
            setP(prev => ({ ...prev, min_weighted_score: 0.50, confidence_floor: 30 }));
        }
    }, [p.sensitivity, panelMode]);

    // ── Run handler ────────────────────────────────────────────────────────
    const handleRun = () => {
        if (!canRun || isRunning) return;
        const overrides = buildOverrides(p);
        trigger(p.sensitivity, overrides);
    };

    // ══════════════════════════════════════════════════════════════════════════
    // RENDER
    // ══════════════════════════════════════════════════════════════════════════

    return (
        <div className="param-panel">
            {/* Mode toggle */}
            <div className="mode-toggle">
                <button className={`mode-toggle__btn ${panelMode === 'normal' ? 'mode-toggle__btn--active' : ''}`}
                    onClick={() => setMode('normal')}>Simple</button>
                <button className={`mode-toggle__btn ${panelMode === 'scientific' ? 'mode-toggle__btn--active' : ''}`}
                    onClick={() => setMode('scientific')}>Advanced</button>
            </div>

            {/* ── NORMAL MODE ──────────────────────────────────────────── */}
            {panelMode === 'normal' && (
                <>
                    {/* Data Source */}
                    <Section title="Data Source" defaultOpen={true}>
                        <div className="radio-group">
                            {[
                                { val: 'synthetic', label: 'Synthetic — AI-generated demo data' },
                                { val: 'real_file', label: 'Real — Satellite data files' },
                                { val: 'gee_live', label: 'Live — Google Earth Engine' },
                            ].map(opt => (
                                <div key={opt.val}
                                    className={`radio-option ${p.data_mode === opt.val ? 'radio-option--active' : ''}`}
                                    onClick={() => set('data_mode', opt.val)}>
                                    <span className={`radio-dot ${p.data_mode === opt.val ? 'radio-dot--active' : ''}`} />
                                    <span>{opt.label}</span>
                                </div>
                            ))}
                        </div>
                    </Section>

                    {/* Sensitivity */}
                    <Section title="Detection Sensitivity" defaultOpen={true}>
                        <div className="radio-group">
                            {[
                                { val: 'low', label: 'Low — High confidence alerts only' },
                                { val: 'medium', label: 'Balanced — Recommended' },
                                { val: 'high', label: 'High — Maximum detection coverage' },
                            ].map(opt => (
                                <div key={opt.val}
                                    className={`radio-option ${p.sensitivity === opt.val ? 'radio-option--active' : ''}`}
                                    onClick={() => set('sensitivity', opt.val)}>
                                    <span className={`radio-dot ${p.sensitivity === opt.val ? 'radio-dot--active' : ''}`} />
                                    <span>{opt.label}</span>
                                </div>
                            ))}
                        </div>
                    </Section>

                    {/* AI Reporter */}
                    <Section title="AI Field Reporter" defaultOpen={true}>
                        <div className="param-row">
                            <div className="param-row__label"><span className="param-row__name">Report Style</span></div>
                            <select value={p.report_style} onChange={e => set('report_style', e.target.value)}>
                                <option value="ranger">Field Ranger (short)</option>
                                <option value="balanced">Analyst (balanced)</option>
                                <option value="research">Research (detailed)</option>
                            </select>
                        </div>
                        <SliderRow label="AI Strictness (Temperature)" value={p.ai_strictness}
                            min={0.10} max={0.70} step={0.05}
                            onChange={v => { set('ai_strictness', v); set('primary_temp', v); }} />
                    </Section>
                </>
            )}

            {/* ── SCIENTIFIC MODE ──────────────────────────────────────── */}
            {panelMode === 'scientific' && (
                <>
                    {/* A: Data & Synthetic */}
                    <Section title="Data & Synthetic Settings">
                        <div className="param-row">
                            <div className="param-row__label"><span className="param-row__name">Data Source</span></div>
                            <select value={p.data_mode} onChange={e => set('data_mode', e.target.value)}>
                                <option value="synthetic">Synthetic</option>
                                <option value="real_file">Real File</option>
                                <option value="gee_live">GEE Live</option>
                            </select>
                        </div>
                        <NumberRow label="Zone Count" value={p.zone_count} min={100} max={2000} onChange={v => set('zone_count', v)} />
                        <SliderRow label="Sensor Noise" value={p.noise_level} min={0.01} max={0.10} step={0.01} onChange={v => set('noise_level', v)} />
                        <SliderRow label="Deforestation %" value={p.deforestation_pct} min={0} max={0.30} step={0.01} onChange={v => set('deforestation_pct', v)} />
                        <SliderRow label="Encroachment %" value={p.encroachment_pct} min={0} max={0.30} step={0.01} onChange={v => set('encroachment_pct', v)} />
                        <SliderRow label="Mining %" value={p.mining_pct} min={0} max={0.20} step={0.01} onChange={v => set('mining_pct', v)} />
                        <NumberRow label="Random Seed" value={p.seed} min={0} max={99999} onChange={v => set('seed', v)} />
                        {!eventValid && <div className="warning-banner warning-banner--error">Event rates exceed 60% — reduce combined percentages.</div>}
                    </Section>

                    {/* B: Ensemble ML */}
                    <Section title="Ensemble Methods" defaultOpen={true}>
                        {/* Isolation Forest */}
                        <ToggleRow label="Isolation Forest" checked={p.iso_enabled} onChange={v => set('iso_enabled', v)} />
                        {p.iso_enabled && <>
                            <SliderRow label="Weight" value={p.iso_weight} min={0} max={1} step={0.05} onChange={v => set('iso_weight', v)} />
                            <SliderRow label="Contamination" value={p.iso_contamination} min={0.01} max={0.30} step={0.01} onChange={v => set('iso_contamination', v)} />
                            <SliderRow label="N Trees" value={p.iso_trees} min={50} max={500} step={10} onChange={v => set('iso_trees', v)} />
                        </>}

                        {/* DBSCAN */}
                        <ToggleRow label="DBSCAN" checked={p.dbscan_enabled} onChange={v => set('dbscan_enabled', v)} />
                        {p.dbscan_enabled && <>
                            <SliderRow label="Weight" value={p.dbscan_weight} min={0} max={1} step={0.05} onChange={v => set('dbscan_weight', v)} />
                            <SliderRow label="Epsilon" value={p.dbscan_eps} min={0.10} max={2.00} step={0.05} onChange={v => set('dbscan_eps', v)} />
                            <SliderRow label="Min Samples" value={p.dbscan_min_samples} min={2} max={15} step={1} onChange={v => set('dbscan_min_samples', v)} />
                        </>}

                        {/* LOF */}
                        <ToggleRow label="Local Outlier Factor" checked={p.lof_enabled} onChange={v => set('lof_enabled', v)} />
                        {p.lof_enabled && <>
                            <SliderRow label="Weight" value={p.lof_weight} min={0} max={1} step={0.05} onChange={v => set('lof_weight', v)} />
                            <SliderRow label="Neighbors" value={p.lof_neighbors} min={5} max={50} step={1} onChange={v => set('lof_neighbors', v)} />
                        </>}

                        {/* KMeans */}
                        <ToggleRow label="KMeans" checked={p.kmeans_enabled} onChange={v => set('kmeans_enabled', v)} />
                        {p.kmeans_enabled && <>
                            <SliderRow label="Weight" value={p.kmeans_weight} min={0} max={1} step={0.05} onChange={v => set('kmeans_weight', v)} />
                            <SliderRow label="Clusters" value={p.kmeans_clusters} min={2} max={15} step={1} onChange={v => set('kmeans_clusters', v)} />
                        </>}

                        <SliderRow label="Min Weighted Score" value={p.min_weighted_score} min={0.20} max={0.90} step={0.05} onChange={v => set('min_weighted_score', v)} />
                        <div className="param-row">
                            <div className="param-row__label"><span className="param-row__name">Feature Scaler</span></div>
                            <select value={p.feature_scaler} onChange={e => set('feature_scaler', e.target.value)}>
                                <option value="robust">Robust</option>
                                <option value="standard">Standard</option>
                                <option value="minmax">MinMax</option>
                            </select>
                        </div>

                        {!anyMethodOn && <div className="warning-banner warning-banner--error">No detection methods active — enable at least one.</div>}
                        <WeightBar sum={ensembleSum} onNormalize={normalizeEnsemble} />
                    </Section>

                    {/* C: Threat Classification */}
                    <Section title="Threat Classification">
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>Deforestation Signature</div>
                        <SliderRow label="Min NDVI Drop" value={p.defor_ndvi_min} min={-0.40} max={-0.05} step={0.01} onChange={v => set('defor_ndvi_min', v)} />
                        <SliderRow label="Max BSI" value={p.defor_bsi_max} min={0.01} max={0.15} step={0.01} onChange={v => set('defor_bsi_max', v)} />
                        <SliderRow label="Max NDBI" value={p.defor_ndbi_max} min={0.01} max={0.15} step={0.01} onChange={v => set('defor_ndbi_max', v)} />

                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, marginTop: 12 }}>Mining Signature</div>
                        <SliderRow label="Min NDVI Drop" value={p.mining_ndvi_min} min={-0.30} max={-0.05} step={0.01} onChange={v => set('mining_ndvi_min', v)} />
                        <SliderRow label="Min BSI Rise" value={p.mining_bsi_rise} min={0.01} max={0.25} step={0.01} onChange={v => set('mining_bsi_rise', v)} />
                        <SliderRow label="Max NDBI" value={p.mining_ndbi_max} min={0.01} max={0.15} step={0.01} onChange={v => set('mining_ndbi_max', v)} />

                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, marginTop: 12 }}>Encroachment Signature</div>
                        <SliderRow label="Min NDVI Drop" value={p.encr_ndvi_min} min={-0.30} max={-0.05} step={0.01} onChange={v => set('encr_ndvi_min', v)} />
                        <SliderRow label="Min NDBI Rise" value={p.encr_ndbi_rise} min={0.01} max={0.20} step={0.01} onChange={v => set('encr_ndbi_rise', v)} />
                        <SliderRow label="Min Nightlight" value={p.encr_nightlight} min={1} max={20} step={1} onChange={v => set('encr_nightlight', v)} />
                    </Section>

                    {/* D: Score Weights */}
                    <Section title="Threat Score Weighting">
                        <SliderRow label="Vegetation (NDVI) Weight" value={p.sw_vegetation} min={0} max={1} step={0.05} onChange={v => set('sw_vegetation', v)} />
                        <SliderRow label="Urban (NDBI) Weight" value={p.sw_urban} min={0} max={1} step={0.05} onChange={v => set('sw_urban', v)} />
                        <SliderRow label="Soil (BSI) Weight" value={p.sw_soil} min={0} max={1} step={0.05} onChange={v => set('sw_soil', v)} />
                        <SliderRow label="Nightlight Weight" value={p.sw_nightlight} min={0} max={1} step={0.05} onChange={v => set('sw_nightlight', v)} />
                        <SliderRow label="Seasonal Proof Weight" value={p.sw_seasonal} min={0} max={1} step={0.05} onChange={v => set('sw_seasonal', v)} />
                        <WeightBar sum={scoreSum} onNormalize={normalizeScoreWeights} />
                    </Section>

                    {/* E: Drift & Temporal */}
                    <Section title="Drift & Temporal Analysis">
                        <SliderRow label="DSR Cutoff Threshold" value={p.dsr_threshold} min={1.0} max={3.0} step={0.10} onChange={v => set('dsr_threshold', v)} />
                        <SliderRow label="Smoothing Window (months)" value={p.smoothing_window} min={1} max={6} step={1} onChange={v => set('smoothing_window', v)} />
                        <SliderRow label="Consecutive Declines Required" value={p.consecutive_required} min={2} max={12} step={1} onChange={v => set('consecutive_required', v)} />
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, marginTop: 8 }}>Drift Score Component Weights</div>
                        <SliderRow label="Vegetation Weight" value={p.drift_ndvi_w} min={0} max={1} step={0.05} onChange={v => set('drift_ndvi_w', v)} />
                        <SliderRow label="Temporal Weight" value={p.drift_temporal_w} min={0} max={1} step={0.05} onChange={v => set('drift_temporal_w', v)} />
                        <SliderRow label="Spatial Weight" value={p.drift_spatial_w} min={0} max={1} step={0.05} onChange={v => set('drift_spatial_w', v)} />
                        <SliderRow label="DSR Weight" value={p.drift_dsr_w} min={0} max={1} step={0.05} onChange={v => set('drift_dsr_w', v)} />
                        <WeightBar sum={driftSum} onNormalize={normalizeDriftWeights} />
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, marginTop: 12 }}>Severity Thresholds</div>
                        <SliderRow label="Moderate" value={p.sev_moderate} min={2.0} max={5.0} step={0.5} onChange={v => set('sev_moderate', v)} />
                        <SliderRow label="High" value={p.sev_high} min={4.0} max={7.0} step={0.5} onChange={v => set('sev_high', v)} />
                        <SliderRow label="Severe" value={p.sev_severe} min={5.0} max={9.0} step={0.5} onChange={v => set('sev_severe', v)} />
                        <SliderRow label="Critical" value={p.sev_critical} min={7.0} max={10.0} step={0.5} onChange={v => set('sev_critical', v)} />
                        {!sevValid && <div className="warning-banner warning-banner--error">Severity thresholds must be in ascending order.</div>}
                    </Section>

                    {/* F: Post-Detection Filters */}
                    <Section title="Post-Detection Filters">
                        <ToggleRow label="Monsoon Filter" checked={p.monsoon_enabled} onChange={v => set('monsoon_enabled', v)} />
                        <ToggleRow label="Recovery Check" checked={p.recovery_enabled} onChange={v => set('recovery_enabled', v)} />
                        <ToggleRow label="Regional Context" checked={p.regional_enabled} onChange={v => set('regional_enabled', v)} />
                        <ToggleRow label="Duration Check" checked={p.duration_enabled} onChange={v => set('duration_enabled', v)} />
                        {p.duration_enabled && <SliderRow label="Min Consecutive Months" value={p.duration_min} min={1} max={8} step={1} onChange={v => set('duration_min', v)} />}
                        <SliderRow label="Confidence Floor" value={p.confidence_floor} min={5} max={95} step={5} onChange={v => set('confidence_floor', v)} />
                        {allFiltersOff && <div className="warning-banner warning-banner--warning">All safety filters disabled — expect many false positives.</div>}
                    </Section>

                    {/* G: Spatial */}
                    <Section title="Spatial Analysis">
                        <SliderRow label="K Neighbors" value={p.k_neighbors} min={3} max={30} step={1} onChange={v => set('k_neighbors', v)} />
                        <SliderRow label="Moran's I Threshold" value={p.morans_threshold} min={0.1} max={0.8} step={0.05} onChange={v => set('morans_threshold', v)} />
                        <SliderRow label="GeoJSON Zone Radius (km)" value={p.geojson_radius} min={0.5} max={5.0} step={0.5} onChange={v => set('geojson_radius', v)} />
                    </Section>

                    {/* H: AI Reports */}
                    <Section title="AI Reporting Engine">
                        <div className="param-row">
                            <div className="param-row__label"><span className="param-row__name">Primary Model</span></div>
                            <select value={p.primary_model} onChange={e => set('primary_model', e.target.value)}>
                                <option value="openai/gpt-oss-120b">GPT OSS 120B (Reasoning)</option>
                                <option value="llama-3.3-70b-versatile">Llama 3.3 70B</option>
                            </select>
                        </div>
                        <SliderRow label="Primary Temperature" value={p.primary_temp} min={0} max={1} step={0.05} onChange={v => set('primary_temp', v)} />
                        <NumberRow label="Primary Max Tokens" value={p.primary_max_tokens} min={100} max={8000} onChange={v => set('primary_max_tokens', v)} />
                        <div className="param-row">
                            <div className="param-row__label"><span className="param-row__name">Fallback Model</span></div>
                            <select value={p.fallback_model} onChange={e => set('fallback_model', e.target.value)}>
                                <option value="llama-3.3-70b-versatile">Llama 3.3 70B</option>
                                <option value="openai/gpt-oss-120b">GPT OSS 120B</option>
                            </select>
                        </div>
                        <SliderRow label="Fallback Temperature" value={p.fallback_temp} min={0} max={1} step={0.05} onChange={v => set('fallback_temp', v)} />
                    </Section>
                </>
            )}

            {/* ── BOTTOM BAR (always visible) ──────────────────────────── */}
            {runError && <div className="warning-banner warning-banner--error">{runError}</div>}

            {panelMode === 'scientific' && !canRun && (
                <div className="warning-banner warning-banner--error">
                    Resolve validation issues before running.
                </div>
            )}

            <button
                className="btn btn--primary btn--run"
                disabled={!canRun || isRunning}
                onClick={handleRun}
            >
                {isRunning ? <><span className="spinner" /> Running Pipeline...</> : 'Run Analysis'}
            </button>

            <button
                className="btn btn--secondary"
                style={{ width: '100%', marginTop: 8 }}
                onClick={() => setP(getDefaults())}
            >
                Reset to Defaults
            </button>
        </div>
    );
}
