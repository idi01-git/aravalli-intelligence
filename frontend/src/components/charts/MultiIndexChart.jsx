import React from 'react';
import ReactECharts from 'echarts-for-react';

/**
 * MultiIndexChart — PRD §18
 * 4 spectral indices (NDVI, NDBI, BSI, Nightlight) overlaid on one graph.
 * Shows correlations: "NDVI dropped at same time BSI spiked → mining".
 */

const LINE_CONFIG = [
    { key: 'ndvi_smoothed', fallback: 'ndvi', name: 'NDVI', color: '#10b981' },
    { key: 'ndbi_smoothed', fallback: 'ndbi', name: 'NDBI', color: '#a855f7' },
    { key: 'bsi_smoothed', fallback: 'bsi', name: 'BSI', color: '#f97316' },
    { key: 'nightlight', fallback: 'nightlight', name: 'Nightlight', color: '#3b82f6' },
];

export default function MultiIndexChart({ data }) {
    if (!data || !data.length) return <div className="loading-state">No index data</div>;

    const labels = data.map(d => d.month_label || d.timestamp);

    const option = {
        backgroundColor: 'transparent',
        grid: { left: 36, right: 12, top: 24, bottom: 28 },
        legend: {
            top: 0,
            right: 0,
            textStyle: { color: '#94a3b8', fontSize: 10 },
            itemWidth: 16,
            itemHeight: 2,
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: '#1a1a2e',
            borderColor: 'rgba(255,255,255,0.1)',
            textStyle: { color: '#e2e8f0', fontSize: 11 },
        },
        xAxis: {
            type: 'category',
            data: labels,
            axisLabel: { fontSize: 9, color: '#64748b', interval: 11 },
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        },
        yAxis: {
            type: 'value',
            axisLabel: { fontSize: 9, color: '#64748b' },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
        },
        series: LINE_CONFIG.map(cfg => ({
            name: cfg.name,
            type: 'line',
            data: data.map(d => d[cfg.key] ?? d[cfg.fallback] ?? null),
            showSymbol: false,
            lineStyle: { color: cfg.color, width: 1.5 },
            itemStyle: { color: cfg.color },
        })),
    };

    return (
        <div className="chart-wrapper">
            <ReactECharts option={option} style={{ height: '100%', width: '100%' }} notMerge lazyUpdate />
        </div>
    );
}
