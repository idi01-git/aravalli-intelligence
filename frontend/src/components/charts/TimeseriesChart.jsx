import React from 'react';
import ReactECharts from 'echarts-for-react';

/**
 * TimeseriesChart — PRD §18 Trend Analysis
 * 84-month NDVI line chart with:
 *   - Solid green: smoothed NDVI
 *   - Dashed blue: raw NDVI
 *   - Shaded band: healthy baseline ±1 std
 *   - Red vertical line: event onset
 *   - Red dots: changepoints
 */
export default function TimeseriesChart({ data, metadata }) {
    if (!data || !data.length) return <div className="loading-state">No timeseries data</div>;

    const labels = data.map(d => d.month_label || d.timestamp);
    const raw = data.map(d => d.ndvi ?? null);
    const smooth = data.map(d => d.ndvi_smoothed ?? null);
    const baseLow = metadata?.healthy_band_min ?? 0;
    const baseHi = metadata?.healthy_band_max ?? 1;

    // Markup lines for changepoints and event onset
    const markLines = [];
    if (metadata?.event_onset_month_index != null) {
        markLines.push({
            xAxis: metadata.event_onset_month_index,
            label: { show: true, formatter: 'Onset', fontSize: 9, color: '#ef4444' },
            lineStyle: { color: '#ef4444', type: 'solid', width: 2 },
        });
    }
    (metadata?.changepoint_months || []).forEach(idx => {
        markLines.push({
            xAxis: idx,
            label: { show: false },
            lineStyle: { color: '#f97316', type: 'dashed', width: 1 },
        });
    });

    const option = {
        backgroundColor: 'transparent',
        grid: { left: 36, right: 12, top: 16, bottom: 28 },
        tooltip: {
            trigger: 'axis',
            backgroundColor: '#1a1a2e',
            borderColor: 'rgba(255,255,255,0.1)',
            textStyle: { color: '#e2e8f0', fontSize: 11 },
        },
        xAxis: {
            type: 'category',
            data: labels,
            axisLabel: {
                fontSize: 9,
                color: '#64748b',
                interval: 11, // Show yearly labels
            },
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        },
        yAxis: {
            type: 'value',
            min: 0,
            max: 1,
            axisLabel: { fontSize: 9, color: '#64748b' },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
        },
        series: [
            // Healthy band (fill between two invisible lines)
            {
                name: 'Healthy Band Upper',
                type: 'line',
                data: data.map(() => baseHi),
                showSymbol: false,
                lineStyle: { opacity: 0 },
                areaStyle: { opacity: 0 },
                stack: 'band',
                z: 1,
            },
            {
                name: 'Healthy Band',
                type: 'line',
                data: data.map(() => baseHi - baseLow),
                showSymbol: false,
                lineStyle: { opacity: 0 },
                areaStyle: {
                    color: 'rgba(16,185,129,0.08)',
                    opacity: 1,
                },
                stack: 'band',
                z: 1,
            },
            // Raw NDVI — dashed blue
            {
                name: 'Raw NDVI',
                type: 'line',
                data: raw,
                showSymbol: false,
                lineStyle: { color: '#60a5fa', width: 1, type: 'dashed' },
                z: 2,
            },
            // Smoothed NDVI — solid green
            {
                name: 'Smoothed NDVI',
                type: 'line',
                data: smooth,
                showSymbol: false,
                lineStyle: { color: '#10b981', width: 2 },
                markLine: markLines.length ? {
                    data: markLines,
                    silent: true,
                    symbol: 'none',
                } : undefined,
                z: 3,
            },
        ],
    };

    return (
        <div className="chart-wrapper">
            <ReactECharts option={option} style={{ height: '100%', width: '100%' }} notMerge lazyUpdate />
        </div>
    );
}
