import React from 'react';
import ReactECharts from 'echarts-for-react';

/**
 * DriversChart — PRD §18 Primary Factors
 * Horizontal bar chart showing top-3 features that drove the detection.
 * Data from GET /api/zones/{zone_id}/importance.
 */
export default function DriversChart({ drivers }) {
    if (!drivers || !drivers.length) return <div className="loading-state">No driver data</div>;

    // drivers = [{ feature_name, importance_score, detail }]
    const names = drivers.map(d => d.feature_name).reverse();
    const scores = drivers.map(d => d.importance_score).reverse();

    const option = {
        backgroundColor: 'transparent',
        grid: { left: 8, right: 40, top: 8, bottom: 8, containLabel: true },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            backgroundColor: '#1a1a2e',
            borderColor: 'rgba(255,255,255,0.1)',
            textStyle: { color: '#e2e8f0', fontSize: 11 },
            formatter: (params) => {
                const p = params[0];
                return `<b>${p.name}</b><br/>Z-score: ${p.value.toFixed(2)}`;
            },
        },
        xAxis: {
            type: 'value',
            axisLabel: { fontSize: 9, color: '#64748b' },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
        },
        yAxis: {
            type: 'category',
            data: names,
            axisLabel: { fontSize: 10, color: '#94a3b8', width: 120, overflow: 'truncate' },
            axisLine: { show: false },
            axisTick: { show: false },
        },
        series: [{
            type: 'bar',
            data: scores,
            barWidth: 14,
            itemStyle: {
                borderRadius: [0, 4, 4, 0],
                color: {
                    type: 'linear', x: 0, y: 0, x2: 1, y2: 0,
                    colorStops: [
                        { offset: 0, color: 'rgba(59,130,246,0.6)' },
                        { offset: 1, color: 'rgba(59,130,246,1)' },
                    ],
                },
            },
            label: {
                show: true,
                position: 'right',
                formatter: '{c}',
                fontSize: 10,
                fontFamily: 'JetBrains Mono',
                color: '#94a3b8',
            },
        }],
    };

    return (
        <div className="chart-wrapper" style={{ height: 120 }}>
            <ReactECharts option={option} style={{ height: '100%', width: '100%' }} notMerge lazyUpdate />
        </div>
    );
}
