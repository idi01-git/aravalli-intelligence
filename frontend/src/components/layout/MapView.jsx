import React, { useRef, useEffect } from 'react';
import maplibregl from 'maplibre-gl';
import useStore from '../../hooks/useStore';
import Legend from './Legend';

/**
 * MapView — PRD §16
 * MapLibre GL with CartoDB Dark Matter basemap.
 * GeoJSON circle layer with data-driven color/size/opacity.
 * Click to select zone → opens ZonePopup.
 */

const THREAT_COLORS = {
    deforestation: '#ef4444',
    mining: '#f97316',
    encroachment: '#a855f7',
    localized_disturbance: '#eab308',
    unclassified: '#6b7280',
};

export default function MapView() {
    const mapContainer = useRef(null);
    const mapRef = useRef(null);
    const geojson = useStore(s => s.geojson);
    const selectZone = useStore(s => s.selectZone);
    const selectedZoneId = useStore(s => s.selectedZoneId);
    const summary = useStore(s => s.summary);

    // Initialize map
    useEffect(() => {
        if (mapRef.current) return;

        const map = new maplibregl.Map({
            container: mapContainer.current,
            style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
            center: [73.5, 25.5],
            zoom: 7,
            maxBounds: [[70, 22], [78, 30]],
            attributionControl: false,
        });

        map.addControl(new maplibregl.NavigationControl(), 'bottom-right');
        mapRef.current = map;

        return () => { map.remove(); mapRef.current = null; };
    }, []);

    // Load GeoJSON data when available
    useEffect(() => {
        const map = mapRef.current;
        if (!map || !geojson) return;

        const onLoad = () => {
            // Remove old source/layers if they exist
            if (map.getLayer('zones-circle')) map.removeLayer('zones-circle');
            if (map.getLayer('zones-selected')) map.removeLayer('zones-selected');
            if (map.getSource('zones')) map.removeSource('zones');

            map.addSource('zones', { type: 'geojson', data: geojson });

            // Main circle layer — PRD §16 data-driven styling
            map.addLayer({
                id: 'zones-circle',
                type: 'circle',
                source: 'zones',
                paint: {
                    'circle-color': [
                        'match', ['get', 'threat_type'],
                        'deforestation', THREAT_COLORS.deforestation,
                        'mining', THREAT_COLORS.mining,
                        'encroachment', THREAT_COLORS.encroachment,
                        'localized_disturbance', THREAT_COLORS.localized_disturbance,
                        'unclassified', THREAT_COLORS.unclassified,
                        '#10b981' // default: healthy
                    ],
                    'circle-radius': [
                        'interpolate', ['linear'], ['coalesce', ['get', 'threat_score'], 0],
                        0, 4, 30, 7, 60, 11, 100, 16
                    ],
                    'circle-opacity': [
                        'interpolate', ['linear'], ['coalesce', ['get', 'confidence'], 0],
                        0, 0.25, 50, 0.5, 80, 0.8, 100, 1.0
                    ],
                    'circle-stroke-width': 0,
                },
            });

            // Selected zone highlight ring
            map.addLayer({
                id: 'zones-selected',
                type: 'circle',
                source: 'zones',
                filter: ['==', ['get', 'zone_id'], ''],
                paint: {
                    'circle-radius': [
                        'interpolate', ['linear'], ['coalesce', ['get', 'threat_score'], 0],
                        0, 7, 30, 10, 60, 14, 100, 19
                    ],
                    'circle-color': 'transparent',
                    'circle-stroke-color': '#ffffff',
                    'circle-stroke-width': 2,
                },
            });

            // Cursor + tooltip on hover
            map.on('mouseenter', 'zones-circle', () => { map.getCanvas().style.cursor = 'pointer'; });
            map.on('mouseleave', 'zones-circle', () => { map.getCanvas().style.cursor = ''; });

            // Click → select zone
            map.on('click', 'zones-circle', (e) => {
                const f = e.features?.[0];
                if (!f) return;
                const id = f.properties.zone_id;
                selectZone(id);
                map.flyTo({ center: e.lngLat, zoom: Math.max(map.getZoom(), 9), duration: 600 });
            });

            // Click empty → deselect
            map.on('click', (e) => {
                const features = map.queryRenderedFeatures(e.point, { layers: ['zones-circle'] });
                if (!features.length) useStore.getState().clearZone();
            });
        };

        if (map.loaded()) onLoad();
        else map.on('load', onLoad);
    }, [geojson, selectZone]);

    // Update selected zone filter
    useEffect(() => {
        const map = mapRef.current;
        if (!map || !map.getLayer('zones-selected')) return;
        map.setFilter('zones-selected', ['==', ['get', 'zone_id'], selectedZoneId || '']);
    }, [selectedZoneId]);

    const threatCount = summary?.total_threats ?? '--';
    const totalZones = summary?.total_zones ?? '--';

    return (
        <div className="map-container">
            <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />

            {/* Top-left badge */}
            <div className="map-badge glass-panel">
                {threatCount} threats detected across {totalZones} zones
            </div>

            {/* Top-right export button */}
            <button
                className="map-export-btn"
                onClick={() => window.open('/api/download/geojson', '_blank')}
            >
                ↓ Export GeoJSON
            </button>

            <Legend />
        </div>
    );
}
