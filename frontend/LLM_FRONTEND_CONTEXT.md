# Aravalli Intelligence — Frontend System Context for LLMs

**Target Audience:** Another Large Language Model (LLM) acting as a coding assistant, React developer, or reviewer.
**Purpose:** Provide complete context of the React SPA architecture, component tree, state management (Zustand), map mechanics (MapLibre GL), charting (ECharts), and the specific logic used to interface with the Python backend pipeline.

---

## 1. System Overview & Tech Stack
The frontend is a single-page application (SPA) acting as a massive control panel and geospatial visualization interface for the Ecological Monitoring Backend.

**Core Tech Stack (from `package.json`):**
- **Framework:** `React` 18.2 with `Vite` bundler (runs on port 3000, proxies `/api` to localhost:8000).
- **State Management:** `zustand` (Single global store in `useStore.js`).
- **Map Engine:** `maplibre-gl` (Used for rendering large GeoJSON payloads natively on the GPU).
- **Data Visualization:** `echarts` + `echarts-for-react` (Used for high-density timeseries graphs).
- **Networking:** `axios` (Client wrapper pointing to the backend API).
- **Styling:** Vanilla CSS (`App.css`). No Tailwind, Material UI, or external UI components are used.

---

## 2. Architecture & Layout (`App.jsx`)
The application has a fixed viewport (no vertical scrolling on the `body`) organized into 4 main regions based on the PRD Layout:
1. **Header:** Top navigation/logo.
2. **MapView (Center/Left 70%):** The main MapLibre interface taking up the majority of the screen.
3. **Right Panel (Right 30%):** A contextual dynamic sidebar.
   - If a zone on the map is clicked, it mounts `ZonePopup.jsx`.
   - If no zone is selected, it mounts the pipeline controller `ParameterPanel.jsx`.
4. **AccuracyBar (Bottom Fixed):** A footer summarizing overall pipeline performance, drift metrics, and true/false positive counts.

---

## 3. Global State (`hooks/useStore.js`)
The entire application relies on a single Zustand store to synchronize UI components without prop-drilling or React Context wrappers.

**Key Store Slices:**
- `selectedZoneId`: Holds the ID of the zone currently clicked on the map. Modifying this triggers the right panel switch.
- `panelMode`: toggles `ParameterPanel` between "normal" (simple UI) and "scientific" (84 deep sliders).
- `summary`, `geojson`, `config`, `health`: Hold data fetched from the backend.
- `boot()`: Hydrates the initial dataset on mount (`App.jsx` `useEffect`).
- `triggerAnalysis(sensitivity, overrides)`: Sets `isRunning` strictly to `true`, sends the POST payload to the backend to launch the ML pipeline, and then refetches all caches (`loadSummary`, `loadGeoJSON`) when the pipeline resolves.

---

## 4. Components Breakdown (`src/components/`)

### A. The Control Room (`ParameterPanel.jsx`)
This is the most complex UI component. It dynamically intercepts user input, validates constraints, normalizes ML ensemble weights, and builds an exhaustive `overrides` dictionary to send to the backend's `/api/analyze`.
1. **Modes:**
   - **Normal:** 3 simple sections (radio buttons for generic use cases like data source, sensitivity preset, and strictness).
   - **Scientific:** An expansive interface overriding all 84 config parameters (Ensemble Weights, Drift Weights, Seasonal Cutoffs, Filters).
2. **Real-time Validation & Normalization:** Prevents user error (e.g., if a user slides the IsolationForest weight, a "Normalize" button appears to mathematically force the 4 ensemble methods to sum to `1.00`). Warns if event percentages > 60%, or if detection filters are entirely disabled.
3. **Override Builder:** The `buildOverrides(p)` function explicitly maps the local UI state dictionary (`p`) into nested JSON matching the backend's `config.yaml` schema perfectly.

### B. The Map Engine (`MapView.jsx`)
Handles 1000+ geographical zones at 60 FPS.
- Instantiates a raw `maplibregl.Map` instance targeting CartoDB's Dark Matter tiles.
- In `useEffect`, it natively connects `useStore.getState().geojson` to a WebGL feature layer (`zones-circle`).
- **Data-Driven Styling:** Does NOT loop or manually render points. The styling uses Mapbox Style Spec arrays directly in the layer Paint property:
   - Sets color using `['match', ['get', 'threat_type'], 'deforestation', '#ef4444'...]`.
   - Modulates circle opacity logarithmically mapped to the backend algorithm's `confidence`.
   - Modulates circle radius explicitly driven by `threat_score`.
- Adds a highlighting transparent rim ring (`zones-selected`) filtered by `selectedZoneId`.

### C. Details Drawer (`ZonePopup.jsx`)
Triggers deep backend API calls (`fetchZone`/`fetchTimeseries`/`fetchImportance`) simultaneously on mount using `Promise.allSettled` to display data specifically for the selected zone.
- Visualizes feature permutations, spectral indices deltas, and whether specific algorithms (e.g., IsolationForest) flagged the point.
- Renders the AI reasoning text received from Groq/GPT/Llama.

### D. Charting Engine (`charts/`)
Heavily utilizes ECharts (`echarts-for-react`) due to its ability to handle 84-month arrays instantly.
- **`TimeseriesChart.jsx`:** Renders 84 months of `raw` vs `smoothed` NDVI. It draws a complex shaded band in the background to visually represent the backend's Dynamic Seasonal Referent (DSR) using `baseLow` / `baseHi` bounds returned by the API.

---

## 5. API Networking (`api/client.js`)
All HTTP traffic flows through `axios` mapped to `/api`. The base URL points relative because Vite's `server.proxy` internally maps `/api` -> `:8000`.

**Endpoints Exposed:**
- `GET /api/summary`: Populates bottom `AccuracyBar`.
- `GET /api/health`: Validates backend connection.
- `GET /api/config`: Reads default backend rules payload.
- `GET /api/zones/geojson`: Loads map points.
- `POST /api/analyze`: Extremely heavy request blocking the UI until the backend ML chain finishes processing.

---

## 6. How To Assist
If you (the LLM reading this context) are asked to modify the frontend:
1. **Never use React Context.** Stick strictly to the `useStore` Zustand pattern if you need state to span distant components.
2. **Never break MapLibre `addLayer` spec.** Modification of MapView colors/radii requires knowledge of Mapbox Style Expression Arrays (e.g., `['interpolate', ['linear'], ['get', 'threat_score']...]`).
3. **Respect Parameter Panel Overrides.** If you add a config parameter to the backend, you must also add a state slice, UI slider, and mapping in `buildOverrides(p)` within `ParameterPanel.jsx`.
4. **Vite Toolchain.** Don't refer to `react-scripts` or `webpack`. It's a Vite build.
