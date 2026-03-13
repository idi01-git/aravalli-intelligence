# Aravalli Intelligence — Advanced Parameter Reference Guide

This document explains every configuration parameter precisely in the order they appear in the frontend's **Advanced (Scientific) Parameter Panel**. 

---

## A. Data & Synthetic Settings
This section controls where the system gets its data and how it generates synthetic anomalies for testing.

*   **Data Source (`data_mode`):**
    *   *Synthetic:* Automatically builds fake 84-month histories for zones to perfectly test the ML models without downloading real satellite images.
    *   *Real File:* Loads actual data from the hardcoded `raw_data.csv`.
    *   *GEE Live:* Triggers a live pull from Google Earth Engine (if the hook is implemented).
*   **Zone Count (`zone_count`):** (Only in Synthetic mode). How many geographical parcels of land to simulate. Higher numbers drastically increase RAM usage but provide better clustering data for the AI. Default: 1000.
*   **Sensor Noise (`noise_level`):** Calculates how much random static to add to synthetic satellite data. Mimics atmospheric haze, bad satellite angles, or partial cloud cover. Default: 0.03.
*   **Deforestation % (`deforestation_pct`):** The exact percentage of the total zones to manually infect with a massive drop in green vegetation (NDVI).
*   **Encroachment % (`encroachment_pct`):** The percentage of zones to infect with a vegetation drop simultaneously paired with a rise in the concrete/urban index (NDBI) and nightlights.
*   **Mining % (`mining_pct`):** The percentage of zones to infect with a vegetation drop paired with a massive spike in the bare soil index (BSI).
*   *(Warning: Combined rates exceeding 60% trigger a UI warning because the math models struggle if the "anomalies" outnumber the "normal" zones).*
*   **Random Seed (`seed`):** A fixed mathematical seed. Ensures that if you run the exact same synthetic parameters twice, the fake data generates identically, so you can test if changing a single ML slider improved or degraded accuracy.

## B. Ensemble Methods
Controls the four unsupervised Machine Learning models that detect anomalies. The final decision is a weighted vote.

*   **Isolation Forest (`iso_enabled`):**
    *   **Weight (`iso_weight`):** How much this model's vote counts. Default 0.35 (35%).
    *   **Contamination (`iso_contamination`):** An explicit guess given to the model about what percentage of the data is completely abnormal. E.g., 10% (0.10).
    *   **N Trees (`iso_trees`):** How many branching decision paths the model draws completely at random to try and isolate points. More trees = slower but more reliable. Default: 200.
*   **DBSCAN (`dbscan_enabled`):**
    *   **Weight (`dbscan_weight`):** Voting power. Default 0.30 (30%).
    *   **Epsilon (`dbscan_eps`):** The physical radius the algorithm draws around a data point to group it. Higher epsilon = looser clusters (flags fewer anomalies).
    *   **Min Samples (`dbscan_min_samples`):** The minimum number of zones that must exist inside the Epsilon radius to form a "healthy" group. Default: 3.
*   **Local Outlier Factor (`lof_enabled`):**
    *   **Weight (`lof_weight`):** Voting power. Default 0.25 (25%).
    *   **Neighbors (`lof_neighbors`):** How many surrounding data points LOF mathematically looks at to decide if a zone is "less dense" than its immediate peers. Default: 20.
*   **KMeans (`kmeans_enabled`):**
    *   **Weight (`kmeans_weight`):** Voting power. Default 0.10 (10%).
    *   **Clusters (`kmeans_clusters`):** The exact number of buckets the algorithm is forced to crush all the zones into. The algorithm then flags the buckets with the fewest zones. Default: 8.
*   **Min Weighted Score (`min_weighted_score`):** The ultimate voting threshold. If the weights of all models that flagged a zone sum to this decimal or higher (e.g., `0.50`), the zone is officially marked as an anomaly. Lowering this increases total detections (more sensitive).
*   **Feature Scaler (`feature_scaler`):** How data is mathematically compressed before feeding to the neural networks.
    *   *Robust:* (Recommended) Ignores extreme outliers so standard math isn't ruined by a single crazy data point.
    *   *Standard:* Normalizes data to a bell curve (mean of 0).
    *   *MinMax:* Crushes all data perfectly between 0.0 and 1.0.

## C. Threat Classification
If a zone is flagged as an anomaly, these hardcoded rules determine what to name the anomaly (Deforestation, Mining, or Encroachment). 

*   **Deforestation Signature:**
    *   **Min NDVI Drop (`defor_ndvi_min`):** Greenness MUST drop by at least this amount (a negative number, e.g., `-0.10`) for the system to classify it here.
    *   **Max BSI (`defor_bsi_max`):** Bare soil index must NOT exceed this amount. If soil rises too high, it's categorized as mining instead. 
    *   **Max NDBI (`defor_ndbi_max`):** Urban index must NOT exceed this amount. If concrete rises too high, it's categorized as encroachment instead.
*   **Mining Signature:**
    *   **Min NDVI Drop (`mining_ndvi_min`):** Greenness MUST drop by at least this much.
    *   **Min BSI Rise (`mining_bsi_rise`):** Bare soil MUST spike rapidly. This is the defining characteristic of an open-pit mine (ripping away trees to expose dirt).
    *   **Max NDBI (`mining_ndbi_max`):** Concrete index must stay low (to avoid confusion with city building).
*   **Encroachment Signature:**
    *   **Min NDVI Drop (`encr_ndvi_min`):** Greenness MUST drop by at least this much.
    *   **Min NDBI Rise (`encr_ndbi_rise`):** Concrete MUST spike. This happens when trees are ripped out to build concrete structures or asphalt.
    *   **Min Nightlight (`encr_nightlight`):** Light emitted at night must hit at least this magnitude, indicating human electrical presence.

## D. Threat Score Weighting
After classifying the threat, the backend assigns a 1-100 severity percentage based on the physical severity of the index changes. These weights determine which index matters most in that final math calculation.

*   **Vegetation (NDVI) Weight (`sw_vegetation`):** Default 30%. How heavily the destruction of plant life impacts the final 1-100 percentage metric.
*   **Urban (NDBI) Weight (`sw_urban`):** Default 20%. How heavily pouring concrete impacts the score.
*   **Soil (BSI) Weight (`sw_soil`):** Default 20%. How heavily exposing dirt impacts the score.
*   **Nightlight Weight (`sw_nightlight`):** Default 15%. How heavily human power generation impacts the score.
*   **Seasonal Proof Weight (`sw_seasonal`):** Default 15%. The amount of absolute mathematical proof we have that the change is NOT caused by winter.

## E. Drift & Temporal Analysis
Controls complex mathematical time-series filters dealing with seasons and history.

*   **DSR Cutoff Threshold (`dsr_threshold`):** Deviation from Seasonal Referent. A Z-Score measuring how weird a month is compared to the exact same calendar month over the last 6 years. E.g., if set to 1.50, any vegetation drop with a DSR lower than 1.50 is declared perfectly natural (just winter or a bad rain month) and is completely ignored.
*   **Smoothing Window (`smoothing_window`):** Replaces raw satellite data with a rolling average (e.g., 3 months). This perfectly irons out temporary cloud-cover that ruined a single month's satellite photo.
*   **Consecutive Declines Required (`consecutive_required`):** A tracker to see if the forest is dying rapidly. E.g., 6 means vegetation declined for 6 consecutive months without a single positive month.
*   **Drift Score Component Weights:** Drift is a secondary 1.0 - 10.0 scale for severity. How should we build this score?
    *   **Vegetation (`drift_ndvi_w`):** Default 35%. Weighs the sheer magnitude of the drop.
    *   **Temporal (`drift_temporal_w`):** Default 25%. Weighs how long the drop has persisted.
    *   **Spatial (`drift_spatial_w`):** Default 20%. Weighs how isolated the event is compared to neighbors.
    *   **DSR Weight (`drift_dsr_w`):** Default 20%. Weighs the mathematical proof that the destruction is artificial.
*   **Severity Thresholds:** Turns the raw 1.0-10.0 drift number into English adjectives.
    *   **Moderate (`sev_moderate`):** Default 3.0.
    *   **High (`sev_high`):** Default 5.0.
    *   **Severe (`sev_severe`):** Default 7.0.
    *   **Critical (`sev_critical`):** Default 8.5. Any drift score above 8.5 is absolute critical destruction.

## F. Post-Detection Filters
If the ML ensemble flags a zone, it must pass through these explicit safety checks, or it is dropped entirely. These are critical to preventing "False Positives" (annoying rangers with bad alerts).

*   **Monsoon Filter (`monsoon_enabled`):** Automatically drops the threat if the incident happened in July/August/September AND vegetation actually went UP immediately afterward (meaning it was just an explosive natural growth cycle misread by the AI as bizarre).
*   **Recovery Check (`recovery_enabled`):** Downgrades the severity of any ongoing threat if the most recent two months show massive, steady forest regrowth (meaning the forest is already healing on its own).
*   **Regional Context (`regional_enabled`):** Drops the threat if the zone is sitting in the exact center of a massive block of identically dead zones. (This immediately cancels out massive regional droughts or state-wide crop failures, ensuring we only capture isolated, targeted crimes).
*   **Duration Check (`duration_enabled`):** Turns on the time-filter check below.
    *   **Min Consecutive Months (`duration_min`):** The threat must remain persistently detected for at minimum this many months (e.g. 3). Drops one-month satellite errors instantly.
*   **Confidence Floor (`confidence_floor`):** An absolute cutoff percentage (e.g., 30). If the backend is less than 30% sure a crime happened, do not show it in the UI. 

## G. Spatial Analysis
Controls the geospatial calculations on coordinates (latitude/longitude) and UI mapping sizes.

*   **K Neighbors (`k_neighbors`):** How many neighboring spots of land to pull when looking for geographic comparisons. (Default: 8).
*   **Moran's I Threshold (`morans_threshold`):** Defines the mathematical line where clustered crimes are declared "organized." If the index clears this number (e.g., 0.30), it indicates that a mining operation is spreading outward perfectly. 
*   **GeoJSON Zone Radius (km) (`geojson_radius`):** When generating the map data for the frontend UI, how large of a boundary to literally physically draw around the GPS point to represent a zone's size.

## H. AI Reporting Engine
Controls the language models that translate the numbers into plain English field reports.

*   **Primary Model (`primary_model`):** The heavyweight reasoning model hitting the API (e.g., `openai/gpt-oss-120b`). Very smart, reads the math deeply.
*   **Primary Temperature (`primary_temp`):** Creativity setting from 0.0 to 1.0 for the heavyweight model. 0.30 is highly recommended so it doesn't hallucinate fake mountains or trees.
*   **Primary Max Tokens (`primary_max_tokens`):** Caps the maximum size of the generated paragraph. 300 is ideal for a field ranger's mobile app.
*   **Fallback Model (`fallback_model`):** A faster, cheaper secondary model (e.g., `llama-3.3-70b-versatile`) that takes over instantly if the heavy reasoning model dies or times out.
*   **Fallback Temperature (`fallback_temp`):** Creativity dial for the backup model.

## Extra: "Normal Mode" Options (UI Only)
When the user switches the UI from "Advanced" back to "Simple", these three high-level macros automatically manipulate the sliders above.
*   **Detection Sensitivity (`sensitivity`):**
    *   *Low:* Sets confidence floor to 50%, minimum ML vote to 0.70. Result: only the absolute worst, most definite destruction appears on the map.
    *   *Medium:* The standard defaults.
    *   *High:* Sets floor to 20%, ML vote to 0.30. Maps every minor dip in the forest canopy.
*   **Report Style (`report_style`):** Instructs the UI to alter the `system prompt` sent to the LLM (e.g., "Write like a field ranger" vs "Write like an academic researcher").
*   **AI Strictness (`ai_strictness`):** Directly controls the Primary Temperature dial.
