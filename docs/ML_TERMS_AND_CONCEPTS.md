# Aravalli Intelligence — ML Models & Concepts Dictionary

This document explains the core Machine Learning models, geospatial indices, and mathematical concepts used in the Aravalli Intelligence backend in **simple, easy-to-understand terms**.

---

## 1. Satellite Indices (The Raw Data)
These are scores computed from satellite imagery by measuring how different types of light (visible, near-infrared, shortwave-infrared) bounce off the earth.

*   **NDVI (Normalized Difference Vegetation Index):** Measures greenness. High = dense, healthy forest. Low = bare dirt, rocks, or city. A sudden drop in NDVI usually means trees were cut down.
*   **NDBI (Normalized Difference Built-up Index):** Measures urban areas or concrete. High = buildings, roads, or construction. If NDVI drops and NDBI rises, it means nature was replaced by human construction (Encroachment).
*   **BSI (Bare Soil Index):** Measures exposed dirt and rock. High = bare ground or open-pit mines. If NDVI drops and BSI rises sharply, it means land was cleared but nothing was built (Deforestation or Mining).
*   **Nightlight:** Measures human-generated light at night. A sudden spike in nightlights inside a forest usually means an illegal settlement or an active mining camp is operating.

---

## 2. Unsupervised Machine Learning Models (The "Ensemble")
Because we don't always have labeled data for "illegal" vs "legal" activities, we use *Unsupervised Anomaly Detection* models to look for things that "just don't look normal" compared to surrounding zones or historical data.

*   **Ensemble Method:** Instead of relying on one AI, we use four different AIs. They "vote" on whether a zone is suspicious. If enough vote yes, the system flags it.
*   **Isolation Forest:** Imagine a game of 20 questions. This model tries to isolate every single zone. "Normal" zones are grouped tightly together so it takes many questions to isolate just one. "Anomalies" (suspicious zones) are weird and far away, so it only takes a few questions to isolate them.
*   **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** This model groups zones that are packed closely together in the data. Any zone that is left out of the main groups—floating all alone in the math space—is flagged as "Noise" (an anomaly).
*   **LOF (Local Outlier Factor):** It looks at a zone and compares its density to its immediate neighbors. If a zone is way less dense than the zones right next to it, it is an outlier.
*   **K-Means Clustering:** We ask the AI to group all zones into 8 distinct buckets based on their behavior. The bucket with the fewest zones usually contains the weirdest, most abnormal behavior. We flag the zones in that specific bucket.

---

## 3. Core Analytical Concepts (The Pipeline Math)

### DSR (Deviation from Seasonal Referent)
*   **What it is:** The most important math in the pipeline. It proves a drop in vegetation is an actual threat, not just winter.
*   **How it works:** Forests naturally drop their leaves in the dry season. DSR looks at the last 6 years of data *specifically for that calendar month*. If August's greenness is normally an 0.8, but this August it is a 0.3, the DSR score will spike massively, proving this is an artificial drop.

### Smoothing / Rolling Mean
*   **What it is:** Cleaning up blurry satellite data.
*   **How it works:** Sometimes a cloud covers the satellite, making the greenness look like it dropped to zero for a single month. To fix this, we average out the last 3 months to create a "smooth" line that ignores temporary clouds.

### Changepoint Detection (The "Pelt" Algorithm)
*   **What it is:** Finding the exact month a disaster happened.
*   **How it works:** The algorithm looks at the 84-month greenness timeline and snips the line whenever the fundamental mathematical structure changes. E.g., if a forest hums along happily for 5 years, then drops to zero and stays there, the algorithm places a red line right at the drop. 

### Spatial Autocorrelation (Moran's I)
*   **What it is:** Checking if a crime is organized or random.
*   **How it works:** If illegal mining is happening in one spot, it's likely happening next door. Moran's I measures if threats are clustered together globally. A high score means threats are spreading like a virus (organized destruction). A low score means threats are totally random (like individual trees dying of disease).

### KNN (K-Nearest Neighbors)
*   **What it is:** Comparing a zone to its geographic neighbors.
*   **How it works:** For every zone on the map, we find the 8 closest neighboring zones. If a zone is dying, but its 8 neighbors are perfectly healthy, the zone is assigned a high "Isolation Score" because whatever is hurting it isn't natural (a drought would hurt the neighbors too).

### Composite Drift Score
*   **What it is:** The ultimate 1-to-10 severity rating for a threat.
*   **How it works:** It takes four things into account: 
    1. How much greenness was lost.
    2. How many months in a row it has stayed dead.
    3. How isolated it is from its healthy neighbors.
    4. The DSR (proof it's not seasonal).
    It blends these together to give rangers a simple 1 to 10 priority score.
