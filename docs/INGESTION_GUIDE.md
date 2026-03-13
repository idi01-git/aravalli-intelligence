# Aravalli Intelligence — Data Ingestion Guide

This document explains how data "fuel" enters the Aravalli Intelligence pipeline. The system supports three distinct ingestion modes, controlled by `config.yaml`.

---

## 🏗️ The "Monitoring Scene"
The "Scene" consists of **1,000 monitoring zones** (each ~3km x 3km). For every zone, the system requires a **7-year timeseries** (84 months) of four spectral indices:
1. **NDVI** (Greenery)
2. **NDBI** (Construction)
3. **BSI** (Soil / Mining)
4. **Nightlight** (Human activity)

---

## 📂 Case 1: Real File (`real_file`)
**The primary mode for the hackathon.**

*   **How it works:** Reads the master CSV located at `data/real_aravalli_7year.csv`.
*   **Speed:** Near-instant (Pandas `read_csv`).
*   **Reliability:** 100% offline. No internet required.
*   **Data Scale:** 84,000 data points.
*   **Configuration:**
    ```yaml
    data:
      mode: "real_file"
      real_file_path: "data/real_aravalli_7year.csv"
    ```

---

## 🧪 Case 2: Synthetic Data (`synthetic`)
**The "Sandbox" mode for testing ML logic.**

*   **How it works:** Generates "mathematically perfect" timeseries using NumPy.
*   **Injection:** The system "injects" events (e.g., a drop in NDVI at month 80) into a percentage of zones.
*   **Purpose:** 
    - Verifying that the **Detection Engine** can correctly find and classify anomalies.
    - Demonstrating the project if the real satellite data has gaps.
*   **Configuration:**
    ```yaml
    data:
      mode: "synthetic"
      synthetic:
        deforestation_pct: 0.12 # Injects 12% deforestation events
    ```

---

## ☁️ Case 3: GEE Live (`gee_live`)
**The "Production" mode for real-world monitoring.**

*   **How it works:** Uses the `google-earth-engine` (GEE) Python API.
*   **The Process:**
    1.  The system sends 1,000 Lat/Lon points to Google Cloud.
    2.  GEE processes the massive Sentinel-2 (Level-2A) image collection.
    3.  GEE calculates the indices (NDVI, etc.) in the cloud.
    4.  The system downloads **only the results** (the indices), not the actual GB-sized images.
*   **Benefit:** Allows monitoring of the Aravallis for the **current month** (Feb 2026) rather than staying stuck in historical data.
*   **Requirement:** Requires a `GEE_SERVICE_ACCOUNT` and active internet.

---

## ⏱️ The Timeframe Slicing Rule
The user selects a timeframe (e.g., 24 months) via the frontend slider. The Backend handles this as follows:

| Mode | Processing Logic | Performance |
| :--- | :--- | :--- |
| **Synthetic** | Generates 84 months, but only processes and returns the selected window. | ⚡ Instant |
| **Real CSV** | Loads the 84,000-row master file, then filters for the selected `start_month` to `end_month`. | ⚡ Instant |
| **GEE Live** | Re-downloads the **entire selected timeframe** for all 1,000 zones from Google Cloud. | ⏳ Slow (Proportional to window) |

> [!WARNING]
> **GEE Live Performance:** Running GEE Live for a 7-year (84-month) window across 1,000 zones will likely exceed 60 seconds and may timeout. It is recommended to use GEE Live only for recent analysis (last 3–6 months) or for individual zone deep-dives.

