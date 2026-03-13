# Aravalli Intelligence — Future Scope & Development Roadmap

This document outlines the strategic roadmap for the Aravalli Intelligence platform. It details potential technical upgrades, feature additions, and scaling architectures designed to transition the project from a hackathon prototype into an enterprise-grade, planetary-scale ecological monitoring system.

---

## 1. Data Ingestion & Satellite Integration
Currently, the system relies heavily on CSV-based historical data or pure mathematical synthetic generation.
*   **Live Google Earth Engine (GEE) Pipeline:** Upgrade the `gee_live` stub in `ingest.py` to automatically authenticate and pull the latest Sentinel-2 (10m resolution) and Landsat 8/9 imagery every 5 days.
*   **SAR (Synthetic Aperture Radar) Integration:** Integrate Sentinel-1 radar data. The Aravalli range experiences heavy cloud cover during the monsoon season, which blinds traditional optical indices (NDVI/NDBI). SAR can penetrate clouds to detect deforestation and mining equipment geometry year-round.
*   **Sub-Meter Resolution Hook:** Add support for commercial satellite APIs (e.g., Planet Labs, Maxar) to allow High-Value Target (HVT) zones to be inspected at 0.5m-resolution once an anomaly is flagged by the free Sentinel data.

## 2. Machine Learning & AI Pipeline
The current system utilizes an incredibly robust Scikit-Learn Unsupervised Ensemble. The next evolution involves deep learning.
*   **Spatio-Temporal Transformers:** Replace the standard Changepoint Detection (`ruptures.Pelt`) with a time-series Transformer or LSTM (Long Short-Term Memory) neural network. These models can understand highly complex, multi-year seasonal dependencies far better than standard statistical baselines.
*   **Vision-Language Models (VLMs):** Currently, the `explain.py` LLM only reads tabular CSV data (e.g., "NDVI dropped 0.15"). In the future, feed the actual before-and-after satellite image tiles to a VLM (like GPT-4 Vision, Claude 3.5 Sonnet, or open-source LLaVA). This allows the AI report to say things like, *"I observe three new unpaved logging roads and heavy machinery near the northern ridge."*
*   **Feedback Loop (Active Learning):** Implement a "Thumbs Up / Thumbs Down" button on the UI for field rangers. If a ranger marks a detection as a False Positive, the backend should log the anomaly signature and use it to re-train the Ensemble models to ignore harmless seasonal anomalies.

## 3. Backend Architecture & Database Scaling
The current architecture uses `pandas` DataFrames heavily and stores data in-memory or in CSV/JSON files, protected by an `asyncio.Lock()`.
*   **PostgreSQL + PostGIS:** Migrate all data from `raw_data.csv` into a proper relational database with spatial extensions. This allows the system to scale from 1,000 zones to 100,000+ zones (covering the entire Indian subcontinent) without overwhelming server RAM.
*   **Asynchronous Message Queue:** Offload the heavy Pandas mathematical pipeline and the LLM API calls from the main FastAPI thread to a background worker queue (e.g., Celery + Redis or RabbitMQ). This will allow multiple users to trigger different analyses simultaneously without locking the server.
*   **Automated CRON Jobs:** Instead of waiting for a user to click "Run Analysis" in the frontend, the backend should automatically run the entire pipeline every night at 2:00 AM, caching the latest detections and LLM field reports so the frontend loads instantly the next morning.

## 4. Frontend Ecosystem & Mobile Expansion
The frontend SPA currently acts as a master data-science control panel.
*   **Role-Based Access Control (RBAC):** Introduce login systems (OAuth/JWT) with distinct views:
    *   *Data Scientist:* Has access to the Advanced Parameter Panel to tweak the 84 ML parameters.
    *   *Field Ranger:* Only sees a simplified mobile-friendly dashboard showing high-confidence alerts, GPS coordinates, and the AI Field Report.
*   **Temporal Playback Slider (Map Time-Machine):** Add a timeline slider to the bottom of the map UI. Users can drag the slider back to 2019 and watch the GeoJSON dots physically change colors and sizes over time, animating the spread of illegal mining chronologically.
*   **Mobile Companion App (Offline Mode):** Build a React Native mobile app for forest rangers. Because the actual Aravalli hills may lack cellular service, the app should download the AI Reports and GPS coordinates for flagged zones while at basecamp, allowing offline navigation to the threat sites.
*   **Multilingual Support (i18n):** Automatically translate the UI and the generated AI Field Reports from English into Hindi and local regional dialects using the LLM, making the tool instantly usable by local forestry departments.
*   **Push Notifications:** Implement Email, SMS, or WhatsApp integration (via Twilio/AWS SNS). If the pipeline runs overnight and detects a "Critical" severity threat (Drift > 8.5), it instantly messages the local authorities.
