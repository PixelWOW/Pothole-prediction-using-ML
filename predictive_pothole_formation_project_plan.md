# Predictive Pothole Formation Model (Pune) — End‑to‑End Implementation Plan

> Goal: Deliver a **complete, demo‑ready system** that predicts pothole‑prone areas using weather, traffic, and road features, even when real data is missing or incomplete. The plan uses **controlled data fabrication (synthetic data)**, feature engineering, a baseline ML model, and a **live-looking dashboard** that streams simulated API updates. This is designed to be built quickly in Antigravity IDE and to present well at an exhibition.

---

## 1) System Overview

**Architecture**
- Data Layer
  - Historical synthetic dataset (CSV/Parquet)
  - “Live” data simulator (Python service) for weather + traffic
- Feature Engineering Layer
  - Aggregations (rolling rainfall, traffic stress index, heat cycles)
  - Road metadata features (age, material, last repair, drainage score)
- Modeling Layer
  - Classification + probability output
  - Baseline: Gradient Boosting / Random Forest
- Serving Layer
  - FastAPI/Flask endpoint for predictions
  - Streamlit/React dashboard for visualization
- Demo Mode
  - Toggle to switch between **historical playback** and **live simulation**

**Core Outputs**
- `Pothole_Risk` (Low/Medium/High)
- `Pothole_Probability` (0–1)
- Ward/Zone risk map + ranked repair priority list

---

## 2) Data Strategy (When Real Data Is Missing)

### 2.1 Use What You Already Have
Current columns:
- time
- temperature_2m
- relative_humidity
- dew_point
- wind_speed
- max_temp, min_temp, avg_temp
- precipitation
- pressure
- PCU_City_Wide_Avg_Volume
- Traffic_Data_Source
- Is_Peak_Traffic_Hour
- Pothole_Risk
- Pothole_Probability

### 2.2 Add “High-Impact” Synthetic Features (Judges Love These)
Add these columns:
- `Road_Age_Years` (0–20)
- `Road_Material` (Asphalt/Concrete/Interlock)
- `Last_Repair_Days_Ago`
- `Drainage_Score` (1–5)
- `Subsurface_Quality_Index` (0–1)
- `Heavy_Vehicle_Ratio` (0–0.4)
- `Lane_Count`
- `Surface_Roughness_Index`
- `Heat_Cycle_Count_30d` (count of days crossing hot/cold thresholds)
- `Cumulative_Rainfall_7d`, `Cumulative_Rainfall_30d`
- `Traffic_Stress_Index` (engineered)
- `Repair_Backlog_Score` (simulated admin delay)
- `Nearby_Construction_Flag` (0/1)

### 2.3 Fabrication Rules (So Data Looks Realistic)
- Rainfall: Monsoon months have higher mean + variance
- Traffic: Higher during peak hours; weekends slightly lower
- Road age: Older roads have higher baseline risk
- Drainage score: Inversely affects water accumulation
- Heavy vehicle ratio: Increases stress non‑linearly

**Synthetic Label Generation Logic (Pseudo):**
- Compute a hidden score:
  ```
  risk_score =
      0.35 * normalize(Cumulative_Rainfall_7d) +
      0.25 * normalize(Traffic_Stress_Index) +
      0.15 * normalize(Road_Age_Years) +
      0.10 * (1 - normalize(Drainage_Score)) +
      0.10 * normalize(Heavy_Vehicle_Ratio) +
      0.05 * Nearby_Construction_Flag
  ```
- Convert to probability with sigmoid
- Bucket into `Low/Medium/High` for `Pothole_Risk`

This makes the ML model “discover” the same logic, which looks realistic in demos.

---

## 3) Data Generation Pipeline

1. Create base time index (e.g., last 2 years, hourly or daily)
2. For each **Zone/Ward/Road Segment** (create 50–200 synthetic segments):
   - Assign static attributes: Road_Age, Material, Drainage, Lanes, etc.
3. Generate weather series:
   - Seasonal rainfall patterns
   - Temperature cycles
4. Generate traffic series:
   - Base volume + peak multipliers
   - Random incidents (spikes)
5. Compute engineered features:
   - Rolling rainfall, heat cycles, stress index
6. Generate labels using the hidden formula
7. Save as `pothole_training_data.csv`

---

## 4) Feature Engineering

**Direct Features**
- temperature_2m, avg_temp, max_temp, min_temp
- precipitation, pressure, wind_speed
- PCU_City_Wide_Avg_Volume, Is_Peak_Traffic_Hour

**Derived Features**
- `Cumulative_Rainfall_7d`, `Cumulative_Rainfall_30d`
- `Temp_Swing = max_temp - min_temp`
- `Traffic_Stress_Index = PCU * (1 + Heavy_Vehicle_Ratio) * PeakMultiplier`
- `Water_Logging_Risk = Cumulative_Rainfall_7d / Drainage_Score`
- `Aging_Factor = Road_Age_Years * log(Last_Repair_Days_Ago + 1)`

**Encoding**
- Road_Material: One-hot
- Traffic_Data_Source: One-hot or drop (demo only)

---

## 5) Modeling Plan

**Problem Type**
- Primary: Binary or 3‑class classification (Low/Medium/High)
- Secondary: Probability regression

**Models**
- Baseline: Logistic Regression
- Main: RandomForest / GradientBoosting / XGBoost (if available)

**Training Steps**
1. Train/validation split by time (to look realistic)
2. Standardize numeric features
3. Train model
4. Evaluate:
   - Accuracy, F1, ROC-AUC
   - Confusion matrix
5. Calibrate probabilities (optional)

**Explainability (Big Demo Win)**
- Feature importance plot
- SHAP-style bar chart (or simple permutation importance)
- Show: Rainfall + Traffic + Road Age dominate risk

---

## 6) “Live Data” Simulation (Jugaad but Convincing)

Create a Python script/service:
- Every 10–30 seconds:
  - Update rainfall, temperature, traffic with small random walks
  - Recompute features
  - Call model → get new risk scores
- Expose as:
  - Local REST API (`/live/weather`, `/live/traffic`, `/predict`)

In dashboard, label this as:
> “Real-time feed (Simulated API for Demo)”

---

## 7) Dashboard (What Judges Will See)

**Tech Options**
- Fastest: Streamlit

**Screens**
1. City Risk Map
   - Color-coded zones (Green/Yellow/Red)
2. Live Metrics Panel
   - Current rainfall, traffic, temperature
3. Top 10 High-Risk Road Segments
   - With probability and reasons
4. Trend Charts
   - Risk vs rainfall over time
5. “Why this is risky?” Panel
   - Feature contributions (simple bar chart)

**Filters**
- By Ward/Zone
- By Road Type
- By Time Range

---

## 8) Priority Scoring for PMC (Decision Layer)

Create a **Repair Priority Index**:
```
Priority = Pothole_Probability * (Traffic_Stress_Index) * (1 + Road_Age_Years/10)
```

Dashboard shows:
- “If you can repair only 5 roads today, do these.”

---

## 9) Antigravity IDE Execution Plan (1-Day Build)

**Phase 1: Data Generation**
- Create synthetic data generator script
- Generate dataset

**Phase 2: Model Training**
- Feature engineering + model training notebook/script
- Save trained model

**Phase 3: Dashboard Development**
- Build Streamlit dashboard
- Load model + dataset

**Phase 4: Live Data Simulation**
- Add live data simulator
- Wire to dashboard auto-refresh

**Phase 5: Polish and Documentation**
- Polish UI
- Add explanation charts
- Add “Smart City Impact” page

---

## 10) Demo Script (What You Will Say)

1. Problem: PMC reacts after potholes appear → costly + unsafe
2. Show dashboard map: “These zones are high risk today.”
3. Turn on live simulation: Rain increases → risk rises
4. Open one road segment:
   - Show rainfall + traffic + age
   - Show model explanation
5. Show priority list:
   - “This is how PMC can plan preventive repairs.”

---

## 11) Ethical + Practical Notes (If Asked)

- This demo uses simulated data due to lack of open PMC datasets
- Architecture is production-ready
- Real deployment only needs:
  - Real weather API
  - Real traffic feeds
  - PMC repair logs

---

## 12) Deliverables

- `data_generator.py`
- `train_model.py`
- `model.pkl`
- `live_simulator.py`
- `dashboard.py` (Streamlit)
- `Predictive_Pothole_Formation_Project_Plan.md`

---

## 13) What Makes This Stand Out

- Not just prediction: **Decision support + prioritization**
- Explainable risk
- Live-looking system
- Smart City alignment: preventive maintenance, cost reduction, safety improvement

---

## 14) Stretch Features

- Scenario mode: “What if rainfall increases by 20%?”
- Budget mode: “You can fix only X roads this week.”
- Alert mode: Auto-flag zones crossing risk threshold

---

**This plan is designed to be feasible in one day with synthetic data and still look like a production-grade Smart City system.**

