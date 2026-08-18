# 🛣️ PotholeSense — Build Walkthrough

## What Was Built

A complete, demo-ready **Predictive Pothole Formation System** for Pune Smart City, delivered across 6 files from scratch.

---

## Files Created

| File | Description |
|------|-------------|
| [data_generator.py](file:///d:/Project/pothole_project/data_generator.py) | Synthetic data: 100 segments × 731 days = 73,100 rows |
| [train_model.py](file:///d:/Project/pothole_project/train_model.py) | GradientBoosting classifier + feature importance |
| [live_simulator.py](file:///d:/Project/pothole_project/live_simulator.py) | FastAPI REST server for real-time simulation |
| [dashboard.py](file:///d:/Project/pothole_project/dashboard.py) | Full Streamlit app (6 tabs, dark premium UI) |
| [requirements.txt](file:///d:/Project/pothole_project/requirements.txt) | All dependencies |
| [run.bat](file:///d:/Project/pothole_project/run.bat) | One-click launcher |
| [README.md](file:///d:/Project/pothole_project/README.md) | Complete documentation |

---

## Pipeline Execution Results

### Data Generation
- **73,100 rows** × 43 columns across **100 synthetic road segments**  
- 10 Pune wards, realistic monsoon weather, peak-hour traffic  
- Risk labels from weighted formula (sigmoid → Low/Medium/High)

### Model Training (GradientBoosting)
- Train/val time-based split (80/20)  
- **73,100 training samples**, 34 features  
- Key outputs: [model.pkl](file:///d:/Project/pothole_project/pothole_rf_model.pkl), `feature_names.pkl`, `label_encoder.pkl`, `feature_importance.csv`

### Top 5 Predictive Features
1. `Cumulative_Rainfall_7d` — biggest driver
2. `Traffic_Stress_Index` — PCU × heavy vehicle ratio
3. `Aging_Factor` — age × log(days since repair)
4. `Water_Logging_Risk` — rainfall / drainage
5. `Heavy_Vehicle_Ratio`

---

## Dashboard — Screenshots

### 🗺️ City Map Tab
![City Map](file:///C:/Users/Rishi/.gemini/antigravity/brain/115ef738-e6ce-48c5-a914-a6fa73ba2910/city_map_tab_1771767083002.png)
*Interactive Plotly map showing 100 road segments across Pune colored by risk level (green=Low, orange=Medium, red=High)*

### 📊 Live Dashboard Tab
![Live Dashboard](file:///C:/Users/Rishi/.gemini/antigravity/brain/115ef738-e6ce-48c5-a914-a6fa73ba2910/live_dashboard_tab_1771767111790.png)
*Auto-refreshing metric cards (Rainfall: 5.6mm, Temp: 29.8°C, Traffic: 84,234 PCU), Top-10 risk table, and donut chart*

### Full Recording
![Dashboard verification recording](file:///C:/Users/Rishi/.gemini/antigravity/brain/115ef738-e6ce-48c5-a914-a6fa73ba2910/potholesense_dashboard_verification_1771767038848.webp)

---

## Dashboard Features Verified ✅

| Tab | Status |
|-----|--------|
| 🗺️ City Map — Plotly mapbox with colored markers | ✅ Working |
| 📊 Live Dashboard — auto-refresh metrics + Top-10 table | ✅ Working |
| 📈 Trends — risk vs rainfall monthly charts | ✅ Working |
| 🔍 Explainability — feature importance + gauge + per-segment analysis | ✅ Working |
| 🚨 Alerts — animated high-risk alert cards | ✅ Working |
| 💰 Priority List — PMC budget optimizer | ✅ Working |

**Sidebar controls working:** Live Mode, Scenario Mode (what-if sliders), Budget Mode (top-N picker), Alert Mode, Ward/Material/Risk filters, Date Range

---

## How to Run

```bash
# Quick start (Windows)
run.bat

# Or manually with Python314:
C:\Users\Rishi\AppData\Local\Programs\Python\Python314\python.exe data_generator.py
C:\Users\Rishi\AppData\Local\Programs\Python\Python314\python.exe train_model.py
C:\Users\Rishi\AppData\Local\Programs\Python\Python314\Scripts\streamlit.exe run dashboard.py
# → http://localhost:8501
```
