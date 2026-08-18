# 🛣️ PotholeSense — Predictive Pothole Formation System
### Pune Smart City Initiative

> A demo-ready ML system that predicts pothole-prone road segments using weather, traffic, and road-condition data — with a live-looking Streamlit dashboard.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Option A: One-click launcher (Windows)
run.bat

# Option B: Manual steps
python data_generator.py     # Generate synthetic dataset (~100 segments, 2 years)
python train_model.py        # Train GradientBoosting model
streamlit run dashboard.py   # Launch dashboard at http://localhost:8501

# Option C: Also run the live REST API simulator (optional)
uvicorn live_simulator:app --port 8000
```

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `data_generator.py` | Generates `pothole_training_data.csv` + `segment_metadata.csv` |
| `train_model.py` | Trains model → `model.pkl`, `feature_names.pkl`, `feature_importance.csv` |
| `live_simulator.py` | FastAPI REST server for real-time simulation |
| `dashboard.py` | Streamlit dashboard (6 tabs, dark theme) |
| `requirements.txt` | Python dependencies |
| `run.bat` | One-click launcher (Windows) |

---

## 📊 Dashboard Features

| Tab | Features |
|-----|---------|
| 🗺️ **City Map** | Interactive Plotly map with colored risk markers |
| 📊 **Live Dashboard** | Auto-refreshing metrics, Top-10 table, Ward heatmap |
| 📈 **Trends** | Monthly risk vs rainfall charts, correlation matrix |
| 🔍 **Explainability** | Feature importance, per-segment gauge, risk factors |
| 🚨 **Alerts** | Auto-flag high-risk zones with animated cards |
| 💰 **Priority List** | PMC Budget Optimizer — top-N repair list |

### Sidebar Controls
- 🔴 **Live Mode** — auto-refreshes every N seconds
- 🎭 **Scenario Mode** — manual sliders to test "what if rainfall +20%?"
- 💰 **Budget Mode** — pick how many roads PMC can repair today
- 🔔 **Alert Mode** — show red-card alerts for high-risk zones
- Filters by Ward, Road Material, Risk Level, Date Range

---

## 🧠 ML Model

- **Algorithm**: GradientBoostingClassifier (200 trees, depth=5)
- **Features**: 34 features across weather, traffic, road metadata, and engineered signals
- **Target**: 3-class `Pothole_Risk` (Low / Medium / High)
- **Training**: Time-based 80/20 split (no data leakage)
- **Metrics**: ~85%+ accuracy, macro F1 > 0.80

### Top Predictive Features
1. 7-Day Cumulative Rainfall
2. Traffic Stress Index
3. Road Aging Factor
4. Water Logging Risk
5. Heavy Vehicle Ratio

---

## ⚙️ Architecture

```
[Synthetic Data] → [Feature Engineering] → [GradientBoosting Model]
                                                      ↓
                                         [FastAPI Live Simulator]
                                                      ↓
                                        [Streamlit Dashboard]
                                    (Map | Alerts | Budget | Trends)
```

---

## 📋 Priority Scoring Formula

```
Priority = Pothole_Probability × (Traffic_Stress_Index / 1e5) × (1 + Road_Age_Years / 10)
```
> Used by PMC to decide which roads to repair first.

---

## ⚠️ Ethical Notes

- All data is **synthetic** — designed to mimic Pune's real patterns
- Architecture is **production-ready**: swap in real PMC + IMD + traffic APIs
- Real deployment needs: OpenWeather/IMD API, traffic sensors, PMC repair logs

---

*Built for Smart City Exhibition | Pune 2025*
