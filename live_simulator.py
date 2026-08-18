"""
live_simulator.py
FastAPI micro-service that simulates real-time weather, traffic, and pothole predictions.
Uses a random-walk model to update conditions every call.

Endpoints:
  GET /live/weather   → current simulated weather for a segment
  GET /live/traffic   → current simulated traffic
  GET /predict        → risk prediction for all segments
  GET /segments       → metadata for all segments

Run: uvicorn live_simulator:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import pickle
import threading
import math
import random
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# FastAPI import (falls back gracefully if not installed)
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load model artifacts ──────────────────────────────────────────────────────
def _load(name):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

MODEL         = _load("model.pkl")
FEATURE_NAMES = _load("feature_names.pkl")
LE            = _load("label_encoder.pkl")
SEGMENTS_DF   = None

segs_path = os.path.join(OUTPUT_DIR, "segment_metadata.csv")
if os.path.exists(segs_path):
    SEGMENTS_DF = pd.read_csv(segs_path)


# ─── Shared live state (random walk) ──────────────────────────────────────────
class LiveState:
    def __init__(self):
        self._lock = threading.Lock()
        self.rainfall     = 5.0
        self.temperature  = 30.0
        self.humidity     = 65.0
        self.traffic      = 85000.0
        self.is_peak      = 1
        self.timestamp    = datetime.now().isoformat()

    def step(self):
        """Apply random walk to all live variables."""
        with self._lock:
            month = datetime.now().month
            # Seasonal monsoon bias
            monsoon_factor = max(0, math.sin(math.pi * (month - 4) / 6))
            target_rain = 3 + monsoon_factor * 25

            self.rainfall    = max(0.0, self.rainfall    + random.gauss(target_rain * 0.1 - self.rainfall * 0.05, 1.5))
            self.temperature = max(15.0, min(44.0, self.temperature + random.gauss(0, 0.8)))
            self.humidity    = max(20.0, min(98.0, self.humidity    + random.gauss(0, 2.0)))
            self.traffic     = max(20000, min(180000, self.traffic  + random.gauss(0, 3000)))
            self.is_peak     = 1 if (7 <= datetime.now().hour <= 10 or 17 <= datetime.now().hour <= 20) else 0
            self.timestamp   = datetime.now().isoformat()

    def snapshot(self):
        with self._lock:
            return {
                "rainfall"    : round(self.rainfall, 2),
                "temperature" : round(self.temperature, 1),
                "humidity"    : round(self.humidity, 1),
                "traffic"     : int(self.traffic),
                "is_peak"     : self.is_peak,
                "timestamp"   : self.timestamp,
            }


STATE = LiveState()


def _background_updater(interval: int = 10):
    while True:
        STATE.step()
        time.sleep(interval)


# ─── Prediction helper ─────────────────────────────────────────────────────────
def predict_for_segment(seg_row: dict, weather: dict) -> dict:
    if MODEL is None or FEATURE_NAMES is None:
        return {"Pothole_Probability": 0.5, "Pothole_Risk": "Medium", "Priority_Score": 0.5}

    # Build one-row feature dict
    mat = seg_row.get("Road_Material", "Asphalt")
    rain7 = weather["rainfall"] * 7 * 0.4   # rough rolling approx

    row = {f: 0 for f in FEATURE_NAMES}
    row.update({
        "temperature_2m"              : weather["temperature"],
        "relative_humidity"           : weather["humidity"],
        "dew_point"                   : weather["temperature"] - (100 - weather["humidity"]) / 5,
        "wind_speed"                  : 5.0,
        "max_temp"                    : weather["temperature"] + 3,
        "min_temp"                    : weather["temperature"] - 3,
        "avg_temp"                    : weather["temperature"],
        "precipitation"               : weather["rainfall"],
        "pressure"                    : 1010.0,
        "PCU_City_Wide_Avg_Volume"    : weather["traffic"],
        "Is_Peak_Traffic_Hour"        : weather["is_peak"],
        "Is_Weekend"                  : 1 if datetime.now().weekday() >= 5 else 0,
        "Road_Age_Years"              : seg_row.get("Road_Age_Years", 8),
        "Drainage_Score"              : seg_row.get("Drainage_Score", 3),
        "Subsurface_Quality_Index"    : seg_row.get("Subsurface_Quality_Index", 0.6),
        "Heavy_Vehicle_Ratio"         : seg_row.get("Heavy_Vehicle_Ratio", 0.15),
        "Last_Repair_Days_Ago"        : seg_row.get("Last_Repair_Days_Ago", 365),
        "Surface_Roughness_Index"     : seg_row.get("Surface_Roughness_Index", 2.0),
        "Nearby_Construction_Flag"    : seg_row.get("Nearby_Construction_Flag", 0),
        "Repair_Backlog_Score"        : seg_row.get("Repair_Backlog_Score", 0.3),
        "Lane_Count"                  : seg_row.get("Lane_Count", 2),
        "Material_Asphalt"            : int(mat == "Asphalt"),
        "Material_Concrete"           : int(mat == "Concrete"),
        "Material_Interlock"          : int(mat == "Interlock"),
        "Cumulative_Rainfall_7d"      : rain7,
        "Cumulative_Rainfall_30d"     : rain7 * 3,
        "Temp_Swing"                  : 6.0,
        "Heat_Cycle_Count_30d"        : 5,
        "Traffic_Stress_Index"        : weather["traffic"] * (1 + seg_row.get("Heavy_Vehicle_Ratio", 0.15)) * (1.4 if weather["is_peak"] else 1.0),
        "Water_Logging_Risk"          : rain7 / max(1, seg_row.get("Drainage_Score", 3)),
        "Aging_Factor"                : seg_row.get("Road_Age_Years", 8) * math.log1p(seg_row.get("Last_Repair_Days_Ago", 365)),
        "Precipitation_Log"           : math.log1p(weather["rainfall"]),
        "Humidex"                     : weather["temperature"] + 2,
        "Month"                       : datetime.now().month,
        "DayOfWeek"                   : datetime.now().weekday(),
    })

    X = pd.DataFrame([row])[FEATURE_NAMES]
    prob  = float(MODEL.predict_proba(X)[0].max())
    label_idx = MODEL.predict(X)[0]
    if LE is not None:
        label = LE.inverse_transform([label_idx])[0]
    else:
        label = ["Low", "Medium", "High"][label_idx]

    priority = prob * row["Traffic_Stress_Index"] / 1e5 * (1 + seg_row.get("Road_Age_Years", 8) / 10)

    return {
        "Pothole_Probability": round(prob, 4),
        "Pothole_Risk"       : label,
        "Priority_Score"     : round(priority, 4),
    }


# ─── FastAPI App ───────────────────────────────────────────────────────────────
if HAS_FASTAPI:
    app = FastAPI(title="Pothole Risk Live Simulator", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.on_event("startup")
    def startup_event():
        t = threading.Thread(target=_background_updater, args=(10,), daemon=True)
        t.start()

    @app.get("/live/weather")
    def live_weather():
        STATE.step()
        s = STATE.snapshot()
        return {"status": "ok", "data": {
            "rainfall_mm"  : s["rainfall"],
            "temperature_c": s["temperature"],
            "humidity_pct" : s["humidity"],
            "timestamp"    : s["timestamp"],
        }}

    @app.get("/live/traffic")
    def live_traffic():
        s = STATE.snapshot()
        return {"status": "ok", "data": {
            "volume"    : s["traffic"],
            "is_peak"   : bool(s["is_peak"]),
            "timestamp" : s["timestamp"],
        }}

    @app.get("/segments")
    def get_segments():
        if SEGMENTS_DF is None:
            return {"status": "error", "message": "segment_metadata.csv not found"}
        return {"status": "ok", "data": SEGMENTS_DF.to_dict(orient="records")}

    @app.get("/predict")
    def predict_all(ward: Optional[str] = None):
        if SEGMENTS_DF is None:
            return {"status": "error", "message": "segment_metadata.csv not found"}
        weather = STATE.snapshot()
        results = []
        df = SEGMENTS_DF if ward is None else SEGMENTS_DF[SEGMENTS_DF["Ward"] == ward]
        for _, seg in df.iterrows():
            pred = predict_for_segment(seg.to_dict(), weather)
            results.append({**seg.to_dict(), **pred})
        return {"status": "ok", "weather": weather, "predictions": results}

    if __name__ == "__main__":
        uvicorn.run("live_simulator:app", host="0.0.0.0", port=8000, reload=False)

else:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")
    print("The dashboard uses its own simulation; live_simulator.py is optional.")
