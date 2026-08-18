"""
data_generator.py
Generates a rich synthetic dataset for the Predictive Pothole Formation project (Pune).
Produces pothole_training_data.csv  + segment_metadata.csv
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ─── Configuration ─────────────────────────────────────────────────────────────
N_SEGMENTS      = 100
START_DATE      = date(2023, 1, 1)
END_DATE        = date(2024, 12, 31)
OUTPUT_DIR      = os.path.dirname(os.path.abspath(__file__))

PUNE_WARDS = [
    "Kasba Peth", "Shivajinagar", "Kothrud", "Hadapsar", "Wanowrie",
    "Aundh", "Baner", "Kondhwa", "Bibwewadi", "Yerawada"
]

ROAD_MATERIALS = ["Asphalt", "Concrete", "Interlock"]

# Approximate geographic bounding boxes for each Pune ward
# (lat_min, lat_max, lon_min, lon_max)
WARD_BOUNDS = {
    "Kasba Peth"  : (18.510, 18.530, 73.852, 73.878),
    "Shivajinagar": (18.525, 18.548, 73.838, 73.862),
    "Kothrud"     : (18.490, 18.530, 73.800, 73.838),
    "Hadapsar"    : (18.480, 18.525, 73.918, 73.968),
    "Wanowrie"    : (18.472, 18.505, 73.868, 73.900),
    "Aundh"       : (18.550, 18.580, 73.818, 73.858),
    "Baner"       : (18.548, 18.580, 73.770, 73.812),
    "Kondhwa"     : (18.448, 18.480, 73.868, 73.918),
    "Bibwewadi"   : (18.458, 18.490, 73.838, 73.870),
    "Yerawada"    : (18.538, 18.568, 73.878, 73.920),
}

# ─── 1. Generate Segment Metadata ──────────────────────────────────────────────
def make_segments(n: int) -> pd.DataFrame:
    seg_ids   = [f"SEG_{i:03d}" for i in range(n)]
    wards     = rng.choice(PUNE_WARDS, size=n)
    materials = rng.choice(ROAD_MATERIALS, size=n, p=[0.60, 0.30, 0.10])

    # Assign coordinates within each ward's geographic bounding box
    lats, lons = [], []
    for ward in wards:
        lat_min, lat_max, lon_min, lon_max = WARD_BOUNDS[ward]
        lats.append(round(float(rng.uniform(lat_min, lat_max)), 5))
        lons.append(round(float(rng.uniform(lon_min, lon_max)), 5))

    data = {
        "Segment_ID"              : seg_ids,
        "Ward"                    : wards,
        "Road_Material"           : materials,
        "Road_Age_Years"          : rng.uniform(0, 20, n).round(1),
        "Lane_Count"              : rng.choice([2, 4, 6], size=n, p=[0.4, 0.4, 0.2]),
        "Drainage_Score"          : rng.integers(1, 6, n),
        "Subsurface_Quality_Index": rng.uniform(0.2, 1.0, n).round(3),
        "Heavy_Vehicle_Ratio"     : rng.uniform(0.0, 0.4, n).round(3),
        "Last_Repair_Days_Ago"    : rng.integers(30, 1200, n),
        "Surface_Roughness_Index" : rng.uniform(0.5, 5.0, n).round(2),
        "Nearby_Construction_Flag": rng.choice([0, 1], size=n, p=[0.75, 0.25]),
        "Repair_Backlog_Score"    : rng.uniform(0, 1, n).round(3),
        "Latitude"                : lats,
        "Longitude"               : lons,
    }
    return pd.DataFrame(data)


# ─── 2. Generate Daily Weather ─────────────────────────────────────────────────
def make_weather(dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(dates)
    months = dates.month.values

    # Pune seasonal rainfall: heavy Jun–Sep
    monsoon = np.sin(np.pi * (months - 4) / 6).clip(0)  # peaks Jul
    base_rain = monsoon * 18 + rng.exponential(1, n)
    precipitation = (base_rain + rng.normal(0, 2, n)).clip(0).round(2)

    # Temperature: hot Mar–May, moderate monsoon, cool Dec–Jan
    temp_base = 28 + 7 * np.sin(np.pi * (months - 2) / 8)
    temperature_2m = (temp_base + rng.normal(0, 1.5, n)).clip(10, 44).round(1)

    # Diurnal range
    temp_swing = rng.uniform(4, 12, n).round(1)
    max_temp = (temperature_2m + temp_swing / 2).clip(20, 46).round(1)
    min_temp = (temperature_2m - temp_swing / 2).clip(8, 35).round(1)
    avg_temp = ((max_temp + min_temp) / 2).round(1)

    relative_humidity = (50 + monsoon * 35 + rng.normal(0, 5, n)).clip(20, 98).round(1)
    dew_point = (temperature_2m - (100 - relative_humidity) / 5).round(1)
    pressure = (1010 + rng.normal(0, 3, n)).round(1)
    wind_speed = rng.exponential(6, n).clip(0, 40).round(1)

    return pd.DataFrame({
        "temperature_2m"   : temperature_2m,
        "relative_humidity": relative_humidity,
        "dew_point"        : dew_point,
        "wind_speed"       : wind_speed,
        "max_temp"         : max_temp,
        "min_temp"         : min_temp,
        "avg_temp"         : avg_temp,
        "precipitation"    : precipitation,
        "pressure"         : pressure,
    }, index=dates)


# ─── 3. Generate Daily Traffic ─────────────────────────────────────────────────
def make_traffic(dates: pd.DatetimeIndex, base_volume: float) -> pd.DataFrame:
    n = len(dates)
    dow = dates.dayofweek.values              # 0=Mon … 6=Sun
    is_weekend = (dow >= 5).astype(int)

    # Weekday peak ~90k, weekend ~65k
    base = base_volume * (1 - 0.25 * is_weekend)
    noise = rng.normal(0, base_volume * 0.05, n)
    volume = (base + noise).clip(20000, 200000).round(0)

    # Peak hour flag (simulate it per-day)
    is_peak = rng.choice([0, 1], size=n, p=[0.38, 0.62])  # more peak days in city

    return pd.DataFrame({
        "PCU_City_Wide_Avg_Volume": volume,
        "Is_Peak_Traffic_Hour"    : is_peak,
        "Is_Weekend"              : is_weekend,
        "Traffic_Data_Source"     : rng.choice([0, 1], size=n),
    }, index=dates)


# ─── 4. Engineered Features ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rolling rainfall
    df["Cumulative_Rainfall_7d"]  = (
        df.groupby("Segment_ID")["precipitation"]
          .transform(lambda x: x.rolling(7,  min_periods=1).sum())
    ).round(2)
    df["Cumulative_Rainfall_30d"] = (
        df.groupby("Segment_ID")["precipitation"]
          .transform(lambda x: x.rolling(30, min_periods=1).sum())
    ).round(2)

    # Temp swing
    df["Temp_Swing"] = (df["max_temp"] - df["min_temp"]).round(1)

    # Heat cycle count (days crossing 35°C in last 30d)
    hot_day = (df["max_temp"] > 35).astype(int)
    df["Heat_Cycle_Count_30d"] = (
        df.groupby("Segment_ID")["max_temp"]
          .transform(lambda x: (x > 35).rolling(30, min_periods=1).sum())
    ).astype(int)

    # Traffic stress index
    peak_mult = 1 + 0.4 * df["Is_Peak_Traffic_Hour"]
    df["Traffic_Stress_Index"] = (
        df["PCU_City_Wide_Avg_Volume"] *
        (1 + df["Heavy_Vehicle_Ratio"]) *
        peak_mult
    ).round(1)

    # Water logging risk
    df["Water_Logging_Risk"] = (
        df["Cumulative_Rainfall_7d"] / df["Drainage_Score"]
    ).round(3)

    # Aging factor
    df["Aging_Factor"] = (
        df["Road_Age_Years"] * np.log1p(df["Last_Repair_Days_Ago"])
    ).round(3)

    # Precipitation log
    df["Precipitation_Log"] = np.log1p(df["precipitation"]).round(3)

    # Humidex
    df["Humidex"] = (
        df["temperature_2m"] + 0.5555 * (
            6.11 * np.exp(5417.7530 * (1/273.16 - 1/(df["dew_point"] + 273.16))) - 10
        )
    ).round(1)

    return df


# ─── 5. Label Generation ───────────────────────────────────────────────────────
def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s * 0.0
    return (s - lo) / (hi - lo)

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    risk_score = (
        0.35 * _minmax(df["Cumulative_Rainfall_7d"]) +
        0.25 * _minmax(df["Traffic_Stress_Index"])   +
        0.15 * _minmax(df["Road_Age_Years"])          +
        0.10 * (1 - _minmax(df["Drainage_Score"]))   +
        0.10 * _minmax(df["Heavy_Vehicle_Ratio"])     +
        0.05 * df["Nearby_Construction_Flag"]         +
        0.03 * _minmax(df["Aging_Factor"])            +
        0.02 * _minmax(df["Water_Logging_Risk"])
    )

    # Sigmoid → probability
    prob = (1 / (1 + np.exp(-8 * (risk_score - 0.45)))).round(4)
    prob = prob.clip(0.02, 0.97)

    # Add small noise
    prob = (prob + rng.normal(0, 0.02, len(prob))).clip(0.01, 0.99).round(4)

    df["Pothole_Probability"] = prob
    df["Pothole_Risk"] = pd.cut(
        prob,
        bins=[0, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    return df


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Predictive Pothole Formation — Data Generator")
    print("=" * 60)

    # Segments
    print("\n[1/5] Generating segment metadata ...")
    segments = make_segments(N_SEGMENTS)
    seg_path = os.path.join(OUTPUT_DIR, "segment_metadata.csv")
    segments.to_csv(seg_path, index=False)
    print(f"      Saved → {seg_path}")

    # Date range
    print("[2/5] Building date index ...")
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    # Base volumes per segment (some roads are busier)
    base_volumes = rng.uniform(40000, 160000, N_SEGMENTS)

    all_dfs = []
    print(f"[3/5] Generating daily records for {N_SEGMENTS} segments × {len(dates)} days ...")
    for i, row in segments.iterrows():
        weather  = make_weather(dates)
        traffic  = make_traffic(dates, base_volume=base_volumes[i])
        day_df   = pd.concat([weather, traffic], axis=1).reset_index()
        day_df.rename(columns={"index": "Date"}, inplace=True)

        # Attach segment static features
        for col in segments.columns:
            day_df[col] = row[col]

        all_dfs.append(day_df)

    df = pd.concat(all_dfs, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # Derived time features
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Day"]       = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    print("[4/5] Engineering features ...")
    df = engineer_features(df)

    print("[5/5] Generating risk labels ...")
    df = generate_labels(df)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "pothole_training_data.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Dataset saved → {out_path}")
    print(f"  Shape         : {df.shape}")
    print(f"  Segments      : {N_SEGMENTS}")
    print(f"  Date range    : {START_DATE} → {END_DATE}")
    risk_dist = df["Pothole_Risk"].value_counts(normalize=True).round(3)
    print(f"\n  Risk distribution:\n{risk_dist.to_string()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
