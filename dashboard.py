"""
dashboard.py
Predictive Pothole Formation Dashboard — Pune Smart City
"""
import os
import pickle
import math
import random
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="PotholeSense — Pune Smart City",
    page_icon="road",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #0f1117;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1e2330;
}
section[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; }

/* App header bar */
header[data-testid="stHeader"] { background-color: #0f1117; }

/* Metric cards */
.kpi-card {
    background: #161b27;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 16px 18px;
    text-align: left;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.kpi-accent { border-left: 3px solid #3b82f6; }
.kpi-accent-warn { border-left: 3px solid #f59e0b; }
.kpi-accent-danger { border-left: 3px solid #ef4444; }
.kpi-accent-ok { border-left: 3px solid #10b981; }

/* Page title */
.page-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.3px;
    margin: 0;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #64748b;
    margin: 2px 0 0 0;
}
.title-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0 14px 0;
    border-bottom: 1px solid #1e2330;
    margin-bottom: 18px;
}
.title-badge {
    background: #1e3a5f;
    color: #60a5fa;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Section headers */
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2330;
    margin-bottom: 12px;
}

/* Alert rows */
.alert-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 0.875rem;
    border: 1px solid transparent;
}
.alert-row-high {
    background: #1c0f0f;
    border-color: #7f1d1d;
    color: #fca5a5;
}
.alert-row-medium {
    background: #1c160a;
    border-color: #78350f;
    color: #fcd34d;
}
.alert-dot-high  { width:8px; height:8px; border-radius:50%; background:#ef4444; flex-shrink:0; }
.alert-dot-med   { width:8px; height:8px; border-radius:50%; background:#f59e0b; flex-shrink:0; }
.alert-id  { font-weight: 600; color: #f1f5f9; }
.alert-meta { color: #94a3b8; font-size: 0.8rem; }
.alert-prob { font-weight: 600; margin-left: auto; }

/* Risk badge */
.rbadge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.rbadge-high   { background:#3b0000; color:#ef4444; border:1px solid #7f1d1d; }
.rbadge-medium { background:#2d1700; color:#f59e0b; border:1px solid #78350f; }
.rbadge-low    { background:#002010; color:#10b981; border:1px solid #065f46; }

/* Status bar */
.status-bar {
    font-size: 0.75rem;
    color: #475569;
    padding: 6px 0;
    display: flex;
    gap: 16px;
    align-items: center;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #10b981; display: inline-block; margin-right: 4px;
}

/* Sidebar section label */
.sb-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin: 12px 0 6px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_artifacts():
    def _load(name):
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None
    return _load("model.pkl"), _load("feature_names.pkl"), _load("label_encoder.pkl")


@st.cache_data(show_spinner=False)
def load_data():
    seg_path  = os.path.join(OUTPUT_DIR, "segment_metadata.csv")
    data_path = os.path.join(OUTPUT_DIR, "pothole_training_data.csv")
    fi_path   = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    segs = pd.read_csv(seg_path)  if os.path.exists(seg_path)  else None
    data = pd.read_csv(data_path, parse_dates=["Date"]) if os.path.exists(data_path) else None
    fi   = pd.read_csv(fi_path)   if os.path.exists(fi_path)   else None
    return segs, data, fi


RISK_COLOR = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}

def risk_color(risk: str) -> str:
    return RISK_COLOR.get(risk, "#64748b")

def risk_badge(risk: str) -> str:
    cls = {"High": "rbadge-high", "Medium": "rbadge-medium", "Low": "rbadge-low"}.get(risk, "")
    return f"<span class='rbadge {cls}'>{risk}</span>"

def priority_index(prob, traffic_stress, road_age):
    return prob * (traffic_stress / 1e5) * (1 + road_age / 10)

FEATURE_NICE = {
    "Cumulative_Rainfall_7d" : "7-Day Rainfall",
    "Traffic_Stress_Index"   : "Traffic Stress",
    "Aging_Factor"           : "Road Aging",
    "Water_Logging_Risk"     : "Water Logging Risk",
    "Heavy_Vehicle_Ratio"    : "Heavy Vehicle Ratio",
    "Drainage_Score"         : "Drainage Score",
    "Road_Age_Years"         : "Road Age (Yrs)",
    "Nearby_Construction_Flag": "Near Construction",
    "Repair_Backlog_Score"   : "Repair Backlog",
    "Precipitation_Log"      : "Rainfall (log)",
    "Humidex"                : "Humidex",
    "Cumulative_Rainfall_30d": "30-Day Rainfall",
    "Surface_Roughness_Index": "Surface Roughness",
    "Heat_Cycle_Count_30d"   : "Heat Cycles (30d)",
}

# Plotly layout defaults for dark theme
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111827",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=0, r=0, t=36, b=0),
)
GRID = dict(gridcolor="#1e2330", zerolinecolor="#1e2330")


# ── Live simulation state ────────────────────────────────────────────────────
if "live_state" not in st.session_state:
    st.session_state.live_state = {
        "rainfall": 8.0, "temperature": 31.0,
        "humidity": 70.0, "traffic": 90000.0,
        "is_peak": 1, "last_update": datetime.now(),
    }

def step_live_state(scenario: dict = None):
    ls = st.session_state.live_state
    month   = datetime.now().month
    monsoon = max(0, math.sin(math.pi * (month - 4) / 6))
    if scenario:
        ls["rainfall"]    = float(scenario.get("rainfall",    ls["rainfall"]))
        ls["temperature"] = float(scenario.get("temperature", ls["temperature"]))
        ls["humidity"]    = float(scenario.get("humidity",    ls["humidity"]))
        ls["traffic"]     = float(scenario.get("traffic",     ls["traffic"]))
    else:
        ls["rainfall"]    = max(0, ls["rainfall"]    + random.gauss(monsoon * 1.5 - ls["rainfall"] * 0.05, 1.5))
        ls["temperature"] = max(15, min(44, ls["temperature"] + random.gauss(0, 0.6)))
        ls["humidity"]    = max(20, min(98, ls["humidity"]    + random.gauss(0, 1.5)))
        ls["traffic"]     = max(20000, min(180000, ls["traffic"] + random.gauss(0, 2500)))
    hr = datetime.now().hour
    ls["is_peak"]     = 1 if (7 <= hr <= 10 or 17 <= hr <= 20) else 0
    ls["last_update"] = datetime.now()


def build_feature_row(seg_row, ls, feature_names):
    rain7 = ls["rainfall"] * 5
    row   = {f: 0.0 for f in feature_names}
    mat   = seg_row.get("Road_Material", "Asphalt")
    row.update({
        "temperature_2m"           : ls["temperature"],
        "relative_humidity"        : ls["humidity"],
        "dew_point"                : ls["temperature"] - (100 - ls["humidity"]) / 5,
        "wind_speed"               : 5.0,
        "max_temp"                 : ls["temperature"] + 3,
        "min_temp"                 : ls["temperature"] - 3,
        "avg_temp"                 : ls["temperature"],
        "precipitation"            : ls["rainfall"],
        "pressure"                 : 1010.0,
        "PCU_City_Wide_Avg_Volume" : ls["traffic"],
        "Is_Peak_Traffic_Hour"     : ls["is_peak"],
        "Is_Weekend"               : 1 if datetime.now().weekday() >= 5 else 0,
        "Road_Age_Years"           : seg_row.get("Road_Age_Years", 8),
        "Drainage_Score"           : seg_row.get("Drainage_Score", 3),
        "Subsurface_Quality_Index" : seg_row.get("Subsurface_Quality_Index", 0.6),
        "Heavy_Vehicle_Ratio"      : seg_row.get("Heavy_Vehicle_Ratio", 0.15),
        "Last_Repair_Days_Ago"     : seg_row.get("Last_Repair_Days_Ago", 365),
        "Surface_Roughness_Index"  : seg_row.get("Surface_Roughness_Index", 2.0),
        "Nearby_Construction_Flag" : seg_row.get("Nearby_Construction_Flag", 0),
        "Repair_Backlog_Score"     : seg_row.get("Repair_Backlog_Score", 0.3),
        "Lane_Count"               : seg_row.get("Lane_Count", 2),
        "Material_Asphalt"         : int(mat == "Asphalt"),
        "Material_Concrete"        : int(mat == "Concrete"),
        "Material_Interlock"       : int(mat == "Interlock"),
        "Cumulative_Rainfall_7d"   : rain7,
        "Cumulative_Rainfall_30d"  : rain7 * 3.5,
        "Temp_Swing"               : 6.0,
        "Heat_Cycle_Count_30d"     : 5,
        "Traffic_Stress_Index"     : ls["traffic"] * (1 + seg_row.get("Heavy_Vehicle_Ratio", 0.15)) * (1.4 if ls["is_peak"] else 1.0),
        "Water_Logging_Risk"       : rain7 / max(1, seg_row.get("Drainage_Score", 3)),
        "Aging_Factor"             : seg_row.get("Road_Age_Years", 8) * math.log1p(seg_row.get("Last_Repair_Days_Ago", 365)),
        "Precipitation_Log"        : math.log1p(ls["rainfall"]),
        "Humidex"                  : ls["temperature"] + 2,
        "Month"                    : datetime.now().month,
        "DayOfWeek"                : datetime.now().weekday(),
    })
    return row


def predict_segment(model, le, feature_names, seg_row, ls) -> dict:
    row = build_feature_row(seg_row, ls, feature_names)
    X   = pd.DataFrame([row])[feature_names]
    prob_arr  = model.predict_proba(X)[0]
    label_idx = model.predict(X)[0]
    label = le.inverse_transform([label_idx])[0] if le else ["Low", "Medium", "High"][label_idx]
    classes  = list(le.classes_) if le else ["High", "Low", "Medium"]
    high_idx = classes.index("High") if "High" in classes else 0
    med_idx  = classes.index("Medium") if "Medium" in classes else 2
    prob = round(min(0.99, max(0.01, float(prob_arr[high_idx] + 0.5 * prob_arr[med_idx]))), 4)
    ts   = row["Traffic_Stress_Index"]
    age  = seg_row.get("Road_Age_Years", 8)
    return {
        "Pothole_Probability" : prob,
        "Pothole_Risk"        : label,
        "Priority_Score"      : round(priority_index(prob, ts, age), 5),
        "Traffic_Stress_Index": round(ts, 0),
    }


def batch_predict(model, le, feature_names, segs_df, ls) -> pd.DataFrame:
    results = []
    for _, seg in segs_df.iterrows():
        pred = predict_segment(model, le, feature_names, seg.to_dict(), ls)
        results.append({**seg.to_dict(), **pred})
    return pd.DataFrame(results)


# ── Main App ─────────────────────────────────────────────────────────────────
def main():
    model, feature_names, le = load_artifacts()
    segs, data, fi = load_data()
    ready = (model is not None and feature_names is not None
             and segs is not None and data is not None)

    # ── Page title ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="title-bar">
        <div>
            <p class="page-title">PotholeSense</p>
            <p class="page-subtitle">Predictive Pothole Formation System &nbsp;|&nbsp; Pune Smart City Initiative</p>
        </div>
        <span class="title-badge">Live</span>
    </div>
    """, unsafe_allow_html=True)

    if not ready:
        st.error("Model or data not found. Run `python data_generator.py` then `python train_model.py` first.")
        st.code("python data_generator.py\npython train_model.py\nstreamlit run dashboard.py", language="bash")
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<p class='page-title' style='font-size:1rem;'>Controls</p>", unsafe_allow_html=True)
        st.markdown("<div class='sb-label'>Modes</div>", unsafe_allow_html=True)
        live_mode     = st.toggle("Live Simulation", value=True)
        scenario_mode = st.toggle("Scenario Mode (What-If)", value=False)
        budget_mode   = st.toggle("Budget Mode", value=False)
        alert_mode    = st.toggle("Alert Mode", value=True)

        st.divider()
        st.markdown("<div class='sb-label'>Filters</div>", unsafe_allow_html=True)
        wards        = ["All Wards"] + sorted(segs["Ward"].unique().tolist())
        sel_ward     = st.selectbox("Ward", wards, label_visibility="collapsed")
        mat_opts     = ["All Materials"] + sorted(segs["Road_Material"].unique().tolist())
        sel_material = st.selectbox("Road Material", mat_opts, label_visibility="collapsed")
        risk_filter  = st.multiselect("Risk Level", ["High", "Medium", "Low"],
                                      default=["High", "Medium", "Low"])

        st.divider()
        st.markdown("<div class='sb-label'>Historical Range</div>", unsafe_allow_html=True)
        min_date   = data["Date"].min().date()
        max_date   = data["Date"].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date,
                                   label_visibility="collapsed")

        if scenario_mode:
            st.divider()
            st.markdown("<div class='sb-label'>Scenario Parameters</div>", unsafe_allow_html=True)
            sc_rain  = st.slider("Rainfall (mm)", 0.0, 80.0, float(st.session_state.live_state["rainfall"]), 0.5)
            sc_temp  = st.slider("Temperature (°C)", 15.0, 45.0, float(st.session_state.live_state["temperature"]), 0.5)
            sc_hum   = st.slider("Humidity (%)", 20, 100, int(st.session_state.live_state["humidity"]))
            sc_traf  = st.slider("Traffic Volume", 20000, 180000, int(st.session_state.live_state["traffic"]), 1000)
            scenario = {"rainfall": sc_rain, "temperature": sc_temp, "humidity": sc_hum, "traffic": sc_traf}
        else:
            scenario = None

        if budget_mode:
            st.divider()
            st.markdown("<div class='sb-label'>Budget Parameters</div>", unsafe_allow_html=True)
            budget_n = st.slider("Roads to Repair", 1, 20, 5)
        else:
            budget_n = 5

        if live_mode:
            st.divider()
            st.markdown("<div class='sb-label'>Refresh Interval</div>", unsafe_allow_html=True)
            refresh_secs = st.slider("Seconds", 5, 60, 15)
        else:
            refresh_secs = None

    # ── State update ──────────────────────────────────────────────────────────
    if scenario_mode:
        step_live_state(scenario=scenario)
    elif live_mode:
        step_live_state()
    ls = st.session_state.live_state

    # ── Predictions ───────────────────────────────────────────────────────────
    # Map: all segments always visible
    all_pred_df = batch_predict(model, le, feature_names, segs, ls)
    all_pred_df = all_pred_df[all_pred_df["Pothole_Risk"].isin(risk_filter)]

    # Other panels: filtered by ward/material
    filtered_segs = segs.copy()
    if sel_ward != "All Wards":
        filtered_segs = filtered_segs[filtered_segs["Ward"] == sel_ward]
    if sel_material != "All Materials":
        filtered_segs = filtered_segs[filtered_segs["Road_Material"] == sel_material]
    pred_df = batch_predict(model, le, feature_names, filtered_segs, ls)
    pred_df = pred_df[pred_df["Pothole_Risk"].isin(risk_filter)]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["City Map", "Live Dashboard", "Trends",
                    "Explainability", "Alerts", "Priority List"])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — CITY MAP
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        col_map, col_panel = st.columns([3, 1])

        with col_panel:
            st.markdown("<div class='section-title'>Risk Summary</div>", unsafe_allow_html=True)
            h = int((all_pred_df["Pothole_Risk"] == "High").sum())
            m = int((all_pred_df["Pothole_Risk"] == "Medium").sum())
            l = int((all_pred_df["Pothole_Risk"] == "Low").sum())
            st.markdown(f"""
            <div class='kpi-card kpi-accent-danger' style='margin-bottom:8px;'>
                <div class='kpi-value' style='color:#ef4444;'>{h}</div>
                <div class='kpi-label'>High Risk Segments</div>
            </div>
            <div class='kpi-card kpi-accent-warn' style='margin-bottom:8px;'>
                <div class='kpi-value' style='color:#f59e0b;'>{m}</div>
                <div class='kpi-label'>Medium Risk Segments</div>
            </div>
            <div class='kpi-card kpi-accent-ok' style='margin-bottom:8px;'>
                <div class='kpi-value' style='color:#10b981;'>{l}</div>
                <div class='kpi-label'>Low Risk Segments</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Circle size = risk probability. Hover for details.")
            if sel_ward != "All Wards":
                st.info(f"Viewing: **{sel_ward}**\nAll 100 segments visible.")

        with col_map:
            if sel_ward != "All Wards":
                ward_segs = segs[segs["Ward"] == sel_ward]
                clat = ward_segs["Latitude"].mean()  if len(ward_segs) else 18.52
                clon = ward_segs["Longitude"].mean() if len(ward_segs) else 73.86
                zoom = 13
            else:
                clat, clon, zoom = 18.52, 73.86, 11

            fig_map = px.scatter_mapbox(
                all_pred_df,
                lat="Latitude", lon="Longitude",
                color="Pothole_Risk",
                color_discrete_map=RISK_COLOR,
                size="Pothole_Probability",
                size_max=18,
                hover_name="Segment_ID",
                hover_data={"Ward": True, "Road_Material": True,
                            "Pothole_Probability": ":.1%",
                            "Road_Age_Years": True,
                            "Latitude": False, "Longitude": False},
                mapbox_style="carto-darkmatter",
                center={"lat": clat, "lon": clon},
                zoom=zoom,
                opacity=0.88,
            )
            fig_map.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(bgcolor="rgba(15,17,23,0.85)",
                            bordercolor="#1e2330", borderwidth=1,
                            font=dict(color="#e2e8f0")),
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_map, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — LIVE DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        pct_high = (pred_df["Pothole_Risk"] == "High").mean() * 100 if len(pred_df) else 0
        status   = "LIVE" if live_mode else "PAUSED"
        scenario_badge = " &nbsp;|&nbsp; Scenario Active" if scenario_mode else ""
        st.markdown(f"""
        <div class='status-bar'>
            <span><span class='live-dot'></span>{status}</span>
            <span>Updated: {ls['last_update'].strftime('%H:%M:%S')}{scenario_badge}</span>
        </div>
        """, unsafe_allow_html=True)

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        cards = [
            (c1, f"{ls['rainfall']:.1f} mm",       "Rainfall",         "kpi-accent"),
            (c2, f"{ls['temperature']:.1f} °C",     "Temperature",      "kpi-accent"),
            (c3, f"{int(ls['humidity'])}%",          "Humidity",         "kpi-accent"),
            (c4, f"{int(ls['traffic']):,}",          "Traffic (PCU)",    "kpi-accent"),
            (c5, f"{pct_high:.0f}%",                 "High Risk Zones",  "kpi-accent-danger"),
        ]
        for col, val, label, cls in cards:
            with col:
                st.markdown(f"""
                <div class='kpi-card {cls}'>
                    <div class='kpi-value'>{val}</div>
                    <div class='kpi-label'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_t, col_g = st.columns([1.3, 1])
        with col_t:
            st.markdown("<div class='section-title'>Top 10 High-Risk Segments</div>", unsafe_allow_html=True)
            top10 = pred_df.sort_values("Pothole_Probability", ascending=False).head(10).copy()
            top10["Risk Level"] = top10["Pothole_Risk"]
            top10["Prob %"]     = (top10["Pothole_Probability"] * 100).round(1).astype(str) + "%"
            top10["Priority"]   = top10["Priority_Score"].round(4)
            show_cols = ["Segment_ID", "Ward", "Risk Level", "Prob %", "Priority",
                         "Road_Age_Years", "Road_Material"]
            st.dataframe(
                top10[show_cols].rename(columns={
                    "Segment_ID": "Segment", "Road_Age_Years": "Age (Yrs)",
                    "Road_Material": "Material"
                }),
                use_container_width=True, hide_index=True,
            )

        with col_g:
            st.markdown("<div class='section-title'>Risk Distribution</div>", unsafe_allow_html=True)
            rc = pred_df["Pothole_Risk"].value_counts().reset_index()
            rc.columns = ["Risk", "Count"]
            fig_pie = px.pie(rc, names="Risk", values="Count", color="Risk",
                             color_discrete_map=RISK_COLOR, hole=0.58)
            fig_pie.update_layout(
                height=280,
                **{k: v for k, v in PLOT_LAYOUT.items() if k != "margin"},
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<div class='section-title'>Ward Risk Comparison</div>", unsafe_allow_html=True)
        ws = pred_df.groupby("Ward").agg(
            Avg_Prob=("Pothole_Probability", "mean"),
            High_Count=("Pothole_Risk", lambda x: (x == "High").sum()),
            Segments=("Segment_ID", "count"),
        ).reset_index().sort_values("Avg_Prob", ascending=False)
        ws["Avg_Prob_Pct"] = (ws["Avg_Prob"] * 100).round(1)
        fig_ward = px.bar(ws, x="Ward", y="Avg_Prob_Pct",
                          color="Avg_Prob_Pct",
                          color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                          text="High_Count",
                          labels={"Avg_Prob_Pct": "Avg Risk (%)"})
        fig_ward.update_traces(texttemplate="%{text} High", textposition="outside")
        fig_ward.update_layout(height=260, coloraxis_showscale=False,
                               xaxis=dict(**GRID), yaxis=dict(**GRID), **PLOT_LAYOUT)
        st.plotly_chart(fig_ward, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — TRENDS
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        try:
            dr_start, dr_end = date_range
        except (TypeError, ValueError):
            dr_start, dr_end = min_date, max_date

        filt_data = data[(data["Date"].dt.date >= dr_start) &
                         (data["Date"].dt.date <= dr_end)].copy()
        if sel_ward != "All Wards":
            filt_data = filt_data[filt_data["Ward"] == sel_ward]
        if sel_material != "All Materials":
            filt_data = filt_data[filt_data["Road_Material"] == sel_material]

        filt_data["YearMonth"] = filt_data["Date"].dt.to_period("M").astype(str)
        monthly = filt_data.groupby("YearMonth").agg(
            Avg_Risk_Prob=("Pothole_Probability", "mean"),
            Avg_Rainfall=("Cumulative_Rainfall_7d", "mean"),
            High_Pct=("Pothole_Risk", lambda x: (x == "High").sum() / len(x) * 100),
        ).reset_index()

        st.markdown("<div class='section-title'>Risk Probability vs 7-Day Rainfall (Monthly)</div>",
                    unsafe_allow_html=True)
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Scatter(x=monthly["YearMonth"], y=monthly["Avg_Risk_Prob"],
                       name="Avg Risk Probability",
                       line=dict(color="#3b82f6", width=2),
                       fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"),
            secondary_y=False,
        )
        fig_trend.add_trace(
            go.Bar(x=monthly["YearMonth"], y=monthly["Avg_Rainfall"],
                   name="7-Day Rainfall", marker_color="rgba(100,116,139,0.4)"),
            secondary_y=True,
        )
        fig_trend.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
            font=dict(color="#94a3b8", family="Inter"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="#94a3b8")),
            xaxis=dict(**GRID), margin=dict(l=0, r=0, t=8, b=0),
        )
        fig_trend.update_yaxes(title_text="Risk Probability", secondary_y=False,
                               gridcolor="#1e2330", color="#64748b")
        fig_trend.update_yaxes(title_text="Rainfall (mm)", secondary_y=True,
                               gridcolor="#1e2330", color="#64748b")
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("<div class='section-title'>High-Risk Segments Over Time (%)</div>",
                    unsafe_allow_html=True)
        fig_high = px.area(monthly, x="YearMonth", y="High_Pct",
                           color_discrete_sequence=["#ef4444"],
                           labels={"High_Pct": "High-Risk %", "YearMonth": "Month"})
        fig_high.update_layout(height=240, xaxis=dict(**GRID), yaxis=dict(**GRID),
                               **PLOT_LAYOUT)
        st.plotly_chart(fig_high, use_container_width=True)

        st.markdown("<div class='section-title'>Feature Correlation Matrix</div>",
                    unsafe_allow_html=True)
        corr_features = ["precipitation", "Cumulative_Rainfall_7d", "Traffic_Stress_Index",
                         "Road_Age_Years", "Drainage_Score", "Heavy_Vehicle_Ratio",
                         "Aging_Factor", "Water_Logging_Risk", "Pothole_Probability"]
        avail = [c for c in corr_features if c in filt_data.columns]
        if len(avail) > 2:
            corr_df = filt_data[avail].sample(min(5000, len(filt_data))).corr()
            fig_corr = px.imshow(corr_df, color_continuous_scale="RdBu_r",
                                 aspect="auto", zmin=-1, zmax=1)
            fig_corr.update_layout(height=400, **PLOT_LAYOUT)
            st.plotly_chart(fig_corr, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — EXPLAINABILITY
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        col_fi, col_seg = st.columns([1.2, 1])

        with col_fi:
            st.markdown("<div class='section-title'>Global Feature Importance</div>",
                        unsafe_allow_html=True)
            if fi is not None:
                fi_top = fi.head(15).copy()
                fi_top["Feature_Nice"] = fi_top["Feature"].map(FEATURE_NICE).fillna(fi_top["Feature"])
                fig_fi = px.bar(fi_top[::-1], x="Importance", y="Feature_Nice",
                                orientation="h", color="Importance",
                                color_continuous_scale=["#1e3a5f", "#3b82f6", "#60a5fa"],
                                labels={"Feature_Nice": "", "Importance": "Importance Score"})
                fig_fi.update_layout(height=400, coloraxis_showscale=False,
                                     xaxis=dict(**GRID), yaxis=dict(**GRID, tickfont=dict(size=11)),
                                     **PLOT_LAYOUT)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.warning("feature_importance.csv not found. Run train_model.py first.")

        with col_seg:
            st.markdown("<div class='section-title'>Segment Analysis</div>",
                        unsafe_allow_html=True)
            seg_options = pred_df["Segment_ID"].tolist()
            sel_seg     = st.selectbox("Select Road Segment", seg_options, index=0)
            seg_row     = pred_df[pred_df["Segment_ID"] == sel_seg].iloc[0]
            risk        = seg_row["Pothole_Risk"]
            prob        = seg_row["Pothole_Probability"]

            st.markdown(f"""
            <div style='background:#161b27; border:1px solid #1e2330; border-left:3px solid {risk_color(risk)};
                        border-radius:6px; padding:14px 16px; margin-bottom:12px;'>
                <div style='font-weight:600; font-size:1rem; color:#f1f5f9; margin-bottom:6px;'>
                    {sel_seg} &nbsp; {risk_badge(risk)}
                </div>
                <div style='color:#64748b; font-size:0.82rem;'>
                    {seg_row['Ward']} &nbsp;|&nbsp;
                    {seg_row['Road_Material']} &nbsp;|&nbsp;
                    {seg_row['Road_Age_Years']:.0f} yrs old
                </div>
                <div style='margin-top:8px; font-size:0.9rem;'>
                    Danger Score: <strong style='color:{risk_color(risk)}'>{prob:.1%}</strong>
                    &nbsp;|&nbsp;
                    Priority: <strong style='color:#f1f5f9'>{seg_row['Priority_Score']:.4f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Danger Score (%)", "font": {"color": "#94a3b8", "size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569",
                             "tickfont": {"color": "#64748b"}},
                    "bar": {"color": risk_color(risk), "thickness": 0.28},
                    "bgcolor": "#111827",
                    "bordercolor": "#1e2330",
                    "steps": [
                        {"range": [0, 35],  "color": "#0d1f17"},
                        {"range": [35, 65], "color": "#1c1607"},
                        {"range": [65, 100],"color": "#1c0808"},
                    ],
                },
                number={"font": {"color": risk_color(risk), "size": 28}, "suffix": "%"},
            ))
            fig_gauge.update_layout(
                height=220, paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"), margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("<div class='section-title'>Risk Factors</div>", unsafe_allow_html=True)
            factors = {
                "7-Day Rainfall (mm)" : round(ls["rainfall"] * 5, 1),
                "Road Age (yrs)"      : seg_row["Road_Age_Years"],
                "Traffic Volume (PCU)": int(ls["traffic"]),
                "Drainage Score"      : seg_row["Drainage_Score"],
                "Heavy Vehicle Ratio" : round(seg_row["Heavy_Vehicle_Ratio"], 3),
                "Days Since Repair"   : int(seg_row["Last_Repair_Days_Ago"]),
            }
            for name, val in factors.items():
                st.markdown(f"- **{name}**: `{val}`")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 — ALERTS
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        if not alert_mode:
            st.info("Enable **Alert Mode** in the sidebar to see live alerts.")
        else:
            high_risk   = pred_df[pred_df["Pothole_Risk"] == "High"].sort_values(
                "Pothole_Probability", ascending=False)
            medium_risk = pred_df[pred_df["Pothole_Risk"] == "Medium"].sort_values(
                "Pothole_Probability", ascending=False)

            if high_risk.empty:
                st.success("No high-risk segments detected at current conditions.")
            else:
                st.markdown(f"<div class='section-title'>{len(high_risk)} High-Risk Segments</div>",
                            unsafe_allow_html=True)
                for _, row in high_risk.head(8).iterrows():
                    st.markdown(f"""
                    <div class='alert-row alert-row-high'>
                        <div class='alert-dot-high'></div>
                        <div>
                            <span class='alert-id'>{row['Segment_ID']}</span>
                            <span class='alert-meta'> &nbsp;|&nbsp; {row['Ward']}
                            &nbsp;|&nbsp; {row['Road_Material']}
                            &nbsp;|&nbsp; {row['Road_Age_Years']:.0f} yrs</span>
                        </div>
                        <span class='alert-prob' style='color:#ef4444;'>{row['Pothole_Probability']:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)

            if not medium_risk.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<div class='section-title'>{len(medium_risk)} Medium-Risk Segments</div>",
                            unsafe_allow_html=True)
                for _, row in medium_risk.head(6).iterrows():
                    st.markdown(f"""
                    <div class='alert-row alert-row-medium'>
                        <div class='alert-dot-med'></div>
                        <div>
                            <span class='alert-id'>{row['Segment_ID']}</span>
                            <span class='alert-meta'> &nbsp;|&nbsp; {row['Ward']}</span>
                        </div>
                        <span class='alert-prob' style='color:#f59e0b;'>{row['Pothole_Probability']:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6 — PRIORITY LIST
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        col_b1, col_b2 = st.columns([2, 1])

        with col_b1:
            st.markdown(f"<div class='section-title'>Top {budget_n} Repair Recommendations</div>",
                        unsafe_allow_html=True)
            st.caption(f"PMC Decision Support — prioritised by danger score, traffic stress, and road age.")
            priority_df = pred_df.sort_values("Priority_Score", ascending=False).head(budget_n).copy()
            priority_df["Rank"]       = range(1, len(priority_df) + 1)
            priority_df["Risk Level"] = priority_df["Pothole_Risk"]
            priority_df["Prob %"]     = (priority_df["Pothole_Probability"] * 100).round(1).astype(str) + "%"
            priority_df["Est. Impact"]= (priority_df["Priority_Score"] * 1000).round(1)
            display_cols = ["Rank", "Segment_ID", "Ward", "Risk Level", "Prob %",
                            "Road_Age_Years", "Road_Material", "Est. Impact"]
            st.dataframe(
                priority_df[display_cols].rename(columns={
                    "Segment_ID": "Segment", "Road_Age_Years": "Age (Yrs)",
                    "Road_Material": "Material",
                }),
                use_container_width=True, hide_index=True,
            )

        with col_b2:
            st.markdown("<div class='section-title'>Ward Distribution</div>",
                        unsafe_allow_html=True)
            ward_impact = priority_df.groupby("Ward")["Priority_Score"].sum().reset_index()
            fig_sun = px.pie(ward_impact, names="Ward", values="Priority_Score",
                             hole=0.45,
                             color_discrete_sequence=["#3b82f6","#f59e0b","#10b981",
                                                       "#8b5cf6","#ef4444","#06b6d4"])
            fig_sun.update_layout(
                height=280,
                **{k: v for k, v in PLOT_LAYOUT.items() if k != "margin"},
                margin=dict(l=0, r=0, t=8, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig_sun, use_container_width=True)

        st.markdown("<div class='section-title'>Priority Score Breakdown</div>",
                    unsafe_allow_html=True)
        fig_pbar = px.bar(priority_df, x="Segment_ID", y="Priority_Score",
                          color="Pothole_Risk", color_discrete_map=RISK_COLOR,
                          labels={"Segment_ID": "Segment", "Priority_Score": "Priority Score"})
        fig_pbar.update_layout(height=280, showlegend=True,
                               xaxis=dict(**GRID), yaxis=dict(**GRID),
                               legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                               **PLOT_LAYOUT)
        st.plotly_chart(fig_pbar, use_container_width=True)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if live_mode and refresh_secs:
        time.sleep(refresh_secs)
        st.rerun()


if __name__ == "__main__":
    main()
