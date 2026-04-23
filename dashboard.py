import streamlit as st
import pandas as pd
import redis
import json
import plotly.express as px
import time
import os

# =================================================================
# 1. CONFIGURATION & CONNECTION
# =================================================================
# Use environment variables to support Docker networking
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
r = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

st.set_page_config(page_title="SecuriteAI Dash", layout="wide")
st.title("🛡️ SecuriteAI: Live Command Center")

# =================================================================
# 2. SYSTEM HEARTBEAT (Real-Time MSE)
# =================================================================
st.subheader("System Heartbeat (MSE Reconstruction Error)")

# Fetch the last 50 MSE scores directly from the Redis stream
mse_stream = r.lrange("securiteai_mse_stream", 0, -1)

if not mse_stream:
    st.info("Waiting for live log ingestion to begin...")
    # Use a baseline value if no data exists yet
    plot_data = [0.0]
else:
    # Convert Redis strings back to floats and reverse to show chronological order
    plot_data = [float(m) for m in mse_stream][::-1]  # type: ignore

# Visualization using Plotly for interactive log-scale analysis
fig = px.line(
    y=plot_data,
    title="Live 'Surprise' Score (Log Scale)",
    labels={"y": "Reconstruction MSE", "x": "Recent Windows"},
)
fig.update_yaxes(type="log")
st.plotly_chart(fig, use_container_width=True)

# =================================================================
# 3. LOG EXPLORER (Anomaly Drill-down)
# =================================================================
st.divider()
st.subheader("🔍 Anomaly Log Explorer (GRC Drill-down)")

# Pull captured anomaly windows for auditor review
recent_data = r.lrange("securiteai_recent_anomalies", 0, -1)

if not recent_data:
    st.info("No anomalies detected in the current window.")
else:
    options = [
        f"{json.loads(d)['timestamp']} - MSE: {json.loads(d)['mse']:.4f}"
        for d in recent_data  # type: ignore
    ]
    selection = st.selectbox("Select Anomaly Incident", options)

    if selection:
        idx = options.index(selection)
        event = json.loads(recent_data[idx])  # type: ignore

        # Display severity and auditor-focused context
        st.error(f"Severity: {event['severity']} | Reason: {event['reason']}")
        st.write("### Root Cause: Window Context")
        st.table(pd.DataFrame(event["logs"]))

# Automated refresh to simulate live telemetry
time.sleep(2)
st.rerun()
