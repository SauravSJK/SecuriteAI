"""
SecuriteAI: Auditor Command Center
---------------------------------
Main Author: Saurav Jayakumar
Description: A real-time Streamlit dashboard for monitoring system health.
It visualizes the reconstruction MSE "heartbeat" and provides a Log Explorer
for granular root-cause analysis.

Key Features:
- Live MSE Heartbeat via Redis Streams.
- Anomaly Drill-down for GRC Auditability.
- Log-scale visualization for high-SNR events.
"""

import streamlit as st
import pandas as pd
import redis
import json
import plotly.express as px
import time
import os

# =================================================================
# 1. CONFIGURATION & REDIS CONNECTION
# =================================================================
# Environment-aware connection logic for Docker/Local flexibility
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
r = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

st.set_page_config(page_title="SecuriteAI Dash", layout="wide")
st.title("🛡️ SecuriteAI: Live Command Center")

# =================================================================
# 2. SYSTEM HEARTBEAT (Real-Time MSE)
# =================================================================
# Provides immediate visibility into the model's 'surprise' levels.
st.subheader("System Heartbeat (MSE Reconstruction Error)")

# Pull the rolling window of MSE scores from the Redis stream
mse_stream = r.lrange("securiteai_mse_stream", 0, -1)

if not mse_stream:
    st.info("Waiting for live log ingestion to begin...")
    plot_data = [0.0]
else:
    # Reverse to chronological order: Left (Oldest) -> Right (Newest)
    plot_data = [float(m) for m in mse_stream][::-1]  # type: ignore

# Interactive log-scale chart for identifying stealthy versus burst anomalies
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
# The primary GRC tool for security auditors to verify threat alerts.
st.divider()
st.subheader("🔍 Anomaly Log Explorer (GRC Drill-down)")

# Access captured incident windows stored by the API
recent_data = r.lrange("securiteai_recent_anomalies", 0, -1)

if not recent_data:
    st.info("No anomalies detected in the current window.")
else:
    # Auditor selection based on timestamp and intensity
    options = [
        f"{json.loads(d)['timestamp']} - MSE: {json.loads(d)['mse']:.4f}"
        for d in recent_data  # type: ignore
    ]
    selection = st.selectbox("Select Anomaly Incident", options)

    if selection:
        idx = options.index(selection)
        event = json.loads(recent_data[idx])  # type: ignore

        # Expose severity and log sequence context for forensic review
        st.error(f"Severity: {event['severity']} | Reason: {event['reason']}")
        st.write("### Root Cause: Window Context")
        # Renders the exact 20 logs that caused the detection
        st.table(pd.DataFrame(event["logs"]))

# Simulated real-time update loop
time.sleep(2)
st.rerun()
