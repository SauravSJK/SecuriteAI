# SecuriteAI: Production-Grade Anomaly Detection for Linux Logs

**SecuriteAI** is an unsupervised deep learning pipeline designed to identify "zero-day" anomalies in Linux system logs. By utilizing an **LSTM-Autoencoder**, the system learns the "temporal grammar" of healthy system behavior and flags deviations in real-time via a REST API.

---

## 🚀 Key Performance Metrics

* **Signal-to-Noise Ratio (SNR):** `~95,000x` — Massive mathematical separation between normal operations and attack states.
* **Detection Rate (Recall):** `100.00%` on high-entropy anomaly bursts.
* **Generalization:** Successfully handles multi-date temporal features through clustered training across 365 days.

---

## 🧠 Engineering Architecture

### 1. Generalizing Temporal Logic
Unlike standard models that overfit to specific dates, SecuriteAI utilizes **Clustered Generation**:
* **Temporal Variety:** Training data is distributed across random clusters over a full year.
* **Cyclical Encoding:** Timestamps are decomposed into **Sine and Cosine pairs** for Hour, Minute, Second, and Day to preserve temporal adjacency (e.g., 23:59 is near 00:01).

### 2. Reconstructive Modeling
* **Encoder:** Compresses a sequence of 20 logs into a 64-dimensional latent bottleneck.
* **Decoder:** Reconstructs the original sequence; healthy logs yield low Mean Squared Error (MSE), while anomalies cause a spike.

### 3. Production Deployment (Docker)
The system is fully containerized for consistency across environments.
* **FastAPI Wrapper:** Serves the model at the `/predict` endpoint.
* **Volume Mounting:** Weights and thresholds are mounted externally to allow for model updates without rebuilding the image.

---

## 📂 Project Structure

| File | Primary Responsibility |
| :--- | :--- |
| `app.py` | FastAPI application for real-time inference. |
| `autoencoder.py` | Defines the 2-layer LSTM Encoder and Decoder classes. |
| `generate_data.py` | Generates a generalized dataset spanning 365 days. |
| `Dockerfile` | Blueprint for creating the isolated container environment. |
| `feat_eng.py` | Orchestrates cyclical time encoding and sliding window creation. |
| `train_test.py` | Manages training and establishes the anomaly threshold. |

---

## 🛠️ Usage Pipeline

1. **Build the Fortress:**
   ```bash
   docker build -t securiteai-app .

2. **Run Inference Server:**
    ```bash
    docker run -d -p 8000:8000 -v "$(pwd)/models:/app/models" securiteai-app

3. **Test API:** Navigate to http://localhost:8000/docs to test the model via the interactive Swagger UI.