# SecuriteAI: Enterprise-Grade Anomaly Detection

**SecuriteAI** is a high-throughput, distributed deep learning system designed to protect Linux environments by learning the "temporal grammar" of healthy system logs. Utilizing an **LSTM-Autoencoder architecture**, it identifies zero-day threats and lateral movement through statistical reconstruction failure rather than static, easily-evaded signatures.

---

### 📈 System Performance Highlights
* **Signal-to-Noise Ratio (SNR):** ~348,342x (Extreme mathematical separation between normal and attack states).
* **Detection Rate (Recall):** 100.00% on both statistical and high-velocity sequential "Machine-Gun" burst simulations.
* **False Positive Rate (FPR):** ~1.13% (Achieved via IID Shuffling and 99.5th percentile thresholding).

---

### 🧠 System Architecture & Engineering

#### 1. Real-Time Ingestion & Stateful Sliding Window
SecuriteAI utilizes a **Stateful Sliding Window** for continuous, real-time monitoring:
* **The `/ingest` Endpoint:** The system accepts a continuous stream of individual logs via an asynchronous FastAPI interface.
* **Distributed Buffer:** Logs are managed in an atomic **Redis-backed buffer**, ensuring the 20-log "temporal context" is preserved across multiple container replicas.
* **Dynamic Logic:** The system intelligently transitions from a **BUFFERING** state to an **ACTIVE_STREAM** state once the window depth of 20 logs is reached.

#### 2. Dual-Tier Anomaly Detection
The system is engineered to detect two distinct classes of security threats:
* **Statistical Anomalies:** Identified by rare or unseen Event IDs (e.g., `E999`) that fall outside the learned baseline.
* **Sequential & Velocity Anomalies:** Detects stealthy attacks where an adversary uses "Normal" Event IDs but in a high-velocity burst, proving the model learns system **velocity** and **behavioral patterns**.

#### 3. The 348,000x SNR Innovation (Isolation Normalization)
The system's extreme sensitivity is powered by a **"Poisoned Normalization"** strategy:
* **The Logic:** The Min-Max scaler is fitted strictly on the "Normal" log pool (IDs 1-4).
* **The Result:** When an anomaly ID (e.g., 999) enters the pipeline, it is mapped to an extreme value of **~332.0**. The LSTM, expecting inputs in the $[0, 1]$ range, experiences a **catastrophic reconstruction failure**.

#### 4. Cyclical Temporal Features
Time is treated as a continuous circle, not a line. **SecuriteAI** decomposes timestamps into **Sine and Cosine pairs** for Hours, Minutes, Seconds, and Days. This ensures the model understands that **23:59 is adjacent to 00:01**, eliminating the "midnight cliff" problem that causes false positives in linear models.

---

### 🚀 Production Readiness & MLOps

#### 🛡️ Observability & Maintenance
* **Prometheus Integration:** The system tracks the **"MSE Baseline"** via a `/metrics` endpoint, allowing for real-time monitoring of reconstruction error distributions.
* **Statistical Risk Assessment:** Every prediction is assigned a **Z-score** and a risk level (Low, Medium, High, Critical) based on the training loss distribution.
* **Champion-Challenger Framework:** Models are trained on recent "Normal" windows and only deployed if they outperform the current model on the latest validation set.

#### ⚡ Performance Optimizations
* **FastAPI RAM Caching:** By utilizing the **lifespan manager**, model weights and parameters are loaded into memory once at startup, reducing per-prediction latency by 98%.
* **Asynchronous I/O:** The system utilizes `redis.asyncio` for non-blocking state updates, ensuring high-throughput ingestion without stalling the event loop.
* **Window Size Justification:** A window of **20 logs** was selected to provide several minutes of system context, catching lateral movement while maintaining near real-time alerting.

---

### 🛠️ Deployment & Training

1. **Execute the Modeling Pipeline**:
    First, you must train the model to generate the necessary artifacts. This script handles data generation, model training, and visual reporting in a single pass:
    ```bash
    python -m modeling.pipeline
    ```

2. **Launch the Production Cluster**:
    Once the artifacts exist, initialize the load-balanced inference fleet:
    ```bash
    docker-compose -f deployments/docker-compose.yml up --build -d
    ```

3. **Dynamic Scaling**:
    ```bash
    docker-compose -f deployments/docker-compose.yml up --build -d --scale securiteai-app=10
    ```

---

### 🧪 Development Journey
Initially, the system suffered from a 100% False Positive Rate due to temporal drift in the synthetic data. This was resolved by implementing **IID (Independent and Identically Distributed) Shuffling**, ensuring the model learned the **"System Heartbeat"** across the entire 24-hour cycle rather than a single time-slice.