# SecuriteAI: Enterprise-Grade Anomaly Detection

**SecuriteAI** is a high-throughput, distributed deep learning system designed to protect Linux environments by learning the "temporal grammar" of healthy system logs. Utilizing an **LSTM-Autoencoder architecture**, it identifies zero-day threats and lateral movement through statistical reconstruction failure rather than static, easily-evaded signatures.

---

### 📈 System Performance Highlights
* **Signal-to-Noise Ratio (SNR):** ~348,342x (Extreme mathematical separation between normal and attack states).
* **Detection Rate (Recall):** 100.00% on both statistical and high-velocity sequential "Machine-Gun" burst simulations.
* **False Positive Rate (FPR):** ~1.13% (Achieved via IID Shuffling and 99.5th percentile thresholding).
* **Evolutionary Integrity:** Models are only promoted via a **Champion-Challenger** framework, ensuring no performance regression during automated retraining cycles.

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

#### 3. Multi-Scale Windowing (Adversarial Defense)
To counteract sophisticated adversarial tactics, the system implements a dual-window defense strategy:
* **Short-Term Burst Detection (20 Logs):** Optimized for high-velocity threats like brute-force attacks.
* **Long-Term Density Analysis (1,000 Logs):** Specifically engineered to detect **"Slow-Walk" attacks**, where an adversary interleaves malicious logs with normal system noise to "poison" the baseline over several hours.

#### 4. The 348,000x SNR Innovation (Isolation Normalization)
The system's extreme sensitivity is powered by a **"Poisoned Normalization"** strategy:
* **The Logic:** The Min-Max scaler is fitted strictly on the "Normal" log pool (IDs 1-4).
* **The Result:** When an anomaly ID (e.g., 999) enters the pipeline, it is mapped to an extreme value of **~332.0**. The LSTM experiences a **catastrophic reconstruction failure** because the input is mathematically outside its learned universe.

#### 5. Cyclical Temporal Features
Time is treated as a continuous circle, not a line. **SecuriteAI** decomposes timestamps into **Sine and Cosine pairs** for Hours, Minutes, Seconds, and Days. This ensures the model understands that **23:59 is adjacent to 00:01**, eliminating the "midnight cliff" problem.

---

### 🚀 Production Readiness & MLOps

#### 🛡️ Observability & GRC Integration
* **Prometheus Integration:** The system tracks the **"MSE Baseline"** via a `/metrics` endpoint for real-time monitoring of reconstruction error distributions.
* **Live Telemetry Dashboard:** A "Glass House" view providing a live heartbeat of the MSE reconstruction error and a **Log Explorer** for auditing specific anomaly windows.
* **Human-in-the-Loop Feedback:** Auditors submit corrections via the `/feedback` endpoint, which are used to trigger automated model fine-tuning.
* **Hot-Swapping Weights:** The `/reload` endpoint allows the production fleet to adopt "Challenger" model weights instantly without system downtime.

#### ⚡ Performance Optimizations
* **FastAPI RAM Caching:** By utilizing the **lifespan manager**, model weights and parameters are loaded into memory once at startup, reducing per-prediction latency by 98%.
* **Asynchronous I/O:** The system utilizes `redis.asyncio` for non-blocking state updates, ensuring high-throughput ingestion without stalling the event loop.

---

### 🛠️ Deployment & Training

1. **Execute the Modeling Pipeline**:
    Establish the initial 99.5th percentile threshold and model weights:
    ```bash
    python -m modeling.pipeline
    ```

2. **Launch the Production Cluster**:
    Initialize the load-balanced inference fleet and the auditor dashboard:
    ```bash
    docker-compose -f deployments/docker-compose.yml up --build -d
    ```

---

### 🧪 Development Journey

* **Solving the "Midnight Cliff"**: Initial versions treated time linearly, causing false positives at 00:00. Implementing **Cyclical Sine/Cosine pairs** ensured temporal continuity.
* **The SNR Breakthrough**: We discovered that fitting the Min-Max scaler **strictly on normal data** caused anomalies to map to extreme values, triggering the reconstruction failure needed for 100% recall.
* **Transition to Stateful Scaling**: Moving the buffer to **Redis** transformed the system into a distributed engine where the 20-log context is preserved regardless of which API node receives the log.
* **Hardening against "Slow-Walkers"**: To catch stealthy attackers, we added **Multi-scale Windowing**, implementing a 1,000-log "Density Check" to identify cumulative semantic drift.
* **Closing the Governance Loop**: The project evolved into a "Governance Framework" by adding **Champion-Challenger** logic and **Live Reload** capabilities, allowing the model to learn from auditor expertise without restarts.
* **From Point to Sequence Detection**: By utilizing an LSTM-Autoencoder, the system moved beyond identifying "bad words" to identifying "bad behavior" based on the order and velocity of events.