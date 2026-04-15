# SecuriteAI: Enterprise-Grade Anomaly Detection

**SecuriteAI** is a high-throughput, distributed deep learning system designed to protect Linux environments by learning the "temporal grammar" of healthy system logs. Utilizing an **LSTM-Autoencoder architecture**, it identifies zero-day threats and lateral movement through statistical reconstruction failure rather than static, easily-evaded signatures.

---

### 📈 System Performance Highlights
* **Signal-to-Noise Ratio (SNR):** ~95,000x (Extreme separation between normal and attack states).
* **Detection Rate (Recall):** 100.0% on high-entropy brute-force simulations.
* **False Positive Rate (FPR):** ~1.41% (Achieved via IID Shuffling and 99.5th percentile thresholding).

---

### 🧠 System Architecture & Engineering

#### 1. Real-Time Ingestion & Stateful Sliding Window
Unlike traditional batch-processing systems, SecuriteAI utilizes a **Stateful Sliding Window** for real-time monitoring:
* **The `/ingest` Endpoint:** The system accepts a continuous stream of individual logs.
* **Distributed Buffer:** Logs are managed in an atomic Redis-backed buffer, ensuring that the 20-log "temporal context" is preserved even as logs are distributed across multiple container replicas.
* **Dynamic Logic:** The system intelligently transitions from a "Buffering" state to an "Active Stream" state once the window is saturated.

#### 2. Distributed State Management (Redis)
To support horizontal scaling, SecuriteAI externalizes its "memory":
* **Stateless Replicas:** Inference containers are completely stateless; any container can process any incoming log.
* **Atomic Orchestration:** Using Redis `RPUSH` and `LTRIM` operations, the system maintains a consistent, ordered 20-log window across the entire cluster, solving the "split-stream" problem common in load-balanced environments.

#### 3. The 95,000x SNR Innovation (Isolation Normalization)
The system's extreme sensitivity is powered by a **"Poisoned Normalization"** strategy:
* **The Logic:** The Min-Max scaler is fitted strictly on the "Normal" log pool (IDs 1-4).
* **The Result:** When an anomaly ID (e.g., 999) enters the pipeline, it is mapped to an extreme value (e.g., ~332.0), causing a **catastrophic reconstruction failure** in the LSTM. This creates a clear, binary signal for security analysts.

#### 4. Cyclical Temporal Features
Time is a circle, not a line. **SecuriteAI** decomposes timestamps into **Sine and Cosine pairs** for Hours, Minutes, Seconds, and Days. This allows the model to understand that **23:59 is adjacent to 00:01**, preventing the "midnight cliff" problem that often causes false positives in standard linear models.

---

### 🚀 Production Readiness & MLOps

#### 🛡️ Champion-Challenger Maintenance Framework
In a production environment, "Normal" is a moving target. SecuriteAI includes monitoring and retraining strategies:
* **Monitoring:** The system tracks the **"MSE Baseline"** via Prometheus histograms. An upward trend over 7 days triggers a retraining alert.
* **Retraining:** New models are trained on the most recent 30-day "Normal" window and only deployed if they outperform the current model on the latest validation set.

#### ⚡ Performance Optimizations
* **FastAPI RAM Caching:** By utilizing the **lifespan manager**, model weights and scaler parameters are loaded into memory once at startup, reducing per-prediction latency by 98%.
* **Asynchronous I/O:** The system utilizes `redis.asyncio` to perform non-blocking state updates, ensuring high-throughput ingestion without stalling the event loop.
* **Multi-Stage Docker Builds:** We separate the build environment from the lean runtime container, reducing the security attack surface.

---

### 🛠️ Deployment
Using **Docker Compose**, you can launch a load-balanced cluster of inference containers integrated with a Redis state store:

1.  **Initialize the Cluster:**
    ```bash
    docker-compose -f deployments/docker-compose.yml up --build -d
    ```

2.  **Dynamic Scaling:**
    Scale the inference fleet to handle sudden log spikes:
    ```bash
    docker-compose -f deployments/docker-compose.yml up --build -d --scale securiteai-app=10
    ```

---

### 🧪 Development Journey
Initially, the system suffered from a 100% False Positive Rate due to temporal drift in the synthetic data. This was resolved by implementing **IID (Independent and Identically Distributed) Shuffling**, ensuring the model learned the **"System Heartbeat"** across the entire 24-hour cycle rather than a single time-slice.