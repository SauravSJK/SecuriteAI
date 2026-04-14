# SecuriteAI: Enterprise Anomaly Detection

**SecuriteAI** is a high-throughput, distributed deep learning system for Linux log analysis. It identifies security breaches by detecting deviations from "Normal" temporal patterns using an LSTM-Autoencoder architecture.

---

## 🚀 Observability & Metrics

The system includes built-in telemetry for monitoring model performance "in the wild":
* **Real-Time MSE Tracking:** Visualized via Prometheus histograms to track reconstruction error distributions.
* **Confidence Scoring:** Utilizes Z-score statistical analysis to assign risk levels ranging from Low to Critical.
* **Automated Alerting:** Asynchronous webhook triggers for instant incident response via background tasks.
* **Metrics Endpoint:** Accessible at `/metrics` for integration with Prometheus and Grafana dashboards.

---

## 🧠 Engineering Architecture

### 1. Modular Directory Structure
The project is organized into a modular hierarchy to separate concerns between core logic, infrastructure, and model artifacts:
* **`src/`**: Contains core model architectures and feature engineering pipelines.
* **`api/`**: Houses the FastAPI inference layer and request validation schemas.
* [cite_start]**`deployments/`**: Contains Docker and Nginx configurations for orchestration.
* **`artifacts/`**: Secure storage for trained weights and normalization parameters.

### 2. Horizontal Scaling & Load Balancing
The system utilizes an **Nginx Load Balancer** to distribute high-volume log traffic across a fleet of identical SecuriteAI inference containers:
* **Strategy:** Implements a Round Robin distribution for optimal resource utilization across the cluster.
* **Redundancy:** Configured with multiple replicas to ensure high availability and high-throughput processing.

### 3. Vectorized Batch Inference
The API is optimized for enterprise log volume by supporting **Log Batching**. This allows the system to utilize parallel processing power, analyzing multiple 20-log sequences in a single mathematical pass.

---

## 🛠️ Deployment (The Scaled Fortress)

Using **Docker Compose**, you can launch the entire load-balanced cluster from the `deployments/` directory.

1.  **Prepare the Environment:**
    Ensure your `artifacts/` directory contains the pre-trained weights (`securiteai_model.pth`) and baseline metrics (`loss_metrics.npy`, `scaler_params.npy`, and `anomaly_threshold.npy`).

2.  **Launch the Cluster:**
    [cite_start]Navigate to the `deployments/` folder and initialize the Nginx controller and inference fleet[cite: 3]:
    ```bash
    docker-compose up --build -d
    ```

3.  **Dynamic Scaling:**
    If log volume increases, scale the inference fleet to 10+ instances instantly to handle the load:
    ```bash
    docker-compose up -d --scale securiteai-app=10
    ```

4.  **Inference & Monitoring:**
    * **Load Balanced API:** `POST http://localhost/predict` (Port 80).
    * **Aggregated Telemetry:** `GET http://localhost/metrics`.