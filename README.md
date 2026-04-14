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
* **`deployments/`**: Contains Docker and Nginx configurations for orchestration.
* **`artifacts/`**: Secure storage for trained weights and normalization parameters.
* **`experiments/`**: Contains scripts for model training, testing, and performance visualization.

### 2. Horizontal Scaling & Load Balancing
The system utilizes an **Nginx Load Balancer** to distribute high-volume log traffic across a fleet of identical SecuriteAI inference containers:
* **Strategy:** Implements a Round Robin distribution for optimal resource utilization across the cluster.
* **Redundancy:** Configured with multiple replicas to ensure high availability and high-throughput processing.

### 3. Vectorized Batch Inference
The API is optimized for enterprise log volume by supporting **Log Batching**. This allows the system to utilize parallel processing power, analyzing multiple 20-log sequences in a single mathematical pass.

---

## 🛠️ Deployment (The Scaled Fortress)

Using **Docker Compose**, you can launch the entire load-balanced cluster from the project root.

1.  **Prepare the Environment:**
    Ensure your `artifacts/` directory contains the pre-trained weights (`securiteai_model.pth`) and baseline metrics (`loss_metrics.npy`, `scaler_params.npy`, and `anomaly_threshold.npy`).

2.  **Launch the Cluster:**
    Navigate to the root folder and initialize the Nginx controller and inference fleet:
    ```bash
    docker-compose -f deployments/docker-compose.yml up --build -d
    ```

3.  **Dynamic Scaling:**
    If log volume increases, scale the inference fleet to 10+ instances instantly to handle the load:
    ```bash
    docker-compose -f deployments/docker-compose.yml up --build -d --scale securiteai-app=10
    ```

4.  **Inference & Monitoring:**
    * **Load Balanced API:** `POST http://localhost/predict` (Port 80).
    * **Interactive Swagger UI:** [http://localhost/docs](http://localhost/docs).
    * **Aggregated Telemetry:** `GET http://localhost/metrics`.

5.  **Stop the Cluster:**
    ```bash
    docker-compose -f deployments/docker-compose.yml down
    ```

---

## 🧪 Development & Training

To update the model or generate new performance reports, run the following commands from the **project root** using the Python module flag:

### 1. Train the Model
This script generates the dataset, trains the LSTM-Autoencoder, and saves the resulting artifacts.
```bash
python -m experiments.train_test
```

### 2. Generate Visual Report
Produces a "Skyscraper" plot showing reconstruction error over time, saved to the `visualizations/` directory.
```bash
python -m experiments.visualization
```