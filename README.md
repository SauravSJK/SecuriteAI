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

## 🧠 Engineering & Model Architecture

### 1. LSTM-Autoencoder Architecture
The core "brain" of the system is an Unsupervised Neural Network designed for sequence modeling:
* **Encoder:** Compresses a sequence of 20 log entries into a low-dimensional "latent space" representation.
* **Decoder:** Attempts to reconstruct the original 20-log sequence from the compressed latent vector.
* **Anomaly Logic:** During training, the model learns the patterns of "normal" logs. When an attack occurs, the reconstruction error (MSE) spikes because the decoder cannot accurately recreate the unfamiliar patterns.

### 2. Feature Engineering Pipeline
Raw logs are transformed into a 9-dimensional numerical tensor:
* **Cyclical Encoding:** Time components (Hour, Minute, Second) and Date components are transformed into $Sin$ and $Cos$ pairs to preserve temporal continuity (e.g., 23:59 being close to 00:01).
* **Normalization:** Event IDs and numerical features are scaled using Min-Max scaling based exclusively on normal training data distributions.

### 3. Modular System Design
The project is organized to separate concerns between core logic, infrastructure, and model artifacts:
* **`src/`**: Core model architectures (`models/`), log cleaning (`processing/`), and synthetic data generation (`utils/`).
* **`api/`**: FastAPI inference layer with strict Pydantic validation for log batches.
* **`deployments/`**: Orchestration logic including `Dockerfile`, `docker-compose.yml`, and `nginx.conf`.
* **`artifacts/`**: Storage for `weights/` (.pth) and `parameters/` (.npy).

---

## 🌐 Scaling & Load Balancing

### 1. Horizontal Scaling
The system utilize an **Nginx Load Balancer** to distribute high-volume log traffic across a fleet of identical SecuriteAI inference containers:
* **Strategy:** Implements a Round Robin distribution for optimal resource utilization.
* **Redundancy:** Configured with three replicas by default to ensure high availability.

### 2. Vectorized Batch Inference
The API is optimized for enterprise volume by supporting **Log Batching**. It processes multiple 20-log sequences in a single mathematical pass, significantly reducing overhead per prediction.

### 3. Memory-Resident Caching
To minimize latency, model weights and baseline statistics are cached in RAM via the FastAPI `lifespan` manager, eliminating slow disk I/O during active inference.

---

## 🛠️ Deployment (The Scaled Fortress)

Using **Docker Compose**, you can launch the entire load-balanced cluster from the project root.

1.  **Prepare the Environment:**
    Ensure your `artifacts/` directory contains the pre-trained weights (`securiteai_model.pth`) and baseline metrics (`loss_metrics.npy`, `scaler_params.npy`, and `anomaly_threshold.npy`).

2.  **Launch the Cluster:**
    Initialize the Nginx controller and the inference fleet:
    ```bash
    docker-compose -f deployments/docker-compose.yml up --build -d
    ```

3.  **Dynamic Scaling:**
    Scale the inference fleet to 10+ instances instantly:
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

To update the model or generate performance reports, run these commands from the **project root**:

### 1. Train the Model
This script generates the synthetic dataset, trains the LSTM-Autoencoder, and saves the resulting artifacts to the `artifacts/` folder.
```bash
python -m experiments.train_test
```

### 2. Generate Visual Report
Produces a "Skyscraper" plot showing reconstruction error over time, saved to the `visualizations/` directory.
```bash
python -m experiments.visualization
```