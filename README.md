# SecuriteAI: Unsupervised Anomaly Detection for Linux Logs

**SecuriteAI** is an unsupervised deep learning pipeline designed to identify "zero-day" anomalies in Linux system logs. Rather than relying on static attack signatures, the system utilizes an **LSTM-Autoencoder** to learn the underlying "temporal grammar" of healthy system behavior.

---

## 🚀 Key Performance Metrics

Based on the final system evaluation using synthetic data, SecuriteAI achieved the following results:

* **Signal-to-Noise Ratio (SNR):** `95,282.77x` — Indicating a massive mathematical separation between normal operations and attack states.
* **Detection Rate (Recall):** `100.00%` on high-entropy anomaly bursts.
* **False Positive Rate (FPR):** `1.41%` on unseen normal system logs.

---

## 🧠 Engineering Architecture

### 1. Reconstructive Modeling
The core architecture is a **Sequence-to-Sequence LSTM-Autoencoder**.
* **Encoder:** Compresses a sequence of 20 logs into a fixed-size latent bottleneck of 64 dimensions.
* **Decoder:** Attempts to reconstruct the original 20-log sequence from the latent vector.
* **Logic:** The model is trained exclusively on normal data; it reconstructs healthy logs with low error but fails significantly when encountering anomalous patterns it has never seen before.

### 2. Cyclical Temporal Engineering
Time is treated as a cyclical feature to preserve temporal adjacency (e.g., ensuring the model understands 23:59 is close to 00:01).
* Timestamps are decomposed into **Sine and Cosine pairs** for the Hour, Minute, Second, and Day dimensions.
* This results in an **8-dimensional temporal feature set** that captures system routines without overfitting to specific dates.

### 3. Isolation-Based Normalization
The system leverages a "poisoned" normalization strategy to maximize sensitivity.
* The **Min-Max scaler** for Event IDs is fitted strictly on the "Normal" training pool.
* When a high-ID anomaly (e.g., `E999`) occurs, its normalized value falls far outside the standard `0.0` to `1.0` range, triggering an immediate spike in reconstruction loss.

---

## 📂 Project Structure

| File | Primary Responsibility |
| :--- | :--- |
| `autoencoder.py` | Defines the 2-layer LSTM Encoder and Decoder classes. |
| `clean_log.py` | Handles timestamp synthesis and numeric Event ID extraction. |
| `feat_eng.py` | Orchestrates cyclical time encoding and sliding window creation. |
| `generate_data.py` | Synthetic log generator (10,000 Normal / 1,000 Anomaly). |
| `train_test.py` | Manages model training (100 epochs) and accuracy reporting. |
| `visualization.py` | Generates "Skyscraper" plots to visualize reconstruction error spikes. |

---

## 🛠️ Usage Pipeline

1.  **Environment Initialization:** Run `train_test.py` to clear previous artifacts and create a fresh `models/` directory.
2.  **Training & Thresholding:** Trains the autoencoder for 100 epochs, establishing an anomaly threshold at the **99.5th percentile** of training loss.
3.  **Inference:** Use the saved model weights and scaler parameters to evaluate incoming logs in real-time.
4.  **Visualization:** Run `visualization.py` to generate a report showing the mathematical "surprise" of the model during simulated attack bursts.