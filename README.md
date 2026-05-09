# 🔍 UPI Fraud Radar - 4th Place
### Real-Time Fraud Detection using Graph Analytics + Databricks

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.5-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-Model_Serving-FF3621?style=flat-square&logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## 📌 Overview

Digital payment systems like UPI process **millions of transactions daily**, making real-time fraud detection critical. Traditional rule-based systems fail because fraudsters adapt quickly and operate in **networks** — not in isolation.

This project models transactions as a **directed graph** (accounts = nodes, money flows = edges), extracts behavioral features per node, and feeds them into a **Random Forest classifier** served via Databricks to flag fraudulent accounts in real time.

---

## 🎯 Problem Statement

| Challenge | Why It Matters |
|---|---|
| Fraudsters adapt to rules | Static thresholds become obsolete quickly |
| Multi-account fraud rings | Patterns invisible when viewed per-transaction |
| Millions of transactions/day | Needs scalable, automated detection |
| Extreme class imbalance | Rare fraud events hard to learn from |

---

## 🧠 Solution Approach
👉 [Watch Demo Video](https://drive.google.com/file/d/1gz2ekK0BxcXS8IEu0HHoeVDGruYi4gBm/view?usp=sharing)

We model the transaction network as a graph and extract **per-node structural features** that capture behavioral anomalies:

```
Transactions → Graph (nodes + edges) → Feature Extraction → ML Model → Fraud Score
```

A node (account) is suspicious if it has:
- High outgoing flow with zero incoming transactions
- No triangles (no reciprocal relationships)
- Isolated cluster (not connected to trusted accounts)
- Disproportionate `amount_out` vs `amount_in`

---

## ⚙️ Architecture

```
User Input (Streamlit)
        │
        ▼
Graph Construction (NetworkX / GraphFrames)
  nodes = unique accounts
  edges = sender → receiver (aggregated)
        │
        ▼
Feature Extraction per node
  degree, tx_count_in, tx_count_out,
  amount_in, amount_out, cluster_size, triangle_count
        │
        ▼
Databricks Model Serving  ←── POST /invocations
        │
        ▼
Fraud Score  [0.0 → 1.0]
  ≥ 0.7  →  🚨 FRAUD
  0.4–0.7 → ⚠️  SUSPECT
  < 0.4  →  ✅ LEGIT
```

---

## ☁️ How Databricks is Used

### ✅ Data Processing
Large-scale transaction data handled using **Apache Spark** with **Delta Lake** tables for nodes and edges.

### ✅ Feature Engineering
Graph-based features computed at scale using **GraphFrames** on Spark clusters.

### ✅ Model Training
**Random Forest** trained with PySpark MLlib. Cross-validation used for hyperparameter tuning.

### ✅ Experiment Tracking
**MLflow** auto-logs metrics, parameters, and model artifacts. Model versions registered in Model Registry.

### ✅ Model Serving
Registered model deployed as a **Databricks REST endpoint**. The Streamlit app POSTs feature vectors and receives fraud scores.

**Request format:**
```json
POST https://<workspace>.cloud.databricks.com/serving-endpoints/<n>/invocations

{
  "dataframe_records": [
    {
      "id": "C1232585256",
      "degree": 1,
      "tx_count_in": 0,
      "tx_count_out": 10,
      "amount_in": 0.0,
      "amount_out": 168083.2,
      "cluster_size": 1,
      "triangle_count": 1
    }
  ]
}
```

---

## 🕸️ Graph Features

| Feature | Description | Fraud Signal |
|---|---|---|
| `degree` | Total connections (in + out) | Exactly 1 or abnormally high |
| `tx_count_in` | Incoming transaction count | Zero while tx_count_out is large |
| `tx_count_out` | Outgoing transaction count | High fan-out to many receivers |
| `amount_in` | Total money received (₹) | Near zero vs. large amount_out |
| `amount_out` | Total money sent (₹) | Disproportionate to amount_in |
| `cluster_size` | Weakly-connected component size | Isolated node (size = 1) |
| `triangle_count` | Triangles the node participates in | Zero = no reciprocal relationships |

---

## 🖥️ Streamlit Frontend

The app has 4 tabs:

| Tab | What it does |
|---|---|
| **Manual Entry** | Add transactions one by one → instant graph + prediction |
| **Bulk / Stream CSV** | Upload CSV, replay at N rows/sec or process all at once |
| **Graph Features** | View extracted node table, distributions, download CSV |
| **Predictions** | Risk scores, fraud alert cards, export results |

**CSV format expected:**
```
sender, receiver, amount, timestamp, type
C1001,  M2001,    250.0,  2026-04-25T10:00:00, P2M
```

---

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone <your-repo-link>
cd <repo-folder>
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run upi_fraud_app.py
```

**4. Configure Databricks (in the sidebar)**
```
Endpoint URL  →  https://<workspace>.cloud.databricks.com/serving-endpoints/<n>/invocations
Access Token  →  dapiXXXXXXXXXXXXXX
```

> 💡 Keep **mock mode ON** while the Databricks endpoint is still deploying — the heuristic predictor gives realistic scores for demo purposes.

---

## 📊 Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | PySpark, Delta Lake, Apache Spark |
| Graph Analytics | NetworkX, GraphFrames |
| ML Training | PySpark MLlib (Random Forest) |
| Experiment Tracking | MLflow |
| Model Serving | Databricks Model Serving (REST API) |
| Frontend | Streamlit |
| Language | Python 3.11 |

---

## ⚠️ Limitations

- Model uses graph-structural features only — no time-series or amount-deviation signals yet
- Dataset is highly imbalanced (fraud events are rare)
- Binary output only — no continuous anomaly score threshold tuning
- Does not currently model temporal patterns (velocity of transactions per hour)

---

## 🚀 Future Improvements

- [ ] Add time-series velocity features (transactions per hour window)
- [ ] Use continuous anomaly scores instead of binary classification
- [ ] Apply SMOTE / class-weight balancing for imbalanced data
- [ ] Incorporate amount-deviation from account's historical baseline
- [ ] Deploy scalable API gateway for production traffic
- [ ] Add GNN (Graph Neural Network) layer for richer node embeddings

---

## 🏁 Conclusion

This project demonstrates how combining **graph analytics**, **machine learning**, and **cloud-based model serving (Databricks)** can power a real-time fraud detection system that goes far beyond traditional rule-based approaches — detecting coordinated fraud rings that would otherwise be invisible at the transaction level.

---

> Built for the Hackathon · Graph Analytics + Databricks + Streamlit
