# CyberFin Nexus

**Privacy-First Cyber-Financial Fusion for Mule Ring Detection using GNNs, Federated Learning & Gemini AI**

> Banks lose **$3.1B annually** to money mule networks. CyberFin Nexus fuses cyber threat intelligence with financial transaction graphs to detect mule rings using Graph Attention Networks — while preserving privacy through Federated Learning.

---

## 🎯 Problem

- **SOC teams** detect phishing/malware but ignore financial patterns
- **AML teams** flag suspicious transactions but miss cyber-attack chains
- **Privacy laws** (GDPR/DPDP) prevent banks from sharing data
- **Criminals evolve** — using VPNs, AI mimicry, device rotation to evade detection

**CyberFin Nexus bridges all gaps** with a unified graph + ensemble approach.

---

## 🏗️ Architecture

```
PostgreSQL ──► Heterogeneous Graph (12-dim features)
                    │
    ┌───────────────┼───────────────┐
    │               │               │
  GAT (60%)    Iso Forest (20%)  Rules (20%)
    │               │               │
    └───────────────┼───────────────┘
                    │
            Ensemble Scorer + Consensus
                    │
    ┌───────────────┼───────────────┐
    │               │               │
 Community      Active          Adversarial
 Detection     Learning         RL Testing
    │               │               │
    └───────────────┼───────────────┘
                    │
         Blockchain Audit Trail
                    │
            7-Tab Dashboard
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🕸️ **Cyber-Financial Graph** | Unified graph connecting accounts ↔ devices ↔ transactions ↔ cyber events |
| 🧠 **GAT Model** | 3-layer, 4-head Graph Attention Network (AUC: 0.983) |
| 🛡️ **Ensemble Defense** | GAT + Isolation Forest + Rule-Based with 2-of-3 consensus |
| 🔎 **Community Detection** | Louvain clustering discovers unknown cross-bank mule rings |
| 🎯 **Active Learning** | Surfaces uncertain accounts for human review |
| 🏦 **Federated Learning** | FedAvg + Differential Privacy (ε=1.0) across 4 banks |
| ⚔️ **Adversarial RL** | 4 attack strategies (velocity, splitting, rotation, mimicry) |
| 🤖 **Gemini AI** | Natural language explanations for flagged accounts |
| 🔗 **Blockchain Audit** | SHA-256 tamper-proof trail for regulators |
| 💾 **PostgreSQL** | Live database with sidebar connection status |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL (optional — falls back to in-memory data)
- [PyTorch](https://pytorch.org/get-started/locally/) with your CUDA/CPU version

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/cyberfin-nexus.git
cd cyberfin-nexus

# Install dependencies
pip install -r requirements.txt
```

### Database Setup (Optional)

```bash
# Create PostgreSQL database
psql -U postgres -c "CREATE DATABASE cyberfin_db;"

# Create .env file
echo "DB_USER=postgres" > .env
echo "DB_PASS=yourpassword" >> .env
echo "DB_HOST=localhost" >> .env
echo "DB_PORT=5432" >> .env
echo "DB_NAME=cyberfin_db" >> .env

# Seed the database with synthetic data
python data_generator.py
```

### Run the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Run Tests

```bash
python test_all.py
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| AUC | **0.983** |
| Precision | **76.1%** |
| Recall | **94.6%** |
| F1 Score | **0.843** |
| Privacy | ε=1.0, GDPR compliant |
| Est. Annual Savings | **$18.5M** |

---

## 🖥️ Dashboard Tabs

1. **🏠 Overview** — Key metrics, training progress, risk distribution
2. **🕸️ Nexus Graph** — Interactive network visualization with risk-colored nodes
3. **🏦 Federated Learning** — Cross-bank training rounds with privacy metrics
4. **⚔️ Attack Simulation** — RL adversarial testing with 4 evasion strategies
5. **🔗 Audit Trail** — Blockchain event log with integrity verification
6. **🚨 Alerts & XAI** — Flagged accounts with Gemini AI explanations
7. **🧬 Defense Analysis** — Ensemble scoring, community detection, active learning queue

---

## 🔧 Tech Stack

- **ML**: PyTorch, PyTorch Geometric, Scikit-learn
- **Graph**: NetworkX (Louvain community detection)
- **Privacy**: Custom FedAvg + Differential Privacy
- **AI**: Google Gemini API
- **Database**: PostgreSQL + SQLAlchemy
- **Frontend**: Streamlit + Plotly
- **Security**: SHA-256 blockchain hash chain

---

## 📁 Project Structure

```
├── app.py                    # 7-tab Streamlit dashboard
├── data_generator.py         # Synthetic data + DB seeding
├── graph_builder.py          # Heterogeneous graph (12-dim features)
├── test_all.py               # Integration tests
├── requirements.txt
├── models/
│   ├── gat_model.py          # 3-layer GAT
│   └── ensemble.py           # IF + Rules + Ensemble
├── federated/
│   └── fl_engine.py          # FedAvg + DP
├── rl/
│   └── adversarial_sim.py    # Adversarial RL
├── blockchain/
│   └── audit_trail.py        # SHA-256 audit chain
└── utils/
    ├── config.py             # Hyperparameters
    ├── metrics.py            # Evaluation metrics
    ├── gemini_helper.py      # Gemini AI integration
    ├── graph_analytics.py    # Community detection
    ├── active_learning.py    # Review queue
    └── db_manager.py         # PostgreSQL manager
```



