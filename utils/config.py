"""
CyberFin Nexus — Configuration & Constants
All hyperparameters, paths, color schemes, and constants in one place.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ─── Data Generation ────────────────────────────────────────────────────
NUM_BANKS = 4
NUM_ACCOUNTS = 250
NUM_DEVICES = 60
NUM_TRANSACTIONS = 600
NUM_CYBER_EVENTS = 400
MULE_RING_COUNT = 6          # number of mule rings to generate
MULE_RING_SIZE = (3, 8)      # min/max accounts per ring
MULE_RATIO = 0.15            # ~15% of accounts are mules

# ─── Attack Types ────────────────────────────────────────────────────────
ATTACK_TYPES = ["phishing", "malware", "brute_force", "credential_stuffing", "session_hijack"]
ATTACK_SEVERITY = {"phishing": 0.7, "malware": 0.9, "brute_force": 0.5,
                   "credential_stuffing": 0.6, "session_hijack": 0.85}

# ─── GNN / GAT Model ────────────────────────────────────────────────────
GAT_INPUT_DIM = 12            # node feature dimension (8 base + 4 temporal)
GAT_HIDDEN_DIM = 32
GAT_OUTPUT_DIM = 1
GAT_HEADS = 4
GAT_LAYERS = 3
GAT_DROPOUT = 0.3
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
TRAIN_EPOCHS = 100
TRAIN_PATIENCE = 15           # early stopping patience

# ─── Federated Learning ─────────────────────────────────────────────────
FL_ROUNDS = 10
FL_LOCAL_EPOCHS = 5
FL_DP_EPSILON = 1.0           # differential privacy budget
FL_DP_DELTA = 1e-5
FL_DP_CLIP_NORM = 1.0

# ─── RL / Adversarial ───────────────────────────────────────────────────
RL_ATTACK_STRATEGIES = [
    "velocity_reduction",      # mules slow tx frequency
    "amount_splitting",        # break large into sub-threshold
    "device_rotation",         # switch devices
    "behavioral_mimicry",      # AI-generated normal patterns
]
RL_RETRAIN_EPOCHS = 10

# ─── Blockchain Audit ───────────────────────────────────────────────────
HASH_ALGORITHM = "sha256"

# ─── Dashboard Colors & Theme ───────────────────────────────────────────
THEME = {
    "bg_primary": "#0a0e17",
    "bg_secondary": "#111827",
    "bg_card": "#1a2332",
    "accent_cyan": "#00f0ff",
    "accent_pink": "#ff006e",
    "accent_green": "#00ff88",
    "accent_orange": "#ff8c00",
    "accent_purple": "#8b5cf6",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "risk_low": "#00ff88",
    "risk_medium": "#ffaa00",
    "risk_high": "#ff006e",
    "risk_critical": "#ff0000",
    "gradient_start": "#00f0ff",
    "gradient_end": "#8b5cf6",
}

# Risk thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.85,
    "critical": 0.95,
}

# ROI assumptions
ROI_AVG_MULE_RING_LOSS = 500_000   # $ average loss per undetected ring
ROI_DETECTION_IMPROVEMENT = 0.25    # 25% improvement over baseline
