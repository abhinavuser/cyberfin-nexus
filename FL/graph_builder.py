"""
Phase 2: Graph Construction per Bank
Converts each bank's CSV into a PyTorch Geometric graph.

Graph structure:
- Nodes: each row (application) with numerical features
- Edges: applications sharing the same categorical attribute value
         are connected (shared-attribute edges)
- Labels: fraud_bool

This creates a natural graph where fraudulent patterns propagate
through shared device_os, payment_type, housing_status, etc.
"""
import csv
import os
import random
import math
from collections import defaultdict

import torch
from torch_geometric.data import Data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Column indices in bank CSVs (from preprocessed.csv)
TARGET_COL = 'fraud_bool'

# Numerical feature columns (used as node features)
NUMERICAL_COLS = [
    'income', 'name_email_similarity', 'prev_address_months_count',
    'current_address_months_count', 'customer_age', 'days_since_request',
    'intended_balcon_amount', 'zip_count_4w', 'velocity_6h',
    'velocity_24h', 'velocity_4w', 'bank_branch_count_8w',
    'date_of_birth_distinct_emails_4w', 'credit_risk_score',
    'email_is_free', 'phone_home_valid', 'phone_mobile_valid',
    'bank_months_count', 'has_other_cards', 'proposed_credit_limit',
    'foreign_request', 'session_length_in_minutes', 'keep_alive_session',
    'device_distinct_emails_8w', 'month',
    'prev_address_months_count_is_missing', 'bank_months_count_is_missing',
]

# Categorical columns (used to build edges)
CATEGORICAL_COLS = [
    'payment_type', 'employment_status', 'housing_status',
    'source', 'device_os',
]

# Max edges per categorical group to keep graph manageable
# (for groups with 100K+ nodes, we sample edges instead of full clique)
MAX_EDGES_PER_GROUP = 50


def build_graph_from_csv(csv_path, max_nodes=None, seed=42):
    """
    Build a PyG Data object from a bank CSV file.

    Strategy: For each categorical column, group nodes by value.
    Within each group, connect each node to K random neighbors
    (sampled edges instead of full clique to keep graph sparse).

    Args:
        csv_path: Path to bank CSV
        max_nodes: Optional limit for faster testing
        seed: Random seed

    Returns:
        PyG Data object with x, y, edge_index, train/val/test masks
    """
    random.seed(seed)
    bank_name = os.path.basename(csv_path).replace('.csv', '')
    print(f"\n  Building graph for {bank_name}...")

    # --- Read CSV ---
    rows = []
    header = None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for i, row in enumerate(reader):
            rows.append(row)
            if max_nodes and i >= max_nodes - 1:
                break

    n_nodes = len(rows)
    n_fraud = sum(1 for r in rows if r[TARGET_COL] == '1')
    print(f"    Nodes: {n_nodes:,} | Fraud: {n_fraud:,} ({n_fraud/n_nodes:.2%})")

    # --- Build node features (numerical columns) ---
    features = []
    for row in rows:
        feat = []
        for col in NUMERICAL_COLS:
            feat.append(float(row[col]))
        features.append(feat)

    x = torch.tensor(features, dtype=torch.float)
    print(f"    Feature dim: {x.shape[1]}")

    # --- Build labels ---
    y = torch.tensor([int(row[TARGET_COL]) for row in rows], dtype=torch.float)

    # --- Build edges from shared categorical attributes ---
    edge_src = []
    edge_dst = []

    for col in CATEGORICAL_COLS:
        # Group node indices by categorical value
        groups = defaultdict(list)
        for idx, row in enumerate(rows):
            groups[row[col]].append(idx)

        n_edges_col = 0
        for val, members in groups.items():
            if len(members) < 2:
                continue

            # For each node in the group, connect to K random neighbors
            k = min(MAX_EDGES_PER_GROUP, len(members) - 1)
            for node in members:
                neighbors = random.sample(members, k + 1)  # +1 because it may include self
                for nb in neighbors:
                    if nb != node:
                        edge_src.append(node)
                        edge_dst.append(nb)
                        n_edges_col += 1
                        if n_edges_col // len(members) >= k:
                            break  # enough per node

        print(f"    Edges from {col}: {n_edges_col:,}")

    # Make bidirectional
    all_src = edge_src + edge_dst
    all_dst = edge_dst + edge_src
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)

    # Remove self-loops and duplicates
    # (simple dedup via set)
    edge_set = set()
    clean_src, clean_dst = [], []
    for s, d in zip(all_src, all_dst):
        if s != d and (s, d) not in edge_set:
            edge_set.add((s, d))
            clean_src.append(s)
            clean_dst.append(d)

    edge_index = torch.tensor([clean_src, clean_dst], dtype=torch.long)
    print(f"    Total edges (deduped, bidirectional): {edge_index.shape[1]:,}")
    print(f"    Avg degree: {edge_index.shape[1] / n_nodes:.1f}")

    # --- Train/Val/Test masks (70/15/15 stratified) ---
    indices = list(range(n_nodes))
    fraud_idx = [i for i in indices if rows[i][TARGET_COL] == '1']
    legit_idx = [i for i in indices if rows[i][TARGET_COL] == '0']
    random.shuffle(fraud_idx)
    random.shuffle(legit_idx)

    def split_list(lst, ratios=(0.7, 0.15, 0.15)):
        n = len(lst)
        t1 = int(n * ratios[0])
        t2 = int(n * (ratios[0] + ratios[1]))
        return lst[:t1], lst[t1:t2], lst[t2:]

    f_train, f_val, f_test = split_list(fraud_idx)
    l_train, l_val, l_test = split_list(legit_idx)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    for idx in f_train + l_train:
        train_mask[idx] = True
    for idx in f_val + l_val:
        val_mask[idx] = True
    for idx in f_test + l_test:
        test_mask[idx] = True

    print(f"    Train: {train_mask.sum().item():,} | Val: {val_mask.sum().item():,} | Test: {test_mask.sum().item():,}")
    print(f"    Train fraud: {y[train_mask].sum().item():.0f} | Val fraud: {y[val_mask].sum().item():.0f} | Test fraud: {y[test_mask].sum().item():.0f}")

    # --- Assemble PyG Data ---
    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data


def build_all_bank_graphs(max_nodes=None, seed=42):
    """
    Build graphs for all 3 banks.

    Args:
        max_nodes: Optional limit per bank (for testing)

    Returns:
        dict {bank_name: PyG Data}
    """
    print("=" * 60)
    print("  PHASE 2: Graph Construction")
    print("=" * 60)

    bank_files = {
        'Bank_A': os.path.join(SCRIPT_DIR, 'bank_A.csv'),
        'Bank_B': os.path.join(SCRIPT_DIR, 'bank_B.csv'),
        'Bank_C': os.path.join(SCRIPT_DIR, 'bank_C.csv'),
    }

    graphs = {}
    for name, path in bank_files.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found. Run split_data.py first.")
            continue
        graphs[name] = build_graph_from_csv(path, max_nodes=max_nodes, seed=seed)

    print(f"\n{'=' * 60}")
    print("  GRAPH CONSTRUCTION COMPLETE")
    print(f"{'=' * 60}")
    for name, data in graphs.items():
        print(f"  {name}: {data.num_nodes:,} nodes, {data.edge_index.shape[1]:,} edges, {data.x.shape[1]} features")

    return graphs


if __name__ == '__main__':
    # Test with small subset first (5000 nodes per bank)
    print("Testing with 5000 nodes per bank...\n")
    graphs = build_all_bank_graphs(max_nodes=5000)

    # Verify graph properties
    for name, data in graphs.items():
        assert data.x.shape[0] == data.y.shape[0], f"{name}: node count mismatch"
        assert data.edge_index.shape[0] == 2, f"{name}: edge_index shape wrong"
        assert data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum() == data.num_nodes, f"{name}: mask coverage"
        assert not torch.isnan(data.x).any(), f"{name}: NaN in features"
    print("\n  All graph validation checks passed!")
