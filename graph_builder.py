"""
CyberFin Nexus — Heterogeneous Graph Builder
Fuses cyber logs + financial transactions into a PyTorch Geometric graph.
Nodes: accounts, devices, external endpoints.
Edges: account↔device (cyber), account→account (transaction), account→external (withdrawal).
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import GAT_INPUT_DIM


def build_graph(data_dict):
    """
    Build a PyTorch Geometric Data object from generated data.

    Returns:
        data: PyG Data object with x (node features), edge_index, y (labels),
              node_ids, node_types, edge_types
    """
    accounts = data_dict["accounts"]
    devices = data_dict["devices"]
    acc_dev_map = data_dict["acc_dev_map"]
    cyber_events = data_dict["cyber_events"]
    transactions = data_dict["transactions"]

    # ─── Collect all unique node IDs ─────────────────────────────────────
    account_ids = accounts.account_id.tolist()
    device_ids = devices.device_id.unique().tolist()

    # External accounts from transactions
    ext_accounts = set()
    for acc in transactions.to_account.unique():
        if acc.startswith("EXT_") or acc not in account_ids:
            ext_accounts.add(acc)
    ext_accounts = sorted(ext_accounts)

    all_node_ids = account_ids + device_ids + ext_accounts
    node_map = {nid: idx for idx, nid in enumerate(all_node_ids)}
    n_nodes = len(all_node_ids)

    # ─── Node types ──────────────────────────────────────────────────────
    node_types = (
        ["account"] * len(account_ids)
        + ["device"] * len(device_ids)
        + ["external"] * len(ext_accounts)
    )

    # ─── Node features (12-dim) ──────────────────────────────────────────
    features = np.zeros((n_nodes, GAT_INPUT_DIM), dtype=np.float32)

    # Account features: [cyber_risk_agg, avg_balance_norm, age_norm, velocity_norm,
    #                     avg_tx_amt_norm, device_count, is_vpn_freq, n_cyber_events,
    #                     burst_score, night_ratio, age_velocity_ratio, timing_entropy]
    for _, row in accounts.iterrows():
        idx = node_map[row.account_id]
        acc_events = cyber_events[cyber_events.account_id == row.account_id]
        acc_devs = acc_dev_map[acc_dev_map.account_id == row.account_id]
        acc_txns = transactions[transactions.from_account == row.account_id]

        cyber_risk = acc_events.severity.mean() if len(acc_events) > 0 else 0.0
        vpn_freq = acc_events.is_vpn.mean() if len(acc_events) > 0 else 0.0
        n_events = min(len(acc_events) / 20.0, 1.0)  # normalize

        # ── Temporal features ────────────────────────────────────────
        # Burst score: fraction of transactions within 2-hour windows of each other
        burst_score = 0.0
        if len(acc_txns) > 1 and 'timestamp' in acc_txns.columns:
            ts = pd.to_datetime(acc_txns['timestamp']).sort_values()
            diffs = ts.diff().dt.total_seconds().dropna()
            burst_score = float((diffs < 7200).sum() / max(len(diffs), 1))

        # Night activity: fraction of transactions between 11PM–5AM
        night_ratio = 0.0
        if len(acc_txns) > 0 and 'hour_of_day' in acc_txns.columns:
            night_count = ((acc_txns.hour_of_day >= 23) | (acc_txns.hour_of_day <= 5)).sum()
            night_ratio = float(night_count / len(acc_txns))

        # Age-velocity suspicion: young account + high velocity = very suspicious
        age_norm = min(row.account_age_days / 3650, 1.0)
        vel_norm = min(row.tx_velocity / 15, 1.0)
        age_vel_ratio = (1.0 - age_norm) * vel_norm  # high when new + fast

        # Timing entropy: low entropy = automated/scripted behaviour
        timing_entropy = 0.5  # default mid-value
        if len(acc_txns) > 2 and 'hour_of_day' in acc_txns.columns:
            hours = acc_txns.hour_of_day.values
            hist, _ = np.histogram(hours, bins=24, range=(0, 24), density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                timing_entropy = min(entropy / np.log2(24), 1.0)

        features[idx] = [
            cyber_risk,
            min(row.avg_balance / 50000, 1.0),
            age_norm,
            vel_norm,
            min(row.avg_tx_amount / 15000, 1.0),
            min(len(acc_devs) / 5.0, 1.0),
            vpn_freq,
            n_events,
            burst_score,
            night_ratio,
            age_vel_ratio,
            timing_entropy,
        ]

    # Device features: [fingerprint_entropy, is_vpn, is_proxy, usage_count, 0,...,0]
    for _, row in devices.iterrows():
        if row.device_id in node_map:
            idx = node_map[row.device_id]
            usage = len(acc_dev_map[acc_dev_map.device_id == row.device_id])
            features[idx, :4] = [
                row.fingerprint_entropy,
                float(row.is_vpn),
                float(row.is_proxy),
                min(usage / 10.0, 1.0),
            ]

    # External features: all zeros (unknown)

    # ─── Labels (accounts only) ──────────────────────────────────────────
    labels = np.zeros(n_nodes, dtype=np.float32)
    for _, row in accounts.iterrows():
        labels[node_map[row.account_id]] = float(row.is_mule)

    # account_mask: which nodes are accounts (for training/eval only on accounts)
    account_mask = np.zeros(n_nodes, dtype=bool)
    for acc_id in account_ids:
        account_mask[node_map[acc_id]] = True

    # ─── Edges ───────────────────────────────────────────────────────────
    edge_sources = []
    edge_targets = []
    edge_types_list = []

    # 1. Account ↔ Device edges (from account-device mapping)
    for _, row in acc_dev_map.iterrows():
        if row.account_id in node_map and row.device_id in node_map:
            s, t = node_map[row.account_id], node_map[row.device_id]
            edge_sources.extend([s, t])
            edge_targets.extend([t, s])
            edge_types_list.extend(["acc_dev", "dev_acc"])

    # 2. Account → Account / External edges (from transactions)
    for _, row in transactions.iterrows():
        if row.from_account in node_map and row.to_account in node_map:
            s, t = node_map[row.from_account], node_map[row.to_account]
            edge_sources.append(s)
            edge_targets.append(t)
            edge_types_list.append("transaction")

    # 3. Cyber event edges (account → device at time of event)
    for _, row in cyber_events.iterrows():
        if row.account_id in node_map and row.device_id in node_map:
            s, t = node_map[row.account_id], node_map[row.device_id]
            edge_sources.append(s)
            edge_targets.append(t)
            edge_types_list.append("cyber_event")

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

    # ─── Build PyG Data ──────────────────────────────────────────────────
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.float),
    )

    # Extra metadata stored as attributes
    data.node_ids = all_node_ids
    data.node_types = node_types
    data.edge_types = edge_types_list
    data.account_mask = torch.tensor(account_mask, dtype=torch.bool)
    data.node_map = node_map
    data.num_nodes = n_nodes

    return data


def partition_by_bank(data_dict, graph_data):
    """Partition graph data by bank for federated learning simulation."""
    accounts = data_dict["accounts"]
    partitions = {}

    for bank_id in sorted(accounts.bank_id.unique()):
        bank_accounts = accounts[accounts.bank_id == bank_id].account_id.tolist()
        bank_acc_indices = [graph_data.node_map[a] for a in bank_accounts if a in graph_data.node_map]

        # Create a mask for this bank's account nodes
        bank_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
        for idx in bank_acc_indices:
            bank_mask[idx] = True

        partitions[bank_id] = {
            "account_ids": bank_accounts,
            "node_indices": bank_acc_indices,
            "bank_mask": bank_mask,
        }

    return partitions


if __name__ == "__main__":
    from data_generator import generate_all_data
    data_dict = generate_all_data()
    graph = build_graph(data_dict)
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    print(f"Features shape: {graph.x.shape}")
    print(f"Labels: {graph.y.sum().item():.0f} mules / {graph.y.shape[0]} total")
    print(f"Account mask: {graph.account_mask.sum().item()} accounts")

    partitions = partition_by_bank(data_dict, graph)
    for bank_id, p in partitions.items():
        print(f"Bank {bank_id}: {len(p['account_ids'])} accounts")
