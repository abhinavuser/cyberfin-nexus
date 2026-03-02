"""
CyberFin Nexus — Graph Analytics
Community detection and subgraph anomaly scoring for discovering
unknown mule rings and suspicious account clusters.
"""

import numpy as np
import networkx as nx
from collections import Counter


def pyg_to_networkx(graph_data, account_mask=None):
    """Convert PyG Data object to NetworkX graph for analytics."""
    G = nx.Graph()

    # Add nodes with attributes
    for i, nid in enumerate(graph_data.node_ids):
        G.add_node(nid, idx=i, node_type=graph_data.node_types[i])

    # Add edges
    ei = graph_data.edge_index.numpy()
    for j in range(ei.shape[1]):
        src = graph_data.node_ids[ei[0, j]]
        dst = graph_data.node_ids[ei[1, j]]
        if src != dst:  # skip self-loops
            G.add_edge(src, dst)

    return G


def detect_communities(graph_data, scores=None, resolution=1.0):
    """
    Run Louvain community detection on the transaction/device graph.

    Returns:
        list of community dicts with members, risk stats, and suspicion score.
    """
    G = pyg_to_networkx(graph_data)

    # Louvain community detection
    try:
        communities = nx.community.louvain_communities(G, resolution=resolution, seed=42)
    except Exception:
        # Fallback: connected components
        communities = list(nx.connected_components(G))

    node_map = graph_data.node_map
    account_mask = graph_data.account_mask.numpy()
    node_types = graph_data.node_types

    results = []
    for comm_id, members in enumerate(communities):
        # Filter to account nodes only
        account_members = [m for m in members if m in node_map and node_types[node_map[m]] == "account"]
        device_members = [m for m in members if m in node_map and node_types[node_map[m]] == "device"]

        if len(account_members) < 2:
            continue  # skip trivial communities

        # Compute community risk metrics
        member_indices = [node_map[m] for m in account_members]
        member_risks = [float(scores[i]) for i in member_indices] if scores is not None else []

        avg_risk = np.mean(member_risks) if member_risks else 0.0
        max_risk = np.max(member_risks) if member_risks else 0.0
        high_risk_count = sum(1 for r in member_risks if r > 0.5)

        # Cross-bank ratio: communities spanning multiple banks are more suspicious
        bank_ids = set()
        for m in account_members:
            if m.startswith("ACC_"):
                parts = m.split("_")
                if len(parts) >= 2:
                    bank_ids.add(parts[1])
        cross_bank = len(bank_ids) > 1

        # Device sharing density: shared devices within the community
        device_sharing_score = min(len(device_members) / max(len(account_members), 1), 1.0)

        # Overall suspicion score
        suspicion = (
            avg_risk * 0.35
            + (high_risk_count / max(len(account_members), 1)) * 0.25
            + (1.0 if cross_bank else 0.0) * 0.20
            + device_sharing_score * 0.20
        )

        results.append({
            "community_id": comm_id,
            "account_members": account_members,
            "device_members": device_members,
            "size": len(account_members),
            "avg_risk": round(float(avg_risk), 3),
            "max_risk": round(float(max_risk), 3),
            "high_risk_count": high_risk_count,
            "cross_bank": cross_bank,
            "banks": sorted(bank_ids),
            "device_sharing_score": round(float(device_sharing_score), 3),
            "suspicion_score": round(float(suspicion), 3),
        })

    # Sort by suspicion score descending
    results.sort(key=lambda x: x["suspicion_score"], reverse=True)

    return results


def get_community_summary(communities):
    """Return high-level statistics about detected communities."""
    if not communities:
        return {"total": 0, "suspicious": 0, "cross_bank": 0, "avg_size": 0}

    suspicious = [c for c in communities if c["suspicion_score"] > 0.3]
    cross_bank = [c for c in communities if c["cross_bank"]]

    return {
        "total": len(communities),
        "suspicious": len(suspicious),
        "cross_bank": len(cross_bank),
        "avg_size": round(np.mean([c["size"] for c in communities]), 1),
        "top_suspicion": communities[0]["suspicion_score"] if communities else 0,
    }
