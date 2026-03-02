"""
CyberFin Nexus — Active Learning
Identifies accounts where the model is most uncertain and prioritizes
them for human review based on multiple risk signals.
"""

import numpy as np


def get_uncertain_accounts(scores, account_mask, node_ids,
                           low_threshold=0.35, high_threshold=0.65):
    """
    Find accounts where the model is most uncertain (scores near 0.5).

    Args:
        scores: numpy array of risk scores.
        account_mask: boolean array for account nodes.
        node_ids: list of node IDs.
        low_threshold: lower bound of uncertainty zone.
        high_threshold: upper bound of uncertainty zone.

    Returns:
        list of dicts with account_id, score, and uncertainty_distance.
    """
    uncertain = []
    for i, nid in enumerate(node_ids):
        if account_mask[i]:
            score = float(scores[i])
            if low_threshold <= score <= high_threshold:
                # Distance from 0.5 → lower distance = more uncertain
                uncertainty = 1.0 - abs(score - 0.5) * 4  # [0, 1]
                uncertain.append({
                    "account_id": nid,
                    "score": round(score, 4),
                    "uncertainty": round(max(uncertainty, 0), 4),
                })

    uncertain.sort(key=lambda x: x["uncertainty"], reverse=True)
    return uncertain


def get_review_queue(scores, features, account_mask, node_ids,
                     graph_data=None, max_items=20):
    """
    Build a prioritized review queue combining uncertainty with
    contextual risk signals.

    Priority = uncertainty × neighbor_risk × feature_suspicion

    Args:
        scores: numpy array of risk scores.
        features: numpy array of node features.
        account_mask: boolean array.
        node_ids: list of node IDs.
        graph_data: PyG Data object for neighbor analysis.
        max_items: maximum queue length.

    Returns:
        list of dicts sorted by priority score.
    """
    uncertain = get_uncertain_accounts(scores, account_mask, node_ids)

    if not uncertain:
        return []

    node_map = {nid: i for i, nid in enumerate(node_ids)}
    ei = graph_data.edge_index.numpy() if graph_data is not None else None

    queue = []
    for item in uncertain:
        idx = node_map.get(item["account_id"])
        if idx is None:
            continue

        f = features[idx]

        # Feature-based suspicion factors
        cyber_risk = float(f[0]) if len(f) > 0 else 0.0
        velocity = float(f[3]) if len(f) > 3 else 0.0
        burst = float(f[8]) if len(f) > 8 else 0.0
        night = float(f[9]) if len(f) > 9 else 0.0
        age_vel = float(f[10]) if len(f) > 10 else 0.0

        feat_suspicion = (cyber_risk * 0.3 + velocity * 0.2
                          + burst * 0.2 + night * 0.15 + age_vel * 0.15)

        # Neighbor risk (average risk of connected nodes)
        neighbor_risk = 0.0
        if ei is not None:
            neighbor_indices = ei[1, ei[0] == idx]
            if len(neighbor_indices) > 0:
                neighbor_scores = [float(scores[n]) for n in neighbor_indices
                                   if n < len(scores)]
                neighbor_risk = np.mean(neighbor_scores) if neighbor_scores else 0.0

        # Combined priority
        priority = (
            item["uncertainty"] * 0.40
            + feat_suspicion * 0.30
            + neighbor_risk * 0.30
        )

        item["feature_suspicion"] = round(float(feat_suspicion), 4)
        item["neighbor_risk"] = round(float(neighbor_risk), 4)
        item["priority"] = round(float(priority), 4)
        item["review_reason"] = _get_reason(item, f)
        queue.append(item)

    queue.sort(key=lambda x: x["priority"], reverse=True)
    return queue[:max_items]


def _get_reason(item, features):
    """Generate a human-readable reason for why this account needs review."""
    reasons = []

    if item["score"] > 0.5:
        reasons.append("slightly elevated risk score")
    else:
        reasons.append("borderline risk score")

    if len(features) > 10 and features[10] > 0.5:
        reasons.append("new account with high activity")
    if len(features) > 8 and features[8] > 0.3:
        reasons.append("burst transaction pattern")
    if len(features) > 9 and features[9] > 0.2:
        reasons.append("unusual night-time activity")
    if features[0] > 0.4:
        reasons.append("elevated cyber risk")
    if item.get("neighbor_risk", 0) > 0.5:
        reasons.append("connected to high-risk accounts")

    if not reasons:
        reasons.append("model uncertainty requires human judgement")

    return "; ".join(reasons[:3])
