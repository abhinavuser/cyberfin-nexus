"""
CyberFin Nexus — Ensemble Model
Combines GAT risk scores with Isolation Forest anomaly detection
and rule-based heuristics for robust mule detection.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """Unsupervised anomaly detector on node feature vectors."""

    def __init__(self, contamination=0.15, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_predict(self, features, account_mask):
        """
        Fit on account node features and predict anomaly scores.

        Args:
            features: numpy array (n_nodes, n_features)
            account_mask: boolean array indicating which nodes are accounts.

        Returns:
            scores: numpy array (n_nodes,) with anomaly scores [0, 1].
                    Higher = more anomalous.
        """
        acc_features = features[account_mask]

        # Scale features for IF
        acc_scaled = self.scaler.fit_transform(acc_features)
        self.model.fit(acc_scaled)
        self._fitted = True

        # score_samples: lower = more anomalous. Negate and normalize to [0,1]
        raw_scores = self.model.score_samples(acc_scaled)
        # Convert: most negative → 1.0, most positive → 0.0
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            norm_scores = 1.0 - (raw_scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.full_like(raw_scores, 0.5)

        # Map back to all nodes
        all_scores = np.zeros(len(features), dtype=np.float32)
        all_scores[account_mask] = norm_scores.astype(np.float32)

        return all_scores


class RuleBasedDetector:
    """
    Domain-knowledge rules for mule detection.

    Rules encode expert knowledge that the GNN might overlook.
    Each rule contributes a score [0, 1]. Final score = weighted average.
    """

    def __init__(self):
        self.rules = [
            ("new_account_high_velocity", 0.25),
            ("low_balance_high_amount", 0.20),
            ("cyber_event_density", 0.20),
            ("device_sharing", 0.15),
            ("night_activity", 0.10),
            ("burst_transactions", 0.10),
        ]

    def predict(self, features, account_mask):
        """
        Apply rule-based scoring to node features.

        Args:
            features: numpy array (n_nodes, 12)
            account_mask: boolean array.

        Returns:
            scores: numpy array (n_nodes,) with rule-based risk scores [0, 1].
        """
        n_nodes = len(features)
        all_scores = np.zeros(n_nodes, dtype=np.float32)
        rule_details = {}

        acc_indices = np.where(account_mask)[0]
        for idx in acc_indices:
            f = features[idx]
            score = 0.0
            details = {}

            # Rule 1: New account + high velocity
            # f[2] = age_norm (low = new), f[3] = velocity_norm (high = suspicious)
            r1 = max(0, (1.0 - f[2]) * f[3])
            details["new_account_high_velocity"] = float(r1)
            score += r1 * 0.25

            # Rule 2: Low balance + high avg tx amount
            # f[1] = balance_norm (low), f[4] = avg_tx_amount_norm (high)
            r2 = max(0, (1.0 - f[1]) * f[4])
            details["low_balance_high_amount"] = float(r2)
            score += r2 * 0.20

            # Rule 3: High cyber event density
            # f[0] = cyber_risk, f[7] = cyber_event_count_norm
            r3 = f[0] * 0.6 + f[7] * 0.4
            details["cyber_event_density"] = float(r3)
            score += r3 * 0.20

            # Rule 4: Multiple devices (device sharing)
            # f[5] = device_count_norm
            r4 = min(f[5] * 1.5, 1.0)
            details["device_sharing"] = float(r4)
            score += r4 * 0.15

            # Rule 5: Night activity (temporal feature 9)
            r5 = f[9] if len(f) > 9 else 0.0
            details["night_activity"] = float(r5)
            score += r5 * 0.10

            # Rule 6: Burst transactions (temporal feature 8)
            r6 = f[8] if len(f) > 8 else 0.0
            details["burst_transactions"] = float(r6)
            score += r6 * 0.10

            all_scores[idx] = min(score, 1.0)
            rule_details[idx] = details

        return all_scores, rule_details


class EnsembleScorer:
    """
    Combines GAT, Isolation Forest, and Rule-Based scores.

    Weights: GAT 60%, IF 20%, Rules 20%
    Consensus flag: 2/3 methods agree on high risk → elevated priority.
    """

    def __init__(self, gat_weight=0.60, if_weight=0.20, rule_weight=0.20):
        self.gat_weight = gat_weight
        self.if_weight = if_weight
        self.rule_weight = rule_weight
        self.if_detector = IsolationForestDetector()
        self.rule_detector = RuleBasedDetector()

    def compute_ensemble(self, gat_scores, features, account_mask):
        """
        Compute ensemble risk scores.

        Args:
            gat_scores: numpy array from GAT model.
            features: numpy array of node features.
            account_mask: boolean array.

        Returns:
            dict with ensemble_scores, individual scores, consensus flags.
        """
        # Individual model scores
        if_scores = self.if_detector.fit_predict(features, account_mask)
        rule_scores, rule_details = self.rule_detector.predict(features, account_mask)

        # Weighted ensemble
        ensemble = (
            self.gat_weight * gat_scores
            + self.if_weight * if_scores
            + self.rule_weight * rule_scores
        )

        # Consensus: how many methods flag as high risk (>0.5)
        gat_flag = (gat_scores > 0.5).astype(int)
        if_flag = (if_scores > 0.5).astype(int)
        rule_flag = (rule_scores > 0.5).astype(int)
        consensus = gat_flag + if_flag + rule_flag  # 0–3

        return {
            "ensemble_scores": ensemble,
            "gat_scores": gat_scores,
            "if_scores": if_scores,
            "rule_scores": rule_scores,
            "rule_details": rule_details,
            "consensus": consensus,
            "weights": {
                "gat": self.gat_weight,
                "isolation_forest": self.if_weight,
                "rules": self.rule_weight,
            },
        }
