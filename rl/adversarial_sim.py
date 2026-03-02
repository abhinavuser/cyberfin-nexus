"""
CyberFin Nexus — Adversarial RL Simulation
Simulates attacker evolution strategies and tests model robustness.
After each attack mutation, retrains GAT incrementally (online learning).
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import RL_ATTACK_STRATEGIES, RL_RETRAIN_EPOCHS, LEARNING_RATE


class AdversarialSimulator:
    """
    Simulates attacker adaptations and tests/retrains the GNN model.

    Attack strategies:
    1. Velocity Reduction - mules slow transactions
    2. Amount Splitting - break large amounts into sub-threshold
    3. Device Rotation - switch devices to break fingerprints
    4. Behavioral Mimicry - AI-generated normal patterns
    """

    def __init__(self, model, graph_data):
        self.original_model = model
        self.graph_data = graph_data
        self.original_x = graph_data.x.clone()
        self.simulation_results = []

    def run_all_attacks(self, retrain_epochs=RL_RETRAIN_EPOCHS, verbose=True):
        """Run all attack strategies and return results."""
        results = []

        for strategy in RL_ATTACK_STRATEGIES:
            if verbose:
                print(f"\n{'='*60}")
                print(f"⚔️  Attack Strategy: {strategy}")
                print(f"{'='*60}")

            result = self.simulate_attack(strategy, retrain_epochs, verbose)
            results.append(result)

        self.simulation_results = results
        return results

    def simulate_attack(self, strategy, retrain_epochs=RL_RETRAIN_EPOCHS, verbose=True):
        """
        Simulate a single attack strategy.

        Returns dict with before/after AUC, detection rates, and adaptation metrics.
        """
        # Baseline: evaluate current model
        baseline_auc, baseline_preds = self._evaluate(self.original_model)

        # Apply attack perturbation to graph features
        attacked_data = self._apply_attack(strategy)

        # Evaluate model under attack (without retraining)
        attack_auc, attack_preds = self._evaluate(self.original_model, attacked_data)

        # Retrain model (RL-like adaptation)
        adapted_model = copy.deepcopy(self.original_model)
        adapted_model, retrain_history = self._retrain(
            adapted_model, attacked_data, retrain_epochs
        )

        # Evaluate adapted model
        adapted_auc, adapted_preds = self._evaluate(adapted_model, attacked_data)

        # Compute detection rates
        account_mask = self.graph_data.account_mask.numpy()
        y_true = self.graph_data.y[self.graph_data.account_mask].numpy()

        baseline_detected = (baseline_preds[account_mask] > 0.5).sum()
        attack_detected = (attack_preds[account_mask] > 0.5).sum()
        adapted_detected = (adapted_preds[account_mask] > 0.5).sum()

        n_mules = int(y_true.sum())

        result = {
            "strategy": strategy,
            "baseline_auc": baseline_auc,
            "under_attack_auc": attack_auc,
            "adapted_auc": adapted_auc,
            "auc_drop": baseline_auc - attack_auc,
            "auc_recovery": adapted_auc - attack_auc,
            "baseline_detections": int(baseline_detected),
            "attack_detections": int(attack_detected),
            "adapted_detections": int(adapted_detected),
            "total_mules": n_mules,
            "retrain_history": retrain_history,
            "resilience_score": adapted_auc / max(baseline_auc, 0.01),
        }

        if verbose:
            print(f"  Baseline AUC:     {baseline_auc:.4f}")
            print(f"  Under Attack AUC: {attack_auc:.4f} (drop: {result['auc_drop']:.4f})")
            print(f"  After Adapt AUC:  {adapted_auc:.4f} (recovery: {result['auc_recovery']:.4f})")
            print(f"  Resilience Score: {result['resilience_score']:.3f}")

        # Restore original features
        self.graph_data.x = self.original_x.clone()

        return result

    def _apply_attack(self, strategy):
        """Apply attack perturbation to graph features."""
        data = copy.deepcopy(self.graph_data)
        account_mask = data.account_mask
        mule_mask = account_mask & (data.y > 0.5)
        mule_indices = mule_mask.nonzero().squeeze()

        if strategy == "velocity_reduction":
            # Mules reduce transaction velocity to look like normal users
            # Feature index 3 = tx_velocity
            noise = torch.randn(len(mule_indices)) * 0.3
            data.x[mule_indices, 3] = torch.clamp(
                data.x[mule_indices, 3] - 0.4 + noise, 0, 1
            )
            # Also reduce cyber risk signal slightly
            data.x[mule_indices, 0] *= 0.6

        elif strategy == "amount_splitting":
            # Mules split large transactions into smaller ones
            # Feature index 4 = avg_tx_amount
            data.x[mule_indices, 4] *= 0.3  # reduce avg amount
            data.x[mule_indices, 3] *= 1.5  # increase velocity (more small txns)
            data.x[mule_indices, 3] = torch.clamp(data.x[mule_indices, 3], 0, 1)

        elif strategy == "device_rotation":
            # Rotates devices to break fingerprint links
            # Feature index 5 = device_count, index 6 = vpn_freq
            data.x[mule_indices, 5] = torch.rand(len(mule_indices)) * 0.3
            data.x[mule_indices, 6] = 0.0  # stop using VPN
            # Reduce fingerprint entropy similarity
            data.x[mule_indices, 7] *= 0.3

        elif strategy == "behavioral_mimicry":
            # AI-generated normal behaviors across all features
            legit_mask = account_mask & (data.y < 0.5)
            legit_indices = legit_mask.nonzero().squeeze()

            # Copy average legit features with small noise
            legit_mean = data.x[legit_indices].mean(dim=0)
            legit_std = data.x[legit_indices].std(dim=0) * 0.5

            for idx in mule_indices:
                noise = torch.randn(data.x.shape[1]) * legit_std
                data.x[idx] = legit_mean + noise
                data.x[idx] = torch.clamp(data.x[idx], 0, 1)

        return data

    def _evaluate(self, model, data=None):
        """Evaluate model and return AUC + raw predictions."""
        if data is None:
            data = self.graph_data

        model.eval()
        with torch.no_grad():
            preds = model(data.x, data.edge_index).squeeze().numpy()
            y_acc = data.y[data.account_mask].numpy()
            pred_acc = preds[data.account_mask.numpy()]

            try:
                auc = roc_auc_score(y_acc, pred_acc)
            except:
                auc = 0.5

        return float(auc), preds

    def _retrain(self, model, data, epochs=RL_RETRAIN_EPOCHS):
        """Incremental retraining (online learning) on attacked data."""
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.5)
        model.train()

        history = {"loss": [], "auc": []}
        y_acc = data.y[data.account_mask]

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            pred_acc = pred[data.account_mask].squeeze()

            loss = F.binary_cross_entropy(pred_acc, y_acc)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                try:
                    auc = roc_auc_score(y_acc.numpy(), pred_acc.detach().numpy())
                except:
                    auc = 0.5

            history["loss"].append(loss.item())
            history["auc"].append(auc)

        return model, history

    def get_summary(self):
        """Return overall adversarial robustness summary."""
        if not self.simulation_results:
            return None

        avg_resilience = np.mean([r["resilience_score"] for r in self.simulation_results])
        worst_attack = min(self.simulation_results, key=lambda r: r["under_attack_auc"])
        best_recovery = max(self.simulation_results, key=lambda r: r["auc_recovery"])

        return {
            "avg_resilience": float(avg_resilience),
            "worst_attack": worst_attack["strategy"],
            "worst_attack_auc": worst_attack["under_attack_auc"],
            "best_recovery": best_recovery["strategy"],
            "best_recovery_gain": best_recovery["auc_recovery"],
            "all_results": self.simulation_results,
        }


if __name__ == "__main__":
    from data_generator import generate_all_data
    from graph_builder import build_graph
    from models.gat_model import CyberFinGAT, train_model

    data_dict = generate_all_data()
    graph = build_graph(data_dict)

    model = CyberFinGAT()
    model, _ = train_model(model, graph, epochs=50, verbose=False)

    sim = AdversarialSimulator(model, graph)
    results = sim.run_all_attacks(verbose=True)

    summary = sim.get_summary()
    print(f"\n{'='*60}")
    print(f"Overall Resilience: {summary['avg_resilience']:.3f}")
    print(f"Worst Attack: {summary['worst_attack']} (AUC: {summary['worst_attack_auc']:.3f})")
