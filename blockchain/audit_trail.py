"""
CyberFin Nexus — Blockchain Audit Trail
Mock hash-chain providing immutable audit logs for regulatory compliance.
Each transaction/decision gets a SHA-256 hash linking to the previous entry.
"""

import hashlib
import json
import time
from datetime import datetime
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import HASH_ALGORITHM


class AuditBlock:
    """Single block in the audit chain."""

    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        """Compute SHA-256 hash of the block contents."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }, sort_keys=True, default=str)

        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }


class AuditChain:
    """
    Immutable blockchain-like audit trail for CyberFin Nexus.

    Records model decisions, risk assessments, and alerts with
    cryptographic linking for tamper-evident forensics.
    """

    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the first block in the chain."""
        genesis = AuditBlock(
            index=0,
            timestamp=datetime.now().isoformat(),
            data={
                "type": "GENESIS",
                "message": "CyberFin Nexus Audit Trail Initialized",
                "version": "1.0.0",
            },
            previous_hash="0" * 64,
        )
        self.chain.append(genesis)

    def add_transaction_record(self, txn_id, from_account, to_account,
                                amount, risk_score, model_decision):
        """Record a transaction assessment in the audit trail."""
        data = {
            "type": "TRANSACTION_ASSESSMENT",
            "txn_id": txn_id,
            "from_account": from_account,
            "to_account": to_account,
            "amount": amount,
            "risk_score": round(float(risk_score), 4),
            "model_decision": model_decision,  # "PASS", "FLAG", "BLOCK"
            "model_version": "GAT-v1",
        }
        self._add_block(data)

    def add_alert(self, alert_type, accounts, confidence, risk_amount, details=""):
        """Record an alert in the audit trail."""
        data = {
            "type": "ALERT",
            "alert_type": alert_type,
            "accounts": accounts if isinstance(accounts, list) else [accounts],
            "confidence": round(float(confidence), 4),
            "estimated_risk": float(risk_amount),
            "details": details,
        }
        self._add_block(data)

    def add_model_event(self, event_type, metrics):
        """Record model training/retraining event."""
        data = {
            "type": "MODEL_EVENT",
            "event_type": event_type,  # "TRAIN", "FL_ROUND", "RL_ADAPT"
            "metrics": {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                       for k, v in metrics.items()},
        }
        self._add_block(data)

    def add_ring_detection(self, ring_id, accounts, confidence, total_risk):
        """Record a mule ring detection."""
        data = {
            "type": "RING_DETECTION",
            "ring_id": ring_id,
            "accounts": accounts,
            "confidence": round(float(confidence), 4),
            "estimated_total_risk": float(total_risk),
            "detection_method": "GNN-GAT Propagation",
        }
        self._add_block(data)

    def _add_block(self, data):
        """Add a new block to the chain."""
        prev_block = self.chain[-1]
        new_block = AuditBlock(
            index=len(self.chain),
            timestamp=datetime.now().isoformat(),
            data=data,
            previous_hash=prev_block.hash,
        )
        self.chain.append(new_block)

    def verify_integrity(self):
        """Verify the entire chain's integrity (no tampering)."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check hash validity
            if current.hash != current.compute_hash():
                return False, f"Block {i}: hash mismatch (tampered)"

            # Check chain linkage
            if current.previous_hash != previous.hash:
                return False, f"Block {i}: broken chain link"

        return True, "Chain integrity verified ✓"

    def get_chain_summary(self):
        """Return summary statistics of the audit trail."""
        types = {}
        for block in self.chain:
            t = block.data.get("type", "UNKNOWN")
            types[t] = types.get(t, 0) + 1

        return {
            "total_blocks": len(self.chain),
            "block_types": types,
            "first_timestamp": self.chain[0].timestamp,
            "last_timestamp": self.chain[-1].timestamp,
            "chain_hash": self.chain[-1].hash,
        }

    def get_alerts(self):
        """Return all alert blocks."""
        return [b.to_dict() for b in self.chain if b.data.get("type") == "ALERT"]

    def get_ring_detections(self):
        """Return all ring detection blocks."""
        return [b.to_dict() for b in self.chain if b.data.get("type") == "RING_DETECTION"]

    def export_json(self):
        """Export full chain as JSON-serializable list."""
        return [b.to_dict() for b in self.chain]

    def __len__(self):
        return len(self.chain)

    def __repr__(self):
        return f"AuditChain(blocks={len(self.chain)})"


def build_audit_trail(data_dict, risk_scores, node_ids, account_mask, mule_rings=None):
    """
    Build a complete audit trail from model predictions.

    Args:
        data_dict: Generated data dictionary
        risk_scores: Model predictions (numpy array)
        node_ids: List of node IDs
        account_mask: Boolean mask for account nodes
        mule_rings: DataFrame of mule ring assignments

    Returns:
        AuditChain object
    """
    chain = AuditChain()
    transactions = data_dict["transactions"]

    # Record model training event
    chain.add_model_event("TRAIN", {"model": "GAT-v1", "status": "completed"})

    # Record transaction assessments
    for _, txn in transactions.head(100).iterrows():  # limit for demo
        # Find risk score for from_account
        from_acc = txn["from_account"]
        risk = 0.0
        for i, nid in enumerate(node_ids):
            if nid == from_acc and account_mask[i]:
                risk = float(risk_scores[i])
                break

        decision = "BLOCK" if risk > 0.85 else ("FLAG" if risk > 0.5 else "PASS")
        chain.add_transaction_record(
            txn_id=txn["txn_id"],
            from_account=from_acc,
            to_account=txn["to_account"],
            amount=float(txn["amount"]),
            risk_score=risk,
            model_decision=decision,
        )

    # Record alerts for high-risk accounts
    for i, nid in enumerate(node_ids):
        if account_mask[i] and risk_scores[i] > 0.7:
            chain.add_alert(
                alert_type="HIGH_RISK_ACCOUNT",
                accounts=nid,
                confidence=float(risk_scores[i]),
                risk_amount=float(risk_scores[i]) * 50000,
                details=f"GAT attention indicates {nid} linked to suspicious cyber activity",
            )

    # Record ring detections
    if mule_rings is not None and len(mule_rings) > 0:
        for ring_id in mule_rings.ring_id.unique():
            members = mule_rings[mule_rings.ring_id == ring_id].account_id.tolist()
            ring_risks = []
            for m in members:
                for i, nid in enumerate(node_ids):
                    if nid == m:
                        ring_risks.append(float(risk_scores[i]))
                        break

            avg_conf = np.mean(ring_risks) if ring_risks else 0.5
            total_risk = avg_conf * len(members) * 50000

            chain.add_ring_detection(
                ring_id=ring_id,
                accounts=members,
                confidence=avg_conf,
                total_risk=total_risk,
            )

    return chain


if __name__ == "__main__":
    # Quick test
    chain = AuditChain()
    chain.add_transaction_record("TXN_001", "ACC_1_0001", "EXT_100", 5000, 0.92, "BLOCK")
    chain.add_alert("HIGH_RISK", ["ACC_1_0001"], 0.95, 50000, "Linked to phishing")
    chain.add_ring_detection("RING_0", ["ACC_1_0001", "ACC_1_0002"], 0.88, 200000)

    valid, msg = chain.verify_integrity()
    print(f"Chain valid: {valid} — {msg}")
    print(f"Summary: {chain.get_chain_summary()}")
