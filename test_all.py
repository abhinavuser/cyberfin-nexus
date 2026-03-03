"""
CyberFin Nexus — Full Integration Test
Tests all modules end-to-end.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("CyberFin Nexus — Full Integration Test")
print("=" * 60)

# 1. Data Generation
print("\n[1/6] Testing Data Generator...")
from data_generator import generate_all_data
data_dict = generate_all_data(seed=42)
for name, df in data_dict.items():
    print(f"  {name}: {df.shape}")
assert len(data_dict["accounts"]) == 250, "Expected 250 accounts"
assert data_dict["accounts"].is_mule.sum() > 0, "Need mule accounts"
print("  ✅ Data generation OK")

# 2. Graph Construction
print("\n[2/6] Testing Graph Builder...")
from graph_builder import build_graph, partition_by_bank
graph = build_graph(data_dict)
print(f"  Nodes: {graph.num_nodes}, Edges: {graph.edge_index.shape[1]}")
print(f"  Features shape: {graph.x.shape}")
print(f"  Account mask: {graph.account_mask.sum().item()} accounts")
print(f"  Mule labels: {graph.y[graph.account_mask].sum().item():.0f} mules")
assert graph.x.shape[1] == 12, "Expected 12 features (8 base + 4 temporal)"
assert graph.edge_index.shape[0] == 2, "Edge index should be 2xE"
print("  ✅ Graph construction OK")

partitions = partition_by_bank(data_dict, graph)
print(f"  Banks partitioned: {list(partitions.keys())}")
print("  ✅ Partitioning OK")

# 3. GAT Model
print("\n[3/6] Testing GAT Model...")
from models.gat_model import CyberFinGAT, train_model, predict
model = CyberFinGAT()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params:,}")

model, history = train_model(model, graph, epochs=100, verbose=False)
scores = predict(model, graph)

from utils.metrics import compute_all_metrics, risk_category
am = graph.account_mask.numpy()
y_true = graph.y[graph.account_mask].numpy()
y_scores = scores[am]
metrics = compute_all_metrics(y_true, y_scores)

print(f"  AUC: {metrics['auc']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall: {metrics['recall']:.4f}")
print(f"  F1: {metrics['f1']:.4f}")
print(f"  Scores range: {y_scores.min():.4f} - {y_scores.max():.4f}")
print(f"  Mule avg: {y_scores[y_true > 0.5].mean():.4f}")
print(f"  Legit avg: {y_scores[y_true < 0.5].mean():.4f}")
print(f"  Training epochs: {len(history['auc'])}")
print("  ✅ GAT model OK")

# 4. Federated Learning
print("\n[4/6] Testing Federated Learning...")
from federated.fl_engine import FederatedLearningEngine
fl_engine = FederatedLearningEngine()
fl_model, fl_metrics = fl_engine.train_federated(
    graph, partitions, n_rounds=5, local_epochs=3, verbose=False
)
final_auc = fl_metrics[-1]["global_auc"]
print(f"  FL Global AUC after 5 rounds: {final_auc:.4f}")
privacy = fl_engine.get_privacy_report()
print(f"  Privacy: epsilon={privacy['epsilon']}, GDPR={privacy['compliant_gdpr']}")
print("  ✅ Federated Learning OK")

# 5. Adversarial Simulation
print("\n[5/6] Testing Adversarial Simulation...")
from rl.adversarial_sim import AdversarialSimulator
sim = AdversarialSimulator(model, graph)
result = sim.simulate_attack("velocity_reduction", verbose=False)
print(f"  Baseline AUC: {result['baseline_auc']:.4f}")
print(f"  Under Attack: {result['under_attack_auc']:.4f}")
print(f"  After Adapt:  {result['adapted_auc']:.4f}")
print(f"  Resilience:   {result['resilience_score']:.3f}")
print("  ✅ Adversarial simulation OK")

# 6. Blockchain Audit Trail
print("\n[6/6] Testing Blockchain Audit Trail...")
from blockchain.audit_trail import AuditChain, build_audit_trail
chain = build_audit_trail(
    data_dict, scores, graph.node_ids,
    graph.account_mask.numpy(), data_dict["mule_rings"]
)
valid, msg = chain.verify_integrity()
summary = chain.get_chain_summary()
print(f"  Chain blocks: {summary['total_blocks']}")
print(f"  Block types: {summary['block_types']}")
print(f"  Integrity: {msg}")
print(f"  Alerts: {len(chain.get_alerts())}")
print(f"  Ring detections: {len(chain.get_ring_detections())}")
assert valid, "Chain integrity check failed!"
print("  ✅ Blockchain audit trail OK")

# Summary
print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED — CyberFin Nexus is ready!")
print("=" * 60)
print(f"\nKey Metrics:")
print(f"  GAT AUC: {metrics['auc']:.4f}")
print(f"  FL  AUC: {final_auc:.4f}")
print(f"  RL Resilience: {result['resilience_score']:.3f}")
print(f"  Audit Blocks: {summary['total_blocks']}")
print(f"\nTo launch dashboard: streamlit run app.py")
