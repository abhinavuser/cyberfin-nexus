"""Quick verification of all new modules."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_all_data
from graph_builder import build_graph
from models.gat_model import CyberFinGAT, train_model, predict
from models.ensemble import EnsembleScorer
from utils.graph_analytics import detect_communities, get_community_summary
from utils.active_learning import get_review_queue

print("1. Generating data...")
data = generate_all_data()

print("2. Building graph...")
g = build_graph(data)
print(f"   Features: {g.x.shape}")

print("3. Training GAT (30 epochs)...")
model = CyberFinGAT()
model, h = train_model(model, g, epochs=30, verbose=False)
scores = predict(model, g)
am = g.account_mask.numpy()
print(f"   AUC: {h['auc'][-1]:.4f}")

print("4. Ensemble scoring...")
e = EnsembleScorer()
r = e.compute_ensemble(scores, g.x.numpy(), am)
print(f"   GAT mean: {r['gat_scores'][am].mean():.3f}")
print(f"   IF mean:  {r['if_scores'][am].mean():.3f}")
print(f"   Rules mean: {r['rule_scores'][am].mean():.3f}")
print(f"   Consensus >= 2: {(r['consensus'][am] >= 2).sum()}")

print("5. Community detection...")
c = detect_communities(g, scores)
cs = get_community_summary(c)
print(f"   Found {cs['total']} communities, {cs['suspicious']} suspicious")
if c:
    print(f"   Top community: #{c[0]['community_id']} suspicion={c[0]['suspicion_score']:.3f}")

print("6. Active learning queue...")
q = get_review_queue(scores, g.x.numpy(), am, g.node_ids, g, max_items=5)
print(f"   Review queue: {len(q)} items")
for item in q[:3]:
    print(f"   {item['account_id']}: score={item['score']:.3f} priority={item['priority']:.3f}")

print("\nALL MODULES VERIFIED OK")
