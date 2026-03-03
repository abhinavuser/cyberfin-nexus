import os
import pandas as pd
from graph_builder import build_graph
from models.gat_model import CyberFinGAT, train_model, predict
from utils.metrics import compute_all_metrics

def evaluate_csv_data(csv_directory):
    """
    Load data from CSV files and evaluate the CyberFinGAT model on it.
    Uses only the first 5000 entries and applies a 70/30 train/test split.
    """
    print(f"Loading first 5000 entries from CSV files in {csv_directory}...")
    try:
        data_dict = {
            "accounts": pd.read_csv(os.path.join(csv_directory, "accounts.csv"), nrows=5000),
            "devices": pd.read_csv(os.path.join(csv_directory, "devices.csv"), nrows=5000),
            "acc_dev_map": pd.read_csv(os.path.join(csv_directory, "acc_dev_map.csv"), nrows=5000),
            "cyber_events": pd.read_csv(os.path.join(csv_directory, "cyber_events.csv"), nrows=5000),
            "transactions": pd.read_csv(os.path.join(csv_directory, "transactions.csv"), nrows=5000),
            "mule_rings": pd.read_csv(os.path.join(csv_directory, "mule_rings.csv"), nrows=5000)
        }
        
        # Ensure timestamp columns are parsed as datetime
        data_dict["cyber_events"]["timestamp"] = pd.to_datetime(data_dict["cyber_events"]["timestamp"])
        data_dict["transactions"]["timestamp"] = pd.to_datetime(data_dict["transactions"]["timestamp"])
        
        print(f"✅ Data loaded successfully. Building graph...")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find one of the required CSV files in '{csv_directory}'")
        print(f"Details: {e}")
        return
    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return
        
    # Build Graph
    graph = build_graph(data_dict)
    n_total_accounts = graph.account_mask.sum().item()
    print(f"Nodes: {graph.x.shape[0]}, Edges: {graph.edge_index.shape[1]}")
    print(f"Total valid accounts in subset: {n_total_accounts}")
    
    import numpy as np
    import torch
    
    # ─── 70/30 Train/Test Split ───
    account_indices = np.where(graph.account_mask.numpy())[0]
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(account_indices)
    
    split_idx = int(len(account_indices) * 0.70)
    train_idx = account_indices[:split_idx]
    test_idx = account_indices[split_idx:]
    
    train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    
    test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    
    print(f"Splitting data: {len(train_idx)} accounts (70%) for training, {len(test_idx)} accounts (30%) for testing.")
    
    # Initialize Model
    model = CyberFinGAT()
    
    print("Training GAT model strictly on the 70% training accounts...")
    # Temporarily override the graph's account mask so it ONLY trains on the 70%
    graph.account_mask = train_mask
    model, history = train_model(model, graph, epochs=100, verbose=False)
    
    print("Making predictions using the 30% holdout test set...")
    scores = predict(model, graph)
    
    # Calculate Accuracy Metrics strictly on the 30% test mask
    test_mask_np = test_mask.numpy()
    y_true_test = graph.y[test_mask_np].numpy()
    y_scores_test = scores[test_mask_np]
    
    metrics = compute_all_metrics(y_true_test, y_scores_test)
    
    print("\n" + "="*45)
    print("📊 30% HOLDOUT TEST DATA PERFORMANCE (max 5000 rows)")
    print("="*45)
    print(f"AUC Score (Discrimination): {metrics['auc']:.4f}")
    print(f"Precision (Correct Guesses): {metrics['precision']:.2%}")
    print(f"Recall (Mules Caught):       {metrics['recall']:.2%}")
    print(f"F1 Score (Balanced):         {metrics['f1']:.4f}")
    print(f"Raw Accuracy (Overall):      {metrics['accuracy']:.2%}")
    print("="*45)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CyberFin model on CSV data")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing the 6 CSV files")
    args = parser.parse_args()
    
    evaluate_csv_data(args.dir)
