"""
CyberFin Nexus — GAT Model
3-layer Graph Attention Network for mule risk prediction.
Multi-head attention propagates cyber risk across the heterogeneous graph.
Includes explainability via attention weight extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    GAT_INPUT_DIM, GAT_HIDDEN_DIM, GAT_OUTPUT_DIM,
    GAT_HEADS, GAT_DROPOUT, LEARNING_RATE, WEIGHT_DECAY,
    TRAIN_EPOCHS, TRAIN_PATIENCE
)


class CyberFinGAT(nn.Module):
    """
    3-layer GAT with multi-head attention for mule risk prediction.

    Architecture:
        Input (8) → GAT(8→32, 4 heads) → GAT(128→32, 4 heads) → GAT(128→1, 1 head or linear)
        With dropout, batch norm, and skip connections.
    """

    def __init__(self, in_dim=GAT_INPUT_DIM, hidden_dim=GAT_HIDDEN_DIM,
                 out_dim=GAT_OUTPUT_DIM, heads=GAT_HEADS, dropout=GAT_DROPOUT):
        super().__init__()

        self.dropout = dropout

        # Layer 1: input → hidden (multi-head)
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        # Layer 2: hidden → hidden (multi-head)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)

        # Layer 3: hidden → output (single head)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, out_dim),
        )

        # Store attention weights for explainability
        self._attention_weights = {}

    def forward(self, x, edge_index, return_attention=False):
        """Forward pass with optional attention weight capture."""

        # Layer 1
        if return_attention:
            x1, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
            self._attention_weights["layer1"] = attn1
        else:
            x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Layer 2
        if return_attention:
            x2, attn2 = self.conv2(x1, edge_index, return_attention_weights=True)
            self._attention_weights["layer2"] = attn2
        else:
            x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Layer 3
        if return_attention:
            x3, attn3 = self.conv3(x2, edge_index, return_attention_weights=True)
            self._attention_weights["layer3"] = attn3
        else:
            x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.elu(x3)

        # Classifier
        out = self.classifier(x3)
        out = torch.sigmoid(out)

        return out

    def get_attention_weights(self):
        """Return stored attention weights from last forward pass."""
        return self._attention_weights

    def get_node_attention_scores(self, edge_index):
        """Compute per-node aggregated attention scores for explainability."""
        if "layer3" not in self._attention_weights:
            return None

        attn_edge_index, attn_weights = self._attention_weights["layer3"]
        n_nodes = edge_index.max().item() + 1

        # Average incoming attention per node
        node_scores = torch.zeros(n_nodes)
        counts = torch.zeros(n_nodes)

        target_nodes = attn_edge_index[1]
        for i in range(len(target_nodes)):
            node_idx = target_nodes[i].item()
            node_scores[node_idx] += attn_weights[i].mean().item()
            counts[node_idx] += 1

        # Normalize
        mask = counts > 0
        node_scores[mask] /= counts[mask]

        return node_scores.numpy()


def train_model(model, data, epochs=TRAIN_EPOCHS, lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY, patience=TRAIN_PATIENCE, verbose=True):
    """
    Train GAT model with early stopping and class-weight balancing.

    Returns:
        model, history (dict with loss, auc per epoch)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class weights for imbalanced data
    account_mask = data.account_mask
    y_accounts = data.y[account_mask]
    n_pos = y_accounts.sum().item()
    n_neg = (1 - y_accounts).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    history = {"loss": [], "auc": []}
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward  (use raw logits for BCEWithLogitsLoss)
        raw_out = model.classifier[:-1](  # skip sigmoid in classifier
            _get_embeddings(model, data.x, data.edge_index)
        )
        # Actually, use forward then undo sigmoid via logit
        pred = model(data.x, data.edge_index)
        pred_accounts = pred[account_mask].squeeze()

        # BCE loss (sigmoid already applied in forward)
        loss = F.binary_cross_entropy(pred_accounts, y_accounts)

        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            pred_np = pred_accounts.detach().numpy()
            y_np = y_accounts.numpy()
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_np, pred_np)
            except:
                auc = 0.0

        history["loss"].append(loss.item())
        history["auc"].append(auc)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | AUC: {auc:.4f}")

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict(model, data):
    """Run inference and return per-node risk scores."""
    model.eval()
    with torch.no_grad():
        scores = model(data.x, data.edge_index, return_attention=True)
    return scores.squeeze().numpy()


def _get_embeddings(model, x, edge_index):
    """Helper to get embeddings before classifier (for attention analysis)."""
    x1 = model.conv1(x, edge_index)
    x1 = model.bn1(x1)
    x1 = F.elu(x1)

    x2 = model.conv2(x1, edge_index)
    x2 = model.bn2(x2)
    x2 = F.elu(x2)

    x3 = model.conv3(x2, edge_index)
    x3 = model.bn3(x3)
    x3 = F.elu(x3)

    return x3


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_generator import generate_all_data
    from graph_builder import build_graph

    data_dict = generate_all_data()
    graph = build_graph(data_dict)

    model = CyberFinGAT()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model, history = train_model(model, graph, epochs=50, verbose=True)
    scores = predict(model, graph)

    account_mask = graph.account_mask.numpy()
    print(f"\nFinal AUC: {history['auc'][-1]:.4f}")
    print(f"Risk scores (accounts): min={scores[account_mask].min():.3f}, "
          f"max={scores[account_mask].max():.3f}, "
          f"mean={scores[account_mask].mean():.3f}")
