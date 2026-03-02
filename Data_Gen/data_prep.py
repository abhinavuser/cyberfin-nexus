"""
Phase 1: Real Data Preparation for WGAN Training
Loads Base.csv, normalizes numerical features, encodes categoricals,
and creates a PyTorch-ready dataset for the WGAN-GP.

Uses streaming approach + csv module (no pandas dependency issues).
Saves fitted scalers for inverse-transform during generation.
"""
import csv
import os
import json
import math
import random
from collections import defaultdict

import torch
from torch.utils.data import TensorDataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CSV = os.path.join(os.path.dirname(SCRIPT_DIR), 'Base.csv')
TRANSFORMS_FILE = os.path.join(SCRIPT_DIR, 'transforms.json')
TENSOR_FILE = os.path.join(SCRIPT_DIR, 'training_data.pt')

# Target column
TARGET = 'fraud_bool'

# Columns to drop (constant / useless from analysis)
DROP_COLS = {'device_fraud_count'}

# Categorical columns (will be one-hot encoded)
CATEGORICAL_COLS = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']

# Max rows to use for training (full 1M is too slow on CPU)
MAX_ROWS = 100_000


def prepare_data(max_rows=MAX_ROWS, batch_size=512, seed=42):
    """
    Load Base.csv, normalize, encode, and return DataLoader + transforms.
    
    Returns:
        dataloader: PyTorch DataLoader with (features, labels)
        transforms: dict with scaler params and encoding maps
        feature_info: dict with column names and dimensions
    """
    random.seed(seed)
    print("=" * 60)
    print("  PHASE 1: Data Preparation for WGAN")
    print("=" * 60)

    # =========================================================
    # PASS 1: Determine column types and collect unique categoricals
    # =========================================================
    print(f"\n[1/4] Scanning {os.path.basename(BASE_CSV)}...")
    
    headers = None
    numerical_cols = []
    cat_unique = {col: set() for col in CATEGORICAL_COLS}
    
    with open(BASE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        # Identify numerical columns
        for col in headers:
            if col == TARGET or col in DROP_COLS or col in CATEGORICAL_COLS:
                continue
            numerical_cols.append(col)
        
        # Collect unique categorical values
        for i, row in enumerate(reader):
            for col in CATEGORICAL_COLS:
                cat_unique[col].add(row[col])
            if i >= max_rows:
                break
    
    print(f"  Numerical columns: {len(numerical_cols)}")
    print(f"  Categorical columns: {len(CATEGORICAL_COLS)}")
    for col in CATEGORICAL_COLS:
        print(f"    {col}: {sorted(cat_unique[col])}")

    # Build one-hot encoding maps
    cat_encoding = {}
    cat_dims = {}
    for col in CATEGORICAL_COLS:
        sorted_vals = sorted(cat_unique[col])
        cat_encoding[col] = {v: i for i, v in enumerate(sorted_vals)}
        cat_dims[col] = len(sorted_vals)
    
    total_cat_dim = sum(cat_dims.values())
    total_num_dim = len(numerical_cols)
    total_feature_dim = total_num_dim + total_cat_dim
    print(f"  Total feature dim: {total_num_dim} numerical + {total_cat_dim} one-hot = {total_feature_dim}")

    # =========================================================
    # PASS 2: Compute min/max for numerical columns (for scaling)
    # =========================================================
    print(f"\n[2/4] Computing scaling statistics (on {max_rows:,} rows)...")
    
    col_min = {col: float('inf') for col in numerical_cols}
    col_max = {col: float('-inf') for col in numerical_cols}
    
    with open(BASE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            for col in numerical_cols:
                try:
                    val = float(row[col])
                    if val < col_min[col]:
                        col_min[col] = val
                    if val > col_max[col]:
                        col_max[col] = val
                except:
                    pass
    
    # Ensure no zero-range columns
    for col in numerical_cols:
        if col_max[col] - col_min[col] < 1e-8:
            col_max[col] = col_min[col] + 1.0

    # =========================================================
    # PASS 3: Build the training tensor
    # =========================================================
    print(f"\n[3/4] Building training tensors...")
    
    features_list = []
    labels_list = []
    n_fraud = 0
    n_total = 0
    
    with open(BASE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            
            feat = []
            
            # Numerical features → MinMax scaled to [0, 1]
            for col in numerical_cols:
                try:
                    val = float(row[col])
                except:
                    val = 0.0
                scaled = (val - col_min[col]) / (col_max[col] - col_min[col])
                scaled = max(0.0, min(1.0, scaled))  # clamp
                feat.append(scaled)
            
            # Categorical features → one-hot
            for col in CATEGORICAL_COLS:
                one_hot = [0.0] * cat_dims[col]
                val = row[col]
                if val in cat_encoding[col]:
                    one_hot[cat_encoding[col][val]] = 1.0
                feat.append(one_hot)
            
            # Flatten (numerical are already flat, one-hots need flattening)
            flat_feat = []
            for f_item in feat:
                if isinstance(f_item, list):
                    flat_feat.extend(f_item)
                else:
                    flat_feat.append(f_item)
            
            label = float(row[TARGET])
            features_list.append(flat_feat)
            labels_list.append(label)
            
            if label == 1.0:
                n_fraud += 1
            n_total += 1
    
    features_tensor = torch.tensor(features_list, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)
    
    print(f"  Loaded: {n_total:,} rows")
    print(f"  Fraud: {n_fraud:,} ({n_fraud/n_total:.2%})")
    print(f"  Feature tensor: {features_tensor.shape}")
    print(f"  Labels tensor: {labels_tensor.shape}")

    # =========================================================
    # Save transforms for inverse-transform during generation
    # =========================================================
    transforms = {
        'numerical_cols': numerical_cols,
        'categorical_cols': CATEGORICAL_COLS,
        'cat_encoding': cat_encoding,
        'cat_dims': cat_dims,
        'col_min': col_min,
        'col_max': col_max,
        'total_num_dim': total_num_dim,
        'total_cat_dim': total_cat_dim,
        'total_feature_dim': total_feature_dim,
        'n_samples': n_total,
        'fraud_ratio': n_fraud / n_total,
    }
    
    with open(TRANSFORMS_FILE, 'w') as f:
        json.dump(transforms, f, indent=2)
    print(f"\n  Saved transforms → {os.path.basename(TRANSFORMS_FILE)}")
    
    # Save tensors
    torch.save({
        'features': features_tensor,
        'labels': labels_tensor,
    }, TENSOR_FILE)
    print(f"  Saved tensors → {os.path.basename(TENSOR_FILE)} ({features_tensor.shape})")
    
    # =========================================================
    # Create DataLoader
    # =========================================================
    print(f"\n[4/4] Creating DataLoader (batch_size={batch_size})...")
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"  Batches per epoch: {len(dataloader)}")
    
    print(f"\n{'=' * 60}")
    print("  PHASE 1 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Feature dim: {total_feature_dim}")
    print(f"  Numerical: {total_num_dim} | Categorical (one-hot): {total_cat_dim}")
    print(f"  Samples: {n_total:,} | Fraud rate: {n_fraud/n_total:.2%}")
    
    return dataloader, transforms


def load_prepared_data(batch_size=512):
    """Load previously prepared data from saved files."""
    with open(TRANSFORMS_FILE, 'r') as f:
        transforms = json.load(f)
    
    saved = torch.load(TENSOR_FILE, weights_only=True)
    dataset = TensorDataset(saved['features'], saved['labels'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataloader, transforms


if __name__ == '__main__':
    dataloader, transforms = prepare_data()
    
    # Quick sanity check
    batch_features, batch_labels = next(iter(dataloader))
    print(f"\n  Sample batch: features={batch_features.shape}, labels={batch_labels.shape}")
    print(f"  Feature range: [{batch_features.min():.4f}, {batch_features.max():.4f}]")
    print(f"  Fraud in batch: {batch_labels.sum().item():.0f}/{batch_labels.shape[0]}")
    print(f"\n  Ready for WGAN training!")
