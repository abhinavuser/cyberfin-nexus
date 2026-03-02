"""
Preprocess Base.csv and output cleaned CSV to GNN folder.

Steps:
1. Drop constant/waste columns (device_fraud_count)
2. Handle missing values (-1 encoding) with median fill + indicator flags
3. Merge rare categories into 'OTHER'
4. Scale numerical features (StandardScaler)
5. Output: GNN/preprocessed.csv
"""
import csv
import math
import random
import os
from collections import defaultdict

INPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'Base.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'preprocessed.csv')

# Columns to drop entirely
DROP_COLS = ['device_fraud_count']

# Columns with -1 as missing value
MISSING_COLS = [
    'prev_address_months_count',
    'bank_months_count',
    'current_address_months_count',
    'session_length_in_minutes',
    'credit_risk_score',
    'device_distinct_emails_8w',
]

# High-missing columns that get an extra is_missing flag
HIGH_MISSING_COLS = ['prev_address_months_count', 'bank_months_count']

# Rare category merges
RARE_MERGES = {
    'payment_type': ['AE'],
    'employment_status': ['CG'],
    'housing_status': ['BG', 'BF'],
}

# Categorical columns (will be label-encoded for CSV output)
CAT_COLS = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']

# ========================
# PASS 1: Compute medians and scaling stats
# ========================
print("[PASS 1/3] Computing medians and scaling statistics...")

# Collect non-missing values for median computation (reservoir sample)
MEDIAN_SAMPLE = 100000
med_reservoirs = {col: [] for col in MISSING_COLS}
random.seed(42)

# Collect all values for mean/std (running computation)
col_sum = defaultdict(float)
col_sum_sq = defaultdict(float)
col_count = defaultdict(int)

total = 0
headers = None

with open(INPUT_FILE, 'r') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    for i, row in enumerate(reader):
        total += 1
        
        # Reservoir sample for medians (only non-missing values)
        for col in MISSING_COLS:
            val = row[col]
            if val != '-1' and val != '-1.0':
                fval = float(val)
                if len(med_reservoirs[col]) < MEDIAN_SAMPLE:
                    med_reservoirs[col].append(fval)
                else:
                    j = random.randint(0, i)
                    if j < MEDIAN_SAMPLE:
                        med_reservoirs[col][j] = fval
        
        # Running stats for all numerical columns
        for col in headers:
            if col in DROP_COLS or col == 'fraud_bool' or col in CAT_COLS:
                continue
            try:
                fval = float(row[col])
                # Don't include -1 missing values in stats
                if col in MISSING_COLS and (row[col] == '-1' or row[col] == '-1.0'):
                    continue
                col_sum[col] += fval
                col_sum_sq[col] += fval * fval
                col_count[col] += 1
            except:
                pass
        
        if (i + 1) % 200000 == 0:
            print(f"  ... {i+1:,} rows")

print(f"  Total rows: {total:,}")

# Compute medians
medians = {}
for col in MISSING_COLS:
    vals = sorted(med_reservoirs[col])
    if vals:
        medians[col] = vals[len(vals) // 2]
    else:
        medians[col] = 0
    print(f"  Median({col}) = {medians[col]:.2f}")

# Compute mean and std for scaling
num_cols_for_scaling = []
col_mean = {}
col_std = {}
for col in headers:
    if col in DROP_COLS or col == 'fraud_bool' or col in CAT_COLS:
        continue
    if col_count.get(col, 0) > 0:
        num_cols_for_scaling.append(col)
        mean = col_sum[col] / col_count[col]
        variance = (col_sum_sq[col] / col_count[col]) - (mean * mean)
        std = math.sqrt(max(variance, 1e-10))
        col_mean[col] = mean
        col_std[col] = std

print(f"  Numerical features to scale: {len(num_cols_for_scaling)}")

# ========================
# PASS 2: Build category encoding maps
# ========================
print("\n[PASS 2/3] Building category encodings...")

# Build label encodings for categorical columns
cat_values = {col: set() for col in CAT_COLS}
with open(INPUT_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for col in CAT_COLS:
            val = row[col]
            # Apply rare merges
            if col in RARE_MERGES and val in RARE_MERGES[col]:
                val = 'OTHER'
            cat_values[col].add(val)

cat_encoding = {}
for col in CAT_COLS:
    sorted_vals = sorted(cat_values[col])
    cat_encoding[col] = {v: i for i, v in enumerate(sorted_vals)}
    print(f"  {col}: {cat_encoding[col]}")

# ========================
# PASS 3: Write preprocessed CSV
# ========================
print(f"\n[PASS 3/3] Writing preprocessed CSV to {OUTPUT_FILE}...")

# Build output header
out_headers = ['fraud_bool']

# Add scaled numerical columns
for col in num_cols_for_scaling:
    out_headers.append(col)

# Add missing indicator flags
for col in HIGH_MISSING_COLS:
    out_headers.append(f'{col}_is_missing')

# Add encoded categorical columns
for col in CAT_COLS:
    out_headers.append(col)

rows_written = 0
with open(INPUT_FILE, 'r') as fin, open(OUTPUT_FILE, 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(out_headers)
    
    for i, row in enumerate(reader):
        out_row = []
        
        # Target
        out_row.append(int(row['fraud_bool']))
        
        # Numerical features (fill missing + scale)
        for col in num_cols_for_scaling:
            val = row[col]
            if col in MISSING_COLS and (val == '-1' or val == '-1.0'):
                fval = medians[col]
            else:
                fval = float(val)
            # StandardScale
            scaled = (fval - col_mean[col]) / col_std[col]
            out_row.append(round(scaled, 6))
        
        # Missing indicator flags
        for col in HIGH_MISSING_COLS:
            is_missing = 1 if (row[col] == '-1' or row[col] == '-1.0') else 0
            out_row.append(is_missing)
        
        # Categorical features (label encoded)
        for col in CAT_COLS:
            val = row[col]
            if col in RARE_MERGES and val in RARE_MERGES[col]:
                val = 'OTHER'
            out_row.append(cat_encoding[col][val])
        
        writer.writerow(out_row)
        rows_written += 1
        
        if (rows_written) % 200000 == 0:
            print(f"  ... {rows_written:,} rows written")

print(f"\n  DONE! Wrote {rows_written:,} rows x {len(out_headers)} columns")
print(f"  Output: {OUTPUT_FILE}")
print(f"\n  Columns ({len(out_headers)}):")
for h in out_headers:
    print(f"    - {h}")
