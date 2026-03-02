"""
Phase 1: Stratified Data Splitting
Splits preprocessed.csv into 3 bank datasets with equal fraud ratios.
"""
import csv
import random
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, 'preprocessed.csv')
BANK_FILES = {
    'Bank_A': os.path.join(SCRIPT_DIR, 'bank_A.csv'),
    'Bank_B': os.path.join(SCRIPT_DIR, 'bank_B.csv'),
    'Bank_C': os.path.join(SCRIPT_DIR, 'bank_C.csv'),
}

NUM_BANKS = 3
SEED = 42


def split_data():
    """
    Stratified split of preprocessed.csv into 3 bank CSVs.
    Fraud and non-fraud rows are shuffled independently,
    then distributed round-robin to ensure equal fraud ratios.
    """
    print("=" * 60)
    print("  PHASE 1: Stratified Data Split (3 Banks)")
    print("=" * 60)

    # --- Read all rows, separate by class ---
    print("\n[1/3] Reading preprocessed.csv...")
    fraud_rows = []
    legit_rows = []
    header = None

    with open(INPUT_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[0] == '1':  # fraud_bool is first column
                fraud_rows.append(row)
            else:
                legit_rows.append(row)

    total = len(fraud_rows) + len(legit_rows)
    print(f"  Total rows: {total:,}")
    print(f"  Fraud:  {len(fraud_rows):,} ({len(fraud_rows)/total:.2%})")
    print(f"  Legit:  {len(legit_rows):,} ({len(legit_rows)/total:.2%})")

    # --- Shuffle both classes ---
    print("\n[2/3] Shuffling and splitting...")
    random.seed(SEED)
    random.shuffle(fraud_rows)
    random.shuffle(legit_rows)

    # --- Distribute round-robin to 3 banks ---
    bank_data = {name: [] for name in BANK_FILES}
    bank_names = list(BANK_FILES.keys())

    for i, row in enumerate(fraud_rows):
        bank_data[bank_names[i % NUM_BANKS]].append(row)

    for i, row in enumerate(legit_rows):
        bank_data[bank_names[i % NUM_BANKS]].append(row)

    # Shuffle within each bank (so fraud isn't clustered at top)
    for name in bank_names:
        random.shuffle(bank_data[name])

    # --- Write bank CSVs ---
    print("\n[3/3] Writing bank CSVs...")
    for name, filepath in BANK_FILES.items():
        rows = bank_data[name]
        n_fraud = sum(1 for r in rows if r[0] == '1')
        n_total = len(rows)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"  {name}: {n_total:>8,} rows | fraud: {n_fraud:>5,} ({n_fraud/n_total:.2%}) | -> {os.path.basename(filepath)}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  SPLIT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Input:  {os.path.basename(INPUT_FILE)}")
    print(f"  Output: {', '.join(os.path.basename(f) for f in BANK_FILES.values())}")
    print(f"  Location: {SCRIPT_DIR}")

    return {name: filepath for name, filepath in BANK_FILES.items()}


if __name__ == '__main__':
    split_data()
