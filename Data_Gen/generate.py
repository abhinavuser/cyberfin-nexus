"""
Phase 4: Generation + Schema Mapping
Uses trained WGAN generator to create 100K realistic accounts across 10 banks,
maps Base.csv features to data_generator.py schema, and generates relational data.

Output: 6 CSVs in Data_Gen/output/ matching existing pipeline format.
"""
import os
import sys
import csv
import json
import random
import math
from datetime import datetime, timedelta

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model import Generator

# Config
NUM_BANKS = 10
ACCOUNTS_PER_BANK = 10000
TOTAL_ACCOUNTS = NUM_BANKS * ACCOUNTS_PER_BANK
MULE_RATIO = 0.15
NUM_DEVICES = 200
NUM_TRANSACTIONS = 12000
NUM_CYBER_EVENTS = 8000
NUM_MULE_RINGS = 20
RING_SIZE = (5, 15)
NOISE_DIM = 128
FEATURE_DIM = 51
NUM_DIM = 25
CAT_DIMS = [5, 7, 7, 2, 5]

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
GENERATOR_PATH = os.path.join(SCRIPT_DIR, 'generator.pt')
TRANSFORMS_PATH = os.path.join(SCRIPT_DIR, 'transforms.json')

ATTACK_TYPES = ["phishing", "malware", "brute_force", "credential_stuffing", "session_hijack"]
ATTACK_SEVERITY = {"phishing": 0.7, "malware": 0.9, "brute_force": 0.5,
                   "credential_stuffing": 0.6, "session_hijack": 0.85}


def inverse_transform_numerical(scaled_val, col_min, col_max):
    """Convert [0,1] scaled value back to original range."""
    return scaled_val * (col_max - col_min) + col_min


def decode_categorical(one_hot_vec, encoding_map):
    """Convert one-hot vector back to categorical label."""
    inv_map = {v: k for k, v in encoding_map.items()}
    idx = one_hot_vec.argmax().item()
    return inv_map.get(idx, list(encoding_map.keys())[0])


def generate_accounts(seed=42):
    """Generate 100K accounts using trained WGAN generator."""
    print("\n[Step 1/6] Generating accounts from WGAN...")

    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load generator
    gen = Generator(noise_dim=NOISE_DIM, feature_dim=FEATURE_DIM,
                    num_dim=NUM_DIM, cat_dims=CAT_DIMS)
    checkpoint = torch.load(GENERATOR_PATH, map_location=device, weights_only=True)
    gen.load_state_dict(checkpoint['generator_state'])
    gen.to(device)
    gen.eval()
    print(f"  Loaded generator (epoch {checkpoint['epoch']}, W-dist={checkpoint['w_dist']:.4f})")

    # Load transforms
    with open(TRANSFORMS_PATH, 'r') as f:
        transforms = json.load(f)

    num_cols = transforms['numerical_cols']
    col_min = transforms['col_min']
    col_max = transforms['col_max']
    cat_encoding = transforms['cat_encoding']
    cat_dims = transforms['cat_dims']

    # Generate in batches
    accounts = []
    batch_size = 1024
    n_generated = 0

    with torch.no_grad():
        while n_generated < TOTAL_ACCOUNTS:
            remaining = TOTAL_ACCOUNTS - n_generated
            bs = min(batch_size, remaining)

            noise = torch.randn(bs, NOISE_DIM, device=device)

            # Decide mule labels: ~15% mule ratio
            labels = torch.zeros(bs, 1, device=device)
            n_mules = int(bs * MULE_RATIO)
            labels[:n_mules] = 1.0

            fake = gen(noise, labels)
            fake = fake.cpu()

            for i in range(bs):
                row = fake[i]
                is_mule = int(labels[i].item())

                # Extract and inverse-transform numerical features
                raw_features = {}
                for j, col in enumerate(num_cols):
                    scaled_val = row[j].item()
                    original_val = inverse_transform_numerical(
                        scaled_val, col_min[col], col_max[col]
                    )
                    raw_features[col] = original_val

                # Decode categoricals
                offset = NUM_DIM
                cat_values = {}
                for col in transforms['categorical_cols']:
                    dim = cat_dims[col]
                    one_hot = row[offset:offset + dim]
                    cat_values[col] = decode_categorical(one_hot, cat_encoding[col])
                    offset += dim

                # === MAP TO EXISTING SCHEMA ===
                bank_id = (n_generated % NUM_BANKS) + 1
                acc_id = f"ACC_{bank_id}_{n_generated:05d}"

                # Map Base.csv features -> data_generator.py schema
                income = max(0., raw_features.get('income', 0.5))
                credit_limit = max(0., raw_features.get('proposed_credit_limit', 500))
                avg_balance = income * credit_limit * 50  # scale to realistic $500-$50K range
                avg_balance = max(500, min(50000, avg_balance))

                age_months = raw_features.get('current_address_months_count', 100)
                account_age_days = max(10, min(3650, abs(age_months)))

                velocity = raw_features.get('velocity_6h', 5000) / 1000
                tx_velocity = max(0.1, min(15, velocity))

                balcon = raw_features.get('intended_balcon_amount', 50)
                avg_tx_amount = max(50, min(15000, abs(balcon) * 100))

                # Adjust based on mule status (reinforce signal)
                if is_mule:
                    avg_balance = min(avg_balance, 5000)
                    account_age_days = min(account_age_days, 180)
                    tx_velocity = max(tx_velocity, 3)
                    avg_tx_amount = max(avg_tx_amount, 2000)
                else:
                    avg_balance = max(avg_balance, 2000)
                    account_age_days = max(account_age_days, 180)
                    tx_velocity = min(tx_velocity, 3)
                    avg_tx_amount = min(avg_tx_amount, 3000)

                accounts.append({
                    "account_id": acc_id,
                    "bank_id": bank_id,
                    "is_mule": is_mule,
                    "avg_balance": round(avg_balance, 2),
                    "account_age_days": int(account_age_days),
                    "tx_velocity": round(tx_velocity, 3),
                    "avg_tx_amount": round(avg_tx_amount, 2),
                })
                n_generated += 1

    n_mules = sum(1 for a in accounts if a['is_mule'] == 1)
    print(f"  Generated: {len(accounts):,} accounts across {NUM_BANKS} banks")
    print(f"  Mules: {n_mules:,} ({n_mules/len(accounts):.1%})")

    return accounts


def generate_devices(seed=42):
    """Generate device profiles."""
    print("\n[Step 2/6] Generating devices...")
    random.seed(seed)
    os_types = ["Windows", "macOS", "Linux", "Android", "iOS"]
    browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
    devices = []
    for i in range(NUM_DEVICES):
        devices.append({
            "device_id": f"DEV_{i:04d}",
            "os": random.choice(os_types),
            "browser": random.choice(browsers),
            "is_vpn": random.random() < 0.2,
            "is_proxy": random.random() < 0.15,
            "fingerprint_entropy": round(random.uniform(0.1, 1.0), 3),
        })
    print(f"  Generated: {len(devices)} devices")
    return devices


def generate_acc_dev_map(accounts, devices, seed=42):
    """Map accounts to devices. Mules share devices for detection signal."""
    print("\n[Step 3/6] Generating account-device mappings...")
    random.seed(seed)
    device_ids = [d['device_id'] for d in devices]
    mule_accs = [a for a in accounts if a['is_mule'] == 1]
    legit_accs = [a for a in accounts if a['is_mule'] == 0]

    # Mules share a small pool
    mule_device_pool = random.sample(device_ids, min(len(device_ids) // 3, len(mule_accs)))

    mappings = set()
    for acc in mule_accs:
        n_devs = random.randint(1, 3)
        for d in random.choices(mule_device_pool, k=n_devs):
            mappings.add((acc['account_id'], d))

    for acc in legit_accs:
        n_devs = random.randint(1, 2)
        for d in random.sample(device_ids, n_devs):
            mappings.add((acc['account_id'], d))

    result = [{"account_id": m[0], "device_id": m[1]} for m in mappings]
    print(f"  Generated: {len(result):,} mappings")
    return result


def generate_cyber_events(accounts, devices, acc_dev_map, seed=42):
    """Generate cyber security events (phishing, malware, etc.)."""
    print("\n[Step 4/6] Generating cyber events...")
    random.seed(seed)
    events = []
    base_time = datetime(2026, 1, 1)

    mule_acc_ids = [a['account_id'] for a in accounts if a['is_mule'] == 1]
    all_acc_ids = [a['account_id'] for a in accounts]
    acc_bank = {a['account_id']: a['bank_id'] for a in accounts}

    # Build device lookup
    acc_devices = {}
    for m in acc_dev_map:
        acc_devices.setdefault(m['account_id'], []).append(m['device_id'])

    for i in range(NUM_CYBER_EVENTS):
        # 70% target mules
        if random.random() < 0.7 and mule_acc_ids:
            target = random.choice(mule_acc_ids)
        else:
            target = random.choice(all_acc_ids)

        attack = random.choice(ATTACK_TYPES)
        severity = ATTACK_SEVERITY[attack] + random.uniform(-0.15, 0.15)
        severity = max(0.05, min(1.0, severity))

        devs = acc_devices.get(target, ['DEV_0000'])
        dev_id = random.choice(devs)

        if random.random() < 0.6:
            ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        else:
            ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

        ts = base_time + timedelta(days=random.randint(0, 60),
                                   hours=random.randint(0, 23),
                                   minutes=random.randint(0, 59))

        events.append({
            "event_id": f"EVT_{i:06d}",
            "timestamp": ts.isoformat(),
            "account_id": target,
            "device_id": dev_id,
            "bank_id": acc_bank.get(target, 1),
            "attack_type": attack,
            "severity": round(severity, 3),
            "source_ip": ip,
            "is_vpn": random.random() < 0.3,
            "is_anomalous": severity > 0.6,
        })

    print(f"  Generated: {len(events):,} events")
    return events


def generate_transactions(accounts, seed=42):
    """Generate financial transactions."""
    print("\n[Step 5/6] Generating transactions...")
    random.seed(seed)
    txns = []
    base_time = datetime(2026, 1, 1)

    mule_accs = [a['account_id'] for a in accounts if a['is_mule'] == 1]
    legit_accs = [a['account_id'] for a in accounts if a['is_mule'] == 0]
    all_accs = [a['account_id'] for a in accounts]
    acc_bank = {a['account_id']: a['bank_id'] for a in accounts}

    for i in range(NUM_TRANSACTIONS):
        if random.random() < 0.4 and mule_accs:
            from_acc = random.choice(mule_accs)
            if random.random() < 0.5:
                to_acc = f"EXT_{random.randint(100, 999)}"
            else:
                to_acc = random.choice(mule_accs + legit_accs[:50])
            amount = round(min(math.exp(random.gauss(8, 1.2)), 100000), 2)
            is_suspicious = 1
        else:
            from_acc = random.choice(legit_accs)
            to_acc = random.choice(all_accs)
            while to_acc == from_acc:
                to_acc = random.choice(all_accs)
            amount = round(min(math.exp(random.gauss(5, 1.5)), 100000), 2)
            is_suspicious = 0

        ts = base_time + timedelta(days=random.randint(0, 60),
                                   hours=random.randint(0, 23),
                                   minutes=random.randint(0, 59))

        txns.append({
            "txn_id": f"TXN_{i:06d}",
            "timestamp": ts.isoformat(),
            "from_account": from_acc,
            "to_account": to_acc,
            "bank_id": acc_bank.get(from_acc, 1),
            "amount": amount,
            "currency": "USD",
            "is_suspicious": is_suspicious,
        })

    print(f"  Generated: {len(txns):,} transactions")
    return txns


def generate_mule_rings(accounts, seed=42):
    """Create mule ring cluster assignments."""
    print("\n[Step 6/6] Generating mule rings...")
    random.seed(seed)
    mule_accs = [a['account_id'] for a in accounts if a['is_mule'] == 1]
    random.shuffle(mule_accs)

    rings = []
    idx = 0
    for ring_id in range(NUM_MULE_RINGS):
        size = random.randint(RING_SIZE[0], RING_SIZE[1])
        members = mule_accs[idx:idx + size]
        idx += size
        if not members:
            break
        for m in members:
            rings.append({"ring_id": f"RING_{ring_id:03d}", "account_id": m})

    print(f"  Generated: {len(rings):,} ring memberships across {ring_id + 1} rings")
    return rings


def save_csv(data, filename, fieldnames):
    """Save list of dicts as CSV."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    return filepath


def generate_all():
    """Full generation pipeline."""
    print("=" * 60)
    print("  PHASE 4: Data Generation + Schema Mapping")
    print("=" * 60)
    print(f"  Scale: {NUM_BANKS} banks x {ACCOUNTS_PER_BANK:,} accounts = {TOTAL_ACCOUNTS:,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate all data
    accounts = generate_accounts()
    devices = generate_devices()
    acc_dev_map = generate_acc_dev_map(accounts, devices)
    cyber_events = generate_cyber_events(accounts, devices, acc_dev_map)
    transactions = generate_transactions(accounts)
    mule_rings = generate_mule_rings(accounts)

    # Save CSVs
    print(f"\n  Saving CSVs to {OUTPUT_DIR}/")
    files = {}
    files['accounts'] = save_csv(accounts, 'accounts.csv',
                                  ['account_id', 'bank_id', 'is_mule', 'avg_balance',
                                   'account_age_days', 'tx_velocity', 'avg_tx_amount'])
    files['devices'] = save_csv(devices, 'devices.csv',
                                 ['device_id', 'os', 'browser', 'is_vpn', 'is_proxy',
                                  'fingerprint_entropy'])
    files['acc_dev_map'] = save_csv(acc_dev_map, 'acc_dev_map.csv',
                                     ['account_id', 'device_id'])
    files['cyber_events'] = save_csv(cyber_events, 'cyber_events.csv',
                                      ['event_id', 'timestamp', 'account_id', 'device_id',
                                       'bank_id', 'attack_type', 'severity', 'source_ip',
                                       'is_vpn', 'is_anomalous'])
    files['transactions'] = save_csv(transactions, 'transactions.csv',
                                      ['txn_id', 'timestamp', 'from_account', 'to_account',
                                       'bank_id', 'amount', 'currency', 'is_suspicious'])
    files['mule_rings'] = save_csv(mule_rings, 'mule_rings.csv',
                                    ['ring_id', 'account_id'])

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  PHASE 4 COMPLETE — Generated Data Summary")
    print(f"{'=' * 60}")
    print(f"  accounts.csv:    {len(accounts):>8,} rows")
    print(f"  devices.csv:     {len(devices):>8,} rows")
    print(f"  acc_dev_map.csv: {len(acc_dev_map):>8,} rows")
    print(f"  cyber_events.csv:{len(cyber_events):>8,} rows")
    print(f"  transactions.csv:{len(transactions):>8,} rows")
    print(f"  mule_rings.csv:  {len(mule_rings):>8,} rows")

    # Per-bank stats
    print(f"\n  Per-bank breakdown:")
    for b in range(1, NUM_BANKS + 1):
        bank_accs = [a for a in accounts if a['bank_id'] == b]
        bank_mules = sum(1 for a in bank_accs if a['is_mule'] == 1)
        print(f"    Bank {b:>2}: {len(bank_accs):,} accounts, {bank_mules} mules ({bank_mules/len(bank_accs):.1%})")

    return files


if __name__ == '__main__':
    generate_all()
