"""
CyberFin Nexus — Synthetic Data Generator
Generates realistic cyber logs, financial transactions, device mappings,
and mule ring structures for 4 banks with 250+ accounts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    NUM_BANKS, NUM_ACCOUNTS, NUM_DEVICES, NUM_TRANSACTIONS,
    NUM_CYBER_EVENTS, MULE_RING_COUNT, MULE_RING_SIZE, MULE_RATIO,
    ATTACK_TYPES, ATTACK_SEVERITY
)


def generate_accounts(n_accounts=NUM_ACCOUNTS, n_banks=NUM_BANKS, mule_ratio=MULE_RATIO, seed=42):
    """Generate account profiles distributed across banks with mule labels."""
    np.random.seed(seed)
    random.seed(seed)

    accounts = []
    n_mules = int(n_accounts * mule_ratio)
    mule_indices = set(random.sample(range(n_accounts), n_mules))

    for i in range(n_accounts):
        bank_id = (i % n_banks) + 1
        is_mule = 1 if i in mule_indices else 0
        acc_id = f"ACC_{bank_id}_{i:04d}"

        # Behavioral features
        if is_mule:
            avg_balance = np.random.uniform(500, 5000)
            account_age_days = np.random.randint(10, 180)
            tx_velocity = np.random.uniform(3, 15)  # txns per day
            avg_tx_amount = np.random.uniform(2000, 15000)
        else:
            avg_balance = np.random.uniform(2000, 50000)
            account_age_days = np.random.randint(180, 3650)
            tx_velocity = np.random.uniform(0.1, 3)
            avg_tx_amount = np.random.uniform(50, 3000)

        accounts.append({
            "account_id": acc_id,
            "bank_id": bank_id,
            "is_mule": is_mule,
            "avg_balance": round(avg_balance, 2),
            "account_age_days": account_age_days,
            "tx_velocity": round(tx_velocity, 3),
            "avg_tx_amount": round(avg_tx_amount, 2),
        })

    return pd.DataFrame(accounts)


def generate_devices(n_devices=NUM_DEVICES, seed=42):
    """Generate device profiles with fingerprint features."""
    np.random.seed(seed)
    devices = []
    os_types = ["Windows", "macOS", "Linux", "Android", "iOS"]
    browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]

    for i in range(n_devices):
        devices.append({
            "device_id": f"DEV_{i:03d}",
            "os": random.choice(os_types),
            "browser": random.choice(browsers),
            "is_vpn": random.random() < 0.2,
            "is_proxy": random.random() < 0.15,
            "fingerprint_entropy": round(np.random.uniform(0.1, 1.0), 3),
        })

    return pd.DataFrame(devices)


def generate_account_device_mapping(accounts_df, devices_df, seed=42):
    """Map accounts to devices. Mules share devices more often (key signal)."""
    np.random.seed(seed)
    random.seed(seed)
    mappings = []

    mule_accounts = accounts_df[accounts_df.is_mule == 1].account_id.tolist()
    legit_accounts = accounts_df[accounts_df.is_mule == 0].account_id.tolist()
    device_ids = devices_df.device_id.tolist()

    # Mules share a smaller pool of devices (suspicious device reuse)
    mule_devices = random.sample(device_ids, min(len(device_ids) // 3, len(mule_accounts)))

    for acc in mule_accounts:
        n_devs = random.randint(1, 3)
        devs = random.choices(mule_devices, k=n_devs)
        for d in devs:
            mappings.append({"account_id": acc, "device_id": d})

    for acc in legit_accounts:
        n_devs = random.randint(1, 2)
        devs = random.sample(device_ids, n_devs)
        for d in devs:
            mappings.append({"account_id": acc, "device_id": d})

    return pd.DataFrame(mappings).drop_duplicates()


def generate_cyber_events(accounts_df, devices_df, acc_dev_map, n_events=NUM_CYBER_EVENTS, seed=42):
    """Generate cyber security events (phishing, malware, etc.) targeting accounts."""
    np.random.seed(seed)
    random.seed(seed)
    events = []
    base_time = datetime(2026, 1, 1)

    mule_accounts = accounts_df[accounts_df.is_mule == 1].account_id.tolist()
    all_accounts = accounts_df.account_id.tolist()

    for i in range(n_events):
        # 70% of cyber events target mule-linked accounts
        if random.random() < 0.7 and mule_accounts:
            target_acc = random.choice(mule_accounts)
        else:
            target_acc = random.choice(all_accounts)

        attack_type = random.choice(ATTACK_TYPES)
        severity = ATTACK_SEVERITY[attack_type] + np.random.uniform(-0.15, 0.15)
        severity = np.clip(severity, 0.05, 1.0)

        # Get a device for this account
        acc_devs = acc_dev_map[acc_dev_map.account_id == target_acc].device_id.tolist()
        device_id = random.choice(acc_devs) if acc_devs else "DEV_000"

        # Generate suspicious IP
        if random.random() < 0.6:
            ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        else:
            ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

        timestamp = base_time + timedelta(
            days=random.randint(0, 60),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        bank_id = accounts_df[accounts_df.account_id == target_acc].bank_id.values[0]

        events.append({
            "event_id": f"EVT_{i:05d}",
            "timestamp": timestamp,
            "account_id": target_acc,
            "device_id": device_id,
            "bank_id": int(bank_id),
            "attack_type": attack_type,
            "severity": round(float(severity), 3),
            "source_ip": ip,
            "is_vpn": random.random() < 0.3,
            "is_anomalous": severity > 0.6,
        })

    return pd.DataFrame(events)


def generate_transactions(accounts_df, n_txns=NUM_TRANSACTIONS, seed=42):
    """Generate financial transactions. Mules have distinctive patterns."""
    np.random.seed(seed)
    random.seed(seed)

    txns = []
    base_time = datetime(2026, 1, 1)

    mule_accounts = accounts_df[accounts_df.is_mule == 1].account_id.tolist()
    legit_accounts = accounts_df[accounts_df.is_mule == 0].account_id.tolist()
    all_accounts = accounts_df.account_id.tolist()

    for i in range(n_txns):
        # 40% of transactions involve at least one mule
        if random.random() < 0.4 and mule_accounts:
            from_acc = random.choice(mule_accounts)
            # Mules often send to external or other mules
            if random.random() < 0.5:
                to_acc = f"EXT_{random.randint(100,999)}"
            else:
                to_acc = random.choice(mule_accounts + legit_accounts[:20])
            amount = round(np.random.lognormal(8, 1.2), 2)  # higher amounts
            is_suspicious = 1
        else:
            from_acc = random.choice(legit_accounts)
            to_acc = random.choice(all_accounts)
            while to_acc == from_acc:
                to_acc = random.choice(all_accounts)
            amount = round(np.random.lognormal(5, 1.5), 2)
            is_suspicious = 0

        amount = min(amount, 100000)  # cap

        timestamp = base_time + timedelta(
            days=random.randint(0, 60),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        from_bank = accounts_df[accounts_df.account_id == from_acc].bank_id.values
        from_bank = int(from_bank[0]) if len(from_bank) > 0 else random.randint(1, 4)

        txns.append({
            "txn_id": f"TXN_{i:05d}",
            "timestamp": timestamp,
            "hour_of_day": timestamp.hour,
            "from_account": from_acc,
            "to_account": to_acc,
            "bank_id": from_bank,
            "amount": amount,
            "currency": "USD",
            "is_suspicious": is_suspicious,
        })

    return pd.DataFrame(txns)


def generate_mule_rings(accounts_df, n_rings=MULE_RING_COUNT, ring_size=MULE_RING_SIZE, seed=42):
    """Generate explicit mule ring cluster assignments."""
    random.seed(seed)
    mule_accounts = accounts_df[accounts_df.is_mule == 1].account_id.tolist()
    random.shuffle(mule_accounts)

    rings = []
    idx = 0
    for ring_id in range(n_rings):
        size = random.randint(ring_size[0], ring_size[1])
        members = mule_accounts[idx:idx + size]
        idx += size
        if not members:
            break
        for m in members:
            rings.append({"ring_id": f"RING_{ring_id}", "account_id": m})

    return pd.DataFrame(rings)


def generate_all_data(seed=42):
    """Generate complete synthetic dataset and return all DataFrames."""
    accounts = generate_accounts(seed=seed)
    devices = generate_devices(seed=seed)
    acc_dev_map = generate_account_device_mapping(accounts, devices, seed=seed)
    cyber_events = generate_cyber_events(accounts, devices, acc_dev_map, seed=seed)
    transactions = generate_transactions(accounts, seed=seed)
    mule_rings = generate_mule_rings(accounts, seed=seed)

    return {
        "accounts": accounts,
        "devices": devices,
        "acc_dev_map": acc_dev_map,
        "cyber_events": cyber_events,
        "transactions": transactions,
        "mule_rings": mule_rings,
    }


if __name__ == "__main__":
    data = generate_all_data()
    for name, df in data.items():
        print(f"{name}: {df.shape}")
        print(df.head(3))
        print()
