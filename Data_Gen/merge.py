"""
Merge all 6 output CSVs into one denormalized dataset.
Each row = 1 account with aggregated features from all tables.
"""
import csv
import os
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def load(filename):
    rows = []
    with open(os.path.join(OUTPUT_DIR, filename), 'r') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def merge():
    print("Merging 6 CSVs into one...")

    # Load all data
    accounts = load('accounts.csv')
    devices = load('devices.csv')
    acc_dev_map = load('acc_dev_map.csv')
    events = load('cyber_events.csv')
    txns = load('transactions.csv')
    rings = load('mule_rings.csv')

    # Build lookups
    device_info = {d['device_id']: d for d in devices}

    # Devices per account
    acc_devices = defaultdict(list)
    for m in acc_dev_map:
        acc_devices[m['account_id']].append(m['device_id'])

    # Events per account
    acc_events = defaultdict(list)
    for e in events:
        acc_events[e['account_id']].append(e)

    # Transactions per account (as sender)
    acc_txns = defaultdict(list)
    for t in txns:
        acc_txns[t['from_account']].append(t)

    # Ring membership
    acc_ring = {}
    for r in rings:
        acc_ring[r['account_id']] = r['ring_id']

    # Build merged rows
    merged = []
    for acc in accounts:
        aid = acc['account_id']
        devs = acc_devices.get(aid, [])
        evts = acc_events.get(aid, [])
        txs = acc_txns.get(aid, [])

        # Device features (aggregate)
        n_devices = len(set(devs))
        n_vpn = sum(1 for d in devs if device_info.get(d, {}).get('is_vpn') == 'True')
        n_proxy = sum(1 for d in devs if device_info.get(d, {}).get('is_proxy') == 'True')
        avg_entropy = 0
        if devs:
            entropies = [float(device_info[d]['fingerprint_entropy']) for d in devs if d in device_info]
            avg_entropy = sum(entropies) / len(entropies) if entropies else 0

        # Event features (aggregate)
        n_events = len(evts)
        n_anomalous = sum(1 for e in evts if e['is_anomalous'] == 'True')
        avg_severity = 0
        if evts:
            avg_severity = sum(float(e['severity']) for e in evts) / len(evts)
        attack_types = len(set(e['attack_type'] for e in evts)) if evts else 0

        # Transaction features (aggregate as sender)
        n_txns = len(txs)
        n_suspicious_txns = sum(1 for t in txs if t['is_suspicious'] == '1')
        total_amount = sum(float(t['amount']) for t in txs)
        avg_amount = total_amount / n_txns if n_txns > 0 else 0
        n_external = sum(1 for t in txs if t['to_account'].startswith('EXT_'))

        # Ring
        ring_id = acc_ring.get(aid, 'NONE')
        in_ring = 1 if ring_id != 'NONE' else 0

        merged.append({
            # Account core
            'account_id': aid,
            'bank_id': acc['bank_id'],
            'is_mule': acc['is_mule'],
            'avg_balance': acc['avg_balance'],
            'account_age_days': acc['account_age_days'],
            'tx_velocity': acc['tx_velocity'],
            'avg_tx_amount': acc['avg_tx_amount'],
            # Device features
            'n_devices': n_devices,
            'n_vpn_devices': n_vpn,
            'n_proxy_devices': n_proxy,
            'avg_device_entropy': round(avg_entropy, 4),
            # Cyber event features
            'n_cyber_events': n_events,
            'n_anomalous_events': n_anomalous,
            'avg_event_severity': round(avg_severity, 4),
            'distinct_attack_types': attack_types,
            # Transaction features
            'n_transactions_sent': n_txns,
            'n_suspicious_txns': n_suspicious_txns,
            'total_amount_sent': round(total_amount, 2),
            'avg_txn_amount': round(avg_amount, 2),
            'n_external_transfers': n_external,
            # Ring
            'ring_id': ring_id,
            'in_ring': in_ring,
        })

    # Write merged CSV
    outfile = os.path.join(OUTPUT_DIR, 'merged_dataset.csv')
    fieldnames = list(merged[0].keys())
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print(f"\nDone! Merged {len(merged):,} rows × {len(fieldnames)} columns")
    print(f"Saved: {outfile}")
    print(f"\nColumns: {fieldnames}")


if __name__ == '__main__':
    merge()
