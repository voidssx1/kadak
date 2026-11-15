"""
data/generate_datasets.py
=========================
Generates realistic, statistically calibrated benchmark datasets for the
three supported NIDS standards:
  1. CICIDS2017  (77 raw flow features + Label)
  2. UNSW-NB15   (42 flow metrics + attack_cat + label)
  3. NSL-KDD     (41 connection features + class)

Outputs:
  - data/cicids2017/cicids2017_benchmark.csv & schema.json
  - data/unsw_nb15/unsw_nb15_benchmark.csv & schema.json
  - data/nsl_kdd/nsl_kdd_benchmark.csv & schema.json
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

# =====================================================================
# 1. CICIDS2017 GENERATOR (77 Raw Features)
# =====================================================================
CICIDS2017_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

def generate_cicids2017(n_samples: int = 25000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n_benign = int(n_samples * 0.60)
    n_dos = int(n_samples * 0.15)
    n_portscan = int(n_samples * 0.12)
    n_bruteforce = int(n_samples * 0.08)
    n_zeroday = n_samples - (n_benign + n_dos + n_portscan + n_bruteforce)
    rows = []

    # 1. BENIGN / NORMAL
    for _ in range(n_benign):
        dur = np.random.exponential(1_500_000) + 1000
        fwd_pkts = int(np.random.geometric(0.15)) + 1
        bwd_pkts = int(np.random.geometric(0.12)) + 1
        fwd_len_mean = np.random.uniform(100, 800)
        bwd_len_mean = np.random.uniform(200, 1400)
        row = {
            "Label": "BENIGN",
            "Destination Port": np.random.choice([80, 443, 53, 8080, 22, 123]),
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len_mean,
            "Total Length of Bwd Packets": bwd_pkts * bwd_len_mean,
            "Fwd Packet Length Max": fwd_len_mean * np.random.uniform(1.2, 2.0),
            "Fwd Packet Length Min": max(20.0, fwd_len_mean * np.random.uniform(0.1, 0.5)),
            "Fwd Packet Length Mean": fwd_len_mean,
            "Fwd Packet Length Std": fwd_len_mean * np.random.uniform(0.2, 0.6),
            "Bwd Packet Length Max": bwd_len_mean * np.random.uniform(1.2, 2.0),
            "Bwd Packet Length Min": max(20.0, bwd_len_mean * np.random.uniform(0.1, 0.4)),
            "Bwd Packet Length Mean": bwd_len_mean,
            "Bwd Packet Length Std": bwd_len_mean * np.random.uniform(0.2, 0.7),
            "Flow Bytes/s": ((fwd_pkts * fwd_len_mean) + (bwd_pkts * bwd_len_mean)) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, (fwd_pkts + bwd_pkts - 1)),
            "Flow IAT Std": (dur / max(1, (fwd_pkts + bwd_pkts - 1))) * np.random.uniform(0.3, 0.8),
            "Flow IAT Max": dur * np.random.uniform(0.4, 0.9),
            "Flow IAT Min": np.random.uniform(1.0, 50.0),
            "Fwd IAT Total": dur * np.random.uniform(0.8, 1.0),
            "Fwd IAT Mean": dur / max(1, fwd_pkts),
            "Fwd IAT Std": (dur / max(1, fwd_pkts)) * np.random.uniform(0.2, 0.7),
            "Fwd IAT Max": dur * np.random.uniform(0.4, 0.8),
            "Fwd IAT Min": np.random.uniform(1.0, 50.0),
            "Bwd IAT Total": dur * np.random.uniform(0.7, 0.95),
            "Bwd IAT Mean": dur / max(1, bwd_pkts),
            "Bwd IAT Std": (dur / max(1, bwd_pkts)) * np.random.uniform(0.2, 0.7),
            "Bwd IAT Max": dur * np.random.uniform(0.4, 0.8),
            "Bwd IAT Min": np.random.uniform(1.0, 50.0),
            "Fwd PSH Flags": float(np.random.choice([0, 1], p=[0.7, 0.3])),
            "Bwd PSH Flags": 0.0,
            "Fwd URG Flags": 0.0,
            "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32,
            "Bwd Header Length": bwd_pkts * 32,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5),
            "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-5),
            "Min Packet Length": 40.0,
            "Max Packet Length": max(fwd_len_mean, bwd_len_mean) * 1.8,
            "Packet Length Mean": (fwd_len_mean + bwd_len_mean) / 2.0,
            "Packet Length Std": 250.0,
            "Packet Length Variance": 250.0 ** 2,
            "FIN Flag Count": float(np.random.choice([0, 1], p=[0.5, 0.5])),
            "SYN Flag Count": float(np.random.choice([0, 1], p=[0.6, 0.4])),
            "RST Flag Count": float(np.random.choice([0, 1], p=[0.95, 0.05])),
            "PSH Flag Count": float(np.random.choice([0, 1], p=[0.4, 0.6])),
            "ACK Flag Count": 1.0,
            "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0,
            "ECE Flag Count": 0.0,
            "Down/Up Ratio": bwd_pkts / fwd_pkts,
            "Average Packet Size": (fwd_len_mean + bwd_len_mean) / 2.0,
            "Avg Fwd Segment Size": fwd_len_mean,
            "Avg Bwd Segment Size": bwd_len_mean,
            "Fwd Header Length.1": fwd_pkts * 32,
            "Fwd Avg Bytes/Bulk": 0.0, "Fwd Avg Packets/Bulk": 0.0, "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_pkts, "Subflow Fwd Bytes": fwd_pkts * fwd_len_mean,
            "Subflow Bwd Packets": bwd_pkts, "Subflow Bwd Bytes": bwd_pkts * bwd_len_mean,
            "Init_Win_bytes_forward": 29200.0, "Init_Win_bytes_backward": 28960.0,
            "act_data_pkt_fwd": max(0, fwd_pkts - 1), "min_seg_size_forward": 32.0,
            "Active Mean": dur * 0.2 if dur > 5e6 else 0.0,
            "Active Std": dur * 0.05 if dur > 5e6 else 0.0,
            "Active Max": dur * 0.25 if dur > 5e6 else 0.0,
            "Active Min": dur * 0.15 if dur > 5e6 else 0.0,
            "Idle Mean": dur * 0.7 if dur > 5e6 else 0.0,
            "Idle Std": dur * 0.1 if dur > 5e6 else 0.0,
            "Idle Max": dur * 0.8 if dur > 5e6 else 0.0,
            "Idle Min": dur * 0.6 if dur > 5e6 else 0.0
        }
        rows.append(row)

    # 2. DoS-SYNFlood
    for _ in range(n_dos):
        dur = np.random.uniform(500, 30_000)
        fwd_pkts = np.random.randint(40, 200)
        bwd_pkts = 0
        fwd_len_mean = 44.0
        row = {
            "Label": "DoS",
            "Destination Port": np.random.choice([80, 443, 8080]),
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": 0,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len_mean,
            "Total Length of Bwd Packets": 0.0,
            "Fwd Packet Length Max": 54.0, "Fwd Packet Length Min": 40.0,
            "Fwd Packet Length Mean": fwd_len_mean, "Fwd Packet Length Std": 2.0,
            "Bwd Packet Length Max": 0.0, "Bwd Packet Length Min": 0.0,
            "Bwd Packet Length Mean": 0.0, "Bwd Packet Length Std": 0.0,
            "Flow Bytes/s": (fwd_pkts * fwd_len_mean) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": fwd_pkts / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, fwd_pkts),
            "Flow IAT Std": (dur / max(1, fwd_pkts)) * 0.1,
            "Flow IAT Max": (dur / max(1, fwd_pkts)) * 1.5, "Flow IAT Min": 1.0,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur / max(1, fwd_pkts),
            "Fwd IAT Std": 5.0, "Fwd IAT Max": dur / max(1, fwd_pkts), "Fwd IAT Min": 1.0,
            "Bwd IAT Total": 0.0, "Bwd IAT Mean": 0.0, "Bwd IAT Std": 0.0, "Bwd IAT Max": 0.0, "Bwd IAT Min": 0.0,
            "Fwd PSH Flags": 0.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": 0,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": 0.0,
            "Min Packet Length": 40.0, "Max Packet Length": 54.0,
            "Packet Length Mean": fwd_len_mean, "Packet Length Std": 2.0,
            "Packet Length Variance": 4.0,
            "FIN Flag Count": 0.0, "SYN Flag Count": 1.0, "RST Flag Count": 0.0,
            "PSH Flag Count": 0.0, "ACK Flag Count": 0.0, "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0, "ECE Flag Count": 0.0, "Down/Up Ratio": 0.0,
            "Average Packet Size": fwd_len_mean, "Avg Fwd Segment Size": fwd_len_mean,
            "Avg Bwd Segment Size": 0.0, "Fwd Header Length.1": fwd_pkts * 32,
            "Fwd Avg Bytes/Bulk": 0.0, "Fwd Avg Packets/Bulk": 0.0, "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_pkts, "Subflow Fwd Bytes": fwd_pkts * fwd_len_mean,
            "Subflow Bwd Packets": 0, "Subflow Bwd Bytes": 0.0,
            "Init_Win_bytes_forward": 1024.0, "Init_Win_bytes_backward": -1.0,
            "act_data_pkt_fwd": 0, "min_seg_size_forward": 32.0,
            "Active Mean": 0.0, "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0
        }
        rows.append(row)

    # 3. PortScan
    for _ in range(n_portscan):
        dur = np.random.uniform(50, 2000)
        fwd_pkts = np.random.choice([1, 2], p=[0.8, 0.2])
        bwd_pkts = np.random.choice([0, 1], p=[0.7, 0.3])
        row = {
            "Label": "PortScan",
            "Destination Port": np.random.randint(1, 65535),
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_pkts * 44.0,
            "Total Length of Bwd Packets": bwd_pkts * 40.0,
            "Fwd Packet Length Max": 44.0, "Fwd Packet Length Min": 44.0,
            "Fwd Packet Length Mean": 44.0, "Fwd Packet Length Std": 0.0,
            "Bwd Packet Length Max": 40.0 if bwd_pkts else 0.0,
            "Bwd Packet Length Min": 40.0 if bwd_pkts else 0.0,
            "Bwd Packet Length Mean": 40.0 if bwd_pkts else 0.0,
            "Bwd Packet Length Std": 0.0,
            "Flow Bytes/s": (fwd_pkts * 44.0 + bwd_pkts * 40.0) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur, "Flow IAT Std": 0.0, "Flow IAT Max": dur, "Flow IAT Min": dur,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur, "Fwd IAT Std": 0.0, "Fwd IAT Max": dur, "Fwd IAT Min": dur,
            "Bwd IAT Total": 0.0, "Bwd IAT Mean": 0.0, "Bwd IAT Std": 0.0, "Bwd IAT Max": 0.0, "Bwd IAT Min": 0.0,
            "Fwd PSH Flags": 0.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": bwd_pkts * 32,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-5),
            "Min Packet Length": 40.0, "Max Packet Length": 44.0,
            "Packet Length Mean": 44.0, "Packet Length Std": 1.0, "Packet Length Variance": 1.0,
            "FIN Flag Count": 0.0, "SYN Flag Count": 1.0,
            "RST Flag Count": float(bwd_pkts == 1),
            "PSH Flag Count": 0.0, "ACK Flag Count": float(bwd_pkts == 1), "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0, "ECE Flag Count": 0.0, "Down/Up Ratio": bwd_pkts / fwd_pkts,
            "Average Packet Size": 44.0, "Avg Fwd Segment Size": 44.0, "Avg Bwd Segment Size": 40.0 if bwd_pkts else 0.0,
            "Fwd Header Length.1": fwd_pkts * 32,
            "Fwd Avg Bytes/Bulk": 0.0, "Fwd Avg Packets/Bulk": 0.0, "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_pkts, "Subflow Fwd Bytes": fwd_pkts * 44.0,
            "Subflow Bwd Packets": bwd_pkts, "Subflow Bwd Bytes": bwd_pkts * 40.0,
            "Init_Win_bytes_forward": 1024.0, "Init_Win_bytes_backward": 0.0 if bwd_pkts else -1.0,
            "act_data_pkt_fwd": 0, "min_seg_size_forward": 32.0,
            "Active Mean": 0.0, "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0
        }
        rows.append(row)

    # 4. BruteForce
    for _ in range(n_bruteforce):
        dur = np.random.uniform(500_000, 3_000_000)
        fwd_pkts = np.random.randint(10, 25)
        bwd_pkts = np.random.randint(8, 20)
        fwd_len_mean = 85.0
        bwd_len_mean = 120.0
        row = {
            "Label": "BruteForce",
            "Destination Port": np.random.choice([22, 21, 3389]),
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts, "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len_mean,
            "Total Length of Bwd Packets": bwd_pkts * bwd_len_mean,
            "Fwd Packet Length Max": 160.0, "Fwd Packet Length Min": 40.0,
            "Fwd Packet Length Mean": fwd_len_mean, "Fwd Packet Length Std": 25.0,
            "Bwd Packet Length Max": 200.0, "Bwd Packet Length Min": 40.0,
            "Bwd Packet Length Mean": bwd_len_mean, "Bwd Packet Length Std": 30.0,
            "Flow Bytes/s": ((fwd_pkts * fwd_len_mean) + (bwd_pkts * bwd_len_mean)) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, (fwd_pkts + bwd_pkts - 1)),
            "Flow IAT Std": 2000.0, "Flow IAT Max": 150_000.0, "Flow IAT Min": 500.0,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur / max(1, fwd_pkts), "Fwd IAT Std": 2000.0,
            "Fwd IAT Max": 150_000.0, "Fwd IAT Min": 500.0,
            "Bwd IAT Total": dur * 0.9, "Bwd IAT Mean": dur / max(1, bwd_pkts), "Bwd IAT Std": 2000.0,
            "Bwd IAT Max": 150_000.0, "Bwd IAT Min": 500.0,
            "Fwd PSH Flags": 1.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": bwd_pkts * 32,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-5),
            "Min Packet Length": 40.0, "Max Packet Length": 200.0,
            "Packet Length Mean": (fwd_len_mean + bwd_len_mean) / 2.0, "Packet Length Std": 35.0,
            "Packet Length Variance": 35.0 ** 2,
            "FIN Flag Count": 1.0, "SYN Flag Count": 1.0, "RST Flag Count": 0.0,
            "PSH Flag Count": 1.0, "ACK Flag Count": 1.0, "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0, "ECE Flag Count": 0.0,
            "Down/Up Ratio": bwd_pkts / fwd_pkts,
            "Average Packet Size": (fwd_len_mean + bwd_len_mean) / 2.0,
            "Avg Fwd Segment Size": fwd_len_mean, "Avg Bwd Segment Size": bwd_len_mean,
            "Fwd Header Length.1": fwd_pkts * 32,
            "Fwd Avg Bytes/Bulk": 0.0, "Fwd Avg Packets/Bulk": 0.0, "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_pkts, "Subflow Fwd Bytes": fwd_pkts * fwd_len_mean,
            "Subflow Bwd Packets": bwd_pkts, "Subflow Bwd Bytes": bwd_pkts * bwd_len_mean,
            "Init_Win_bytes_forward": 65535.0, "Init_Win_bytes_backward": 65535.0,
            "act_data_pkt_fwd": fwd_pkts - 2, "min_seg_size_forward": 32.0,
            "Active Mean": 0.0, "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0
        }
        rows.append(row)

    # 5. Infiltration-ZeroDay
    for _ in range(n_zeroday):
        dur = np.random.uniform(10_000, 3_000_000)
        fwd_pkts = np.random.randint(15, 60)
        bwd_pkts = np.random.randint(1, 4)
        fwd_len_mean = np.random.uniform(900, 1480)
        bwd_len_mean = 40.0
        row = {
            "Label": "Infiltration_ZeroDay",
            "Destination Port": np.random.choice([4444, 1337, 8443, 31337]),
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts, "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len_mean,
            "Total Length of Bwd Packets": bwd_pkts * bwd_len_mean,
            "Fwd Packet Length Max": 1500.0, "Fwd Packet Length Min": 200.0,
            "Fwd Packet Length Mean": fwd_len_mean, "Fwd Packet Length Std": 300.0,
            "Bwd Packet Length Max": 50.0, "Bwd Packet Length Min": 40.0,
            "Bwd Packet Length Mean": bwd_len_mean, "Bwd Packet Length Std": 5.0,
            "Flow Bytes/s": ((fwd_pkts * fwd_len_mean) + (bwd_pkts * bwd_len_mean)) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, (fwd_pkts + bwd_pkts - 1)),
            "Flow IAT Std": (dur / max(1, (fwd_pkts + bwd_pkts - 1))) * 2.5,
            "Flow IAT Max": dur * 0.9, "Flow IAT Min": 1.0,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur / max(1, fwd_pkts),
            "Fwd IAT Std": (dur / max(1, fwd_pkts)) * 2.0, "Fwd IAT Max": dur * 0.85, "Fwd IAT Min": 1.0,
            "Bwd IAT Total": dur * 0.5, "Bwd IAT Mean": dur / max(1, bwd_pkts),
            "Bwd IAT Std": 100.0, "Bwd IAT Max": dur * 0.5, "Bwd IAT Min": 100.0,
            "Fwd PSH Flags": 1.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 1.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": bwd_pkts * 32,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-5),
            "Min Packet Length": 40.0, "Max Packet Length": 1500.0,
            "Packet Length Mean": fwd_len_mean * 0.9, "Packet Length Std": 450.0,
            "Packet Length Variance": 450.0 ** 2,
            "FIN Flag Count": 0.0, "SYN Flag Count": 1.0, "RST Flag Count": 0.0,
            "PSH Flag Count": 1.0, "ACK Flag Count": 1.0, "URG Flag Count": 1.0,
            "CWE Flag Count": 1.0, "ECE Flag Count": 0.0,
            "Down/Up Ratio": bwd_pkts / fwd_pkts, "Average Packet Size": fwd_len_mean * 0.9,
            "Avg Fwd Segment Size": fwd_len_mean, "Avg Bwd Segment Size": bwd_len_mean,
            "Fwd Header Length.1": fwd_pkts * 32,
            "Fwd Avg Bytes/Bulk": 1000.0, "Fwd Avg Packets/Bulk": 2.0, "Fwd Avg Bulk Rate": 500.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_pkts, "Subflow Fwd Bytes": fwd_pkts * fwd_len_mean,
            "Subflow Bwd Packets": bwd_pkts, "Subflow Bwd Bytes": bwd_pkts * bwd_len_mean,
            "Init_Win_bytes_forward": 4096.0, "Init_Win_bytes_backward": 4096.0,
            "act_data_pkt_fwd": fwd_pkts - 1, "min_seg_size_forward": 32.0,
            "Active Mean": 0.0, "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# =====================================================================
# 2. UNSW-NB15 GENERATOR (42 Features + Labels)
# =====================================================================
UNSW_NB15_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
    "attack_cat", "label"
]

def generate_unsw_nb15(n_samples: int = 25000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n_normal = int(n_samples * 0.65)
    n_generic = int(n_samples * 0.12)
    n_exploits = int(n_samples * 0.09)
    n_fuzzers = int(n_samples * 0.08)
    n_dos = n_samples - (n_normal + n_generic + n_exploits + n_fuzzers)
    rows = []

    # Normal traffic
    for _ in range(n_normal):
        dur = np.random.exponential(0.8) + 0.001
        spkts = int(np.random.geometric(0.12)) + 1
        dpkts = int(np.random.geometric(0.10)) + 1
        smean = np.random.uniform(60, 500)
        dmean = np.random.uniform(100, 1200)
        sbytes = int(spkts * smean)
        dbytes = int(dpkts * dmean)
        rows.append({
            "dur": round(dur, 6),
            "proto": np.random.choice(["tcp", "udp", "arp", "ospf"], p=[0.75, 0.20, 0.03, 0.02]),
            "service": np.random.choice(["http", "dns", "ftp", "smtp", "ssh", "-"], p=[0.45, 0.25, 0.05, 0.05, 0.05, 0.15]),
            "state": np.random.choice(["FIN", "CON", "INT", "REQ"], p=[0.7, 0.2, 0.05, 0.05]),
            "spkts": spkts, "dpkts": dpkts, "sbytes": sbytes, "dbytes": dbytes,
            "rate": round((spkts + dpkts) / max(1e-4, dur), 4),
            "sttl": int(np.random.choice([64, 128, 255], p=[0.6, 0.3, 0.1])),
            "dttl": int(np.random.choice([64, 128, 255], p=[0.7, 0.2, 0.1])),
            "sload": round((sbytes * 8) / max(1e-4, dur), 2),
            "dload": round((dbytes * 8) / max(1e-4, dur), 2),
            "sloss": int(np.random.poisson(0.1)),
            "dloss": int(np.random.poisson(0.1)),
            "sinpkt": round(dur / max(1, spkts) * 1000, 3),
            "dinpkt": round(dur / max(1, dpkts) * 1000, 3),
            "sjit": round(np.random.uniform(0.5, 30.0), 3),
            "djit": round(np.random.uniform(0.5, 30.0), 3),
            "swin": 255, "stcpb": int(np.random.randint(1e8, 4e9)),
            "dtcpb": int(np.random.randint(1e8, 4e9)), "dwin": 255,
            "tcprtt": round(np.random.uniform(0.001, 0.05), 6),
            "synack": round(np.random.uniform(0.0005, 0.025), 6),
            "ackdat": round(np.random.uniform(0.0005, 0.025), 6),
            "smean": round(smean, 1), "dmean": round(dmean, 1),
            "trans_depth": int(np.random.choice([0, 1], p=[0.7, 0.3])),
            "response_body_len": int(np.random.choice([0, 500, 2048], p=[0.6, 0.3, 0.1])),
            "ct_srv_src": np.random.randint(1, 10), "ct_state_ttl": 0,
            "ct_dst_ltm": np.random.randint(1, 8), "ct_src_dport_ltm": np.random.randint(1, 5),
            "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": np.random.randint(1, 6),
            "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": int(np.random.choice([0, 1], p=[0.7, 0.3])),
            "ct_src_ltm": np.random.randint(1, 8), "ct_srv_dst": np.random.randint(1, 8),
            "is_sm_ips_ports": 0,
            "attack_cat": "Normal", "label": 0
        })

    # Attacks (Generic, Exploits, Fuzzers, DoS)
    attack_configs = [
        ("Generic", n_generic, "INT", 0.00001, 2, 0, 100, 0, 254, 0),
        ("Exploits", n_exploits, "FIN", 0.15, 8, 6, 1200, 400, 64, 60),
        ("Fuzzers", n_fuzzers, "INT", 0.05, 12, 1, 800, 50, 64, 0),
        ("DoS", n_dos, "INT", 0.001, 50, 0, 44, 0, 254, 0),
    ]

    for cat_name, count, def_state, dur_base, spkts_c, dpkts_c, sbytes_c, dbytes_c, sttl_c, dttl_c in attack_configs:
        for _ in range(count):
            dur = max(1e-5, dur_base * np.random.uniform(0.5, 2.0))
            spkts = int(spkts_c * np.random.uniform(0.8, 1.5))
            dpkts = int(dpkts_c * np.random.uniform(0.5, 1.5)) if dpkts_c > 0 else 0
            sbytes = int(sbytes_c * spkts)
            dbytes = int(dbytes_c * dpkts) if dpkts > 0 else 0
            rows.append({
                "dur": round(dur, 6),
                "proto": "tcp" if cat_name != "Generic" else np.random.choice(["udp", "ospf"]),
                "service": "-" if cat_name in ("Generic", "DoS") else "http",
                "state": def_state,
                "spkts": spkts, "dpkts": dpkts, "sbytes": sbytes, "dbytes": dbytes,
                "rate": round((spkts + dpkts) / max(1e-4, dur), 4),
                "sttl": sttl_c, "dttl": dttl_c,
                "sload": round((sbytes * 8) / max(1e-4, dur), 2),
                "dload": round((dbytes * 8) / max(1e-4, dur), 2),
                "sloss": int(spkts * 0.05), "dloss": int(dpkts * 0.05),
                "sinpkt": round(dur / max(1, spkts) * 1000, 3),
                "dinpkt": round(dur / max(1, dpkts) * 1000, 3) if dpkts else 0.0,
                "sjit": round(np.random.uniform(0.1, 5.0), 3),
                "djit": round(np.random.uniform(0.1, 5.0), 3),
                "swin": 255 if dpkts else 0, "stcpb": int(np.random.randint(1e7, 1e9)),
                "dtcpb": int(np.random.randint(1e7, 1e9)) if dpkts else 0,
                "dwin": 255 if dpkts else 0,
                "tcprtt": 0.0 if not dpkts else 0.015,
                "synack": 0.0 if not dpkts else 0.008,
                "ackdat": 0.0 if not dpkts else 0.007,
                "smean": round(sbytes / max(1, spkts), 1),
                "dmean": round(dbytes / max(1, dpkts), 1) if dpkts else 0.0,
                "trans_depth": 0, "response_body_len": 0,
                "ct_srv_src": np.random.randint(5, 30), "ct_state_ttl": 2 if cat_name == "Generic" else 1,
                "ct_dst_ltm": np.random.randint(5, 30), "ct_src_dport_ltm": np.random.randint(5, 20),
                "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": np.random.randint(5, 25),
                "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0,
                "ct_src_ltm": np.random.randint(5, 25), "ct_srv_dst": np.random.randint(5, 25),
                "is_sm_ips_ports": 0,
                "attack_cat": cat_name, "label": 1
            })

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# =====================================================================
# 3. NSL-KDD GENERATOR (41 Connection Features + Class)
# =====================================================================
NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class"
]

def generate_nsl_kdd(n_samples: int = 25000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n_normal = int(n_samples * 0.60)
    n_neptune = int(n_samples * 0.18)  # SYN Flood DoS
    n_satan = int(n_samples * 0.10)    # Probe / PortScan
    n_smurf = int(n_samples * 0.07)    # ICMP flood
    n_warez = n_samples - (n_normal + n_neptune + n_satan + n_smurf) # R2L
    rows = []

    # Normal connections
    for _ in range(n_normal):
        dur = int(np.random.exponential(1.5))
        src_bytes = int(np.random.exponential(350)) + 40
        dst_bytes = int(np.random.exponential(2500)) + 40
        rows.append({
            "duration": dur,
            "protocol_type": np.random.choice(["tcp", "udp", "icmp"], p=[0.8, 0.15, 0.05]),
            "service": np.random.choice(["http", "smtp", "ftp_data", "private", "domain_u"], p=[0.6, 0.15, 0.1, 0.05, 0.1]),
            "flag": "SF",
            "src_bytes": src_bytes, "dst_bytes": dst_bytes,
            "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
            "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
            "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
            "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0,
            "count": np.random.randint(1, 15), "srv_count": np.random.randint(1, 15),
            "serror_rate": 0.0, "srv_serror_rate": 0.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": round(np.random.uniform(0.0, 0.2), 2),
            "dst_host_count": np.random.randint(50, 255), "dst_host_srv_count": np.random.randint(150, 255),
            "dst_host_same_srv_rate": round(np.random.uniform(0.85, 1.0), 2),
            "dst_host_diff_srv_rate": round(np.random.uniform(0.0, 0.05), 2),
            "dst_host_same_src_port_rate": round(np.random.uniform(0.0, 0.1), 2),
            "dst_host_srv_diff_host_rate": round(np.random.uniform(0.0, 0.05), 2),
            "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
            "class": "normal"
        })

    # Neptune (SYN Flood DoS)
    for _ in range(n_neptune):
        rows.append({
            "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "S0",
            "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
            "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
            "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
            "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0,
            "count": np.random.randint(100, 350), "srv_count": np.random.randint(1, 15),
            "serror_rate": 1.0, "srv_serror_rate": 1.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
            "same_srv_rate": round(np.random.uniform(0.02, 0.1), 2),
            "diff_srv_rate": round(np.random.uniform(0.05, 0.15), 2),
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 255, "dst_host_srv_count": np.random.randint(1, 15),
            "dst_host_same_srv_rate": round(np.random.uniform(0.01, 0.08), 2),
            "dst_host_diff_srv_rate": round(np.random.uniform(0.06, 0.12), 2),
            "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
            "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
            "class": "neptune"
        })

    # Satan (Probe / PortScan)
    for _ in range(n_satan):
        rows.append({
            "duration": 0, "protocol_type": "tcp", "service": np.random.choice(["private", "other", "remote_job"]),
            "flag": "REJ", "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
            "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
            "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
            "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0,
            "count": np.random.randint(150, 400), "srv_count": 1,
            "serror_rate": 0.0, "srv_serror_rate": 0.0,
            "rerror_rate": 1.0, "srv_rerror_rate": 1.0,
            "same_srv_rate": 0.01, "diff_srv_rate": 0.15, "srv_diff_host_rate": 0.0,
            "dst_host_count": 255, "dst_host_srv_count": 1,
            "dst_host_same_srv_rate": 0.01, "dst_host_diff_srv_rate": 0.12,
            "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 1.0, "dst_host_srv_rerror_rate": 1.0,
            "class": "satan"
        })

    # Smurf (ICMP Echo Flood)
    for _ in range(n_smurf):
        rows.append({
            "duration": 0, "protocol_type": "icmp", "service": "ecr_i", "flag": "SF",
            "src_bytes": 1032, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
            "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
            "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
            "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0,
            "count": 511, "srv_count": 511,
            "serror_rate": 0.0, "srv_serror_rate": 0.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
            "dst_host_count": 255, "dst_host_srv_count": 255,
            "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
            "class": "smurf"
        })

    # Warezclient (R2L payload transfer)
    for _ in range(n_warez):
        rows.append({
            "duration": int(np.random.exponential(10.0)),
            "protocol_type": "tcp", "service": "ftp", "flag": "SF",
            "src_bytes": np.random.randint(1000, 5000), "dst_bytes": np.random.randint(5000, 20000),
            "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 2,
            "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
            "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 1,
            "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 1,
            "count": np.random.randint(1, 10), "srv_count": np.random.randint(1, 10),
            "serror_rate": 0.0, "srv_serror_rate": 0.0,
            "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
            "dst_host_count": 255, "dst_host_srv_count": np.random.randint(5, 50),
            "dst_host_same_srv_rate": round(np.random.uniform(0.1, 0.4), 2),
            "dst_host_diff_srv_rate": round(np.random.uniform(0.01, 0.05), 2),
            "dst_host_same_src_port_rate": round(np.random.uniform(0.0, 0.1), 2),
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
            "class": "warezclient"
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    print("=" * 70)
    print("📦 Generating Calibrated NIDS Datasets for All 3 Supported Standards...")
    print("=" * 70)

    # 1. CICIDS2017
    cicids_dir = DATA_DIR / "cicids2017"
    cicids_dir.mkdir(parents=True, exist_ok=True)
    print("\n[1/3] Generating CICIDS2017 Benchmark (77 Raw Features)...")
    df_cicids = generate_cicids2017(n_samples=25000)
    cicids_csv = cicids_dir / "cicids2017_benchmark.csv"
    df_cicids.to_csv(cicids_csv, index=False)
    with open(cicids_dir / "schema.json", "w") as f:
        json.dump({
            "dataset": "CICIDS2017",
            "total_samples": len(df_cicids),
            "features_count": len(CICIDS2017_FEATURES),
            "features": CICIDS2017_FEATURES,
            "target_label": "Label",
            "classes": df_cicids["Label"].value_counts().to_dict()
        }, f, indent=2)
    print(f"      Saved -> {cicids_csv} ({len(df_cicids)} rows)")
    print(f"      Classes: {df_cicids['Label'].value_counts().to_dict()}")

    # 2. UNSW-NB15
    unsw_dir = DATA_DIR / "unsw_nb15"
    unsw_dir.mkdir(parents=True, exist_ok=True)
    print("\n[2/3] Generating UNSW-NB15 Benchmark (42 Flow Metrics)...")
    df_unsw = generate_unsw_nb15(n_samples=25000)
    unsw_csv = unsw_dir / "unsw_nb15_benchmark.csv"
    df_unsw.to_csv(unsw_csv, index=False)
    with open(unsw_dir / "schema.json", "w") as f:
        json.dump({
            "dataset": "UNSW-NB15",
            "total_samples": len(df_unsw),
            "features_count": len(UNSW_NB15_FEATURES) - 2,
            "features": [f for f in UNSW_NB15_FEATURES if f not in ("attack_cat", "label")],
            "target_labels": ["attack_cat", "label"],
            "classes": df_unsw["attack_cat"].value_counts().to_dict()
        }, f, indent=2)
    print(f"      Saved -> {unsw_csv} ({len(df_unsw)} rows)")
    print(f"      Classes: {df_unsw['attack_cat'].value_counts().to_dict()}")

    # 3. NSL-KDD
    nsl_dir = DATA_DIR / "nsl_kdd"
    nsl_dir.mkdir(parents=True, exist_ok=True)
    print("\n[3/3] Generating NSL-KDD Benchmark (41 Connection Features)...")
    df_nsl = generate_nsl_kdd(n_samples=25000)
    nsl_csv = nsl_dir / "nsl_kdd_benchmark.csv"
    df_nsl.to_csv(nsl_csv, index=False)
    with open(nsl_dir / "schema.json", "w") as f:
        json.dump({
            "dataset": "NSL-KDD",
            "total_samples": len(df_nsl),
            "features_count": len(NSL_KDD_FEATURES) - 1,
            "features": [f for f in NSL_KDD_FEATURES if f != "class"],
            "target_label": "class",
            "classes": df_nsl["class"].value_counts().to_dict()
        }, f, indent=2)
    print(f"      Saved -> {nsl_csv} ({len(df_nsl)} rows)")
    print(f"      Classes: {df_nsl['class'].value_counts().to_dict()}")

    print("\n" + "=" * 70)
    print("🎉 All 3 NIDS datasets successfully generated in separate folders under data/!")
    print("=" * 70)


if __name__ == "__main__":
    main()
