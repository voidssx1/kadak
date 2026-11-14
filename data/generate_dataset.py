"""
data/generate_dataset.py
========================
Generates a realistic, statistically calibrated CICIDS2017 benchmark dataset
for training and evaluating the NIDS Dual-Tier ML Architecture.

Features:
  - 77 Raw CICIDS2017 schema flow metrics (durations, IATs, packet lengths, TCP flags)
  - 5 Distinct Traffic Classes:
      1. BENIGN / NORMAL (60%): Diverse realistic browsing, DNS, streaming, normal variance
      2. DoS-SYNFlood (15%): High packet rates, tiny packet lengths, extreme SYN flag density
      3. PortScan (12%): High destination port range, 1-2 packets per flow, rapid pacing
      4. BruteForce (8%): Fixed-length authentication bursts on port 22/80, regular IAT
      5. Infiltration-ZeroDay (5%): Novel subspace anomaly with unseen timing distributions
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# The 77 standard CICIDS2017 feature names matching industry tooling
RAW_77_FEATURES = [
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

def generate_synthetic_cicids2017(n_samples: int = 25000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # Class Distribution
    n_benign = int(n_samples * 0.60)
    n_dos = int(n_samples * 0.15)
    n_portscan = int(n_samples * 0.12)
    n_bruteforce = int(n_samples * 0.08)
    n_zeroday = n_samples - (n_benign + n_dos + n_portscan + n_bruteforce)
    
    rows = []
    
    # 1. BENIGN / NORMAL (Diverse, realistic distributions)
    for _ in range(n_benign):
        dur = np.random.exponential(1_500_000) + 1000  # microseconds
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
            "Flow IAT Max": dur * np.random.uniform(0.4, 0.8),
            "Flow IAT Min": np.random.uniform(50, 500),
            "Fwd IAT Total": dur * np.random.uniform(0.8, 1.0),
            "Fwd IAT Mean": dur / max(1, fwd_pkts),
            "Fwd IAT Std": (dur / max(1, fwd_pkts)) * 0.5,
            "Fwd IAT Max": dur * 0.7,
            "Fwd IAT Min": np.random.uniform(50, 500),
            "Bwd IAT Total": dur * np.random.uniform(0.7, 0.95),
            "Bwd IAT Mean": dur / max(1, bwd_pkts),
            "Bwd IAT Std": (dur / max(1, bwd_pkts)) * 0.5,
            "Bwd IAT Max": dur * 0.7,
            "Bwd IAT Min": np.random.uniform(50, 500),
            "Fwd PSH Flags": float(np.random.choice([0, 1], p=[0.7, 0.3])),
            "Bwd PSH Flags": 0.0,
            "Fwd URG Flags": 0.0,
            "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32,
            "Bwd Header Length": bwd_pkts * 32,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5),
            "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-5),
            "Min Packet Length": 20.0,
            "Max Packet Length": max(bwd_len_mean * 1.5, fwd_len_mean * 1.5),
            "Packet Length Mean": (fwd_len_mean + bwd_len_mean) / 2.0,
            "Packet Length Std": abs(bwd_len_mean - fwd_len_mean) * 0.5,
            "Packet Length Variance": (abs(bwd_len_mean - fwd_len_mean) * 0.5) ** 2,
            "FIN Flag Count": float(np.random.choice([0, 1], p=[0.4, 0.6])),
            "SYN Flag Count": float(np.random.choice([0, 1], p=[0.2, 0.8])),
            "RST Flag Count": float(np.random.choice([0, 1], p=[0.95, 0.05])),
            "PSH Flag Count": float(np.random.choice([0, 1], p=[0.6, 0.4])),
            "ACK Flag Count": float(np.random.choice([0, 1], p=[0.1, 0.9])),
            "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0,
            "ECE Flag Count": 0.0,
            "Down/Up Ratio": bwd_pkts / max(1, fwd_pkts),
            "Average Packet Size": (fwd_len_mean + bwd_len_mean) / 2.0,
            "Avg Fwd Segment Size": fwd_len_mean,
            "Avg Bwd Segment Size": bwd_len_mean,
            "Fwd Header Length.1": fwd_pkts * 32,
            "Fwd Avg Bytes/Bulk": 0.0, "Fwd Avg Packets/Bulk": 0.0, "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_pkts,
            "Subflow Fwd Bytes": fwd_pkts * fwd_len_mean,
            "Subflow Bwd Packets": bwd_pkts,
            "Subflow Bwd Bytes": bwd_pkts * bwd_len_mean,
            "Init_Win_bytes_forward": float(np.random.choice([29200, 65535, 14600])),
            "Init_Win_bytes_backward": float(np.random.choice([29200, 65535, 14600])),
            "act_data_pkt_fwd": max(0, fwd_pkts - 1),
            "min_seg_size_forward": 20.0,
            "Active Mean": dur * 0.2 if dur > 5_000_000 else 0.0,
            "Active Std": 0.0, "Active Max": dur * 0.2 if dur > 5_000_000 else 0.0, "Active Min": dur * 0.2 if dur > 5_000_000 else 0.0,
            "Idle Mean": dur * 0.8 if dur > 5_000_000 else 0.0,
            "Idle Std": 0.0, "Idle Max": dur * 0.8 if dur > 5_000_000 else 0.0, "Idle Min": dur * 0.8 if dur > 5_000_000 else 0.0
        }
        rows.append(row)
        
    # 2. DoS-SYNFlood (Extreme packet rate, tiny packets, SYN count = 1, ACK count = 0)
    for _ in range(n_dos):
        dur = np.random.uniform(500, 50_000)  # very short burst
        fwd_pkts = np.random.randint(50, 500)
        bwd_pkts = 0  # Victim never responds or dropped
        fwd_len_mean = np.random.uniform(40, 64)  # pure headers, zero payload
        
        row = {
            "Label": "DoS",
            "Destination Port": 80,
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": 0,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len_mean,
            "Total Length of Bwd Packets": 0.0,
            "Fwd Packet Length Max": 64.0, "Fwd Packet Length Min": 40.0,
            "Fwd Packet Length Mean": fwd_len_mean, "Fwd Packet Length Std": 2.0,
            "Bwd Packet Length Max": 0.0, "Bwd Packet Length Min": 0.0,
            "Bwd Packet Length Mean": 0.0, "Bwd Packet Length Std": 0.0,
            "Flow Bytes/s": (fwd_pkts * fwd_len_mean) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": fwd_pkts / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, fwd_pkts),
            "Flow IAT Std": 5.0, "Flow IAT Max": 20.0, "Flow IAT Min": 1.0,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur / max(1, fwd_pkts),
            "Fwd IAT Std": 5.0, "Fwd IAT Max": 20.0, "Fwd IAT Min": 1.0,
            "Bwd IAT Total": 0.0, "Bwd IAT Mean": 0.0, "Bwd IAT Std": 0.0,
            "Bwd IAT Max": 0.0, "Bwd IAT Min": 0.0,
            "Fwd PSH Flags": 0.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": 0,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": 0.0,
            "Min Packet Length": 40.0, "Max Packet Length": 64.0,
            "Packet Length Mean": fwd_len_mean, "Packet Length Std": 2.0,
            "Packet Length Variance": 4.0,
            "FIN Flag Count": 0.0, "SYN Flag Count": 1.0, "RST Flag Count": 0.0,
            "PSH Flag Count": 0.0, "ACK Flag Count": 0.0, "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0, "ECE Flag Count": 0.0,
            "Down/Up Ratio": 0.0, "Average Packet Size": fwd_len_mean,
            "Avg Fwd Segment Size": fwd_len_mean, "Avg Bwd Segment Size": 0.0,
            "Fwd Header Length.1": fwd_pkts * 32,
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

    # 3. PortScan (Scans across destination ports, 1 or 2 packets per flow, RST or ICMP response)
    for _ in range(n_portscan):
        dur = np.random.uniform(10, 500)  # Micro-duration
        fwd_pkts = np.random.choice([1, 2], p=[0.8, 0.2])
        bwd_pkts = np.random.choice([0, 1], p=[0.5, 0.5])
        port = np.random.randint(1, 65535)
        
        row = {
            "Label": "PortScan",
            "Destination Port": port,
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_pkts * 44.0,
            "Total Length of Bwd Packets": bwd_pkts * 40.0,
            "Fwd Packet Length Max": 44.0, "Fwd Packet Length Min": 44.0,
            "Fwd Packet Length Mean": 44.0, "Fwd Packet Length Std": 0.0,
            "Bwd Packet Length Max": 40.0 if bwd_pkts else 0.0, "Bwd Packet Length Min": 40.0 if bwd_pkts else 0.0,
            "Bwd Packet Length Mean": 40.0 if bwd_pkts else 0.0, "Bwd Packet Length Std": 0.0,
            "Flow Bytes/s": ((fwd_pkts * 44.0) + (bwd_pkts * 40.0)) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur, "Flow IAT Std": 0.0, "Flow IAT Max": dur, "Flow IAT Min": dur,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur, "Fwd IAT Std": 0.0, "Fwd IAT Max": dur, "Fwd IAT Min": dur,
            "Bwd IAT Total": 0.0, "Bwd IAT Mean": 0.0, "Bwd IAT Std": 0.0, "Bwd IAT Max": 0.0, "Bwd IAT Min": 0.0,
            "Fwd PSH Flags": 0.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": bwd_pkts * 32,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-5),
            "Min Packet Length": 40.0 if bwd_pkts else 44.0, "Max Packet Length": 44.0,
            "Packet Length Mean": 42.0 if bwd_pkts else 44.0, "Packet Length Std": 2.0 if bwd_pkts else 0.0,
            "Packet Length Variance": 4.0 if bwd_pkts else 0.0,
            "FIN Flag Count": 0.0, "SYN Flag Count": 1.0, "RST Flag Count": float(bwd_pkts),
            "PSH Flag Count": 0.0, "ACK Flag Count": float(bwd_pkts), "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0, "ECE Flag Count": 0.0,
            "Down/Up Ratio": bwd_pkts / fwd_pkts, "Average Packet Size": 42.0,
            "Avg Fwd Segment Size": 44.0, "Avg Bwd Segment Size": 40.0 if bwd_pkts else 0.0,
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

    # 4. BruteForce (Targeting Port 22 SSH or 80 HTTP, fixed payload length, rhythmic IAT)
    for _ in range(n_bruteforce):
        dur = np.random.uniform(500_000, 2_000_000)
        fwd_pkts = np.random.randint(8, 25)
        bwd_pkts = np.random.randint(6, 20)
        fwd_len_mean = 120.0  # Password attempt payload
        bwd_len_mean = 95.0   # Failure response
        
        row = {
            "Label": "BruteForce",
            "Destination Port": 22,
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len_mean,
            "Total Length of Bwd Packets": bwd_pkts * bwd_len_mean,
            "Fwd Packet Length Max": 160.0, "Fwd Packet Length Min": 40.0,
            "Fwd Packet Length Mean": fwd_len_mean, "Fwd Packet Length Std": 30.0,
            "Bwd Packet Length Max": 140.0, "Bwd Packet Length Min": 40.0,
            "Bwd Packet Length Mean": bwd_len_mean, "Bwd Packet Length Std": 25.0,
            "Flow Bytes/s": ((fwd_pkts * fwd_len_mean) + (bwd_pkts * bwd_len_mean)) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, (fwd_pkts + bwd_pkts - 1)),
            "Flow IAT Std": 100.0,  # extremely rhythmic timing
            "Flow IAT Max": (dur / max(1, (fwd_pkts + bwd_pkts - 1))) * 1.1,
            "Flow IAT Min": (dur / max(1, (fwd_pkts + bwd_pkts - 1))) * 0.9,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur / max(1, fwd_pkts),
            "Fwd IAT Std": 50.0, "Fwd IAT Max": (dur / max(1, fwd_pkts)) * 1.1,
            "Fwd IAT Min": (dur / max(1, fwd_pkts)) * 0.9,
            "Bwd IAT Total": dur * 0.9, "Bwd IAT Mean": dur / max(1, bwd_pkts),
            "Bwd IAT Std": 50.0, "Bwd IAT Max": (dur / max(1, bwd_pkts)) * 1.1,
            "Bwd IAT Min": (dur / max(1, bwd_pkts)) * 0.9,
            "Fwd PSH Flags": 1.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": bwd_pkts * 32,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-5),
            "Min Packet Length": 40.0, "Max Packet Length": 160.0,
            "Packet Length Mean": (fwd_len_mean + bwd_len_mean) / 2.0,
            "Packet Length Std": 28.0, "Packet Length Variance": 784.0,
            "FIN Flag Count": 1.0, "SYN Flag Count": 1.0, "RST Flag Count": 0.0,
            "PSH Flag Count": 1.0, "ACK Flag Count": 1.0, "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0, "ECE Flag Count": 0.0,
            "Down/Up Ratio": bwd_pkts / fwd_pkts, "Average Packet Size": (fwd_len_mean + bwd_len_mean) / 2.0,
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

    # 5. Infiltration-ZeroDay (Out-of-distribution manifold departure, unusual ratios and timing)
    for _ in range(n_zeroday):
        dur = np.random.uniform(10_000, 3_000_000)
        fwd_pkts = np.random.randint(15, 60)
        bwd_pkts = np.random.randint(1, 4)  # High asymmetric exfiltration
        fwd_len_mean = np.random.uniform(900, 1480)  # Heavy unusual staging payloads
        bwd_len_mean = 40.0
        
        row = {
            "Label": "Infiltration_ZeroDay",
            "Destination Port": np.random.choice([4444, 1337, 8443, 31337]),
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len_mean,
            "Total Length of Bwd Packets": bwd_pkts * bwd_len_mean,
            "Fwd Packet Length Max": 1500.0, "Fwd Packet Length Min": 200.0,
            "Fwd Packet Length Mean": fwd_len_mean, "Fwd Packet Length Std": 300.0,
            "Bwd Packet Length Max": 50.0, "Bwd Packet Length Min": 40.0,
            "Bwd Packet Length Mean": bwd_len_mean, "Bwd Packet Length Std": 5.0,
            "Flow Bytes/s": ((fwd_pkts * fwd_len_mean) + (bwd_pkts * bwd_len_mean)) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, (fwd_pkts + bwd_pkts - 1)),
            "Flow IAT Std": (dur / max(1, (fwd_pkts + bwd_pkts - 1))) * 2.5,  # Extreme timing jitter
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
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_file_csv = out_dir / "cicids2017_benchmark.csv"
    out_file_parquet = out_dir / "cicids2017_benchmark.parquet"
    print(f"[*] Generating benchmark dataset with 25,000 flows...")
    df = generate_synthetic_cicids2017(n_samples=25000)
    df.to_csv(out_file_csv, index=False)
    print(f"[+] CSV Dataset saved successfully -> {out_file_csv}")
    try:
        df.to_parquet(out_file_parquet, index=False)
        print(f"[+] Parquet Dataset saved successfully -> {out_file_parquet}")
    except Exception as e:
        print(f"[*] Parquet optional export skipped: {e}")
    print(f"[+] Class distribution:\n{df['Label'].value_counts()}")

