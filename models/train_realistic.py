#!/usr/bin/env python3
"""
NIDS 2.0 synthetic demonstration trainer.
This script generates artificial flows and must not be used to claim real-world,
dataset benchmark, or production intrusion-detection performance.
"""

import os
# Threading safety for macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODELS_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using compute device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate Synthetic Demonstration Dataset with Gaussian Jitter
# ─────────────────────────────────────────────────────────────────────────────
def generate_realistic_dataset(n_samples=25000, seed=42):
    np.random.seed(seed)
    n_benign = int(n_samples * 0.60)     # 15,000
    n_dos = int(n_samples * 0.15)        # 3,750
    n_portscan = int(n_samples * 0.12)   # 3,000
    n_bruteforce = int(n_samples * 0.08) # 2,000
    n_zeroday = int(n_samples * 0.05)    # 1,250
    
    rows = []
    
    # 1. BENIGN Flows
    for i in range(n_benign):
        dur = np.random.uniform(500, 10_000_000)
        fwd_pkts = np.random.randint(1, 30)
        bwd_pkts = np.random.randint(1, 40)
        fwd_len = np.random.uniform(40, 1400)
        bwd_len = np.random.uniform(200, 1400)
        
        syn_flag = 1.0
        ack_flag = 1.0
        rst_flag = 0.0
        dst_port = int(np.random.choice([80, 443, 53, 8080]))
        
        # Realistic Edge Case A: 0.4% benign failed handshake / connection reset (looks like port probe)
        if i < int(n_benign * 0.004):
            fwd_pkts = 1
            bwd_pkts = 0
            dur = np.random.uniform(10, 500)
            syn_flag = 1.0
            ack_flag = 0.0
            rst_flag = 1.0
            dst_port = int(np.random.randint(1024, 65535))
        # Realistic Edge Case B: 0.3% legitimate admin SSH traffic on port 22
        elif i < int(n_benign * 0.007):
            dst_port = 22
            fwd_pkts = np.random.randint(8, 20)
            bwd_pkts = np.random.randint(6, 16)
            dur = np.random.uniform(500_000, 1_500_000)
            
        jitter = np.random.normal(1.0, 0.04)
        dur = max(10.0, dur * jitter)
        
        rows.append({
            'Label': 'BENIGN', 'Destination Port': dst_port,
            'Flow Duration': dur, 'Total Fwd Packets': fwd_pkts, 'Total Backward Packets': bwd_pkts,
            'Total Length of Fwd Packets': fwd_pkts * fwd_len, 'Total Length of Bwd Packets': bwd_pkts * bwd_len,
            'Fwd Packet Length Max': fwd_len * 1.5, 'Fwd Packet Length Min': 20.0,
            'Fwd Packet Length Mean': fwd_len, 'Fwd Packet Length Std': fwd_len * 0.4,
            'Bwd Packet Length Max': bwd_len * 1.5, 'Bwd Packet Length Min': 20.0,
            'Bwd Packet Length Mean': bwd_len, 'Bwd Packet Length Std': bwd_len * 0.4,
            'Flow Bytes/s': ((fwd_pkts * fwd_len) + (bwd_pkts * bwd_len)) / (dur / 1e6 + 1e-5),
            'Flow Packets/s': (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            'Flow IAT Mean': dur / max(1, fwd_pkts + bwd_pkts), 'Flow IAT Std': 500.0 * np.random.normal(1.0, 0.05),
            'Flow IAT Max': dur * 0.7, 'Flow IAT Min': 50.0,
            'Fwd IAT Total': dur, 'Fwd IAT Mean': dur / max(1, fwd_pkts), 'Fwd IAT Std': 200.0,
            'Fwd IAT Max': dur * 0.7, 'Fwd IAT Min': 50.0,
            'Bwd IAT Total': dur * 0.8, 'Bwd IAT Mean': dur / max(1, bwd_pkts), 'Bwd IAT Std': 200.0,
            'Bwd IAT Max': dur * 0.7, 'Bwd IAT Min': 50.0,
            'Fwd PSH Flags': float(np.random.choice([0, 1], p=[0.7, 0.3])), 'Bwd PSH Flags': 0.0,
            'Fwd URG Flags': 0.0, 'Bwd URG Flags': 0.0,
            'Fwd Header Length': fwd_pkts * 32, 'Bwd Header Length': bwd_pkts * 32,
            'Fwd Packets/s': fwd_pkts / (dur / 1e6 + 1e-5), 'Bwd Packets/s': bwd_pkts / (dur / 1e6 + 1e-5),
            'Min Packet Length': 20.0, 'Max Packet Length': max(fwd_len, bwd_len) * 1.5,
            'Packet Length Mean': (fwd_len + bwd_len) / 2.0, 'Packet Length Std': abs(bwd_len - fwd_len) * 0.5,
            'Packet Length Variance': (abs(bwd_len - fwd_len) * 0.5) ** 2,
            'FIN Flag Count': 1.0, 'SYN Flag Count': syn_flag, 'RST Flag Count': rst_flag,
            'PSH Flag Count': 1.0, 'ACK Flag Count': ack_flag, 'URG Flag Count': 0.0,
            'CWE Flag Count': 0.0, 'ECE Flag Count': 0.0, 'Down/Up Ratio': bwd_pkts / max(1, fwd_pkts),
            'Average Packet Size': (fwd_len + bwd_len) / 2.0, 'Avg Fwd Segment Size': fwd_len,
            'Avg Bwd Segment Size': bwd_len, 'Fwd Header Length.1': fwd_pkts * 32,
            'Fwd Avg Bytes/Bulk': 0.0, 'Fwd Avg Packets/Bulk': 0.0, 'Fwd Avg Bulk Rate': 0.0,
            'Bwd Avg Bytes/Bulk': 0.0, 'Bwd Avg Packets/Bulk': 0.0, 'Bwd Avg Bulk Rate': 0.0,
            'Subflow Fwd Packets': fwd_pkts, 'Subflow Fwd Bytes': fwd_pkts * fwd_len,
            'Subflow Bwd Packets': bwd_pkts, 'Subflow Bwd Bytes': bwd_pkts * bwd_len,
            'Init_Win_bytes_forward': 65535.0, 'Init_Win_bytes_backward': 65535.0,
            'act_data_pkt_fwd': max(0, fwd_pkts - 1), 'min_seg_size_forward': 32.0,
            'Active Mean': dur * 0.2 if dur > 5e6 else 0.0, 'Active Std': 0.0,
            'Active Max': dur * 0.2 if dur > 5e6 else 0.0, 'Active Min': dur * 0.2 if dur > 5e6 else 0.0,
            'Idle Mean': dur * 0.8 if dur > 5e6 else 0.0, 'Idle Std': 0.0,
            'Idle Max': dur * 0.8 if dur > 5e6 else 0.0, 'Idle Min': dur * 0.8 if dur > 5e6 else 0.0
        })
        
    # 2. DoS (SYN Flood)
    for i in range(n_dos):
        dur = np.random.uniform(500, 50_000)
        fwd_pkts = np.random.randint(50, 400)
        ack_flag = 0.0
        
        # Realistic Edge Case C: 0.5% ACK flood or low-rate SYN/ACK flood
        if i < int(n_dos * 0.005):
            ack_flag = 1.0
            fwd_pkts = np.random.randint(20, 50)
            
        jitter = np.random.normal(1.0, 0.03)
        dur = max(100.0, dur * jitter)
        
        rows.append({
            'Label': 'DoS', 'Destination Port': 80, 'Flow Duration': dur,
            'Total Fwd Packets': fwd_pkts, 'Total Backward Packets': 0,
            'Total Length of Fwd Packets': fwd_pkts * 54.0, 'Total Length of Bwd Packets': 0.0,
            'Fwd Packet Length Max': 64.0, 'Fwd Packet Length Min': 40.0, 'Fwd Packet Length Mean': 54.0, 'Fwd Packet Length Std': 2.0,
            'Bwd Packet Length Max': 0.0, 'Bwd Packet Length Min': 0.0, 'Bwd Packet Length Mean': 0.0, 'Bwd Packet Length Std': 0.0,
            'Flow Bytes/s': (fwd_pkts * 54.0) / (dur / 1e6 + 1e-5), 'Flow Packets/s': fwd_pkts / (dur / 1e6 + 1e-5),
            'Flow IAT Mean': dur / max(1, fwd_pkts), 'Flow IAT Std': 2.0, 'Flow IAT Max': 10.0, 'Flow IAT Min': 1.0,
            'Fwd IAT Total': dur, 'Fwd IAT Mean': dur / max(1, fwd_pkts), 'Fwd IAT Std': 2.0, 'Fwd IAT Max': 10.0, 'Fwd IAT Min': 1.0,
            'Bwd IAT Total': 0.0, 'Bwd IAT Mean': 0.0, 'Bwd IAT Std': 0.0, 'Bwd IAT Max': 0.0, 'Bwd IAT Min': 0.0,
            'Fwd PSH Flags': 0.0, 'Bwd PSH Flags': 0.0, 'Fwd URG Flags': 0.0, 'Bwd URG Flags': 0.0,
            'Fwd Header Length': fwd_pkts * 32, 'Bwd Header Length': 0,
            'Fwd Packets/s': fwd_pkts / (dur / 1e6 + 1e-5), 'Bwd Packets/s': 0.0,
            'Min Packet Length': 40.0, 'Max Packet Length': 64.0, 'Packet Length Mean': 54.0, 'Packet Length Std': 2.0,
            'Packet Length Variance': 4.0, 'FIN Flag Count': 0.0, 'SYN Flag Count': 1.0, 'RST Flag Count': 0.0,
            'PSH Flag Count': 0.0, 'ACK Flag Count': 0.0, 'URG Flag Count': 0.0, 'CWE Flag Count': 0.0, 'ECE Flag Count': 0.0,
            'Down/Up Ratio': 0.0, 'Average Packet Size': 54.0, 'Avg Fwd Segment Size': 54.0, 'Avg Bwd Segment Size': 0.0,
            'Fwd Header Length.1': fwd_pkts * 32, 'Fwd Avg Bytes/Bulk': 0.0, 'Fwd Avg Packets/Bulk': 0.0, 'Fwd Avg Bulk Rate': 0.0,
            'Bwd Avg Bytes/Bulk': 0.0, 'Bwd Avg Packets/Bulk': 0.0, 'Bwd Avg Bulk Rate': 0.0,
            'Subflow Fwd Packets': fwd_pkts, 'Subflow Fwd Bytes': fwd_pkts * 54.0, 'Subflow Bwd Packets': 0, 'Subflow Bwd Bytes': 0.0,
            'Init_Win_bytes_forward': 1024.0, 'Init_Win_bytes_backward': -1.0, 'act_data_pkt_fwd': 0, 'min_seg_size_forward': 32.0,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0.0, 'Active Min': 0.0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0.0, 'Idle Min': 0.0
        })
        
    # 3. PortScan
    for i in range(n_portscan):
        dur = np.random.uniform(10, 400)
        fwd_pkts = np.random.choice([1, 2], p=[0.8, 0.2])
        
        # 0.3% slow scan probe
        if i < int(n_portscan * 0.003):
            dur = np.random.uniform(2000, 15000)
            
        jitter = np.random.normal(1.0, 0.03)
        dur = max(5.0, dur * jitter)
        
        rows.append({
            'Label': 'PortScan', 'Destination Port': int(np.random.randint(1, 65535)), 'Flow Duration': dur,
            'Total Fwd Packets': fwd_pkts, 'Total Backward Packets': 0,
            'Total Length of Fwd Packets': fwd_pkts * 44.0, 'Total Length of Bwd Packets': 0.0,
            'Fwd Packet Length Max': 44.0, 'Fwd Packet Length Min': 44.0, 'Fwd Packet Length Mean': 44.0, 'Fwd Packet Length Std': 0.0,
            'Bwd Packet Length Max': 0.0, 'Bwd Packet Length Min': 0.0, 'Bwd Packet Length Mean': 0.0, 'Bwd Packet Length Std': 0.0,
            'Flow Bytes/s': (fwd_pkts * 44.0) / (dur / 1e6 + 1e-5), 'Flow Packets/s': fwd_pkts / (dur / 1e6 + 1e-5),
            'Flow IAT Mean': dur, 'Flow IAT Std': 0.0, 'Flow IAT Max': dur, 'Flow IAT Min': dur,
            'Fwd IAT Total': dur, 'Fwd IAT Mean': dur, 'Fwd IAT Std': 0.0, 'Fwd IAT Max': dur, 'Fwd IAT Min': dur,
            'Bwd IAT Total': 0.0, 'Bwd IAT Mean': 0.0, 'Bwd IAT Std': 0.0, 'Bwd IAT Max': 0.0, 'Bwd IAT Min': 0.0,
            'Fwd PSH Flags': 0.0, 'Bwd PSH Flags': 0.0, 'Fwd URG Flags': 0.0, 'Bwd URG Flags': 0.0,
            'Fwd Header Length': fwd_pkts * 32, 'Bwd Header Length': 0,
            'Fwd Packets/s': fwd_pkts / (dur / 1e6 + 1e-5), 'Bwd Packets/s': 0.0,
            'Min Packet Length': 44.0, 'Max Packet Length': 44.0, 'Packet Length Mean': 44.0, 'Packet Length Std': 0.0,
            'Packet Length Variance': 0.0, 'FIN Flag Count': 0.0, 'SYN Flag Count': 1.0, 'RST Flag Count': 1.0,
            'PSH Flag Count': 0.0, 'ACK Flag Count': 0.0, 'URG Flag Count': 0.0, 'CWE Flag Count': 0.0, 'ECE Flag Count': 0.0,
            'Down/Up Ratio': 0.0, 'Average Packet Size': 44.0, 'Avg Fwd Segment Size': 44.0, 'Avg Bwd Segment Size': 0.0,
            'Fwd Header Length.1': fwd_pkts * 32, 'Fwd Avg Bytes/Bulk': 0.0, 'Fwd Avg Packets/Bulk': 0.0, 'Fwd Avg Bulk Rate': 0.0,
            'Bwd Avg Bytes/Bulk': 0.0, 'Bwd Avg Packets/Bulk': 0.0, 'Bwd Avg Bulk Rate': 0.0,
            'Subflow Fwd Packets': fwd_pkts, 'Subflow Fwd Bytes': fwd_pkts * 44.0, 'Subflow Bwd Packets': 0, 'Subflow Bwd Bytes': 0.0,
            'Init_Win_bytes_forward': 1024.0, 'Init_Win_bytes_backward': -1.0, 'act_data_pkt_fwd': 0, 'min_seg_size_forward': 32.0,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0.0, 'Active Min': 0.0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0.0, 'Idle Min': 0.0
        })
        
    # 4. BruteForce
    for i in range(n_bruteforce):
        dur = np.random.uniform(500_000, 1_800_000)
        fwd_pkts = np.random.randint(8, 20)
        bwd_pkts = np.random.randint(6, 16)
        
        # 0.5% boundary brute-force
        if i < int(n_bruteforce * 0.005):
            fwd_pkts = np.random.randint(4, 7)
            bwd_pkts = np.random.randint(3, 5)
            
        jitter = np.random.normal(1.0, 0.03)
        dur = max(100.0, dur * jitter)
        
        rows.append({
            'Label': 'BruteForce', 'Destination Port': 22, 'Flow Duration': dur,
            'Total Fwd Packets': fwd_pkts, 'Total Backward Packets': bwd_pkts,
            'Total Length of Fwd Packets': fwd_pkts * 120.0, 'Total Length of Bwd Packets': bwd_pkts * 90.0,
            'Fwd Packet Length Max': 150.0, 'Fwd Packet Length Min': 40.0, 'Fwd Packet Length Mean': 120.0, 'Fwd Packet Length Std': 20.0,
            'Bwd Packet Length Max': 130.0, 'Bwd Packet Length Min': 40.0, 'Bwd Packet Length Mean': 90.0, 'Bwd Packet Length Std': 20.0,
            'Flow Bytes/s': ((fwd_pkts * 120.0) + (bwd_pkts * 90.0)) / (dur / 1e6 + 1e-5), 'Flow Packets/s': (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            'Flow IAT Mean': dur / max(1, fwd_pkts + bwd_pkts), 'Flow IAT Std': 80.0, 'Flow IAT Max': dur * 0.3, 'Flow IAT Min': 50.0,
            'Fwd IAT Total': dur, 'Fwd IAT Mean': dur / max(1, fwd_pkts), 'Fwd IAT Std': 50.0, 'Fwd IAT Max': dur * 0.3, 'Fwd IAT Min': 50.0,
            'Bwd IAT Total': dur * 0.9, 'Bwd IAT Mean': dur / max(1, bwd_pkts), 'Bwd IAT Std': 50.0, 'Bwd IAT Max': dur * 0.3, 'Bwd IAT Min': 50.0,
            'Fwd PSH Flags': 1.0, 'Bwd PSH Flags': 0.0, 'Fwd URG Flags': 0.0, 'Bwd URG Flags': 0.0,
            'Fwd Header Length': fwd_pkts * 32, 'Bwd Header Length': bwd_pkts * 32,
            'Fwd Packets/s': fwd_pkts / (dur / 1e6 + 1e-5), 'Bwd Packets/s': bwd_pkts / (dur / 1e6 + 1e-5),
            'Min Packet Length': 40.0, 'Max Packet Length': 150.0, 'Packet Length Mean': 105.0, 'Packet Length Std': 20.0,
            'Packet Length Variance': 400.0, 'FIN Flag Count': 1.0, 'SYN Flag Count': 1.0, 'RST Flag Count': 0.0,
            'PSH Flag Count': 1.0, 'ACK Flag Count': 1.0, 'URG Flag Count': 0.0, 'CWE Flag Count': 0.0, 'ECE Flag Count': 0.0,
            'Down/Up Ratio': bwd_pkts / max(1, fwd_pkts), 'Average Packet Size': 105.0, 'Avg Fwd Segment Size': 120.0, 'Avg Bwd Segment Size': 90.0,
            'Fwd Header Length.1': fwd_pkts * 32, 'Fwd Avg Bytes/Bulk': 0.0, 'Fwd Avg Packets/Bulk': 0.0, 'Fwd Avg Bulk Rate': 0.0,
            'Bwd Avg Bytes/Bulk': 0.0, 'Bwd Avg Packets/Bulk': 0.0, 'Bwd Avg Bulk Rate': 0.0,
            'Subflow Fwd Packets': fwd_pkts, 'Subflow Fwd Bytes': fwd_pkts * 120.0, 'Subflow Bwd Packets': bwd_pkts, 'Subflow Bwd Bytes': bwd_pkts * 90.0,
            'Init_Win_bytes_forward': 65535.0, 'Init_Win_bytes_backward': 65535.0, 'act_data_pkt_fwd': fwd_pkts - 2, 'min_seg_size_forward': 32.0,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0.0, 'Active Min': 0.0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0.0, 'Idle Min': 0.0
        })
        
    # 5. Infiltration_ZeroDay
    for _ in range(n_zeroday):
        dur = np.random.uniform(50_000, 2_000_000)
        fwd_pkts = np.random.randint(20, 50)
        bwd_pkts = np.random.randint(1, 3)
        rows.append({
            'Label': 'Infiltration_ZeroDay', 'Destination Port': 4444, 'Flow Duration': dur,
            'Total Fwd Packets': fwd_pkts, 'Total Backward Packets': bwd_pkts,
            'Total Length of Fwd Packets': fwd_pkts * 1200.0, 'Total Length of Bwd Packets': bwd_pkts * 40.0,
            'Fwd Packet Length Max': 1480.0, 'Fwd Packet Length Min': 200.0, 'Fwd Packet Length Mean': 1200.0, 'Fwd Packet Length Std': 250.0,
            'Bwd Packet Length Max': 50.0, 'Bwd Packet Length Min': 40.0, 'Bwd Packet Length Mean': 40.0, 'Bwd Packet Length Std': 5.0,
            'Flow Bytes/s': ((fwd_pkts * 1200.0) + (bwd_pkts * 40.0)) / (dur / 1e6 + 1e-5), 'Flow Packets/s': (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            'Flow IAT Mean': dur / max(1, fwd_pkts + bwd_pkts), 'Flow IAT Std': (dur / max(1, fwd_pkts + bwd_pkts)) * 2.5,
            'Flow IAT Max': dur * 0.9, 'Flow IAT Min': 1.0,
            'Fwd IAT Total': dur, 'Fwd IAT Mean': dur / max(1, fwd_pkts), 'Fwd IAT Std': (dur / max(1, fwd_pkts)) * 2.0,
            'Fwd IAT Max': dur * 0.9, 'Fwd IAT Min': 1.0,
            'Bwd IAT Total': dur * 0.4, 'Bwd IAT Mean': dur / max(1, bwd_pkts), 'Bwd IAT Std': 10.0, 'Bwd IAT Max': dur * 0.4, 'Bwd IAT Min': 10.0,
            'Fwd PSH Flags': 1.0, 'Bwd PSH Flags': 0.0, 'Fwd URG Flags': 1.0, 'Bwd URG Flags': 0.0,
            'Fwd Header Length': fwd_pkts * 32, 'Bwd Header Length': bwd_pkts * 32,
            'Fwd Packets/s': fwd_pkts / (dur / 1e6 + 1e-5), 'Bwd Packets/s': bwd_pkts / (dur / 1e6 + 1e-5),
            'Min Packet Length': 40.0, 'Max Packet Length': 1480.0, 'Packet Length Mean': 1100.0, 'Packet Length Std': 300.0,
            'Packet Length Variance': 90000.0, 'FIN Flag Count': 0.0, 'SYN Flag Count': 1.0, 'RST Flag Count': 0.0,
            'PSH Flag Count': 1.0, 'ACK Flag Count': 1.0, 'URG Flag Count': 1.0, 'CWE Flag Count': 0.0, 'ECE Flag Count': 0.0,
            'Down/Up Ratio': bwd_pkts / max(1, fwd_pkts), 'Average Packet Size': 1100.0, 'Avg Fwd Segment Size': 1200.0, 'Avg Bwd Segment Size': 40.0,
            'Fwd Header Length.1': fwd_pkts * 32, 'Fwd Avg Bytes/Bulk': 1000.0, 'Fwd Avg Packets/Bulk': 2.0, 'Fwd Avg Bulk Rate': 500.0,
            'Bwd Avg Bytes/Bulk': 0.0, 'Bwd Avg Packets/Bulk': 0.0, 'Bwd Avg Bulk Rate': 0.0,
            'Subflow Fwd Packets': fwd_pkts, 'Subflow Fwd Bytes': fwd_pkts * 1200.0, 'Subflow Bwd Packets': bwd_pkts, 'Subflow Bwd Bytes': bwd_pkts * 40.0,
            'Init_Win_bytes_forward': 4096.0, 'Init_Win_bytes_backward': 4096.0, 'act_data_pkt_fwd': fwd_pkts - 1, 'min_seg_size_forward': 32.0,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0.0, 'Active Min': 0.0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0.0, 'Idle Min': 0.0
        })
        
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

print("⏳ Generating synthetic demonstration flows (not a real dataset)...")
df_raw = generate_realistic_dataset(n_samples=25000, seed=42)
df_raw.to_csv(DATA_DIR / "cicids2017_benchmark.csv", index=False)
print("✅ Saved to data/cicids2017_benchmark.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering (13 Derived Interaction Features)
# ─────────────────────────────────────────────────────────────────────────────
def add_engineered_features(df):
    d = df.copy()
    fwd_pkts = d['Total Fwd Packets'].astype(float)
    bwd_pkts = d['Total Backward Packets'].astype(float)
    fwd_bytes = d['Total Length of Fwd Packets'].astype(float)
    bwd_bytes = d['Total Length of Bwd Packets'].astype(float)
    tot_pkts = (fwd_pkts + bwd_pkts).replace(0, 1.0)
    tot_bytes = (fwd_bytes + bwd_bytes).replace(0, 1.0)
    
    d['fwd_bwd_pkt_ratio'] = fwd_pkts / (bwd_pkts + 1.0)
    d['fwd_bwd_byte_ratio'] = fwd_bytes / (bwd_bytes + 1.0)
    d['pkt_len_skew_proxy'] = (d['Packet Length Mean'].astype(float) - d['Min Packet Length'].astype(float)) / (d['Packet Length Std'].astype(float) + 1.0)
    d['byte_per_packet'] = tot_bytes / tot_pkts
    tot_flags = (d['FIN Flag Count'] + d['SYN Flag Count'] + d['RST Flag Count'] + d['PSH Flag Count'] + d['ACK Flag Count'] + d['URG Flag Count']).astype(float)
    d['flag_density'] = tot_flags / tot_pkts
    d['fwd_iat_jitter'] = d['Fwd IAT Std'].astype(float) / (d['Fwd IAT Mean'].astype(float) + 1.0)
    d['flow_pkt_rate_log'] = np.log1p(np.maximum(0, d['Flow Packets/s'].astype(float)))
    d['flow_byte_rate_log'] = np.log1p(np.maximum(0, d['Flow Bytes/s'].astype(float)))
    d['fwd_pkt_len_spread'] = d['Fwd Packet Length Max'].astype(float) - d['Fwd Packet Length Min'].astype(float)
    d['bwd_pkt_len_spread'] = d['Bwd Packet Length Max'].astype(float) - d['Bwd Packet Length Min'].astype(float)
    d['fwd_pkt_ratio'] = fwd_pkts / tot_pkts
    d['bwd_pkt_ratio'] = bwd_pkts / tot_pkts
    d['active_idle_ratio'] = d['Active Mean'].astype(float) / (d['Idle Mean'].astype(float) + 1.0)
    return d

df_eng = add_engineered_features(df_raw)
feature_cols = [c for c in df_eng.columns if c != 'Label']
df_eng[feature_cols] = df_eng[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

with open(MODELS_DIR / "feature_list.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"✅ Schema saved: {len(feature_cols)} features")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Train/Test Split (Zero-Day Held-out)
# ─────────────────────────────────────────────────────────────────────────────
df_known = df_eng[df_eng['Label'] != 'Infiltration_ZeroDay'].copy()
df_zeroday = df_eng[df_eng['Label'] == 'Infiltration_ZeroDay'].copy()

label_mapping = {'BENIGN': 0, 'DoS': 1, 'PortScan': 2, 'BruteForce': 3}
X_known = df_known[feature_cols].values
y_known = np.array([label_mapping[lbl] for lbl in df_known['Label'].values])

X_train, X_test, y_train, y_test = train_test_split(
    X_known, y_known, test_size=0.20, random_state=42, stratify=y_known
)

benign_train_idx = (y_train == label_mapping['BENIGN'])
scaler = RobustScaler(quantile_range=(5.0, 95.0))
scaler.fit(X_train[benign_train_idx])

X_train_scaled = np.clip(scaler.transform(X_train), -10.0, 10.0)
X_test_scaled = np.clip(scaler.transform(X_test), -10.0, 10.0)
X_zeroday_scaled = np.clip(scaler.transform(df_zeroday[feature_cols].values), -10.0, 10.0)

with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train Regularized LightGBM (Tier-1)
# ─────────────────────────────────────────────────────────────────────────────
lgb_train = lgb.Dataset(X_train_scaled, label=y_train, feature_name=feature_cols)
lgb_test = lgb.Dataset(X_test_scaled, label=y_test, reference=lgb_train, feature_name=feature_cols)

params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'learning_rate': 0.08,
    'num_leaves': 25,
    'max_depth': 6,
    'min_child_samples': 45,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'reg_alpha': 0.4,
    'reg_lambda': 0.8,
    'verbose': -1,
    'random_state': 42
}

lgbm_model = lgb.train(params, lgb_train, num_boost_round=90, valid_sets=[lgb_test])
y_test_pred_prob = lgbm_model.predict(X_test_scaled)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

lgb_acc = accuracy_score(y_test, y_test_pred) * 100.0
print("\n" + "=" * 60)
print(f"Synthetic demonstration split accuracy (not real-traffic validation): {lgb_acc:.2f}%")
print("=" * 60)
print(classification_report(y_test, y_test_pred, target_names=list(label_mapping.keys()), digits=4))

with open(MODELS_DIR / "lightgbm_model.pkl", "wb") as f:
    pickle.dump(lgbm_model, f)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train Deep Autoencoder (Tier-2)
# ─────────────────────────────────────────────────────────────────────────────
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim=91, bottleneck_dim=16):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

X_benign_train = torch.tensor(X_train_scaled[benign_train_idx], dtype=torch.float32)
X_benign_test = torch.tensor(X_test_scaled[y_test == label_mapping['BENIGN']], dtype=torch.float32)

model_ae = DeepAutoencoder(input_dim=len(feature_cols), bottleneck_dim=16).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model_ae.parameters(), lr=1e-3, weight_decay=1e-5)

batch_size = 128
epochs = 50
n_samples = len(X_benign_train)
loss_history = []

print("🚀 Training Deep Autoencoder on Benign Normal Manifold...")
for epoch in range(1, epochs + 1):
    model_ae.train()
    perm = torch.randperm(n_samples)
    epoch_loss = 0.0
    for i in range(0, n_samples, batch_size):
        indices = perm[i:i + batch_size]
        batch_x = X_benign_train[indices].to(device)
        optimizer.zero_grad()
        recon, _ = model_ae(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_x)
    epoch_loss /= n_samples
    loss_history.append(epoch_loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Reconstruction MSE Loss: {epoch_loss:.6f}")

model_ae.eval()
with torch.no_grad():
    recon_val, latent_benign = model_ae(X_benign_test.to(device))
    err_benign = torch.mean((X_benign_test.to(device) - recon_val) ** 2, dim=1).cpu().numpy()

threshold = float(np.percentile(err_benign, 99.0))
print(f"✅ Operating Anomaly Threshold (tau) calibrated at 99.0th percentile: {threshold:.4f}")

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
kmeans.fit(latent_benign.cpu().numpy())
prototypes = kmeans.cluster_centers_.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluate Zero-Day Quarantine
# ─────────────────────────────────────────────────────────────────────────────
with torch.no_grad():
    x_zero_t = torch.tensor(X_zeroday_scaled, dtype=torch.float32).to(device)
    recon_zero, _ = model_ae(x_zero_t)
    err_zero = torch.mean((x_zero_t - recon_zero) ** 2, dim=1).cpu().numpy()

quarantine_rate = float(np.mean(err_zero >= threshold)) * 100.0
print(f"Synthetic zero-day demonstration flag rate (not zero-day validation): {quarantine_rate:.2f}%")

torch.save(model_ae.state_dict(), MODELS_DIR / "autoencoder.pt")

metadata = {
    "version": "2.0.0",
    "input_features_count": len(feature_cols),
    "anomaly_threshold": round(threshold, 4),
    "evaluation_scope": "synthetic-generated-demonstration-only",
    "real_traffic_metrics_validated": False,
    "synthetic_demo_zero_day_flag_rate": round(quarantine_rate, 2),
    "synthetic_demo_test_accuracy": round(lgb_acc, 2),
    "label_mapping": label_mapping,
    "prototypes": prototypes
}
with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Render Realistic Evaluation Dashboard
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.patch.set_facecolor('#0b0e14')

for ax in axes:
    ax.set_facecolor('#11151c')

fig.suptitle(f'NIDS 2.0 — Synthetic Demonstration (Split Accuracy: {lgb_acc:.2f}%; Not Real-Traffic Validation)',
             fontsize=14, fontweight='bold', color='#e2e8f0', y=1.03)

# Panel 1: Loss Curve
epochs_range = list(range(1, len(loss_history) + 1))
axes[0].plot(epochs_range, loss_history, color='#3b82f6', linewidth=2.2, label='Train MSE Loss')
axes[0].fill_between(epochs_range, loss_history, alpha=0.18, color='#3b82f6')
axes[0].set_title('Autoencoder Training Convergence', fontsize=12, fontweight='bold', color='#f1f5f9')
axes[0].set_xlabel('Epoch', color='#94a3b8')
axes[0].set_ylabel('Reconstruction MSE Loss', color='#94a3b8')
axes[0].legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155')
axes[0].grid(True, alpha=0.15)

# Panel 2: Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False,
            annot_kws={'size': 11, 'fontweight': 'bold'},
            xticklabels=list(label_mapping.keys()), yticklabels=list(label_mapping.keys()))
axes[1].set_title(f'LightGBM Confusion Matrix ({lgb_acc:.2f}% Acc)', fontsize=12, fontweight='bold', color='#f1f5f9')
axes[1].set_xlabel('Predicted Label', color='#94a3b8')
axes[1].set_ylabel('True Label', color='#94a3b8')

# Panel 3: Separation Histogram
axes[2].hist(err_benign, bins=50, alpha=0.7, color='#10b981', label=f'Benign (mean={np.mean(err_benign):.4f})', density=True)
axes[2].hist(err_zero, bins=50, alpha=0.7, color='#a855f7', label=f'Synthetic zero-day demo (mean={np.mean(err_zero):.4f})', density=True)
axes[2].axvline(threshold, color='#ef4444', linestyle='--', linewidth=2.2, label=f'τ = {threshold:.4f}')
axes[2].set_title(f'Synthetic demo flag rate ({quarantine_rate:.1f}%; not validation)', fontsize=12, fontweight='bold', color='#f1f5f9')
axes[2].set_xlabel('Reconstruction MSE Loss', color='#94a3b8')
axes[2].set_ylabel('Density', color='#94a3b8')
axes[2].legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155')
axes[2].grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'evaluation_dashboard.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved synthetic demonstration dashboard; split accuracy={lgb_acc:.2f}% is not real-traffic validation.")
print("Synthetic demonstration training completed.")
