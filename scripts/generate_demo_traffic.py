"""
scripts/generate_demo_traffic.py
================================
Generates a realistic, labeled SOC demo traffic feed (demo_traffic.jsonl)
from the validated benchmark dataset (data/cicids2017/cicids2017_benchmark.csv):
  60% Benign / NORMAL (600 flows)
  15% PortScan (150 flows)
  15% DoS + Brute Force (150 flows)
  10% Infiltration / Zero-Day (100 flows)
Total: 1,000 flows
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "data" / "cicids2017" / "cicids2017_benchmark.csv"
OUTPUT_PATH = ROOT_DIR / "demo_traffic.jsonl"

FEATURE_TO_RUST_MAP = {
    "Flow Duration": "flow_duration_micros",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_packets_per_sec",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_total",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_total",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_length",
    "Bwd Header Length": "bwd_header_length",
    "Fwd Packets/s": "fwd_packets_per_sec",
    "Bwd Packets/s": "bwd_packets_per_sec",
    "Min Packet Length": "min_packet_length",
    "Max Packet Length": "max_packet_length",
    "Packet Length Mean": "packet_length_mean",
    "Packet Length Std": "packet_length_std",
    "Packet Length Variance": "packet_length_variance",
    "FIN Flag Count": "fin_flag_count",
    "SYN Flag Count": "syn_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "URG Flag Count": "urg_flag_count",
    "CWE Flag Count": "cwe_flag_count",
    "ECE Flag Count": "ece_flag_count",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "average_packet_size",
    "Avg Fwd Segment Size": "avg_fwd_segment_size",
    "Avg Bwd Segment Size": "avg_bwd_segment_size",
    "Fwd Avg Bytes/Bulk": "fwd_avg_bytes_bulk",
    "Fwd Avg Packets/Bulk": "fwd_avg_packets_bulk",
    "Fwd Avg Bulk Rate": "fwd_avg_bulk_rate",
    "Bwd Avg Bytes/Bulk": "bwd_avg_bytes_bulk",
    "Bwd Avg Packets/Bulk": "bwd_avg_packets_bulk",
    "Bwd Avg Bulk Rate": "bwd_avg_bulk_rate",
    "Subflow Fwd Packets": "subflow_fwd_packets",
    "Subflow Fwd Bytes": "subflow_fwd_bytes",
    "Subflow Bwd Packets": "subflow_bwd_packets",
    "Subflow Bwd Bytes": "subflow_bwd_bytes",
    "Init_Win_bytes_forward": "init_win_bytes_forward",
    "Init_Win_bytes_backward": "init_win_bytes_backward",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "min_seg_size_forward": "min_seg_size_forward",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}


def assign_network_meta(label: str, idx: int, proto_val: float = 6.0) -> dict:
    """Generates realistic SOC demonstration network metadata."""
    proto_str = "TCP" if proto_val == 6.0 else ("UDP" if proto_val == 17.0 else "ICMP")

    if label in ("BENIGN", "Normal"):
        src_ip = f"192.168.10.{10 + (idx % 25)}"
        dst_ip = "192.168.10.50" if (idx % 2 == 0) else f"192.168.10.{51 + (idx % 3)}"
        src_port = 40000 + (idx * 17 % 20000)
        dst_port = 443 if (idx % 3 == 0) else (80 if (idx % 3 == 1) else 53)
    elif label == "PortScan":
        src_ip = "172.16.0.100"
        dst_ip = "192.168.10.50"
        src_port = 55000 + (idx % 1000)
        ports = [21, 22, 23, 25, 80, 110, 135, 139, 143, 443, 445, 1433, 3306, 3389, 8080]
        dst_port = ports[idx % len(ports)]
    elif label == "DoS":
        src_ip = f"172.16.0.{101 + (idx % 5)}"
        dst_ip = "192.168.10.50"
        src_port = 45000 + (idx * 31 % 15000)
        dst_port = 80 if (idx % 2 == 0) else 443
    elif label == "BruteForce":
        src_ip = "172.16.0.150"
        dst_ip = "192.168.10.22" if (idx % 2 == 0) else "192.168.10.21"
        dst_port = 22 if (idx % 2 == 0) else 21
        src_port = 51000 + (idx % 1000)
    elif "ZeroDay" in label or "Infiltration" in label:
        src_ip = "192.168.10.199"
        dst_ip = "192.168.10.51"
        src_port = 52000 + (idx % 1000)
        dst_port = 4444 if (idx % 2 == 0) else 1337
    else:
        src_ip = "192.168.10.15"
        dst_ip = "192.168.10.50"
        src_port = 48000 + idx
        dst_port = 80

    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": proto_str,
    }


def main():
    if not CSV_PATH.exists():
        print(f"Error: Benchmark dataset not found at {CSV_PATH}. Run python data/generate_datasets.py first!")
        return

    print(f"Reading benchmark dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    label_col = "Label" if "Label" in df.columns else "StandardLabel"

    df_norm = df[df[label_col].isin(["BENIGN", "Normal"])].sample(n=min(600, len(df[df[label_col].isin(["BENIGN", "Normal"])])), random_state=42)
    df_scan = df[df[label_col] == "PortScan"].sample(n=min(150, len(df[df[label_col] == "PortScan"])), random_state=42)
    df_dos = df[df[label_col] == "DoS"].sample(n=min(100, len(df[df[label_col] == "DoS"])), random_state=42)
    df_brute = df[df[label_col].isin(["BruteForce", "Brute Force"])].sample(n=min(50, len(df[df[label_col].isin(["BruteForce", "Brute Force"])])), random_state=42)
    df_infil = df[df[label_col].isin(["Infiltration_ZeroDay", "Infiltration"])].sample(n=min(100, len(df[df[label_col].isin(["Infiltration_ZeroDay", "Infiltration"])])), random_state=42)

    demo_df = pd.concat([df_norm, df_scan, df_dos, df_brute, df_infil]).sample(
        frac=1.0, random_state=42
    ).reset_index(drop=True)

    print(f"\nTotal sampled demo flows: {len(demo_df)}")
    print(demo_df[label_col].value_counts())

    base_time = datetime(2026, 11, 15, 10, 0, 0)
    current_time = base_time

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for idx, row in demo_df.iterrows():
            lbl = row[label_col]
            meta = assign_network_meta(lbl, idx)

            flow_dur_micros = float(row.get("Flow Duration", 0.0))
            flow_dur_secs = flow_dur_micros / 1e6

            delta_ms = np.random.randint(100, 500)
            current_time += timedelta(milliseconds=delta_ms)

            flow_rec = {
                "flow_id": f"demo-flow-{idx + 1:04d}",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "src_ip": meta["src_ip"],
                "dst_ip": meta["dst_ip"],
                "src_port": meta["src_port"],
                "dst_port": meta["dst_port"],
                "protocol": meta["protocol"],
                "flow_duration_micros": flow_dur_micros,
                "flow_duration_secs": round(flow_dur_secs, 4),
                "ground_truth_label": lbl,
            }

            for feat_col, rust_key in FEATURE_TO_RUST_MAP.items():
                flow_rec[rust_key] = float(row.get(feat_col, 0.0))

            for feat_col in FEATURE_TO_RUST_MAP.keys():
                flow_rec[feat_col] = float(row.get(feat_col, 0.0))

            out_f.write(json.dumps(flow_rec) + "\n")

    print(f"\nSuccessfully generated {len(demo_df)} records to {OUTPUT_PATH}!")


if __name__ == "__main__":
    main()
