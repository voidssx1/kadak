"""
backend/adapter.py
==================
Adapter bridge connecting:
  Rust Flow Record (JSON / JSONL) or Web Flow Payload
        ↓
  RAW_77_FEATURES (CICIDS2017 schema with semantic normalization)
        ↓
  14 Engineered Interaction Features (Ratios, Densities, Jitters, Rates)
        ↓
  91-feature canonical input aligned with models/feature_list.json
        ↓
  RobustScaler [-10, 10] normalized vector
"""

import os
# Fix for OpenMP crashes across various environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("nids.adapter")


def parse_protocol(raw: Dict[str, Any]) -> float:
    """Normalize protocol representation to integer numeric value (6=TCP, 17=UDP, 1=ICMP)."""
    if "Protocol" in raw:
        try:
            return float(raw["Protocol"])
        except (ValueError, TypeError):
            pass

    proto_str = str(raw.get("protocol", "")).strip().upper()
    unsw_proto = str(raw.get("unsw_proto", "")).strip()

    if proto_str == "TCP" or unsw_proto == "6":
        return 6.0
    if proto_str == "UDP" or unsw_proto == "17":
        return 17.0
    if "ICMP" in proto_str or unsw_proto in ("1", "58"):
        return 1.0
    if unsw_proto.isdigit():
        return float(unsw_proto)
    return 6.0  # Default to TCP


def _clip_flag(val: Any) -> float:
    """Clips TCP flag count to binary 0.0 or 1.0 matching CICIDS2017 training semantics."""
    try:
        f = float(val)
        return 1.0 if f > 0.0 else 0.0
    except (ValueError, TypeError):
        return 0.0


def normalize_flow_dict(flow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a flow dictionary coming from Rust engine, replay file, or API payload
    into the canonical 77-feature CICIDS2017 schema with correct casing and types.
    """
    duration_micros = float(flow.get("flow_duration_micros", flow.get("Flow Duration", 0.0)))

    # CICFlowMeter semantics: flows shorter than 5s (5,000,000 micros) have 0 active/idle
    if duration_micros < 5_000_000:
        active_mean, active_std, active_max, active_min = 0.0, 0.0, 0.0, 0.0
        idle_mean, idle_std, idle_max, idle_min = 0.0, 0.0, 0.0, 0.0
    else:
        active_mean = float(flow.get("active_mean", flow.get("Active Mean", 0.0)))
        active_std = float(flow.get("active_std", flow.get("Active Std", 0.0)))
        active_max = float(flow.get("active_max", flow.get("Active Max", 0.0)))
        active_min = float(flow.get("active_min", flow.get("Active Min", 0.0)))
        idle_mean = float(flow.get("idle_mean", flow.get("Idle Mean", 0.0)))
        idle_std = float(flow.get("idle_std", flow.get("Idle Std", 0.0)))
        idle_max = float(flow.get("idle_max", flow.get("Idle Max", 0.0)))
        idle_min = float(flow.get("idle_min", flow.get("Idle Min", 0.0)))

    dst_port = float(flow.get("destination_port", flow.get("dst_port", flow.get("Destination Port", 80.0))))
    tot_fwd_pkts = float(flow.get("total_fwd_packets", flow.get("Total Fwd Packets", 1.0)))
    tot_bwd_pkts = float(flow.get("total_bwd_packets", flow.get("Total Backward Packets", 0.0)))
    tot_fwd_bytes = float(flow.get("total_fwd_bytes", flow.get("Total Length of Fwd Packets", flow.get("Fwd Packets Length Total", 0.0))))
    tot_bwd_bytes = float(flow.get("total_bwd_bytes", flow.get("Total Length of Bwd Packets", flow.get("Bwd Packets Length Total", 0.0))))

    return {
        "Destination Port": dst_port,
        "Flow Duration": duration_micros,
        "Total Fwd Packets": tot_fwd_pkts,
        "Total Backward Packets": tot_bwd_pkts,
        "Total Length of Fwd Packets": tot_fwd_bytes,
        "Total Length of Bwd Packets": tot_bwd_bytes,
        "Fwd Packet Length Max": float(flow.get("fwd_pkt_len_max", flow.get("Fwd Packet Length Max", 0.0))),
        "Fwd Packet Length Min": float(flow.get("fwd_pkt_len_min", flow.get("Fwd Packet Length Min", 0.0))),
        "Fwd Packet Length Mean": float(flow.get("fwd_pkt_len_mean", flow.get("Fwd Packet Length Mean", 0.0))),
        "Fwd Packet Length Std": float(flow.get("fwd_pkt_len_std", flow.get("Fwd Packet Length Std", 0.0))),
        "Bwd Packet Length Max": float(flow.get("bwd_pkt_len_max", flow.get("Bwd Packet Length Max", 0.0))),
        "Bwd Packet Length Min": float(flow.get("bwd_pkt_len_min", flow.get("Bwd Packet Length Min", 0.0))),
        "Bwd Packet Length Mean": float(flow.get("bwd_pkt_len_mean", flow.get("Bwd Packet Length Mean", 0.0))),
        "Bwd Packet Length Std": float(flow.get("bwd_pkt_len_std", flow.get("Bwd Packet Length Std", 0.0))),
        "Flow Bytes/s": float(flow.get("flow_bytes_per_sec", flow.get("Flow Bytes/s", 0.0))),
        "Flow Packets/s": float(flow.get("flow_packets_per_sec", flow.get("Flow Packets/s", 0.0))),
        "Flow IAT Mean": float(flow.get("flow_iat_mean", flow.get("Flow IAT Mean", 0.0))),
        "Flow IAT Std": float(flow.get("flow_iat_std", flow.get("Flow IAT Std", 0.0))),
        "Flow IAT Max": float(flow.get("flow_iat_max", flow.get("Flow IAT Max", 0.0))),
        "Flow IAT Min": float(flow.get("flow_iat_min", flow.get("Flow IAT Min", 0.0))),
        "Fwd IAT Total": float(flow.get("fwd_iat_total", flow.get("Fwd IAT Total", 0.0))),
        "Fwd IAT Mean": float(flow.get("fwd_iat_mean", flow.get("Fwd IAT Mean", 0.0))),
        "Fwd IAT Std": float(flow.get("fwd_iat_std", flow.get("Fwd IAT Std", 0.0))),
        "Fwd IAT Max": float(flow.get("fwd_iat_max", flow.get("Fwd IAT Max", 0.0))),
        "Fwd IAT Min": float(flow.get("fwd_iat_min", flow.get("Fwd IAT Min", 0.0))),
        "Bwd IAT Total": float(flow.get("bwd_iat_total", flow.get("Bwd IAT Total", 0.0))),
        "Bwd IAT Mean": float(flow.get("bwd_iat_mean", flow.get("Bwd IAT Mean", 0.0))),
        "Bwd IAT Std": float(flow.get("bwd_iat_std", flow.get("Bwd IAT Std", 0.0))),
        "Bwd IAT Max": float(flow.get("bwd_iat_max", flow.get("Bwd IAT Max", 0.0))),
        "Bwd IAT Min": float(flow.get("bwd_iat_min", flow.get("Bwd IAT Min", 0.0))),
        "Fwd PSH Flags": _clip_flag(flow.get("fwd_psh_flags", flow.get("Fwd PSH Flags", 0.0))),
        "Bwd PSH Flags": _clip_flag(flow.get("bwd_psh_flags", flow.get("Bwd PSH Flags", 0.0))),
        "Fwd URG Flags": _clip_flag(flow.get("fwd_urg_flags", flow.get("Fwd URG Flags", 0.0))),
        "Bwd URG Flags": _clip_flag(flow.get("bwd_urg_flags", flow.get("Bwd URG Flags", 0.0))),
        "Fwd Header Length": float(flow.get("fwd_header_length", flow.get("Fwd Header Length", tot_fwd_pkts * 32.0))),
        "Bwd Header Length": float(flow.get("bwd_header_length", flow.get("Bwd Header Length", tot_bwd_pkts * 32.0))),
        "Fwd Packets/s": float(flow.get("fwd_packets_per_sec", flow.get("Fwd Packets/s", 0.0))),
        "Bwd Packets/s": float(flow.get("bwd_packets_per_sec", flow.get("Bwd Packets/s", 0.0))),
        "Min Packet Length": float(flow.get("min_packet_length", flow.get("min_pkt_len", flow.get("Min Packet Length", 0.0)))),
        "Max Packet Length": float(flow.get("max_packet_length", flow.get("max_pkt_len", flow.get("Max Packet Length", 0.0)))),
        "Packet Length Mean": float(flow.get("packet_length_mean", flow.get("pkt_len_mean", flow.get("Packet Length Mean", 0.0)))),
        "Packet Length Std": float(flow.get("packet_length_std", flow.get("pkt_len_std", flow.get("Packet Length Std", 0.0)))),
        "Packet Length Variance": float(flow.get("packet_length_variance", flow.get("pkt_len_variance", flow.get("Packet Length Variance", 0.0)))),
        "FIN Flag Count": _clip_flag(flow.get("fin_flag_count", flow.get("FIN Flag Count", 0.0))),
        "SYN Flag Count": _clip_flag(flow.get("syn_flag_count", flow.get("SYN Flag Count", 0.0))),
        "RST Flag Count": _clip_flag(flow.get("rst_flag_count", flow.get("RST Flag Count", 0.0))),
        "PSH Flag Count": _clip_flag(flow.get("psh_flag_count", flow.get("PSH Flag Count", 0.0))),
        "ACK Flag Count": _clip_flag(flow.get("ack_flag_count", flow.get("ACK Flag Count", 0.0))),
        "URG Flag Count": _clip_flag(flow.get("urg_flag_count", flow.get("URG Flag Count", 0.0))),
        "CWE Flag Count": _clip_flag(flow.get("cwe_flag_count", flow.get("CWE Flag Count", 0.0))),
        "ECE Flag Count": _clip_flag(flow.get("ece_flag_count", flow.get("ECE Flag Count", 0.0))),
        "Down/Up Ratio": float(flow.get("down_up_ratio", flow.get("Down/Up Ratio", 0.0))),
        "Average Packet Size": float(flow.get("average_packet_size", flow.get("avg_pkt_size", flow.get("Average Packet Size", 0.0)))),
        "Avg Fwd Segment Size": float(flow.get("avg_fwd_segment_size", flow.get("Avg Fwd Segment Size", 0.0))),
        "Avg Bwd Segment Size": float(flow.get("avg_bwd_segment_size", flow.get("Avg Bwd Segment Size", 0.0))),
        "Fwd Header Length.1": float(flow.get("fwd_header_length_1", flow.get("Fwd Header Length.1", tot_fwd_pkts * 32.0))),
        "Fwd Avg Bytes/Bulk": float(flow.get("fwd_avg_bytes_bulk", flow.get("Fwd Avg Bytes/Bulk", 0.0))),
        "Fwd Avg Packets/Bulk": float(flow.get("fwd_avg_packets_bulk", flow.get("Fwd Avg Packets/Bulk", 0.0))),
        "Fwd Avg Bulk Rate": float(flow.get("fwd_avg_bulk_rate", flow.get("Fwd Avg Bulk Rate", 0.0))),
        "Bwd Avg Bytes/Bulk": float(flow.get("bwd_avg_bytes_bulk", flow.get("Bwd Avg Bytes/Bulk", 0.0))),
        "Bwd Avg Packets/Bulk": float(flow.get("bwd_avg_packets_bulk", flow.get("Bwd Avg Packets/Bulk", 0.0))),
        "Bwd Avg Bulk Rate": float(flow.get("bwd_avg_bulk_rate", flow.get("Bwd Avg Bulk Rate", 0.0))),
        "Subflow Fwd Packets": float(flow.get("subflow_fwd_packets", flow.get("Subflow Fwd Packets", tot_fwd_pkts))),
        "Subflow Fwd Bytes": float(flow.get("subflow_fwd_bytes", flow.get("Subflow Fwd Bytes", tot_fwd_bytes))),
        "Subflow Bwd Packets": float(flow.get("subflow_bwd_packets", flow.get("Subflow Bwd Packets", tot_bwd_pkts))),
        "Subflow Bwd Bytes": float(flow.get("subflow_bwd_bytes", flow.get("Subflow Bwd Bytes", tot_bwd_bytes))),
        "Init_Win_bytes_forward": float(flow.get("init_win_bytes_forward", flow.get("init_win_bytes_fwd", flow.get("Init_Win_bytes_forward", 29200.0)))),
        "Init_Win_bytes_backward": float(flow.get("init_win_bytes_backward", flow.get("init_win_bytes_bwd", flow.get("Init_Win_bytes_backward", 28960.0)))),
        "act_data_pkt_fwd": float(flow.get("act_data_pkt_fwd", flow.get("act_data_pkt_forward", 0.0))),
        "min_seg_size_forward": float(flow.get("min_seg_size_forward", flow.get("min_seg_size_fwd", 32.0))),
        "Active Mean": active_mean,
        "Active Std": active_std,
        "Active Max": active_max,
        "Active Min": active_min,
        "Idle Mean": idle_mean,
        "Idle Std": idle_std,
        "Idle Max": idle_max,
        "Idle Min": idle_min,
    }


class FeatureAdapter:
    def __init__(self, models_dir: Union[str, Path] = None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent.parent / "models"
        self.models_dir = Path(models_dir)

        # 1. Load Feature Schema List
        feature_list_path = self.models_dir / "feature_list.json"
        if not feature_list_path.exists():
            raise FileNotFoundError(f"Feature list schema not found at {feature_list_path}")
        with open(feature_list_path, "r", encoding="utf-8") as f:
            self.expected_features: List[str] = json.load(f)

        # 2. Load RobustScaler
        scaler_path = self.models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"RobustScaler not found at {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives the 14 interaction metrics from raw flow telemetry:
        timing jitter, flag densities, asymmetric traffic ratios, and rate logs.
        """
        d = df.copy()

        # Extract primary metrics with safe defaults
        fwd_pkts = d.get('Total Fwd Packets', pd.Series(1.0, index=d.index)).astype(float)
        bwd_pkts = d.get('Total Backward Packets', pd.Series(0.0, index=d.index)).astype(float)
        fwd_bytes = d.get('Total Length of Fwd Packets', pd.Series(0.0, index=d.index)).astype(float)
        bwd_bytes = d.get('Total Length of Bwd Packets', pd.Series(0.0, index=d.index)).astype(float)
        tot_pkts = (fwd_pkts + bwd_pkts).replace(0, 1.0)
        tot_bytes = (fwd_bytes + bwd_bytes).replace(0, 1.0)

        # 1. Asymmetry & Packet Ratios
        d['fwd_bwd_pkt_ratio'] = fwd_pkts / (bwd_pkts + 1.0)
        d['fwd_bwd_byte_ratio'] = fwd_bytes / (bwd_bytes + 1.0)
        d['fwd_pkt_ratio'] = fwd_pkts / tot_pkts
        d['bwd_pkt_ratio'] = bwd_pkts / tot_pkts

        # 2. Packet Length Distributions
        pkt_mean = d.get('Packet Length Mean', pd.Series(0.0, index=d.index)).astype(float)
        pkt_min = d.get('Min Packet Length', pd.Series(0.0, index=d.index)).astype(float)
        pkt_std = d.get('Packet Length Std', pd.Series(0.0, index=d.index)).astype(float)
        d['pkt_len_skew_proxy'] = (pkt_mean - pkt_min) / (pkt_std + 1.0)
        d['byte_per_packet'] = tot_bytes / tot_pkts

        fwd_max = d.get('Fwd Packet Length Max', pd.Series(0.0, index=d.index)).astype(float)
        fwd_min = d.get('Fwd Packet Length Min', pd.Series(0.0, index=d.index)).astype(float)
        bwd_max = d.get('Bwd Packet Length Max', pd.Series(0.0, index=d.index)).astype(float)
        bwd_min = d.get('Bwd Packet Length Min', pd.Series(0.0, index=d.index)).astype(float)
        d['fwd_pkt_len_spread'] = fwd_max - fwd_min
        d['bwd_pkt_len_spread'] = bwd_max - bwd_min

        # 3. Protocol Flag Density
        fin_f = d.get('FIN Flag Count', pd.Series(0.0, index=d.index)).astype(float)
        syn_f = d.get('SYN Flag Count', pd.Series(0.0, index=d.index)).astype(float)
        rst_f = d.get('RST Flag Count', pd.Series(0.0, index=d.index)).astype(float)
        psh_f = d.get('PSH Flag Count', pd.Series(0.0, index=d.index)).astype(float)
        ack_f = d.get('ACK Flag Count', pd.Series(0.0, index=d.index)).astype(float)
        urg_f = d.get('URG Flag Count', pd.Series(0.0, index=d.index)).astype(float)
        tot_flags = fin_f + syn_f + rst_f + psh_f + ack_f + urg_f
        d['flag_density'] = tot_flags / tot_pkts

        # 4. Timing Jitter & Log Rate Scalings
        fwd_iat_std = d.get('Fwd IAT Std', pd.Series(0.0, index=d.index)).astype(float)
        fwd_iat_mean = d.get('Fwd IAT Mean', pd.Series(0.0, index=d.index)).astype(float)
        d['fwd_iat_jitter'] = fwd_iat_std / (fwd_iat_mean + 1.0)

        flow_pkts_s = d.get('Flow Packets/s', pd.Series(0.0, index=d.index)).astype(float)
        flow_bytes_s = d.get('Flow Bytes/s', pd.Series(0.0, index=d.index)).astype(float)
        d['flow_pkt_rate_log'] = np.log1p(np.maximum(0, flow_pkts_s))
        d['flow_byte_rate_log'] = np.log1p(np.maximum(0, flow_bytes_s))

        # 5. Duty Cycle Ratio
        active_mean = d.get('Active Mean', pd.Series(0.0, index=d.index)).astype(float)
        idle_mean = d.get('Idle Mean', pd.Series(0.0, index=d.index)).astype(float)
        d['active_idle_ratio'] = active_mean / (idle_mean + 1.0)

        return d

    def transform(self, raw_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> np.ndarray:
        """
        Accepts raw flow dicts or DataFrame, normalizes fields, performs feature engineering,
        aligns feature order with schema, replaces invalid values, and scales.
        Returns:
            np.ndarray: Scaled feature array of shape (N, 91), clipped to [-10, 10].
        """
        if isinstance(raw_data, dict):
            normalized = [normalize_flow_dict(raw_data)]
            df = pd.DataFrame(normalized)
        elif isinstance(raw_data, list):
            normalized = [normalize_flow_dict(f) if isinstance(f, dict) else f for f in raw_data]
            df = pd.DataFrame(normalized)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
        else:
            raise ValueError("Input data must be a dict, list of dicts, or pandas DataFrame.")

        # 1. Feature Engineering
        df_eng = self.engineer_features(df)

        # 2. Align Columns (Missing columns filled with 0.0)
        for col in self.expected_features:
            if col not in df_eng.columns:
                df_eng[col] = 0.0

        df_aligned = df_eng[self.expected_features]

        # 3. Clean NaN / Inf values
        df_cleaned = df_aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 4. Scale using RobustScaler fitted on Benign manifold
        X_scaled = self.scaler.transform(df_cleaned.values)

        # 5. Clip extreme values to prevent exploding gradients/activations
        X_clipped = np.clip(X_scaled, -10.0, 10.0)

        return X_clipped
