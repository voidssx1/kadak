"""
🛡️ NIDS 2.0 Feature Adapter
Translates raw network flow metrics into the calibrated 91-feature vector 
matching the model's exact feature schema and normalization bounds.
"""

import os
# Fix for macOS Python 3.14 OpenMP crash (libomp SIGSEGV)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import pickle


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
        Derives the 13 interaction metrics from raw flow telemetry:
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
        Accepts raw flow dicts or DataFrame, performs feature engineering,
        aligns feature order with schema, replaces invalid values, and scales.
        Returns:
            np.ndarray: Scaled feature array of shape (N, 91), clipped to [-10, 10].
        """
        if isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        elif isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
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
