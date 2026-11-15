"""
🛡️ NIDS 2.0 Dual-Tier Risk Engine
Orchestrates microsecond real-time inference across:
1. Tier-1 Supervised LightGBM (Known Threat Signatures)
2. Tier-2 Unsupervised Deep Autoencoder (Zero-Day Anomaly Detection)
3. Latent Prototype Distance (Manifold Novelty Metric)
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
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import torch
import torch.nn as nn

from backend.adapter import FeatureAdapter


# ─────────────────────────────────────────────────────────────────────────────
# 1. PyTorch Deep Autoencoder Architecture (Matches Trained Weights)
# ─────────────────────────────────────────────────────────────────────────────
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 91, bottleneck_dim: int = 16):
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

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


# ─────────────────────────────────────────────────────────────────────────────
# 2. Risk Engine Core Class
# ─────────────────────────────────────────────────────────────────────────────
class RiskEngine:
    def __init__(self, models_dir: Union[str, Path] = None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent.parent / "models"
        self.models_dir = Path(models_dir)

        # Device Selection (MPS on Apple Silicon, CUDA on GPU, else CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Initialize Feature Adapter
        self.adapter = FeatureAdapter(self.models_dir)

        # 1. Load Metadata & Thresholds
        meta_path = self.models_dir / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata: Dict[str, Any] = json.load(f)

        self.input_dim = self.metadata.get("input_features_count", 91)
        self.threshold = float(self.metadata.get("anomaly_threshold", 0.0687))
        self.label_mapping = self.metadata.get("label_mapping", {
            "BENIGN": 0, "DoS": 1, "PortScan": 2, "BruteForce": 3
        })
        self.inv_label_mapping = {v: k for k, v in self.label_mapping.items()}

        # 8 Benign cluster prototypes in 16-D latent space
        self.prototypes = np.array(self.metadata.get("prototypes", []), dtype=np.float32)

        # 2. Load LightGBM Model
        lgb_path = self.models_dir / "lightgbm_model.pkl"
        with open(lgb_path, "rb") as f:
            self.lgbm_model = pickle.load(f)

        # 3. Load Deep Autoencoder Weights
        self.autoencoder = DeepAutoencoder(input_dim=self.input_dim, bottleneck_dim=16).to(self.device)
        ae_weights_path = self.models_dir / "autoencoder.pt"
        self.autoencoder.load_state_dict(torch.load(ae_weights_path, map_location=self.device))
        self.autoencoder.eval()  # Set to evaluation mode for BatchNorm inference

    def inspect_flow(self, raw_flow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspects a single network flow in real-time.
        Returns full threat triage, confidence, reconstruction MSE, and quarantine decision.
        """
        batch_results = self.inspect_batch([raw_flow])
        return batch_results[0]

    def inspect_batch(self, raw_flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Vectorized dual-tier evaluation across a batch of incoming flows.
        """
        if not raw_flows:
            return []

        start_time = time.perf_counter()

        # Step 1: Feature Transformation & Scaling
        X_scaled = self.adapter.transform(raw_flows)
        n_samples = len(X_scaled)

        # Step 2: Tier-1 Supervised Inference (LightGBM)
        # Returns probability distribution across [BENIGN, DoS, PortScan, BruteForce]
        lgbm_probs = self.lgbm_model.predict(X_scaled)
        known_pred_indices = np.argmax(lgbm_probs, axis=1)
        known_confidences = lgbm_probs[np.arange(n_samples), known_pred_indices]

        # Step 3: Tier-2 Unsupervised Inference (Deep Autoencoder)
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            recon, latent = self.autoencoder(x_tensor)
            recon_errors = torch.mean((x_tensor - recon) ** 2, dim=1).cpu().numpy()
            latent_np = latent.cpu().numpy()

        # Step 4: Prototype Distance (Latent Space Novelty)
        if len(self.prototypes) > 0:
            # Pairwise euclidean distance to 8 cluster centroids
            # latent_np shape: (N, 16), prototypes shape: (8, 16)
            dists = np.linalg.norm(latent_np[:, None, :] - self.prototypes[None, :, :], axis=2)
            min_proto_dists = np.min(dists, axis=1)
        else:
            min_proto_dists = np.zeros(n_samples)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        per_flow_latency_ms = round(total_latency_ms / max(1, n_samples), 3)

        # Step 5: Triaging & Decision Matrix
        results = []
        for i in range(n_samples):
            pred_class_idx = int(known_pred_indices[i])
            pred_class_name = self.inv_label_mapping.get(pred_class_idx, "BENIGN")
            confidence = float(known_confidences[i])
            recon_err = float(recon_errors[i])
            proto_dist = float(min_proto_dists[i])

            flow_meta = raw_flows[i]
            src_ip = flow_meta.get("Source IP", flow_meta.get("src_ip", "192.168.1.105"))
            dst_ip = flow_meta.get("Destination IP", flow_meta.get("dst_ip", "10.0.0.1"))
            dst_port = flow_meta.get("Destination Port", flow_meta.get("dst_port", 80))

            # ── DUAL-TIER ARBITRATION LOGIC ─────────────────────────────
            # Case A: Known Signature Attack (LightGBM high-confidence non-Benign)
            if pred_class_name != "BENIGN" and confidence >= 0.70:
                classification = pred_class_name
                detection_tier = "Tier-1 Supervised (LightGBM Signature)"
                is_anomaly = True
                is_zero_day = False
                threat_level = "CRITICAL" if pred_class_name == "DoS" else "HIGH"
                quarantine_action = "BLOCK_FLOW"
                anomaly_score = min(100.0, round(confidence * 100.0, 1))
                reason = f"Known attack signature detected matching {pred_class_name} with {confidence*100:.1f}% confidence."

            # Case B: Zero-Day Threat (Autoencoder Reconstruction Error >= Threshold)
            elif recon_err >= self.threshold:
                classification = "Infiltration_ZeroDay"
                detection_tier = "Tier-2 Unsupervised (Deep Autoencoder Manifold Departure)"
                is_anomaly = True
                is_zero_day = True
                threat_level = "CRITICAL"
                quarantine_action = "QUARANTINE_HOST"
                # Scaled anomaly score based on how far error departs from threshold
                departure_ratio = recon_err / max(1e-5, self.threshold)
                anomaly_score = min(100.0, round(min(100.0, 60.0 + (departure_ratio * 2.5)), 1))
                reason = (f"Reconstruction MSE loss ({recon_err:.4f}) breached anomaly boundary "
                          f"tau ({self.threshold:.4f}) by {departure_ratio:.1f}x. Novelty distance: {proto_dist:.2f}.")

            # Case C: Benign Normal Traffic
            else:
                classification = "BENIGN"
                detection_tier = "Dual-Tier Clear (Normal Manifold)"
                is_anomaly = False
                is_zero_day = False
                threat_level = "LOW"
                quarantine_action = "ALLOW"
                anomaly_score = round(max(0.0, (recon_err / max(1e-5, self.threshold)) * 20.0), 1)
                reason = f"Traffic pattern conforms to normal benign manifold (Reconstruction Error: {recon_err:.4f} < {self.threshold:.4f})."

            results.append({
                "flow_id": f"flow_{int(time.time()*1000)}_{i}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_ip": src_ip,
                "destination_ip": dst_ip,
                "destination_port": int(dst_port),
                "classification": classification,
                "threat_level": threat_level,
                "detection_tier": detection_tier,
                "quarantine_action": quarantine_action,
                "confidence": round(confidence, 4),
                "reconstruction_error": round(recon_err, 6),
                "anomaly_threshold": self.threshold,
                "anomaly_score": anomaly_score,
                "prototype_distance": round(proto_dist, 4),
                "is_anomaly": is_anomaly,
                "is_zero_day": is_zero_day,
                "reason": reason,
                "latency_ms": per_flow_latency_ms
            })

        return results
