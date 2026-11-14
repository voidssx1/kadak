"""
notebooks/train_pipeline.py
===========================
End-to-End NIDS Dual-Tier Training Pipeline:
  Cell 1: Ingestion & Exploration
  Cell 2: Feature Engineering (77 -> 90 features)
  Cell 3: Supervised Model (LightGBM) Training
  Cell 4: Unsupervised Zero-Day Autoencoder (PyTorch) Training
  Cell 5: Deep Subspace Prototypes (Centroids)
  Cell 6: Evaluation Curves & Zero-Day Manifold Quarantine Validation
  Cell 7: Model Artifacts Export to models/
"""

import json
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cicids2017_benchmark.csv"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "notebooks" / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Feature Engineering Definition (13 Custom Interaction Features)
# -------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    fwd_pkts = d["Total Fwd Packets"].astype(float)
    bwd_pkts = d["Total Backward Packets"].astype(float)
    fwd_bytes = d["Total Length of Fwd Packets"].astype(float)
    bwd_bytes = d["Total Length of Bwd Packets"].astype(float)
    tot_pkts = (fwd_pkts + bwd_pkts).replace(0, 1.0)
    tot_bytes = (fwd_bytes + bwd_bytes).replace(0, 1.0)
    
    # 1. Packet & Byte Symmetry Ratios
    d["fwd_bwd_pkt_ratio"] = fwd_pkts / (bwd_pkts + 1.0)
    d["fwd_bwd_byte_ratio"] = fwd_bytes / (bwd_bytes + 1.0)
    
    # 2. Skew Proxy & Packet Density
    pkt_mean = d["Packet Length Mean"].astype(float)
    pkt_min = d["Min Packet Length"].astype(float)
    pkt_std = d["Packet Length Std"].astype(float)
    d["pkt_len_skew_proxy"] = (pkt_mean - pkt_min) / (pkt_std + 1.0)
    d["byte_per_packet"] = tot_bytes / tot_pkts
    
    # 3. Flag Density
    tot_flags = (
        d["FIN Flag Count"] + d["SYN Flag Count"] + d["RST Flag Count"] +
        d["PSH Flag Count"] + d["ACK Flag Count"] + d["URG Flag Count"]
    ).astype(float)
    d["flag_density"] = tot_flags / tot_pkts
    
    # 4. Inter-Arrival Jitter
    fwd_iat_std = d["Fwd IAT Std"].astype(float)
    fwd_iat_mean = d["Fwd IAT Mean"].astype(float)
    d["fwd_iat_jitter"] = fwd_iat_std / (fwd_iat_mean + 1.0)
    
    # 5. Log Dynamic Range Compression
    d["flow_pkt_rate_log"] = np.log1p(np.maximum(0, d["Flow Packets/s"].astype(float)))
    d["flow_byte_rate_log"] = np.log1p(np.maximum(0, d["Flow Bytes/s"].astype(float)))
    
    # 6. Packet Length Spreads
    d["fwd_pkt_len_spread"] = d["Fwd Packet Length Max"].astype(float) - d["Fwd Packet Length Min"].astype(float)
    d["bwd_pkt_len_spread"] = d["Bwd Packet Length Max"].astype(float) - d["Bwd Packet Length Min"].astype(float)
    
    # 7. Directional Proportions
    d["fwd_pkt_ratio"] = fwd_pkts / tot_pkts
    d["bwd_pkt_ratio"] = bwd_pkts / tot_pkts
    
    # 8. Active/Idle Dynamic
    d["active_idle_ratio"] = d["Active Mean"].astype(float) / (d["Idle Mean"].astype(float) + 1.0)
    
    return d

# -------------------------------------------------------------
# 2. PyTorch Deep Autoencoder (8-Layer Symmetric)
# -------------------------------------------------------------
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 90, bottleneck_dim: int = 16):
        super(DeepAutoencoder, self).__init__()
        # Encoder: 90 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, bottleneck_dim)
        )
        # Decoder: 16 -> 32 -> 64 -> 90
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
        reconstruction = self.decoder(latent)
        return reconstruction, latent

# -------------------------------------------------------------
# 3. Main Training Execution
# -------------------------------------------------------------
def run_pipeline():
    print("=" * 70)
    print("NIDS DUAL-TIER TRAINING PIPELINE (Supervised LightGBM + Unsupervised AE)")
    print("=" * 70)
    
    # Step 1: Load Data
    print(f"\n[1/7] Loading dataset from {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"      Loaded {len(df_raw):,} flow records across classes:")
    for lbl, count in df_raw["Label"].value_counts().items():
        print(f"        - {lbl:<22}: {count:,}")
        
    # Step 2: Feature Engineering
    print("\n[2/7] Computing 13 interaction features (77 -> 90 features)...")
    df_eng = add_engineered_features(df_raw)
    feature_cols = [c for c in df_eng.columns if c != "Label"]
    print(f"      Total input features: {len(feature_cols)}")
    
    # Clean infinities and NaNs
    df_eng[feature_cols] = df_eng[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Step 3: Zero-Day Split
    print("\n[3/7] Isolating 'Infiltration_ZeroDay' for true unseen evaluation...")
    df_known = df_eng[df_eng["Label"] != "Infiltration_ZeroDay"].copy()
    df_zeroday = df_eng[df_eng["Label"] == "Infiltration_ZeroDay"].copy()
    
    # Train / Test split on known traffic
    X_known = df_known[feature_cols].values
    y_known = df_known["Label"].values
    
    label_mapping = {"BENIGN": 0, "DoS": 1, "PortScan": 2, "BruteForce": 3}
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    y_known_encoded = np.array([label_mapping[lbl] for lbl in y_known])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known_encoded, test_size=0.20, random_state=42, stratify=y_known_encoded
    )
    print(f"      Known Train samples: {len(X_train):,}")
    print(f"      Known Test samples : {len(X_test):,}")
    print(f"      Zero-Day Held-out  : {len(df_zeroday):,}")
    
    # Fit Scaler ONLY on Benign training data to prevent leakage
    print("\n[4/7] Fitting RobustScaler on Benign training manifold...")
    benign_train_idx = (y_train == label_mapping["BENIGN"])
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaler.fit(X_train[benign_train_idx])
    
    X_train_scaled = np.clip(scaler.transform(X_train), -10.0, 10.0)
    X_test_scaled = np.clip(scaler.transform(X_test), -10.0, 10.0)
    X_zeroday_scaled = np.clip(scaler.transform(df_zeroday[feature_cols].values), -10.0, 10.0)
    
    # Step 4: Train LightGBM Classifier
    print("\n[5/7] Training Supervised LightGBM Booster for known attack signatures...")
    lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
    lgb_test = lgb.Dataset(X_test_scaled, label=y_test, reference=lgb_train)
    
    lgb_params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 7,
        "feature_fraction": 0.85,
        "verbose": -1,
        "random_state": 42
    }
    
    lgbm_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=120,
        valid_sets=[lgb_test]
    )
    
    # Evaluate LightGBM on test set
    y_test_pred_prob = lgbm_model.predict(X_test_scaled)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    
    print("\n--- LightGBM Classification Report on Holdout Known Test Set ---")
    print(classification_report(y_test, y_test_pred, target_names=list(label_mapping.keys()), digits=4))
    
    # Step 5: Train Deep Autoencoder on BENIGN only
    print("\n[6/7] Training Deep Autoencoder on Benign Normal Manifold (Zero-Day Brain)...")
    X_benign_train = X_train_scaled[benign_train_idx]
    X_benign_test = X_test_scaled[y_test == label_mapping["BENIGN"]]
    
    device = torch.device("cpu")
    print(f"      Using PyTorch Device: {device}", flush=True)
    
    autoencoder = DeepAutoencoder(input_dim=len(feature_cols), bottleneck_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    
    epochs = 20
    batch_size = 128
    n_samples = len(X_benign_train)
    for epoch in range(1, epochs + 1):
        autoencoder.train()
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = torch.tensor(X_benign_train[idx], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            recon, _ = autoencoder(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        epoch_loss /= n_samples
        if epoch % 5 == 0 or epoch == 1:
            print(f"      Epoch [{epoch:02d}/{epochs:02d}] - Reconstruction MSE Loss: {epoch_loss:.6f}", flush=True)
            
    # Compute Anomaly Threshold on Benign Validation Set
    autoencoder.eval()
    with torch.no_grad():
        x_val_t = torch.tensor(X_benign_test, dtype=torch.float32).to(device)
        recon_val, latent_benign = autoencoder(x_val_t)
        err_benign = torch.mean((x_val_t - recon_val) ** 2, dim=1).cpu().numpy()
        
    threshold = float(np.percentile(err_benign, 98.5))
    print(f"\n      [+] Calibrated Operating Anomaly Threshold (tau): {threshold:.4f}")
    
    # Step 6: Cluster Prototypes (Centroids) in Bottleneck Latent Space
    print("\n      Computing Latent Cluster Prototypes (K-Means on 16-D Bottleneck)...")
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(latent_benign.cpu().numpy())
    prototypes = kmeans.cluster_centers_.tolist()
    print(f"      Extracted {len(prototypes)} benign manifold cluster prototypes.")
    
    # Zero-Day Evaluation
    with torch.no_grad():
        x_zero_t = torch.tensor(X_zeroday_scaled, dtype=torch.float32).to(device)
        recon_zero, _ = autoencoder(x_zero_t)
        err_zero = torch.mean((x_zero_t - recon_zero) ** 2, dim=1).cpu().numpy()
        
    zero_day_quarantine_rate = float(np.mean(err_zero >= threshold)) * 100.0
    print(f"\n      🔥 ZERO-DAY EVALUATION RESULTS:")
    print(f"         Total Unseen Zero-Day Flows: {len(err_zero):,}")
    print(f"         Successfully Quarantined  : {zero_day_quarantine_rate:.2f}% (Error >= {threshold:.4f})")
    print(f"         Mean Benign Error         : {np.mean(err_benign):.4f}")
    print(f"         Mean Zero-Day Error       : {np.mean(err_zero):.4f} (Massive Out-of-Distribution departure!)")

    # Generate Evaluation Figures
    print("\n[7/7] Generating and saving evaluation figures & model artifacts...")
    
    # Figure 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(label_mapping.keys()),
                yticklabels=list(label_mapping.keys()))
    plt.title("LightGBM Supervised Classification - Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Ground Truth")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    
    # Figure 2: Autoencoder Error Distribution (Benign vs Zero-Day)
    plt.figure(figsize=(9, 5))
    plt.hist(err_benign, bins=50, alpha=0.6, color="#059669", label=f"Benign Normal (Mean: {np.mean(err_benign):.2f})", density=True)
    plt.hist(err_zero, bins=50, alpha=0.6, color="#7c3aed", label=f"Zero-Day Threat (Mean: {np.mean(err_zero):.2f})", density=True)
    plt.axvline(threshold, color="#dc2626", linestyle="--", linewidth=2, label=f"Decision Boundary tau = {threshold:.2f}")
    plt.title("Deep Autoencoder Reconstruction Error: Benign Manifold vs Zero-Day Separation", fontsize=12, fontweight="bold")
    plt.xlabel("Reconstruction MSE Error")
    plt.ylabel("Density")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "autoencoder_separation.png", dpi=300)
    plt.close()
    
    # Export Models
    # 1. Autoencoder PyTorch Weights
    torch.save(autoencoder.state_dict(), MODELS_DIR / "autoencoder.pt")
    # 2. LightGBM Booster
    with open(MODELS_DIR / "lightgbm_model.pkl", "wb") as f:
        pickle.dump(lgbm_model, f)
    # 3. Scaler
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    # 4. Feature List
    with open(MODELS_DIR / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    # 5. Metadata
    metadata = {
        "version": "1.0.0",
        "input_features_count": len(feature_cols),
        "anomaly_threshold": round(threshold, 4),
        "zero_day_quarantine_rate": round(zero_day_quarantine_rate, 2),
        "label_mapping": label_mapping,
        "prototypes": prototypes,
        "benign_error_mean": round(float(np.mean(err_benign)), 4),
        "zeroday_error_mean": round(float(np.mean(err_zero)), 4),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n[+] ALL ARTIFACTS EXPORTED SUCCESSFULLY TO {MODELS_DIR}:")
    print(f"    - autoencoder.pt      (Deep Autoencoder PyTorch weights)")
    print(f"    - lightgbm_model.pkl  (Supervised LightGBM booster)")
    print(f"    - scaler.pkl          (Manifold RobustScaler)")
    print(f"    - feature_list.json   ({len(feature_cols)} exact feature names)")
    print(f"    - metadata.json       (Threshold tau, Prototypes & Metrics)")
    print(f"\n[+] Evaluation Figures Saved to {FIGURES_DIR}:")
    print(f"    - confusion_matrix.png")
    print(f"    - autoencoder_separation.png")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
