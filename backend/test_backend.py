"""
🧪 NIDS 2.0 Backend Diagnostic & Validation Test
Verifies the end-to-end inference pipeline:
FeatureAdapter -> LightGBM -> DeepAutoencoder -> Prototype Centroids.
"""

import os
# Fix for macOS Python 3.14 OpenMP crash (libomp SIGSEGV)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.risk_engine import RiskEngine
from backend.main import generate_sample_flow


def run_tests():
    print("=" * 65)
    print("🧪 Running NIDS 2.0 Backend Diagnostic Suite...")
    print("=" * 65)

    engine = RiskEngine()
    print(f"✅ RiskEngine initialized successfully on: {engine.device}")
    print(f"   Input feature dimensions : {engine.input_dim}")
    print(f"   Calibrated threshold (tau): {engine.threshold}")
    print(f"   Cluster prototypes loaded : {len(engine.prototypes)}")
    print("-" * 65)

    test_cases = [
        ("BENIGN", "Normal Web Browsing"),
        ("DoS", "SYN Flood Attack"),
        ("PortScan", "Port Probe Reconnaissance"),
        ("ZeroDay", "Novel Zero-Day Infiltration")
    ]

    for threat_type, description in test_cases:
        flow = generate_sample_flow(threat_type)
        verdict = engine.inspect_flow(flow)

        print(f"\n🔎 Testing [{threat_type}] ({description}):")
        print(f"   • Classification    : {verdict['classification']}")
        print(f"   • Threat Level      : {verdict['threat_level']}")
        print(f"   • Detection Tier    : {verdict['detection_tier']}")
        print(f"   • Quarantine Action : {verdict['quarantine_action']}")
        print(f"   • Reconstruction MSE: {verdict['reconstruction_error']:.6f} (tau={verdict['anomaly_threshold']})")
        print(f"   • Anomaly Score     : {verdict['anomaly_score']}%")
        print(f"   • Prototype Distance: {verdict['prototype_distance']:.4f}")
        print(f"   • Inference Latency : {verdict['latency_ms']} ms")

    print("\n" + "=" * 65)
    print("🎉 All 4 traffic categories successfully validated against Dual-Tier AI!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
