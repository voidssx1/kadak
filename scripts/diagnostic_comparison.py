"""
scripts/diagnostic_comparison.py
================================
Runs diagnostic comparison across Dual-Tier AI evaluations on:
  1. Benchmark Normal & Attack samples (from data/cicids2017/cicids2017_benchmark.csv)
  2. Live flow sample fixtures (from scratch or generated telemetry)
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.risk_engine import RiskEngine
from backend.main import generate_sample_flow


def main():
    print("=" * 70)
    print("🔍 Kadak NIDS Dual-Tier Diagnostic Verification")
    print("=" * 70)

    engine = RiskEngine()
    print(f"[*] RiskEngine initialized on: {engine.device}")
    print(f"[*] Calibrated threshold (tau): {engine.threshold}")
    print(f"[*] Prototypes count: {len(engine.prototypes)}")

    test_classes = ["BENIGN", "DoS", "PortScan", "ZeroDay"]
    for tc in test_classes:
        flow = generate_sample_flow(tc)
        res = engine.inspect_flow(flow)
        print(f"\n[Test Class: {tc}]")
        print(f"  • Predicted Class : {res['classification']}")
        print(f"  • Threat Level    : {res['threat_level']}")
        print(f"  • Action Taken    : {res['quarantine_action']}")
        print(f"  • Recon MSE       : {res['reconstruction_error']:.6f} (vs tau={res['anomaly_threshold']})")
        print(f"  • Anomaly Score   : {res['anomaly_score']}%")
        print(f"  • Proto Distance  : {res['prototype_distance']:.4f}")
        print(f"  • Latency         : {res['latency_ms']} ms")

    # Benchmark CSV verification if present
    csv_path = ROOT_DIR / "data" / "cicids2017" / "cicids2017_benchmark.csv"
    if csv_path.exists():
        print(f"\n[*] Evaluating sample batch from benchmark CSV: {csv_path.name}")
        df = pd.read_csv(csv_path)
        sample_rows = df.sample(n=min(20, len(df)), random_state=42).to_dict(orient="records")
        results = engine.inspect_batch(sample_rows)
        correct = 0
        for row, r in zip(sample_rows, results):
            gt = row.get("Label", "UNKNOWN")
            pred = r["classification"]
            match = (gt == pred) or (gt in ("BENIGN", "Normal") and pred == "BENIGN")
            if match:
                correct += 1
        print(f"[+] Sample Batch Accuracy: {correct}/{len(results)} ({(correct/len(results))*100:.1f}%)")

    print("\n" + "=" * 70)
    print("🎉 Diagnostic evaluation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
