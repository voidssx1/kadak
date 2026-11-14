"""
🚀 NIDS 2.0 Real-Time Traffic Emulation Agent (Python Alternative to Rust)
Continuously streams realistic synthetic flows to the FastAPI backend
and displays live triage verdicts with latency benchmarks.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path
import requests

# Add project root to sys.path so 'backend' can be imported anywhere
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.main import generate_sample_flow

BACKEND_URL = "http://127.0.0.1:8000/api/v1/inspect/flow"


def main():
    print("=" * 70)
    print("🚀 NIDS 2.0 Live Traffic Agent")
    print(f"   Target Backend Endpoint: {BACKEND_URL}")
    print("=" * 70)

    patterns = ["BENIGN", "BENIGN", "BENIGN", "DoS", "BENIGN", "PortScan", "ZeroDay"]

    count = 1
    try:
        while True:
            import random
            ptype = random.choice(patterns)
            flow = generate_sample_flow(ptype)

            t0 = time.perf_counter()
            try:
                resp = requests.post(BACKEND_URL, json={"flow_data": flow}, timeout=1.0)
                roundtrip_ms = (time.perf_counter() - t0) * 1000.0

                if resp.status_code == 200:
                    r = resp.json()["result"]
                    classification = r["classification"]
                    threat_level = r["threat_level"]
                    tier = r["detection_tier"]
                    mse = r["reconstruction_error"]
                    action = r["quarantine_action"]

                    icons = {
                        "BENIGN": "🟢",
                        "DoS": "🔴",
                        "PortScan": "🟠",
                        "Infiltration_ZeroDay": "🟣 [ZERO-DAY]"
                    }
                    icon = icons.get(classification, "⚠️")

                    print(f"[{count:03d}] {icon} Class: {classification:22s} | Threat: {threat_level:8s} | "
                          f"Action: {action:16s} | MSE: {mse:.4f} | RTT: {roundtrip_ms:.2f}ms")
                else:
                    print(f"[{count:03d}] ⚠️ Error {resp.status_code}: {resp.text}")

            except requests.exceptions.ConnectionError:
                print(f"[{count:03d}] 🛑 Could not connect to {BACKEND_URL}. Ensure FastAPI is running on port 8000!")
                time.sleep(2.0)

            count += 1
            time.sleep(1.2)

    except KeyboardInterrupt:
        print("\n🛑 Traffic agent stopped.")


if __name__ == "__main__":
    main()
