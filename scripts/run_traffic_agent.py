"""
scripts/run_traffic_agent.py
============================
Autonomous Live Traffic & Attack Emulation Stream Runner.
Continuously streams realistic synthetic flows to the FastAPI backend
and displays live triage verdicts with microsecond latency benchmarks.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

# Add project root to sys.path so 'backend' can be imported anywhere
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.main import generate_sample_flow

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/v1/inspect/flow")


def main():
    print("=" * 70)
    print("🚀 Kadak NIDS Live Traffic & Attack Emulation Agent")
    print(f"   Target Backend Endpoint: {BACKEND_URL}")
    print("=" * 70)

    patterns = ["BENIGN", "BENIGN", "BENIGN", "DoS", "BENIGN", "PortScan", "ZeroDay"]

    count = 1
    try:
        while True:
            ptype = random.choice(patterns)
            flow = generate_sample_flow(ptype)

            payload = json.dumps({"flow_data": flow}).encode("utf-8")
            req = urllib.request.Request(
                BACKEND_URL,
                data=payload,
                headers={"Content-Type": "application/json"}
            )

            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    roundtrip_ms = (time.perf_counter() - t0) * 1000.0
                    data = json.loads(resp.read().decode("utf-8"))
                    r = data.get("result", {})
                    classification = r.get("classification", "UNKNOWN")
                    threat_level = r.get("threat_level", "INFO")
                    mse = r.get("reconstruction_error", 0.0)
                    action = r.get("quarantine_action", "ALLOW")

                    icons = {
                        "BENIGN": "🟢",
                        "DoS": "🔴",
                        "PortScan": "🟠",
                        "Infiltration_ZeroDay": "🟣 [ZERO-DAY]"
                    }
                    icon = icons.get(classification, "⚠️")

                    print(f"[{count:03d}] {icon} Class: {classification:22s} | Threat: {threat_level:8s} | "
                          f"Action: {action:16s} | MSE: {mse:.4f} | RTT: {roundtrip_ms:.2f}ms")
            except urllib.error.URLError:
                print(f"[{count:03d}] 🛑 Could not connect to {BACKEND_URL}. Ensure FastAPI backend is running!")
                time.sleep(2.0)
            except Exception as e:
                print(f"[{count:03d}] ⚠️ Error: {e}")

            count += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n🛑 Traffic agent stopped.")


if __name__ == "__main__":
    main()
