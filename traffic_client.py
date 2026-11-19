#!/usr/bin/env python3
"""
🌐 NIDS 2.0 Remote Traffic Sender (For Client / Demonstration Machine)
Zero external dependencies required (Uses Python standard library urllib).
Works out of the box on Linux (NixOS, Ubuntu, Arch), macOS, and Windows.

Usage:
    python traffic_client.py --target 192.168.1.6 --type BENIGN
    python traffic_client.py --target 192.168.1.6 --type DoS
    python traffic_client.py --target 192.168.1.6 --type PortScan
    python traffic_client.py --target 192.168.1.6 --type ZeroDay
    python traffic_client.py --target 192.168.1.6 --loop
"""

import sys
import time
import argparse
import json
import urllib.request
import urllib.error

def send_traffic(target_ip: str, port: int, threat_type: str):
    url = f"http://{target_ip}:{port}/api/v1/simulate/flow?threat_type={threat_type}"
    req = urllib.request.Request(url, method="POST")
    req.add_header("User-Agent", "NIDS-Remote-Client/2.0")

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            data = json.loads(response.read().decode())
            verdict = data.get("verdict", {})
            
            print(f"[{time.strftime('%H:%M:%S')}] 🚀 Sent [{threat_type.upper():<7}] -> Target: {target_ip}:{port}")
            print(f"   ↳ Verdict     : {verdict.get('classification')} ({verdict.get('threat_level')})")
            print(f"   ↳ Action      : {verdict.get('quarantine_action')}")
            print(f"   ↳ AI Tier     : {verdict.get('detection_tier')}")
            print(f"   ↳ Recon MSE   : {verdict.get('reconstruction_error', 0):.4f} (Threshold tau={verdict.get('anomaly_threshold')})")
            print(f"   ↳ Latency RTT : {rtt_ms:.2f} ms")
            print("-" * 65)
            return True
    except urllib.error.URLError as e:
        print(f"❌ Connection failed to http://{target_ip}:{port} : {e.reason}")
        print("💡 Ensure backend is running with: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def interactive_menu(target_ip: str, port: int):
    print("=" * 65)
    print("🛡️  NIDS 2.0 REMOTE TRAFFIC GENERATOR")
    print(f"🎯 Target Server: http://{target_ip}:{port}")
    print("=" * 65)
    print("Select packet flow type to inject:")
    print("  1. BENIGN    (Normal Web Browsing - Safe)")
    print("  2. DoS       (SYN Flood Attack Signature - Blocked)")
    print("  3. PortScan  (Reconnaissance Probe - Blocked)")
    print("  4. ZeroDay   (Novel Latent Anomaly - Quarantined)")
    print("  5. Continuous Mixed Stream (Simulate real network)")
    print("  0. Exit")
    print("=" * 65)

    type_map = {"1": "BENIGN", "2": "DoS", "3": "PortScan", "4": "ZeroDay"}

    while True:
        try:
            choice = input("\nEnter choice [1-5, 0 to quit]: ").strip()
            if choice == "0":
                print("Exiting.")
                break
            elif choice in type_map:
                send_traffic(target_ip, port, type_map[choice])
            elif choice == "5":
                print("\n⚡ Starting continuous network emulation (Press Ctrl+C to stop)...")
                import random
                types = ["BENIGN", "BENIGN", "DoS", "BENIGN", "PortScan", "ZeroDay"]
                try:
                    while True:
                        t = random.choice(types)
                        send_traffic(target_ip, port, t)
                        time.sleep(random.uniform(0.8, 2.0))
                except KeyboardInterrupt:
                    print("\nStream stopped.")
            else:
                print("Invalid choice. Enter 1, 2, 3, 4, 5, or 0.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

def main():
    parser = argparse.ArgumentParser(description="NIDS 2.0 Remote Traffic Sender")
    parser.add_argument("--target", default="127.0.0.1", help="Target Mac/Server IP (e.g. 192.168.1.6)")
    parser.add_argument("--port", type=int, default=8000, help="Target backend port (default: 8000)")
    parser.add_argument("--type", choices=["BENIGN", "DoS", "PortScan", "ZeroDay"], help="Traffic type to inject")
    parser.add_argument("--loop", action="store_true", help="Continuously send mixed traffic")

    args = parser.parse_args()

    if args.loop:
        import random
        types = ["BENIGN", "BENIGN", "DoS", "BENIGN", "PortScan", "ZeroDay"]
        print(f"⚡ Streaming mixed traffic to {args.target}:{args.port} (Ctrl+C to stop)...")
        try:
            while True:
                send_traffic(args.target, args.port, random.choice(types))
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopped.")
    elif args.type:
        send_traffic(args.target, args.port, args.type)
    else:
        interactive_menu(args.target, args.port)

if __name__ == "__main__":
    main()
