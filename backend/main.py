"""
backend/main.py
===============
Unified Production FastAPI Backend for Kadak Autonomous NIDS (v1.1).

Integrates:
  - Wire Data Plane (Rust engine pipe / live libpcap sniffer)
  - Dual-Tier AI Risk Engine (Tier-1 LightGBM + Tier-2 Deep Autoencoder)
  - 8 Latent Space KMeans Prototype Centroids
  - Full Operation Modes: Live Sniffer, Captured Replay, Demo/Lab Feed
  - Comprehensive REST Endpoints (/api/classify, /api/stats, /api/history, /api/mode, /api/replay/*)
  - High-Throughput Real-Time WebSockets (/ws and /ws/telemetry)
  - Modern Frontend SOC Host (/)
"""

import os
# Fix for OpenMP crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.risk_engine import RiskEngine

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nids.backend")

# Base directory paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "data"

# Global state
risk_engine: Optional[RiskEngine] = None
active_connections: List[WebSocket] = []

MAX_HISTORY_ENTRIES = 500
alert_history: deque = deque(maxlen=MAX_HISTORY_ENTRIES)

stats = {
    "total_flows": 0,
    "normal_count": 0,
    "benign_count": 0,
    "known_count": 0,
    "zeroday_count": 0,
    "unknown_count": 0,
    "suspicious_count": 0,
    "malicious_count": 0,
    "critical_count": 0,
    "total_risk_sum": 0.0,
    "average_risk": 0.0,
    "avg_latency_ms": 0.0,
    "total_latency_ms": 0.0,
}

# Operational Modes
MODE_LIVE = "live"
MODE_CAPTURED = "captured"
MODE_DEMO = "demo"

MODE_CONFIG = {
    MODE_LIVE: {
        "name": "LIVE CAPTURE",
        "description": "Actual current packets captured via Rust libpcap engine",
        "source_file": "None (libpcap stream)",
    },
    MODE_CAPTURED: {
        "name": "CAPTURED REPLAY",
        "description": "Historical background network capture replay (live_flows.jsonl)",
        "source_file": "demo_traffic.jsonl",
    },
    MODE_DEMO: {
        "name": "DEMO / LAB REPLAY",
        "description": "Controlled SOC demonstration feed from benchmark dataset",
        "source_file": "demo_traffic.jsonl",
    },
}

current_mode: str = os.getenv("FLOW_REPLAY_MODE", MODE_DEMO)
live_interface: str = os.getenv("CAPTURE_INTERFACE", "virbr0")
replay_delay: float = float(os.getenv("FLOW_REPLAY_DELAY", "0.15"))
is_replaying: bool = False
replay_task: Optional[asyncio.Task] = None
live_bridge_proc: Optional[asyncio.subprocess.Process] = None


def get_stats_payload() -> Dict[str, Any]:
    return {
        "total_flows": stats["total_flows"],
        "normal_count": stats["normal_count"],
        "benign_count": stats["benign_count"],
        "known_count": stats["known_count"],
        "zeroday_count": stats["zeroday_count"],
        "unknown_count": stats["zeroday_count"],
        "suspicious_count": stats["suspicious_count"],
        "malicious_count": stats["malicious_count"],
        "critical_count": stats["critical_count"],
        "total_risk_sum": round(stats["total_risk_sum"], 1),
        "average_risk": stats["average_risk"],
        "avg_latency_ms": stats["avg_latency_ms"],
        "mode": current_mode,
        "mode_name": MODE_CONFIG.get(current_mode, {}).get("name", current_mode.upper()),
        "interface": live_interface if current_mode == MODE_LIVE else None,
        "replaying": is_replaying,
    }


def reset_stats_and_history():
    """Resets memory queues and aggregate metrics."""
    alert_history.clear()
    stats["total_flows"] = 0
    stats["normal_count"] = 0
    stats["benign_count"] = 0
    stats["known_count"] = 0
    stats["zeroday_count"] = 0
    stats["unknown_count"] = 0
    stats["suspicious_count"] = 0
    stats["malicious_count"] = 0
    stats["critical_count"] = 0
    stats["total_risk_sum"] = 0.0
    stats["average_risk"] = 0.0
    stats["avg_latency_ms"] = 0.0
    stats["total_latency_ms"] = 0.0


def generate_sample_flow(threat_type: str = "BENIGN") -> Dict[str, Any]:
    """Generates synthetic flow conforming to the CICIDS2017 schema distribution with all 77 features."""
    threat_type = threat_type.upper()

    if threat_type == "DOS":
        dur = random.uniform(500, 30000)
        fwd_pkts = random.randint(100, 500)
        fwd_len = 54.0
        return {
            "Source IP": f"192.168.1.{random.randint(10, 250)}",
            "Destination IP": "10.0.0.1",
            "Destination Port": 80,
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": 0,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len,
            "Total Length of Bwd Packets": 0.0,
            "Fwd Packet Length Max": 64.0, "Fwd Packet Length Min": 40.0, "Fwd Packet Length Mean": fwd_len, "Fwd Packet Length Std": 2.0,
            "Bwd Packet Length Max": 0.0, "Bwd Packet Length Min": 0.0, "Bwd Packet Length Mean": 0.0, "Bwd Packet Length Std": 0.0,
            "Flow Bytes/s": (fwd_pkts * fwd_len) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": fwd_pkts / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, fwd_pkts), "Flow IAT Std": 2.0, "Flow IAT Max": 10.0, "Flow IAT Min": 1.0,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur / max(1, fwd_pkts), "Fwd IAT Std": 2.0, "Fwd IAT Max": 10.0, "Fwd IAT Min": 1.0,
            "Bwd IAT Total": 0.0, "Bwd IAT Mean": 0.0, "Bwd IAT Std": 0.0, "Bwd IAT Max": 0.0, "Bwd IAT Min": 0.0,
            "Fwd PSH Flags": 0.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": 0,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": 0.0,
            "Min Packet Length": 40.0, "Max Packet Length": 64.0, "Packet Length Mean": fwd_len, "Packet Length Std": 2.0,
            "Packet Length Variance": 4.0,
            "FIN Flag Count": 0.0, "SYN Flag Count": 1.0, "RST Flag Count": 0.0, "PSH Flag Count": 0.0, "ACK Flag Count": 0.0, "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0, "ECE Flag Count": 0.0, "Down/Up Ratio": 0.0, "Average Packet Size": fwd_len,
            "Avg Fwd Segment Size": fwd_len, "Avg Bwd Segment Size": 0.0, "Fwd Header Length.1": fwd_pkts * 32,
            "Fwd Avg Bytes/Bulk": 0.0, "Fwd Avg Packets/Bulk": 0.0, "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_pkts, "Subflow Fwd Bytes": fwd_pkts * fwd_len,
            "Subflow Bwd Packets": 0, "Subflow Bwd Bytes": 0.0,
            "Init_Win_bytes_forward": 1024.0, "Init_Win_bytes_backward": -1.0,
            "act_data_pkt_fwd": 0, "min_seg_size_forward": 32.0,
            "Active Mean": 0.0, "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0
        }

    elif threat_type == "PORTSCAN":
        dur = random.uniform(20, 300)
        return {
            "Source IP": f"172.16.0.{random.randint(2, 50)}",
            "Destination IP": "10.0.0.1",
            "Destination Port": random.randint(1, 65535),
            "Flow Duration": dur,
            "Total Fwd Packets": 1,
            "Total Backward Packets": 0,
            "Total Length of Fwd Packets": 44.0,
            "Total Length of Bwd Packets": 0.0,
            "Fwd Packet Length Max": 44.0, "Fwd Packet Length Min": 44.0, "Fwd Packet Length Mean": 44.0, "Fwd Packet Length Std": 0.0,
            "Bwd Packet Length Max": 0.0, "Bwd Packet Length Min": 0.0, "Bwd Packet Length Mean": 0.0, "Bwd Packet Length Std": 0.0,
            "Flow Bytes/s": 44.0 / (dur / 1e6 + 1e-5),
            "Flow Packets/s": 1.0 / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur, "Flow IAT Std": 0.0, "Flow IAT Max": dur, "Flow IAT Min": dur,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur, "Fwd IAT Std": 0.0, "Fwd IAT Max": dur, "Fwd IAT Min": dur,
            "Bwd IAT Total": 0.0, "Bwd IAT Mean": 0.0, "Bwd IAT Std": 0.0, "Bwd IAT Max": 0.0, "Bwd IAT Min": 0.0,
            "Fwd PSH Flags": 0.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": 32, "Bwd Header Length": 0,
            "Fwd Packets/s": 1.0 / (dur / 1e6 + 1e-5), "Bwd Packets/s": 0.0,
            "Min Packet Length": 44.0, "Max Packet Length": 44.0, "Packet Length Mean": 44.0, "Packet Length Std": 0.0,
            "Packet Length Variance": 0.0,
            "FIN Flag Count": 0.0, "SYN Flag Count": 1.0, "RST Flag Count": 1.0, "PSH Flag Count": 0.0, "ACK Flag Count": 0.0, "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0, "ECE Flag Count": 0.0, "Down/Up Ratio": 0.0, "Average Packet Size": 44.0,
            "Avg Fwd Segment Size": 44.0, "Avg Bwd Segment Size": 0.0, "Fwd Header Length.1": 32,
            "Fwd Avg Bytes/Bulk": 0.0, "Fwd Avg Packets/Bulk": 0.0, "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": 1, "Subflow Fwd Bytes": 44.0,
            "Subflow Bwd Packets": 0, "Subflow Bwd Bytes": 0.0,
            "Init_Win_bytes_forward": 1024.0, "Init_Win_bytes_backward": -1.0,
            "act_data_pkt_fwd": 0, "min_seg_size_forward": 32.0,
            "Active Mean": 0.0, "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0
        }

    elif threat_type in ["ZERODAY", "INFILTRATION", "INFILTRATION_ZERODAY"]:
        dur = random.uniform(80000, 1500000)
        fwd_pkts = random.randint(30, 60)
        bwd_pkts = 2
        fwd_len = 1250.0
        bwd_len = 40.0
        return {
            "Source IP": f"198.51.100.{random.randint(1, 100)}",
            "Destination IP": "10.0.0.1",
            "Destination Port": 4444,
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len,
            "Total Length of Bwd Packets": bwd_pkts * bwd_len,
            "Fwd Packet Length Max": 1480.0, "Fwd Packet Length Min": 200.0, "Fwd Packet Length Mean": fwd_len, "Fwd Packet Length Std": 260.0,
            "Bwd Packet Length Max": 50.0, "Bwd Packet Length Min": 40.0, "Bwd Packet Length Mean": bwd_len, "Bwd Packet Length Std": 5.0,
            "Flow Bytes/s": ((fwd_pkts * fwd_len) + (bwd_pkts * bwd_len)) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, fwd_pkts + bwd_pkts), "Flow IAT Std": (dur / max(1, fwd_pkts + bwd_pkts)) * 2.5, "Flow IAT Max": dur * 0.9, "Flow IAT Min": 1.0,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur / max(1, fwd_pkts), "Fwd IAT Std": (dur / max(1, fwd_pkts)) * 2.0, "Fwd IAT Max": dur * 0.9, "Fwd IAT Min": 1.0,
            "Bwd IAT Total": dur * 0.4, "Bwd IAT Mean": dur / max(1, bwd_pkts), "Bwd IAT Std": 10.0, "Bwd IAT Max": dur * 0.4, "Bwd IAT Min": 10.0,
            "Fwd PSH Flags": 1.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 1.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": bwd_pkts * 32,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-5),
            "Min Packet Length": 40.0, "Max Packet Length": 1480.0, "Packet Length Mean": 1100.0, "Packet Length Std": 300.0,
            "Packet Length Variance": 90000.0,
            "FIN Flag Count": 0.0, "SYN Flag Count": 1.0, "RST Flag Count": 0.0, "PSH Flag Count": 1.0, "ACK Flag Count": 1.0, "URG Flag Count": 1.0,
            "CWE Flag Count": 1.0, "ECE Flag Count": 0.0, "Down/Up Ratio": bwd_pkts / fwd_pkts, "Average Packet Size": 1100.0,
            "Avg Fwd Segment Size": fwd_len, "Avg Bwd Segment Size": bwd_len, "Fwd Header Length.1": fwd_pkts * 32,
            "Fwd Avg Bytes/Bulk": 1000.0, "Fwd Avg Packets/Bulk": 2.0, "Fwd Avg Bulk Rate": 500.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_pkts, "Subflow Fwd Bytes": fwd_pkts * fwd_len,
            "Subflow Bwd Packets": bwd_pkts, "Subflow Bwd Bytes": bwd_pkts * bwd_len,
            "Init_Win_bytes_forward": 4096.0, "Init_Win_bytes_backward": 4096.0,
            "act_data_pkt_fwd": fwd_pkts - 1, "min_seg_size_forward": 32.0,
            "Active Mean": 0.0, "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0
        }

    else:  # BENIGN Normal Browsing (Matching exact dataset distribution)
        dur = random.uniform(500000, 2500000)
        fwd_pkts = random.randint(4, 15)
        bwd_pkts = random.randint(4, 15)
        fwd_len = random.uniform(200, 600)
        bwd_len = random.uniform(300, 900)
        return {
            "Source IP": f"192.168.1.{random.randint(5, 80)}",
            "Destination IP": "10.0.0.1",
            "Destination Port": random.choice([443, 80, 53, 22]),
            "Flow Duration": dur,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": fwd_pkts * fwd_len,
            "Total Length of Bwd Packets": bwd_pkts * bwd_len,
            "Fwd Packet Length Max": fwd_len * 1.5, "Fwd Packet Length Min": 20.0, "Fwd Packet Length Mean": fwd_len, "Fwd Packet Length Std": fwd_len * 0.4,
            "Bwd Packet Length Max": bwd_len * 1.5, "Bwd Packet Length Min": 20.0, "Bwd Packet Length Mean": bwd_len, "Bwd Packet Length Std": bwd_len * 0.4,
            "Flow Bytes/s": ((fwd_pkts * fwd_len) + (bwd_pkts * bwd_len)) / (dur / 1e6 + 1e-5),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (dur / 1e6 + 1e-5),
            "Flow IAT Mean": dur / max(1, fwd_pkts + bwd_pkts), "Flow IAT Std": 500.0, "Flow IAT Max": dur * 0.7, "Flow IAT Min": 50.0,
            "Fwd IAT Total": dur, "Fwd IAT Mean": dur / max(1, fwd_pkts), "Fwd IAT Std": 200.0, "Fwd IAT Max": dur * 0.7, "Fwd IAT Min": 50.0,
            "Bwd IAT Total": dur * 0.8, "Bwd IAT Mean": dur / max(1, bwd_pkts), "Bwd IAT Std": 200.0, "Bwd IAT Max": dur * 0.7, "Bwd IAT Min": 50.0,
            "Fwd PSH Flags": 0.0, "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd_pkts * 32, "Bwd Header Length": bwd_pkts * 32,
            "Fwd Packets/s": fwd_pkts / (dur / 1e6 + 1e-5), "Bwd Packets/s": bwd_pkts / (dur / 1e6 + 1e-5),
            "Min Packet Length": 20.0, "Max Packet Length": max(fwd_len, bwd_len) * 1.5,
            "Packet Length Mean": (fwd_len + bwd_len) / 2.0, "Packet Length Std": abs(bwd_len - fwd_len) * 0.5,
            "Packet Length Variance": (abs(bwd_len - fwd_len) * 0.5) ** 2,
            "FIN Flag Count": 1.0, "SYN Flag Count": 1.0, "RST Flag Count": 0.0,
            "PSH Flag Count": 1.0, "ACK Flag Count": 1.0, "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0, "ECE Flag Count": 0.0, "Down/Up Ratio": bwd_pkts / fwd_pkts,
            "Average Packet Size": (fwd_len + bwd_len) / 2.0, "Avg Fwd Segment Size": fwd_len, "Avg Bwd Segment Size": bwd_len,
            "Fwd Header Length.1": fwd_pkts * 32,
            "Fwd Avg Bytes/Bulk": 0.0, "Fwd Avg Packets/Bulk": 0.0, "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_pkts, "Subflow Fwd Bytes": fwd_pkts * fwd_len,
            "Subflow Bwd Packets": bwd_pkts, "Subflow Bwd Bytes": bwd_pkts * bwd_len,
            "Init_Win_bytes_forward": 65535.0, "Init_Win_bytes_backward": 65535.0,
            "act_data_pkt_fwd": max(0, fwd_pkts - 1), "min_seg_size_forward": 32.0,
            "Active Mean": 0.0, "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0
        }


# ─────────────────────────────────────────────────────────────────────────────
# Background Replay & Live Sniffer Management
# ─────────────────────────────────────────────────────────────────────────────
async def broadcast_raw(payload: Dict[str, Any]):
    if not active_connections:
        return
    text = json.dumps(payload)
    disconnected = []
    for ws in list(active_connections):
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_connections:
            active_connections.remove(ws)


async def broadcast_verdict(alert: Dict[str, Any]):
    """Broadcasts single flow verdict to all connected WebSockets in both v1.0 and v1.1 formats."""
    st = get_stats_payload()
    payload = {
        "type": "flow_alert",
        "data": alert,
        "stats": st,
    }
    await broadcast_raw(payload)


def get_risk_engine() -> RiskEngine:
    """Returns global RiskEngine instance or initializes it lazily if not loaded."""
    global risk_engine
    if risk_engine is None:
        logger.info("Initializing Kadak NIDS RiskEngine on-demand...")
        risk_engine = RiskEngine(MODELS_DIR)
    return risk_engine


def process_flow_record(raw_flow: Dict[str, Any]) -> Dict[str, Any]:
    """Runs single flow through risk_engine and updates stats/history."""
    engine = get_risk_engine()
    verdict = engine.inspect_flow(raw_flow)

    src_ip = str(raw_flow.get("src_ip", raw_flow.get("Source IP", verdict.get("source_ip", "192.168.1.100"))))
    dst_ip = str(raw_flow.get("dst_ip", raw_flow.get("Destination IP", verdict.get("destination_ip", "10.0.0.1"))))
    src_port = int(raw_flow.get("src_port", raw_flow.get("Source Port", verdict.get("source_port", 40000 + random.randint(1, 20000)))))
    dst_port = int(raw_flow.get("dst_port", raw_flow.get("destination_port", raw_flow.get("Destination Port", verdict.get("destination_port", 80)))))
    protocol = str(raw_flow.get("protocol", "TCP")).upper()

    flow_dur = float(raw_flow.get("flow_duration_secs", float(raw_flow.get("Flow Duration", 0.0)) / 1e6))
    tot_pkts = int(raw_flow.get("total_fwd_packets", raw_flow.get("Total Fwd Packets", 1)) + raw_flow.get("total_bwd_packets", raw_flow.get("Total Backward Packets", 1)))
    tot_bytes = int(raw_flow.get("total_fwd_bytes", raw_flow.get("Total Length of Fwd Packets", 100)) + raw_flow.get("total_bwd_bytes", raw_flow.get("Total Length of Bwd Packets", 100)))

    cls = verdict.get("classification", "BENIGN")
    conf = float(verdict.get("confidence", 0.95))
    recon_err = float(verdict.get("reconstruction_error", 0.0))
    proto_dist = float(verdict.get("prototype_distance", 1.0))
    is_zd = bool(verdict.get("is_zero_day", False) or "ZERO" in cls.upper())
    is_anom = bool(verdict.get("is_anomaly", False))
    lat = float(verdict.get("latency_ms", 1.0))

    if cls == "BENIGN":
        pred_label = "Normal"
        decision = "NORMAL"
        action = "ALLOW"
        risk_score = round(min(15.0, (recon_err / max(1e-5, engine.threshold)) * 10.0), 1)
    elif is_zd:
        pred_label = "UNKNOWN"
        decision = "UNKNOWN ATTACK"
        action = "QUARANTINE_HOST"
        risk_score = round(min(100.0, 75.0 + (proto_dist * 4.0)), 1)
    else:
        pred_label = cls
        if cls in ("DoS", "BruteForce"):
            decision = "CRITICAL"
        else:
            decision = "MALICIOUS"
        action = "BLOCK_FLOW"
        risk_score = round(min(100.0, max(75.0, conf * 95.0)), 1)

    alert = {
        "flow_id": raw_flow.get("flow_id", f"flow-{int(time.time()*1000)}-{random.randint(100, 999)}"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "flow_duration_secs": round(flow_dur, 4),
        "total_packets": tot_pkts,
        "total_bytes": tot_bytes,
        "prediction": pred_label,
        "confidence": round(conf, 4),
        "anomaly": is_anom,
        "anomaly_score": round(recon_err, 4),
        "threshold": round(engine.threshold, 4),
        "risk_score": risk_score,
        "decision": decision,
        "action": action,
        "reason": verdict.get("reason", ""),
        # v1.1 compatibility fields
        "source_ip": src_ip,
        "destination_ip": dst_ip,
        "destination_port": dst_port,
        "classification": cls,
        "threat_level": verdict.get("threat_level", "INFO"),
        "detection_tier": verdict.get("detection_tier", "Dual-Tier AI"),
        "quarantine_action": action,
        "reconstruction_error": round(recon_err, 6),
        "prototype_distance": round(proto_dist, 4),
        "latency_ms": lat,
    }

    # Update Statistics
    stats["total_flows"] += 1
    if decision == "NORMAL":
        stats["normal_count"] += 1
        stats["benign_count"] += 1
    elif decision == "UNKNOWN ATTACK":
        stats["zeroday_count"] += 1
        stats["unknown_count"] += 1
        stats["critical_count"] += 1
        stats["malicious_count"] += 1
    else:
        stats["known_count"] += 1
        stats["malicious_count"] += 1
        if decision == "CRITICAL":
            stats["critical_count"] += 1
        else:
            stats["suspicious_count"] += 1

    stats["total_latency_ms"] += lat
    stats["avg_latency_ms"] = round(stats["total_latency_ms"] / max(1, stats["total_flows"]), 2)

    stats["total_risk_sum"] += risk_score
    stats["average_risk"] = round(stats["total_risk_sum"] / max(1, stats["total_flows"]), 1)

    alert_history.append(alert)
    return alert


async def start_live_bridge():
    """Starts the real-time Rust engine capture piped into the backend."""
    global live_bridge_proc
    if live_bridge_proc is not None and live_bridge_proc.returncode is None:
        return

    binary_candidates = [
        ROOT_DIR / "engine" / "target" / "release" / "network-analyzer",
        ROOT_DIR / "engine" / "target" / "debug" / "network-analyzer",
        ROOT_DIR / "target" / "release" / "network-analyzer",
    ]
    bin_path = next((b for b in binary_candidates if b.exists()), None)
    if not bin_path:
        logger.warning("Rust engine binary not found. Standalone capture or simulation will be used.")
        return

    cmd = f"{bin_path} -i {live_interface} --flow-timeout 1 -f jsonl | python3 scripts/pipe_to_backend.py"
    logger.info(f"Launching live packet capture bridge: {cmd}")
    try:
        live_bridge_proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        logger.error(f"Failed to launch live capture bridge: {e}")
        live_bridge_proc = None


async def replay_loop():
    """Continuously streams realistic benchmark or recorded traffic to active sockets."""
    global is_replaying
    patterns = ["BENIGN", "BENIGN", "BENIGN", "DoS", "BENIGN", "PortScan", "BENIGN", "ZeroDay"]

    logger.info(f"Replay worker active (delay={replay_delay}s)...")
    while is_replaying:
        ptype = random.choice(patterns)
        raw_flow = generate_sample_flow(ptype)
        alert = process_flow_record(raw_flow)
        await broadcast_verdict(alert)
        await asyncio.sleep(replay_delay)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Lifespan & Application Setup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global risk_engine
    logger.info("🚀 Initializing Kadak NIDS Dual-Tier AI Risk Engine...")
    risk_engine = RiskEngine(MODELS_DIR)
    logger.info(f"✅ Risk Engine successfully loaded on: {risk_engine.device}!")
    logger.info(f"   Calibrated Threshold (tau): {risk_engine.threshold}")
    logger.info(f"   Loaded Latent Prototypes: {len(risk_engine.prototypes)} centroids")

    # Start replay loop if in demo mode
    if current_mode == MODE_DEMO:
        asyncio.create_task(_start_demo_replay())

    yield

    logger.info("🛑 Shutting down Kadak Backend...")
    active_connections.clear()


async def _start_demo_replay():
    global is_replaying, replay_task
    await asyncio.sleep(1.0)
    is_replaying = True
    replay_task = asyncio.create_task(replay_loop())


app = FastAPI(
    title="🛡️ Kadak NIDS — Autonomous Network Defense API",
    description="Dual-Tier AI Architecture: Supervised LightGBM Signatures + Unsupervised Deep Autoencoder Zero-Day Defense",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static directory if exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# REST API Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    engine = get_risk_engine()
    return {
        "status": "HEALTHY",
        "service": "Kadak Autonomous NIDS",
        "version": "1.1.0",
        "device": str(engine.device),
        "tau_threshold": engine.threshold,
        "prototypes_count": len(engine.prototypes),
        "mode": current_mode,
        "is_replaying": is_replaying,
    }


@app.get("/api/status")
async def get_status():
    engine = get_risk_engine()
    return {
        "mode": current_mode,
        "interface": live_interface,
        "is_replaying": is_replaying,
        "replay_delay": replay_delay,
        "total_flows": stats["total_flows"],
        "connected_clients": len(active_connections),
        "tau_threshold": engine.threshold,
    }


@app.get("/api/stats")
async def get_stats():
    return stats


@app.get("/api/history")
async def get_history(limit: int = 100):
    hist = list(alert_history)
    return hist[-limit:]


@app.get("/api/v1/models/info")
async def get_models_info():
    engine = get_risk_engine()
    return {
        "device": str(engine.device),
        "input_dimensions": engine.input_dim,
        "anomaly_threshold_tau": engine.threshold,
        "label_mapping": engine.label_mapping,
        "cluster_prototypes_count": len(engine.prototypes),
        "architecture": {
            "tier1": "LightGBM GBDT (Known Signatures)",
            "tier2": "PyTorch 8-Layer Deep Autoencoder (Zero-Day Anomaly Detection)",
            "latent_space": "16-D Bottleneck KMeans Prototype Centroids",
        },
    }


class FlowPayload(BaseModel):
    flow_data: Dict[str, Any] = Field(default_factory=dict)


@app.post("/api/classify")
@app.post("/api/v1/inspect/flow")
async def classify_flow(payload: Optional[Dict[str, Any]] = None, body: Optional[FlowPayload] = None):
    raw_flow = {}
    if body and body.flow_data:
        raw_flow = body.flow_data
    elif payload:
        raw_flow = payload.get("flow", payload.get("flow_data", payload))

    if not raw_flow:
        raw_flow = generate_sample_flow("BENIGN")

    verdict = process_flow_record(raw_flow)
    await broadcast_verdict(verdict)
    return {"status": "SUCCESS", "result": verdict}


class BatchPayload(BaseModel):
    flows: List[Dict[str, Any]]


@app.post("/api/v1/inspect/batch")
async def inspect_batch(batch: BatchPayload):
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk Engine initializing")
    results = [process_flow_record(f) for f in batch.flows]
    for r in results:
        await broadcast_verdict(r)
    return {"status": "SUCCESS", "count": len(results), "results": results}


@app.post("/api/v1/simulate/flow")
async def simulate_flow(threat_type: str = Query("BENIGN")):
    raw_flow = generate_sample_flow(threat_type)
    verdict = process_flow_record(raw_flow)
    await broadcast_verdict(verdict)
    return {"status": "SUCCESS", "threat_injected": threat_type, "verdict": verdict}


# Replay & Mode Endpoints
@app.post("/api/mode")
@app.post("/api/replay/mode")
async def set_mode(
    mode: Optional[str] = Query(None),
    payload: Optional[Dict[str, Any]] = None,
):
    global current_mode, is_replaying, replay_task
    target_mode = (mode or (payload and payload.get("mode")) or "").lower()
    if target_mode not in (MODE_LIVE, MODE_CAPTURED, MODE_DEMO):
        raise HTTPException(status_code=400, detail="Invalid mode. Choose: live, captured, demo")

    current_mode = target_mode
    if current_mode == MODE_LIVE:
        is_replaying = False
        await start_live_bridge()
    elif current_mode in (MODE_CAPTURED, MODE_DEMO):
        if live_bridge_proc and live_bridge_proc.returncode is None:
            live_bridge_proc.terminate()
        is_replaying = True
        if replay_task is None or replay_task.done():
            replay_task = asyncio.create_task(replay_loop())

    st = get_stats_payload()
    await broadcast_raw({
        "type": "mode_changed",
        "mode": current_mode,
        "mode_name": MODE_CONFIG[current_mode]["name"],
        "stats": st,
    })

    return {
        "status": "switched",
        "mode": current_mode,
        "mode_name": MODE_CONFIG[current_mode]["name"],
        "source_file": MODE_CONFIG[current_mode].get("source_file", ""),
        "interface": live_interface if current_mode == MODE_LIVE else None,
        "stats": st,
    }


@app.post("/api/replay/start")
async def start_replay_feed(delay: Optional[float] = Query(None)):
    global is_replaying, replay_delay, replay_task
    if delay is not None and delay > 0:
        replay_delay = delay
    is_replaying = True
    if replay_task is None or replay_task.done():
        replay_task = asyncio.create_task(replay_loop())
    return {"status": "started", "is_replaying": is_replaying, "delay": replay_delay}


@app.post("/api/replay/stop")
async def stop_replay_feed():
    global is_replaying
    is_replaying = False
    return {"status": "stopped", "is_replaying": is_replaying}


@app.post("/api/replay/reset")
async def reset_replay_feed():
    reset_stats_and_history()
    st = get_stats_payload()
    await broadcast_raw({
        "type": "reset",
        "stats": st,
    })
    return {"status": "reset", "stats": st}


@app.post("/api/replay/speed")
async def change_speed(payload: Dict[str, Any]):
    global replay_delay
    speed = float(payload.get("speed", 1.0))
    if speed <= 0:
        raise HTTPException(status_code=400, detail="Speed must be > 0")
    # Base delay = 0.15s, speed multiplier reduces delay
    replay_delay = max(0.01, 0.15 / speed)
    return {"status": "SUCCESS", "speed_multiplier": speed, "replay_delay": replay_delay}


# ─────────────────────────────────────────────────────────────────────────────
# WebSockets
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        engine = get_risk_engine()
        # Send initial handshake with status and recent history
        init_payload = {
            "type": "initial_state",
            "stats": get_stats_payload(),
            "recent_alerts": list(alert_history)[-30:],
            "recent_history": list(alert_history)[-30:],
            "tau_threshold": engine.threshold if engine else 0.0687,
            "mode": current_mode,
            "prototypes": engine.prototypes.tolist() if engine and len(engine.prototypes) > 0 else [],
        }
        await websocket.send_text(json.dumps(init_payload))

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
                continue
            try:
                msg = json.loads(data) if data.startswith("{") else {"action": data}
                if msg.get("action") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))
                elif msg.get("action") == "inject":
                    ttype = msg.get("threat", "BENIGN")
                    flow = generate_sample_flow(ttype)
                    verdict = process_flow_record(flow)
                    await broadcast_verdict(verdict)
            except Exception:
                pass
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        logger.debug(f"WebSocket closed: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# Frontend UI Host
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Kadak NIDS 1.1 Backend Running. Frontend index.html not found.</h1>")
