"""
🛡️ NIDS 2.0 Production FastAPI Backend
Provides high-performance REST APIs & Real-Time WebSocket Telemetry
for the Dual-Tier AI Autonomous Network Intrusion Detection System.
"""

import os
# Fix for macOS Python 3.14 OpenMP crash (libomp SIGSEGV)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import json
import random
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.risk_engine import RiskEngine

# Global in-memory Risk Engine instance
risk_engine: Optional[RiskEngine] = None

# Active WebSocket telemetry connections
active_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Preloads PyTorch Autoencoder and LightGBM models
    into memory at startup to guarantee microsecond inference.
    """
    global risk_engine
    print("🚀 Initializing NIDS 2.0 Dual-Tier AI Risk Engine...")
    risk_engine = RiskEngine()
    print(f"✅ Risk Engine successfully loaded on device: {risk_engine.device}!")
    print(f"   Calibrated Anomaly Threshold (tau): {risk_engine.threshold}")
    print(f"   Loaded Latent Prototypes: {len(risk_engine.prototypes)} centroids")
    yield
    print("🛑 Shutting down NIDS 2.0 Backend...")
    active_connections.clear()


app = FastAPI(
    title="🛡️ NIDS 2.0 — Autonomous Network Intrusion Detection API",
    description="Dual-Tier AI Security Architecture: Supervised LightGBM + Unsupervised Deep Autoencoder",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local dev & frontend dashboard (Phase 3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Request & Response Schemas
# ─────────────────────────────────────────────────────────────────────────────
class FlowInspectionRequest(BaseModel):
    flow_data: Dict[str, Any] = Field(
        ...,
        description="Raw flow telemetry dictionary containing packet counts, durations, and flag metrics."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "flow_data": {
                    "Destination Port": 80,
                    "Flow Duration": 1500000,
                    "Total Fwd Packets": 10,
                    "Total Backward Packets": 8,
                    "Total Length of Fwd Packets": 1200.0,
                    "Total Length of Bwd Packets": 4500.0,
                    "Flow Packets/s": 12.0,
                    "Flow Bytes/s": 3800.0,
                    "SYN Flag Count": 1,
                    "ACK Flag Count": 1
                }
            }
        }
    }


class BatchInspectionRequest(BaseModel):
    flows: List[Dict[str, Any]] = Field(..., description="List of raw flow telemetry dictionaries.")


# ─────────────────────────────────────────────────────────────────────────────
# REST Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """System health check and loaded engine status."""
    return {
        "status": "ONLINE",
        "service": "NIDS 2.0 Dual-Tier AI Backend",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(risk_engine.device) if risk_engine else "UNKNOWN",
        "models_loaded": {
            "lightgbm_supervised": risk_engine is not None,
            "deep_autoencoder_unsupervised": risk_engine is not None,
            "feature_adapter": risk_engine is not None
        }
    }


@app.get("/api/v1/models/info")
def get_model_info():
    """Returns model metadata, anomaly threshold, prototypes, and label mapping."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk Engine not yet initialized.")
    return {
        "metadata": risk_engine.metadata,
        "operating_threshold_tau": risk_engine.threshold,
        "features_count": risk_engine.input_dim,
        "latent_prototypes_count": len(risk_engine.prototypes),
        "supported_classes": list(risk_engine.label_mapping.keys()) + ["Infiltration_ZeroDay"]
    }


async def broadcast_flow(verdict: Dict[str, Any]):
    """Broadcasts a processed flow verdict to all connected SOC frontend dashboards."""
    if not active_connections:
        return
    payload = json.dumps({
        "event_type": "PACKET_INSPECTED",
        "telemetry": verdict
    })
    for conn in list(active_connections):
        try:
            await conn.send_text(payload)
        except Exception:
            if conn in active_connections:
                active_connections.remove(conn)


@app.post("/api/v1/inspect/flow")
async def inspect_single_flow(request: FlowInspectionRequest):
    """
    Real-time single flow inspection.
    Triage order: Tier-1 (LightGBM) -> Tier-2 (Deep Autoencoder) -> Prototype Distance.
    Broadcasts verdict live to SOC dashboard.
    """
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk Engine not yet initialized.")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, risk_engine.inspect_flow, request.flow_data)
    await broadcast_flow(result)
    return {"status": "SUCCESS", "result": result}


@app.post("/api/v1/inspect/batch")
async def inspect_batch_flows(request: BatchInspectionRequest):
    """Vectorized high-throughput batch inspection."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk Engine not yet initialized.")
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, risk_engine.inspect_batch, request.flows)
    for res in results:
        await broadcast_flow(res)
    return {
        "status": "SUCCESS",
        "flows_processed": len(results),
        "results": results
    }


# ─────────────────────────────────────────────────────────────────────────────
# Traffic Simulation Generator (Great for Testing & Live Demos)
# ─────────────────────────────────────────────────────────────────────────────
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


@app.post("/api/v1/simulate/flow")
async def simulate_traffic_flow(threat_type: str = Query("BENIGN", description="BENIGN | DoS | PortScan | ZeroDay")):
    """Generates and inspects a sample traffic flow of the requested type on-the-fly and broadcasts to dashboard."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk Engine not yet initialized.")
    flow = generate_sample_flow(threat_type)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, risk_engine.inspect_flow, flow)
    await broadcast_flow(result)
    return {"simulated_threat_type": threat_type, "flow": flow, "verdict": result}


# ─────────────────────────────────────────────────────────────────────────────
# Real-Time WebSocket Telemetry Channel (For SOC Dashboard)
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/telemetry")
async def websocket_telemetry_feed(websocket: WebSocket):
    """
    Subscribes connected SOC frontend dashboards to live wire telemetry.
    Passively listens and only receives flows when actual packets arrive.
    """
    await websocket.accept()
    active_connections.append(websocket)
    print(f"📡 New SOC Dashboard client connected! (Active dashboards: {len(active_connections)})")

    try:
        while True:
            # Passive wait for client ping/messages; actual traffic is pushed by broadcast_flow()
            await websocket.receive_text()

    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print(f"🔌 SOC Dashboard client disconnected. (Remaining: {len(active_connections)})")
    except Exception as e:
        if websocket in active_connections:
            active_connections.remove(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# Mount Frontend Static Directory (Serves Dashboard at http://127.0.0.1:8000/)
# ─────────────────────────────────────────────────────────────────────────────
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    print(f"🎨 SOC Dashboard mounted from: {frontend_path}")
