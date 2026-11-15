/**
 * 🛡️ Kadak NIDS 1.1 Autonomous SOC Dashboard Logic
 * Manages real-time WebSocket telemetry, Chart.js streams,
 * interactive threat injection, mode switching, and latent prototype visualizations.
 */

// Global Dashboard State
const BACKEND_BASE = (window.location.port === "8000" && (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"))
  ? ""
  : "http://127.0.0.1:8000";

const WS_BASE = (window.location.port === "8000" && (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"))
  ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`
  : "ws://127.0.0.1:8000";

const state = {
  totalFlows: 0,
  benignCount: 0,
  knownCount: 0,
  zeroDayCount: 0,
  latencies: [],
  threshold: 0.0687,
  prototypes: [],
  flowsHistory: [],
  isStreamPaused: false,
  currentMode: "demo",
  charts: {
    mseTimeline: null,
    distribution: null
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Audio Synthesizer for Threat Alerts (Web Audio API)
// ─────────────────────────────────────────────────────────────────────────────
let audioCtx = null;
function getAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioCtx;
}

function playAlertSound(type = "zeroday") {
  try {
    const ctx = getAudioContext();
    if (ctx.state === 'suspended') ctx.resume();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (type === "zeroday") {
      // Ominous low-pitch dual-tone alarm
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(220, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(110, ctx.currentTime + 0.35);
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
      osc.start();
      osc.stop(ctx.currentTime + 0.35);
    } else {
      // Subtle alert for known signatures
      osc.type = "sine";
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      gain.gain.setValueAtTime(0.06, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15);
      osc.start();
      osc.stop(ctx.currentTime + 0.15);
    }
  } catch (e) {
    // Audio optional if restricted by browser policy
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialization & Metadata Loading
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  await fetchModelMetadata();
  await fetchSystemStatus();
  initWebSocket();
  bindEventHandlers();
});

async function fetchModelMetadata() {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/v1/models/info`);
    if (!res.ok) throw new Error("Backend not available");
    const data = await res.json();

    state.threshold = data.anomaly_threshold_tau || 0.0687;
    const tauEl = document.getElementById("val-tau");
    if (tauEl) tauEl.textContent = `τ = ${state.threshold.toFixed(4)}`;

    const protoCountEl = document.getElementById("val-prototypes");
    if (protoCountEl) protoCountEl.textContent = `${data.cluster_prototypes_count || 8} Centroids`;

    const metaRes = await fetch(`${BACKEND_BASE}/health`);
    if (metaRes.ok) {
      const hData = await metaRes.json();
      const hwEl = document.getElementById("val-hardware");
      if (hwEl && hData.device) {
        hwEl.innerHTML = `<span class="dot dot-green"></span> ${hData.device.toUpperCase()}`;
      }
    }
  } catch (err) {
    console.warn("⚠️ Could not load /api/v1/models/info:", err);
  }
}

async function fetchSystemStatus() {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/status`);
    if (res.ok) {
      const data = await res.json();
      state.currentMode = data.mode || "demo";
      updateModeButtons(state.currentMode);
      const ifaceEl = document.getElementById("txt-interface");
      if (ifaceEl && data.interface) ifaceEl.textContent = data.interface;
    }
  } catch (e) {
    console.warn("Could not fetch /api/status:", e);
  }
}

function updateModeButtons(mode) {
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
  const activeBtn = document.getElementById(`btn-mode-${mode}`);
  if (activeBtn) activeBtn.classList.add("active");
}

async function switchMode(mode) {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode })
    });
    if (res.ok) {
      state.currentMode = mode;
      updateModeButtons(mode);
      console.log(`[+] Mode switched to: ${mode}`);
    }
  } catch (err) {
    console.error("Error switching mode:", err);
  }
}

async function changeSpeed(speed) {
  try {
    await fetch(`${BACKEND_BASE}/api/replay/speed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speed: parseFloat(speed) })
    });
  } catch (err) {
    console.error("Error changing speed:", err);
  }
}

async function resetMetrics() {
  try {
    await fetch(`${BACKEND_BASE}/api/replay/reset`, { method: "POST" });
    state.totalFlows = 0;
    state.benignCount = 0;
    state.knownCount = 0;
    state.zeroDayCount = 0;
    state.latencies = [];
    state.flowsHistory = [];
    updateKPICards();

    const tbody = document.getElementById("flow-tbody");
    if (tbody) tbody.innerHTML = "";

    if (state.charts.mseTimeline) {
      state.charts.mseTimeline.data.labels = [];
      state.charts.mseTimeline.data.datasets[0].data = [];
      state.charts.mseTimeline.data.datasets[1].data = [];
      state.charts.mseTimeline.update();
    }
    if (state.charts.distribution) {
      state.charts.distribution.data.datasets[0].data = [1, 0, 0, 0];
      state.charts.distribution.update();
    }
  } catch (err) {
    console.error("Error resetting metrics:", err);
  }
}

function triggerThreatBanner(flow) {
  const container = document.getElementById("threat-banner-container");
  const title = document.getElementById("threat-banner-title");
  const desc = document.getElementById("threat-banner-desc");
  const action = document.getElementById("threat-banner-action");

  if (!container) return;

  if (flow.is_zero_day || flow.classification === "Infiltration_ZeroDay") {
    title.textContent = "CRITICAL ZERO-DAY ANOMALY BREACH INTERCEPTED";
    desc.textContent = `Flow from ${flow.source_ip} departed benign manifold with MSE: ${flow.reconstruction_error.toFixed(4)} (Threshold: ${state.threshold}). Isolation enacted.`;
    action.className = "badge badge-zeroday";
    action.textContent = "QUARANTINE_HOST";
  } else {
    title.textContent = `HIGH-SEVERITY SIGNATURE MATCH: ${flow.classification}`;
    desc.textContent = `Malicious pattern detected from ${flow.source_ip} to port ${flow.destination_port}. Kernel hardware drop applied.`;
    action.className = "badge badge-dos";
    action.textContent = "BLOCK_FLOW";
  }

  container.style.display = "block";
  // Auto-dismiss after 8 seconds
  clearTimeout(container._dismissTimer);
  container._dismissTimer = setTimeout(dismissThreatBanner, 8000);
}

function dismissThreatBanner() {
  const container = document.getElementById("threat-banner-container");
  if (container) container.style.display = "none";
}

function renderPrototypes(prototypes) {
  const container = document.getElementById("prototypes-container");
  if (!container || !prototypes) return;
  container.innerHTML = "";

  prototypes.slice(0, 8).forEach((p, idx) => {
    const chip = document.createElement("div");
    chip.className = "proto-chip";
    const coordSummary = Array.isArray(p) ? `[${p.slice(0, 3).map(n => Number(n).toFixed(2)).join(", ")}...]` : `Centroid #${idx+1}`;
    chip.innerHTML = `
      <div class="proto-id">PROTOTYPE #${idx + 1}</div>
      <div class="proto-coords">${coordSummary}</div>
      <div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">16-D Bottleneck Centroid</div>
    `;
    container.appendChild(chip);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart.js Visualizations Setup
// ─────────────────────────────────────────────────────────────────────────────
function initCharts() {
  // Chart 1: Real-time Autoencoder Reconstruction MSE Timeline
  const ctxTimeline = document.getElementById("chart-mse-timeline")?.getContext("2d");
  if (ctxTimeline) {
    state.charts.mseTimeline = new Chart(ctxTimeline, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Reconstruction MSE",
            data: [],
            borderColor: "#06b6d4",
            backgroundColor: "rgba(6, 182, 212, 0.1)",
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: [],
            borderWidth: 2
          },
          {
            label: "Threshold (τ = 0.0687)",
            data: [],
            borderColor: "#ef4444",
            borderWidth: 1.5,
            borderDash: [5, 4],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 }, maxTicksLimit: 8 }
          },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 } },
            suggestedMin: 0,
            suggestedMax: 0.12
          }
        },
        plugins: {
          legend: {
            labels: { color: "#94a3b8", font: { family: "Outfit", size: 10 }, boxWidth: 12 }
          }
        }
      }
    });
  }

  // Chart 2: Threat Distribution Donut
  const ctxDist = document.getElementById("chart-distribution")?.getContext("2d");
  if (ctxDist) {
    state.charts.distribution = new Chart(ctxDist, {
      type: "doughnut",
      data: {
        labels: ["BENIGN (Safe)", "DoS (Flood)", "PortScan", "Infiltration Zero-Day"],
        datasets: [{
          data: [1, 0, 0, 0],
          backgroundColor: ["#10b981", "#ef4444", "#f59e0b", "#a855f7"],
          borderColor: "#0d1322",
          borderWidth: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "65%",
        plugins: {
          legend: {
            position: "right",
            labels: { color: "#94a3b8", font: { family: "Outfit", size: 10.5 }, boxWidth: 12 }
          }
        }
      }
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Real-Time WebSocket Telemetry Receiver
// ─────────────────────────────────────────────────────────────────────────────
function initWebSocket() {
  const wsUrl = `${WS_BASE}/ws/telemetry`;
  const wsBadge = document.getElementById("ws-status-badge");

  let socket;
  try {
    socket = new WebSocket(wsUrl);
  } catch (e) {
    console.warn("WebSocket init error:", e);
    return;
  }

  socket.onopen = () => {
    if (wsBadge) wsBadge.innerHTML = `<span class="dot dot-green pulse"></span> LIVE WS`;
    console.log("📡 Connected to Kadak NIDS Telemetry Stream!");
  };

  socket.onmessage = (event) => {
    if (state.isStreamPaused) return;

    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "init") {
        if (msg.tau_threshold) state.threshold = msg.tau_threshold;
        if (msg.prototypes) renderPrototypes(msg.prototypes);
        if (msg.recent_history && Array.isArray(msg.recent_history)) {
          msg.recent_history.forEach(f => handleIncomingFlow(f, false));
        }
      } else if (msg.type === "flow_telemetry" && msg.data) {
        handleIncomingFlow(msg.data);
      } else if (msg.event_type === "PACKET_INSPECTED" && msg.telemetry) {
        handleIncomingFlow(msg.telemetry);
      }
    } catch (e) {
      console.error("JSON parse error on WebSocket message:", e);
    }
  };

  socket.onclose = () => {
    if (wsBadge) wsBadge.innerHTML = `<span class="dot" style="background:#ef4444;"></span> RECONNECTING`;
    console.log("🔌 WebSocket closed. Reconnecting in 3s...");
    setTimeout(initWebSocket, 3000);
  };

  socket.onerror = (err) => {
    console.warn("WebSocket error:", err);
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Ingest & Process Telemetry Event
// ─────────────────────────────────────────────────────────────────────────────
function handleIncomingFlow(flow, playSound = true) {
  if (!flow) return;

  state.totalFlows++;
  state.flowsHistory.unshift(flow);
  if (state.flowsHistory.length > 100) state.flowsHistory.pop();

  const cls = flow.classification || "BENIGN";
  if (cls === "BENIGN" || flow.quarantine_action === "ALLOW") {
    state.benignCount++;
  } else if (flow.is_zero_day || cls === "Infiltration_ZeroDay") {
    state.zeroDayCount++;
    if (playSound) {
      playAlertSound("zeroday");
      triggerThreatBanner(flow);
    }
  } else {
    state.knownCount++;
    if (playSound) {
      playAlertSound("known");
      triggerThreatBanner(flow);
    }
  }

  state.latencies.push(flow.latency_ms || 9.5);
  if (state.latencies.length > 30) state.latencies.shift();

  updateKPICards();
  insertTableRow(flow);
  updateCharts(flow);
  updateSecurityPosture();
}

function updateSecurityPosture() {
  const el = document.getElementById("val-posture");
  if (!el) return;

  if (state.zeroDayCount > 0) {
    el.className = "status-val text-purple";
    el.innerHTML = `<span class="dot" style="background:#a855f7;"></span> CRITICAL ZERO-DAY`;
  } else if (state.knownCount > 0) {
    el.className = "status-val text-orange";
    el.innerHTML = `<span class="dot" style="background:#f97316;"></span> ELEVATED THREAT`;
  } else {
    el.className = "status-val text-green";
    el.innerHTML = `<span class="dot dot-green"></span> GUARDED`;
  }
}

function updateKPICards() {
  const tf = document.getElementById("kpi-total-flows");
  if (tf) tf.textContent = state.totalFlows.toLocaleString();

  const bc = document.getElementById("kpi-benign-count");
  if (bc) bc.textContent = state.benignCount.toLocaleString();

  const benignPct = state.totalFlows > 0 ? ((state.benignCount / state.totalFlows) * 100).toFixed(1) : "100.0";
  const bp = document.getElementById("kpi-benign-pct");
  if (bp) bp.textContent = `${benignPct}%`;

  const kc = document.getElementById("kpi-known-count");
  if (kc) kc.textContent = state.knownCount.toLocaleString();

  const zc = document.getElementById("kpi-zeroday-count");
  if (zc) zc.textContent = state.zeroDayCount.toLocaleString();

  const avgLat = state.latencies.reduce((a, b) => a + b, 0) / Math.max(1, state.latencies.length);
  const al = document.getElementById("kpi-avg-latency");
  if (al) al.innerHTML = `${avgLat.toFixed(1)} <span class="unit">ms</span>`;
}

function insertTableRow(flow) {
  const tbody = document.getElementById("flow-tbody");
  if (!tbody) return;

  const filterEl = document.getElementById("filter-classification");
  const filterVal = filterEl ? filterEl.value : "ALL";

  if (filterVal !== "ALL" && flow.classification !== filterVal) {
    return;
  }

  const tr = document.createElement("tr");
  tr.className = "new-row-flash";
  tr.onclick = () => openForensicsModal(flow);

  let badgeClass = "badge-benign";
  let actionClass = "action-allow";
  let actionText = "ALLOW";

  if (flow.classification === "DoS") {
    badgeClass = "badge-dos";
    actionClass = "action-block";
    actionText = "BLOCK";
  } else if (flow.classification === "PortScan") {
    badgeClass = "badge-portscan";
    actionClass = "action-block";
    actionText = "BLOCK";
  } else if (flow.classification === "Infiltration_ZeroDay" || flow.is_zero_day) {
    badgeClass = "badge-zeroday";
    actionClass = "action-quarantine";
    actionText = "QUARANTINE";
  }

  const timeStr = flow.timestamp ? flow.timestamp.split("T")[1]?.replace("Z", "") : new Date().toLocaleTimeString();
  const reconErr = typeof flow.reconstruction_error === "number" ? flow.reconstruction_error : 0.0;
  const mseColor = reconErr >= state.threshold ? "#ef4444" : "#10b981";
  const barWidth = Math.min(100, Math.max(5, (reconErr / (state.threshold * 4)) * 100));

  tr.innerHTML = `
    <td>${timeStr || new Date().toLocaleTimeString()}</td>
    <td>${flow.source_ip || "192.168.1.100"}</td>
    <td>${flow.destination_ip || "10.0.0.1"}:${flow.destination_port || 80}</td>
    <td><span class="badge ${badgeClass}">${flow.classification}</span></td>
    <td><span class="tier-label">${flow.detection_tier}</span></td>
    <td>
      <div class="mse-bar-container">
        <span class="mse-val" style="color: ${mseColor}">${reconErr.toFixed(4)}</span>
        <div class="mse-bar">
          <div class="mse-fill" style="width: ${barWidth}%; background: ${mseColor};"></div>
        </div>
      </div>
    </td>
    <td><span class="action-pill ${actionClass}">${actionText}</span></td>
  `;

  tbody.insertBefore(tr, tbody.firstChild);

  // Keep table rows limited for DOM efficiency
  while (tbody.children.length > 50) {
    tbody.removeChild(tbody.lastChild);
  }
}

function updateCharts(flow) {
  const reconErr = typeof flow.reconstruction_error === "number" ? flow.reconstruction_error : 0.0;

  // 1. Update Reconstruction MSE Timeline
  if (state.charts.mseTimeline) {
    const chart = state.charts.mseTimeline;
    const timeLabel = new Date().toLocaleTimeString().split(" ")[0];

    chart.data.labels.push(timeLabel);
    chart.data.datasets[0].data.push(reconErr);
    chart.data.datasets[1].data.push(state.threshold);

    const ptColor = reconErr >= state.threshold ? "#ef4444" : "#06b6d4";
    chart.data.datasets[0].pointBackgroundColor.push(ptColor);

    if (chart.data.labels.length > 25) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
      chart.data.datasets[1].data.shift();
      chart.data.datasets[0].pointBackgroundColor.shift();
    }
    chart.update();
  }

  // 2. Update Threat Distribution Donut
  if (state.charts.distribution) {
    const chart = state.charts.distribution;
    const dosCount = state.flowsHistory.filter(f => f.classification === "DoS").length;
    const scanCount = state.flowsHistory.filter(f => f.classification === "PortScan").length;

    chart.data.datasets[0].data = [
      Math.max(1, state.benignCount),
      dosCount,
      scanCount,
      state.zeroDayCount
    ];
    chart.update();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive Attack Injection Handlers
// ─────────────────────────────────────────────────────────────────────────────
function bindEventHandlers() {
  document.getElementById("btn-inject-benign")?.addEventListener("click", () => injectThreat("BENIGN"));
  document.getElementById("btn-inject-dos")?.addEventListener("click", () => injectThreat("DoS"));
  document.getElementById("btn-inject-portscan")?.addEventListener("click", () => injectThreat("PortScan"));
  document.getElementById("btn-inject-zeroday")?.addEventListener("click", () => injectThreat("ZeroDay"));

  const toggleBtn = document.getElementById("btn-toggle-stream");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", async () => {
      state.isStreamPaused = !state.isStreamPaused;
      const textSpan = document.getElementById("stream-btn-text");
      if (state.isStreamPaused) {
        textSpan.textContent = "▶️ Resume Feed";
        toggleBtn.classList.add("paused");
        await fetch(`${BACKEND_BASE}/api/replay/stop`, { method: "POST" });
      } else {
        textSpan.textContent = "⏸️ Pause Feed";
        toggleBtn.classList.remove("paused");
        await fetch(`${BACKEND_BASE}/api/replay/start`, { method: "POST" });
      }
    });
  }

  document.getElementById("btn-close-modal")?.addEventListener("click", () => {
    document.getElementById("forensics-modal")?.classList.remove("open");
  });

  document.getElementById("filter-classification")?.addEventListener("change", () => {
    // Re-render table with current history
    const tbody = document.getElementById("flow-tbody");
    if (tbody) {
      tbody.innerHTML = "";
      state.flowsHistory.forEach(f => insertTableRow(f));
    }
  });
}

async function injectThreat(threatType) {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/v1/simulate/flow?threat_type=${threatType}`, {
      method: "POST"
    });
    if (res.ok) {
      const data = await res.json();
      console.log(`[+] Injected threat [${threatType}]:`, data.verdict);
    }
  } catch (err) {
    console.error("Threat injection error:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forensics Inspection Modal
// ─────────────────────────────────────────────────────────────────────────────
function openForensicsModal(flow) {
  const modal = document.getElementById("forensics-modal");
  const content = document.getElementById("modal-content");
  if (!modal || !content) return;

  const isAnom = flow.reconstruction_error >= state.threshold;
  const departureRatio = (flow.reconstruction_error / Math.max(1e-5, state.threshold)).toFixed(1);

  content.innerHTML = `
    <div class="forensics-grid">
      <!-- Left Column: Core Verdict & Mathematical Proof -->
      <div class="forensics-col">
        <div class="modal-section-title">SECURITY VERDICT & DUAL-TIER TRIAGE</div>
        <div class="forensics-kv">
          <span class="f-key">Classification</span>
          <span class="f-val text-cyan font-bold">${flow.classification}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Threat Level</span>
          <span class="f-val ${flow.threat_level === 'CRITICAL' ? 'text-red font-bold' : flow.threat_level === 'HIGH' ? 'text-orange' : 'text-green'}">${flow.threat_level}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Arbitration Tier</span>
          <span class="f-val">${flow.detection_tier}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Quarantine Action</span>
          <span class="f-val font-bold ${flow.quarantine_action === 'QUARANTINE_HOST' ? 'text-purple' : flow.quarantine_action === 'BLOCK_FLOW' ? 'text-orange' : 'text-green'}">
            ${flow.quarantine_action}
          </span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Classification Confidence</span>
          <span class="f-val">${((flow.confidence || 0.99) * 100).toFixed(1)}%</span>
        </div>

        <div class="modal-section-title" style="margin-top: 18px;">MATHEMATICAL ANOMALY PROOF</div>
        <div class="forensics-kv">
          <span class="f-key">Autoencoder Reconstruction MSE</span>
          <span class="f-val font-mono ${isAnom ? 'text-red' : 'text-green'} font-bold">${(flow.reconstruction_error || 0).toFixed(6)}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Calibrated Threshold (τ)</span>
          <span class="f-val font-mono">${state.threshold.toFixed(6)}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Manifold Departure Ratio</span>
          <span class="f-val font-mono ${departureRatio > 1.0 ? 'text-purple font-bold' : 'text-green'}">${departureRatio}x ${departureRatio > 1.0 ? '(Breached Anomaly Boundary)' : '(Conforms to Manifold)'}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Novelty Distance to Prototypes</span>
          <span class="f-val font-mono">${(flow.prototype_distance || 0).toFixed(4)}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Inference Latency</span>
          <span class="f-val font-mono">${flow.latency_ms || 9.5} ms</span>
        </div>
      </div>

      <!-- Right Column: Network 5-Tuple & Context -->
      <div class="forensics-col">
        <div class="modal-section-title">NETWORK 5-TUPLE & REASONING</div>
        <div class="forensics-kv">
          <span class="f-key">Source IP</span>
          <span class="f-val font-mono">${flow.source_ip}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Destination IP:Port</span>
          <span class="f-val font-mono">${flow.destination_ip}:${flow.destination_port}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Protocol</span>
          <span class="f-val font-mono">${flow.protocol || "TCP"}</span>
        </div>
        <div class="forensics-kv">
          <span class="f-key">Timestamp (UTC)</span>
          <span class="f-val font-mono">${flow.timestamp || new Date().toISOString()}</span>
        </div>

        <div class="modal-section-title" style="margin-top: 18px;">FORENSIC RATIONALE</div>
        <div style="font-size: 11.5px; color: #cbd5e1; line-height: 1.5; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 6px; border: 1px solid var(--border-subtle);">
          ${flow.reason || "Autonomous decision based on LightGBM probability and Autoencoder reconstruction loss departure."}
        </div>
      </div>
    </div>
  `;

  modal.classList.add("open");
}
