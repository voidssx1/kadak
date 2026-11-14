/**
 * 🛡️ NIDS 2.0 Autonomous SOC Dashboard Logic
 * Manages real-time WebSocket telemetry, Chart.js streams,
 * interactive attack injections, and latent prototype visualizations.
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
  charts: {
    mseTimeline: null,
    distribution: null
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Audio Synthesizer for Threat Alerts (Web Audio API)
// ─────────────────────────────────────────────────────────────────────────────
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function playAlertSound(type = "zeroday") {
  try {
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);

    if (type === "zeroday") {
      // Ominous low-pitch dual-tone alarm
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(220, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(110, audioCtx.currentTime + 0.35);
      gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.35);
    } else {
      // Subtle blip for known signatures
      osc.type = "sine";
      osc.frequency.setValueAtTime(440, audioCtx.currentTime);
      gain.gain.setValueAtTime(0.05, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.15);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.15);
    }
  } catch (e) {
    // Audio optional if blocked by browser autoplay policy
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialization & Metadata Loading
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  await fetchModelMetadata();
  initWebSocket();
  bindEventHandlers();
});

async function fetchModelMetadata() {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/v1/models/info`);
    if (!res.ok) throw new Error("Backend not available");
    const data = await res.json();

    state.threshold = data.operating_threshold_tau || 0.0687;
    document.getElementById("val-tau").textContent = `τ = ${state.threshold.toFixed(4)}`;

    state.prototypes = data.metadata?.prototypes || [];
    document.getElementById("val-prototypes").textContent = `${state.prototypes.length} Centroids`;

    renderPrototypes(state.prototypes);

    // Update threshold line on MSE chart
    if (state.charts.mseTimeline) {
      state.charts.mseTimeline.options.plugins.annotation = {
        annotations: {
          line1: {
            type: 'line',
            yMin: state.threshold,
            yMax: state.threshold,
            borderColor: '#ef4444',
            borderWidth: 2,
            borderDash: [6, 4],
            label: {
              content: `τ Threshold (${state.threshold})`,
              display: true,
              position: 'end'
            }
          }
        }
      };
      state.charts.mseTimeline.update();
    }
  } catch (err) {
    console.warn("⚠️ Could not load /api/v1/models/info:", err);
  }
}

function renderPrototypes(prototypes) {
  const container = document.getElementById("prototypes-container");
  container.innerHTML = "";

  prototypes.slice(0, 8).forEach((p, idx) => {
    const chip = document.createElement("div");
    chip.className = "proto-chip";
    const coordSummary = `[${p.slice(0, 3).map(n => n.toFixed(2)).join(", ")}...]`;
    chip.innerHTML = `
      <div class="proto-id">PROTOTYPE #${idx + 1}</div>
      <div class="proto-coords">${coordSummary}</div>
      <div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">16-D Centroid</div>
    `;
    container.appendChild(chip);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart.js Visualizations Setup
// ─────────────────────────────────────────────────────────────────────────────
function initCharts() {
  // Chart 1: Real-time Autoencoder Reconstruction MSE Timeline
  const ctxTimeline = document.getElementById("chart-mse-timeline").getContext("2d");
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
      animation: { duration: 250 },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 }, maxTicksLimit: 8 }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 } },
          suggestedMin: 0,
          suggestedMax: 0.15
        }
      },
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { family: "Outfit", size: 10 }, boxWidth: 12 }
        }
      }
    }
  });

  // Chart 2: Threat Distribution Donut
  const ctxDist = document.getElementById("chart-distribution").getContext("2d");
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
    wsBadge.innerHTML = `<span class="dot dot-green pulse"></span> LIVE WS`;
    console.log("📡 Connected to NIDS 2.0 Telemetry Stream!");
  };

  socket.onmessage = (event) => {
    if (state.isStreamPaused) return;

    try {
      const msg = JSON.parse(event.data);
      if (msg.event_type === "PACKET_INSPECTED" && msg.telemetry) {
        handleIncomingFlow(msg.telemetry);
      }
    } catch (e) {
      console.error("JSON parse error on WebSocket message:", e);
    }
  };

  socket.onclose = () => {
    wsBadge.innerHTML = `<span class="dot" style="background:#ef4444;"></span> RECONNECTING`;
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
function handleIncomingFlow(flow) {
  // Prevent duplicate insertion if already processed
  if (flow.flow_id && state.flowsHistory.some(f => f.flow_id === flow.flow_id)) return;

  state.totalFlows++;
  state.flowsHistory.unshift(flow);
  if (state.flowsHistory.length > 80) state.flowsHistory.pop();

  // 1. Update KPI Counts
  const cls = flow.classification;
  if (cls === "BENIGN") {
    state.benignCount++;
  } else if (cls === "Infiltration_ZeroDay") {
    state.zeroDayCount++;
    playAlertSound("zeroday");
  } else {
    state.knownCount++;
    playAlertSound("known");
  }

  state.latencies.push(flow.latency_ms || 9.5);
  if (state.latencies.length > 30) state.latencies.shift();

  updateKPICards();

  // 2. Insert Row in Flow Table
  insertTableRow(flow);

  // 3. Update Real-Time Charts
  updateCharts(flow);
}

function updateKPICards() {
  document.getElementById("kpi-total-flows").textContent = state.totalFlows.toLocaleString();
  document.getElementById("kpi-benign-count").textContent = state.benignCount.toLocaleString();

  const benignPct = state.totalFlows > 0 ? ((state.benignCount / state.totalFlows) * 100).toFixed(1) : "100.0";
  document.getElementById("kpi-benign-pct").textContent = `${benignPct}%`;

  document.getElementById("kpi-known-count").textContent = state.knownCount.toLocaleString();
  document.getElementById("kpi-zeroday-count").textContent = state.zeroDayCount.toLocaleString();

  const avgLat = state.latencies.reduce((a, b) => a + b, 0) / Math.max(1, state.latencies.length);
  document.getElementById("kpi-avg-latency").innerHTML = `${avgLat.toFixed(1)} <span class="unit">ms</span>`;
}

function insertTableRow(flow) {
  const tbody = document.getElementById("flow-tbody");
  const filterVal = document.getElementById("filter-classification").value;

  // Filter check
  if (filterVal !== "ALL" && flow.classification !== filterVal) {
    return;
  }

  const tr = document.createElement("tr");
  tr.className = "new-row-flash";
  tr.onclick = () => openForensicsModal(flow);

  // Badges & styling mapping
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
  } else if (flow.classification === "Infiltration_ZeroDay") {
    badgeClass = "badge-zeroday";
    actionClass = "action-quarantine";
    actionText = "QUARANTINE";
  }

  const timeStr = flow.timestamp ? flow.timestamp.split("T")[1].replace("Z", "") : new Date().toLocaleTimeString();
  const mseColor = flow.reconstruction_error >= state.threshold ? "#ef4444" : "#10b981";
  const barWidth = Math.min(100, Math.max(5, (flow.reconstruction_error / (state.threshold * 4)) * 100));

  tr.innerHTML = `
    <td>${timeStr}</td>
    <td>${flow.source_ip}</td>
    <td>${flow.destination_ip}:${flow.destination_port}</td>
    <td><span class="badge ${badgeClass}">${flow.classification}</span></td>
    <td><span class="tier-label">${flow.detection_tier}</span></td>
    <td>
      <div class="mse-bar-container">
        <span class="mse-val" style="color: ${mseColor}">${flow.reconstruction_error.toFixed(4)}</span>
        <div class="mse-bar">
          <div class="mse-fill" style="width: ${barWidth}%; background: ${mseColor};"></div>
        </div>
      </div>
    </td>
    <td><span class="action-pill ${actionClass}">${actionText}</span></td>
  `;

  tbody.insertBefore(tr, tbody.firstChild);

  // Keep table at max 40 rows
  while (tbody.children.length > 40) {
    tbody.removeChild(tbody.lastChild);
  }
}

function updateCharts(flow) {
  const chartMSE = state.charts.mseTimeline;
  const timeLabel = new Date().toLocaleTimeString().split(" ")[0];

  chartMSE.data.labels.push(timeLabel);
  chartMSE.data.datasets[0].data.push(flow.reconstruction_error);
  chartMSE.data.datasets[1].data.push(state.threshold);

  // Color point dynamically
  const ptColor = flow.classification === "Infiltration_ZeroDay" ? "#a855f7" :
                  flow.classification !== "BENIGN" ? "#ef4444" : "#06b6d4";
  chartMSE.data.datasets[0].pointBackgroundColor.push(ptColor);

  if (chartMSE.data.labels.length > 25) {
    chartMSE.data.labels.shift();
    chartMSE.data.datasets[0].data.shift();
    chartMSE.data.datasets[1].data.shift();
    chartMSE.data.datasets[0].pointBackgroundColor.shift();
  }
  chartMSE.update("none");

  // Update Donut
  const chartDist = state.charts.distribution;
  chartDist.data.datasets[0].data = [
    state.benignCount,
    state.flowsHistory.filter(f => f.classification === "DoS").length,
    state.flowsHistory.filter(f => f.classification === "PortScan").length,
    state.zeroDayCount
  ];
  chartDist.update();
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive Attack Injection Trigger Handlers
// ─────────────────────────────────────────────────────────────────────────────
function bindEventHandlers() {
  document.getElementById("btn-inject-benign").addEventListener("click", () => injectThreat("BENIGN"));
  document.getElementById("btn-inject-dos").addEventListener("click", () => injectThreat("DoS"));
  document.getElementById("btn-inject-portscan").addEventListener("click", () => injectThreat("PortScan"));
  document.getElementById("btn-inject-zeroday").addEventListener("click", () => injectThreat("ZeroDay"));

  // Stream Pause / Resume Toggle
  const toggleBtn = document.getElementById("btn-toggle-stream");
  toggleBtn.addEventListener("click", () => {
    state.isStreamPaused = !state.isStreamPaused;
    document.getElementById("stream-btn-text").textContent = state.isStreamPaused ? "▶️ Resume Feed" : "⏸️ Pause Feed";
    toggleBtn.style.borderColor = state.isStreamPaused ? "#f59e0b" : "rgba(100, 116, 139, 0.3)";
  });

  // Modal Close
  document.getElementById("btn-close-modal").addEventListener("click", () => {
    document.getElementById("forensics-modal").classList.remove("open");
  });

  // Filter Dropdown
  document.getElementById("filter-classification").addEventListener("change", () => {
    const tbody = document.getElementById("flow-tbody");
    tbody.innerHTML = "";
    state.flowsHistory.forEach(f => insertTableRow(f));
  });
}

async function injectThreat(type) {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/v1/simulate/flow?threat_type=${encodeURIComponent(type)}`, {
      method: "POST"
    });
    if (!res.ok) throw new Error("Simulation endpoint error");
    const data = await res.json();
    if (data.verdict) {
      handleIncomingFlow(data.verdict);
    }
  } catch (err) {
    console.error("Failed to inject threat:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forensics Detail Modal
// ─────────────────────────────────────────────────────────────────────────────
function openForensicsModal(flow) {
  const modal = document.getElementById("forensics-modal");
  const content = document.getElementById("modal-content");

  const departureRatio = (flow.reconstruction_error / Math.max(1e-5, flow.anomaly_threshold)).toFixed(1);

  content.innerHTML = `
    <div class="forensics-grid">
      <div class="forensic-item">
        <div class="forensic-label">SOURCE IP ADDRESS</div>
        <div class="forensic-val text-cyan">${flow.source_ip}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">TARGET DESTINATION</div>
        <div class="forensic-val">${flow.destination_ip}:${flow.destination_port}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">AI CLASSIFICATION</div>
        <div class="forensic-val text-purple">${flow.classification}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">OPERATIONAL ACTION</div>
        <div class="forensic-val text-red">${flow.quarantine_action}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">RECONSTRUCTION MSE ERROR</div>
        <div class="forensic-val">${flow.reconstruction_error}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">OPERATING THRESHOLD (τ)</div>
        <div class="forensic-val">${flow.anomaly_threshold} (${departureRatio}x departure)</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">LATENT PROTOTYPE DISTANCE</div>
        <div class="forensic-val">${flow.prototype_distance}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">INFERENCE ROUNDTRIP</div>
        <div class="forensic-val">${flow.latency_ms} ms</div>
      </div>
    </div>

    <div class="forensic-reason">
      <strong>AI Triaging Rationale:</strong><br>
      ${flow.reason}
    </div>
  `;

  modal.classList.add("open");
}
