/**
 * nids> mission control — Enterprise SOC Dashboard Logic
 * Integrates real-time WebSocket telemetry, dual-tier LightGBM + Autoencoder
 * inference stream, interactive threat simulation, and forensics triage drawer.
 * Strictly adheres to Splunk SIEM enterprise design standards.
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
    eventsTimeline: null,
    distribution: null
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
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
    if (!res.ok) throw new Error("Backend unavailable");
    const data = await res.json();

    state.threshold = data.operating_threshold_tau || 0.0687;
    const tauFormatted = `τ = ${state.threshold.toFixed(4)}`;
    
    document.getElementById("val-tau").textContent = tauFormatted;
    const cardTau = document.getElementById("card-tau-stat");
    if (cardTau) cardTau.textContent = tauFormatted;
    const sumThresh = document.getElementById("summary-threshold");
    if (sumThresh) sumThresh.textContent = state.threshold.toFixed(4);

    state.prototypes = data.metadata?.prototypes || [];
    document.getElementById("val-prototypes").textContent = state.prototypes.length || 8;

    renderPrototypes(state.prototypes);
  } catch (err) {
    console.warn("Notice: /api/v1/models/info running in standard fallback mode:", err);
  }
}

function renderPrototypes(prototypes) {
  const container = document.getElementById("prototypes-container");
  if (!container) return;
  container.innerHTML = "";

  const items = prototypes && prototypes.length > 0 ? prototypes : Array.from({ length: 8 }, (_, i) => [0.12 * (i + 1), -0.05 * (i + 1), 0.22]);

  items.slice(0, 8).forEach((p, idx) => {
    const cell = document.createElement("div");
    cell.className = "proto-cell";
    const coordStr = Array.isArray(p) ? `[${p.slice(0, 3).map(n => n.toFixed(2)).join(", ")}...]` : "[0.15, -0.08, 0.41...]";
    cell.innerHTML = `
      <div class="proto-label">CENTROID #${idx + 1}</div>
      <div class="proto-val">${coordStr}</div>
    `;
    container.appendChild(cell);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart.js Setup (Splunk Enterprise Clean Dark Theme)
// ─────────────────────────────────────────────────────────────────────────────
function initCharts() {
  const gridColor = "#1a1e28";
  const tickColor = "#5e6878";
  const monoFont = { family: "JetBrains Mono", size: 10 };
  const sansFont = { family: "Inter", size: 10.5 };

  // 1. Autoencoder Anomaly Departure vs Threshold Curve (Splunk SLA curve)
  const ctxMSE = document.getElementById("chart-mse-timeline").getContext("2d");
  state.charts.mseTimeline = new Chart(ctxMSE, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Flow MSE",
          data: [],
          borderColor: "#22c55e",
          borderWidth: 1.8,
          pointRadius: 2,
          pointHoverRadius: 4,
          fill: false,
          tension: 0.25
        },
        {
          label: "Threshold (τ)",
          data: [],
          borderColor: "#e55353",
          borderWidth: 1.2,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 150 },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: tickColor, font: monoFont, maxTicksLimit: 6 }
        },
        y: {
          grid: { color: gridColor },
          ticks: { color: tickColor, font: monoFont },
          suggestedMin: 0,
          suggestedMax: 0.12
        }
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: { color: "#8c96a5", font: sansFont, boxWidth: 10 }
        }
      }
    }
  });

  // 2. Notable Events Over Time (Multi-line Timeline)
  const ctxEvents = document.getElementById("chart-events-timeline").getContext("2d");
  state.charts.eventsTimeline = new Chart(ctxEvents, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Benign",
          data: [],
          borderColor: "#22c55e",
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.3
        },
        {
          label: "DoS",
          data: [],
          borderColor: "#e55353",
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.3
        },
        {
          label: "PortScan",
          data: [],
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.3
        },
        {
          label: "Zero-Day",
          data: [],
          borderColor: "#a78bfa",
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 150 },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: tickColor, font: monoFont, maxTicksLimit: 6 }
        },
        y: {
          grid: { color: gridColor },
          ticks: { color: tickColor, font: monoFont, stepSize: 1 },
          suggestedMin: 0
        }
      },
      plugins: {
        legend: { display: false } // Custom legend already rendered in HTML
      }
    }
  });

  // 3. Events By Status Donut Chart (Splunk SIEM Donut)
  const ctxDist = document.getElementById("chart-distribution").getContext("2d");
  state.charts.distribution = new Chart(ctxDist, {
    type: "doughnut",
    data: {
      labels: ["High (DoS/ZeroDay)", "Medium (PortScan)", "Low (Benign)"],
      datasets: [{
        data: [0, 0, 1],
        backgroundColor: ["#e55353", "#f59e0b", "#22c55e"],
        borderColor: "#12141a",
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "74%",
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Real-Time WebSocket Telemetry
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
    wsBadge.textContent = "LIVE WS";
    console.log("Connected to nids> mission control live stream.");
  };

  socket.onmessage = (event) => {
    if (state.isStreamPaused) return;

    try {
      const msg = JSON.parse(event.data);
      if (msg.event_type === "PACKET_INSPECTED" && msg.telemetry) {
        handleIncomingFlow(msg.telemetry);
      }
    } catch (e) {
      console.error("JSON parse error:", e);
    }
  };

  socket.onclose = () => {
    wsBadge.textContent = "RECONNECTING";
    setTimeout(initWebSocket, 3000);
  };

  socket.onerror = (err) => {
    console.warn("WebSocket error:", err);
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Process Telemetry Event
// ─────────────────────────────────────────────────────────────────────────────
function handleIncomingFlow(flow) {
  if (flow.flow_id && state.flowsHistory.some(f => f.flow_id === flow.flow_id)) return;

  state.totalFlows++;
  state.flowsHistory.unshift(flow);
  if (state.flowsHistory.length > 100) state.flowsHistory.pop();

  const cls = flow.classification;
  if (cls === "BENIGN") {
    state.benignCount++;
  } else if (cls === "Infiltration_ZeroDay") {
    state.zeroDayCount++;
  } else {
    state.knownCount++;
  }

  state.latencies.push(flow.latency_ms || 0.8);
  if (state.latencies.length > 30) state.latencies.shift();

  updateKPICards(flow);
  insertTableRow(flow);
  updateCharts(flow);
}

function updateKPICards(latestFlow) {
  document.getElementById("kpi-total-flows").textContent = state.totalFlows.toLocaleString();
  document.getElementById("kpi-benign-count").textContent = state.benignCount.toLocaleString();

  const benignPct = state.totalFlows > 0 ? ((state.benignCount / state.totalFlows) * 100).toFixed(1) : "100.0";
  document.getElementById("kpi-benign-pct").textContent = `${benignPct}%`;

  document.getElementById("kpi-known-count").textContent = state.knownCount.toLocaleString();
  document.getElementById("kpi-zeroday-count").textContent = state.zeroDayCount.toLocaleString();

  const avgLat = state.latencies.reduce((a, b) => a + b, 0) / Math.max(1, state.latencies.length);
  document.getElementById("kpi-avg-latency").innerHTML = `${avgLat.toFixed(1)}<span class="kpi-unit">ms</span>`;
  const perfTriage = document.getElementById("perf-mean-triage");
  if (perfTriage) perfTriage.innerHTML = `${avgLat.toFixed(1)}<span class="unit">ms</span>`;

  const maxLat = Math.max(...state.latencies);
  const perfMax = document.getElementById("perf-max-roundtrip");
  if (perfMax) perfMax.innerHTML = `${maxLat.toFixed(1)}<span class="unit">ms</span>`;

  // Enterprise Cybersecurity ROI Valuation (Ponemon / IBM Security Metrics):
  // Tier-1 Known Blocked Attacks prevent ~$850 in remediation/analyst hours
  // Tier-2 Zero-Day Quarantined Threats prevent ~$4,200 in critical breach/downtime damages
  const rawDollars = (state.knownCount * 850) + (state.zeroDayCount * 4200);
  let dollarsText = "$0";
  if (rawDollars >= 1000000) {
    dollarsText = `$${(rawDollars / 1000000).toFixed(2)}M`;
  } else if (rawDollars >= 1000) {
    dollarsText = `$${rawDollars.toLocaleString()}`;
  } else if (rawDollars > 0) {
    dollarsText = `$${rawDollars}`;
  }
  const dollarsElem = document.getElementById("kpi-dollars-saved");
  if (dollarsElem) dollarsElem.textContent = dollarsText;

  if (latestFlow) {
    const mseVal = latestFlow.reconstruction_error;
    document.getElementById("kpi-latest-mse").textContent = mseVal.toFixed(4);

    const isExceeded = mseVal >= state.threshold;
    const mseStatus = document.getElementById("kpi-mse-status");
    const sumVerdict = document.getElementById("summary-tau-verdict");

    if (isExceeded) {
      mseStatus.className = "kpi-delta delta-red";
      mseStatus.innerHTML = `<span class="delta-arrow">▲</span> &gt; τ`;
      if (sumVerdict) {
        sumVerdict.className = "summary-val text-red";
        sumVerdict.textContent = "Exceeded (Anomaly)";
      }
    } else {
      mseStatus.className = "kpi-delta delta-green";
      mseStatus.innerHTML = `<span class="delta-arrow">▼</span> &lt; τ`;
      if (sumVerdict) {
        sumVerdict.className = "summary-val text-green";
        sumVerdict.textContent = "Nominal (Within τ)";
      }
    }
  }
}

function insertTableRow(flow) {
  const tbody = document.getElementById("flow-tbody");
  const filterVal = document.getElementById("filter-classification").value;

  if (filterVal !== "ALL" && flow.classification !== filterVal) {
    return;
  }

  const tr = document.createElement("tr");
  tr.onclick = () => openForensicsModal(flow);

  let badgeClass = "badge-benign";
  let actionClass = "act-allow";
  let actionText = "ALLOW";

  if (flow.classification === "DoS") {
    badgeClass = "badge-dos";
    actionClass = "act-block";
    actionText = "BLOCK";
  } else if (flow.classification === "PortScan") {
    badgeClass = "badge-portscan";
    actionClass = "act-block";
    actionText = "BLOCK";
  } else if (flow.classification === "Infiltration_ZeroDay") {
    badgeClass = "badge-zeroday";
    actionClass = "act-quarantine";
    actionText = "QUARANTINE";
  }

  const timeStr = flow.timestamp ? flow.timestamp.split("T")[1].replace("Z", "").slice(0, 8) : new Date().toLocaleTimeString();
  const mseColor = flow.reconstruction_error >= state.threshold ? "var(--color-red)" : "var(--color-green)";

  tr.innerHTML = `
    <td>${timeStr}</td>
    <td>${flow.source_ip}</td>
    <td>${flow.destination_ip}:${flow.destination_port}</td>
    <td><span class="badge-siem ${badgeClass}">${flow.classification}</span></td>
    <td>${flow.detection_tier}</td>
    <td style="color:${mseColor}">${flow.reconstruction_error.toFixed(4)}</td>
    <td><span class="action-pill ${actionClass}">${actionText}</span></td>
  `;

  tbody.insertBefore(tr, tbody.firstChild);

  while (tbody.children.length > 50) {
    tbody.removeChild(tbody.lastChild);
  }
}

function updateCharts(flow) {
  const timeLabel = new Date().toLocaleTimeString().split(" ")[0];

  // 1. Update MSE Departure Chart
  const chartMSE = state.charts.mseTimeline;
  chartMSE.data.labels.push(timeLabel);
  chartMSE.data.datasets[0].data.push(flow.reconstruction_error);
  chartMSE.data.datasets[1].data.push(state.threshold);

  if (chartMSE.data.labels.length > 20) {
    chartMSE.data.labels.shift();
    chartMSE.data.datasets[0].data.shift();
    chartMSE.data.datasets[1].data.shift();
  }
  chartMSE.update("none");

  // 2. Update Notable Events Over Time Chart
  const chartEvents = state.charts.eventsTimeline;
  chartEvents.data.labels.push(timeLabel);

  const benignRate = flow.classification === "BENIGN" ? 1 : 0;
  const dosRate = flow.classification === "DoS" ? 1 : 0;
  const scanRate = flow.classification === "PortScan" ? 1 : 0;
  const zeroRate = flow.classification === "Infiltration_ZeroDay" ? 1 : 0;

  chartEvents.data.datasets[0].data.push(benignRate);
  chartEvents.data.datasets[1].data.push(dosRate);
  chartEvents.data.datasets[2].data.push(scanRate);
  chartEvents.data.datasets[3].data.push(zeroRate);

  if (chartEvents.data.labels.length > 20) {
    chartEvents.data.labels.shift();
    chartEvents.data.datasets.forEach(d => d.data.shift());
  }
  chartEvents.update("none");

  // 3. Update Splunk Donut Chart & Center Stats
  const highCount = state.zeroDayCount + state.flowsHistory.filter(f => f.classification === "DoS").length;
  const medCount = state.flowsHistory.filter(f => f.classification === "PortScan").length;
  const lowCount = state.benignCount;

  const chartDist = state.charts.distribution;
  chartDist.data.datasets[0].data = [highCount, medCount, lowCount];
  chartDist.update("none");

  document.getElementById("donut-center-count").textContent = state.totalFlows.toLocaleString();
  document.getElementById("qty-high").textContent = highCount;
  document.getElementById("qty-medium").textContent = medCount;
  document.getElementById("qty-low").textContent = lowCount;
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive Attack Injections (Operations Console)
// ─────────────────────────────────────────────────────────────────────────────
function bindEventHandlers() {
  document.getElementById("btn-inject-benign").addEventListener("click", () => injectThreat("BENIGN"));
  document.getElementById("btn-inject-dos").addEventListener("click", () => injectThreat("DoS"));
  document.getElementById("btn-inject-portscan").addEventListener("click", () => injectThreat("PortScan"));
  document.getElementById("btn-inject-zeroday").addEventListener("click", () => injectThreat("ZeroDay"));

  const toggleBtn = document.getElementById("btn-toggle-stream");
  toggleBtn.addEventListener("click", () => {
    state.isStreamPaused = !state.isStreamPaused;
    document.getElementById("stream-btn-text").textContent = state.isStreamPaused ? "Resume Stream" : "Pause Stream";
    toggleBtn.style.color = state.isStreamPaused ? "var(--color-amber)" : "var(--text-secondary)";
  });

  document.getElementById("btn-close-modal").addEventListener("click", () => {
    document.getElementById("forensics-modal").classList.remove("open");
  });

  document.getElementById("filter-classification").addEventListener("change", () => {
    const tbody = document.getElementById("flow-tbody");
    tbody.innerHTML = "";
    state.flowsHistory.forEach(f => insertTableRow(f));
  });

  // Smooth Navigation Link Handling
  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", (e) => {
      document.querySelectorAll(".nav-item").forEach(item => item.classList.remove("active"));
      link.parentElement.classList.add("active");
    });
  });
}

async function injectThreat(type) {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/v1/simulate/flow?threat_type=${encodeURIComponent(type)}`, {
      method: "POST"
    });
    if (!res.ok) throw new Error("Threat injection error");
    const data = await res.json();
    if (data.verdict) {
      handleIncomingFlow(data.verdict);
    }
  } catch (err) {
    console.error("Threat injection failure:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forensic Modal Details
// ─────────────────────────────────────────────────────────────────────────────
function openForensicsModal(flow) {
  const modal = document.getElementById("forensics-modal");
  const content = document.getElementById("modal-content");

  const departureRatio = (flow.reconstruction_error / Math.max(1e-5, flow.anomaly_threshold)).toFixed(2);

  // ── Build SHAP / Anomaly Attribution block ─────────────────────────────
  let shapBlock = "";
  const attrs = flow.shap_attributions;
  if (attrs && attrs.length > 0) {
    const isZeroDay = flow.is_zero_day;
    const maxAbs = Math.max(...attrs.map(a => a.abs_value), 1e-9);
    const bars = attrs.map(a => {
      const pct = Math.max(3, Math.min(100, ((a.abs_value / maxAbs) * 100))).toFixed(1);
      const isPositive = a.shap_value >= 0;
      const color = isZeroDay
        ? "var(--color-purple, #a855f7)"
        : (isPositive ? "var(--color-red)" : "var(--color-green)");
      const sign = isPositive ? "+" : "";
      return `
        <div class="shap-row">
          <div class="shap-feat" title="${a.feature}">${a.feature}</div>
          <div class="shap-bar-wrap">
            <div class="shap-bar" style="width:${pct}%;background:${color}"></div>
          </div>
          <div class="shap-val" style="color:${color}">${sign}${a.shap_value.toFixed(4)}</div>
        </div>`;
    }).join("");

    const title = isZeroDay
      ? "ANOMALY FEATURE ATTRIBUTIONS (AUTOENCODER DEPARTURE)"
      : "SHAP FEATURE ATTRIBUTIONS";
    const subtitle = isZeroDay
      ? `Top-10 Anomaly Drivers — Reconstruction MSE: ${flow.reconstruction_error.toFixed(4)} &gt; τ ${flow.anomaly_threshold} (${departureRatio}x departure)`
      : "Top-10 drivers — Tier-1 LightGBM local explanation";
    const legend = isZeroDay
      ? `<span style="color:var(--color-purple, #a855f7)">▮ Anomaly Departure from benign baseline</span>`
      : `<span style="color:var(--color-red)">▮ Pushes toward threat</span>
         <span style="color:var(--color-green)">▮ Pushes toward benign</span>`;

    shapBlock = `
      <div class="shap-section">
        <div class="shap-header">
          <span class="shap-title">${title}</span>
          <span class="shap-subtitle">${subtitle}</span>
        </div>
        <div class="shap-legend">
          ${legend}
        </div>
        <div class="shap-chart">${bars}</div>
      </div>`;
  }

  content.innerHTML = `
    <div class="forensics-grid">
      <div class="forensic-item">
        <div class="forensic-label">SOURCE IP</div>
        <div class="forensic-val">${flow.source_ip}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">DESTINATION</div>
        <div class="forensic-val">${flow.destination_ip}:${flow.destination_port}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">CLASSIFICATION</div>
        <div class="forensic-val" style="color:var(--color-purple)">${flow.classification}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">ACTION</div>
        <div class="forensic-val" style="color:var(--color-red)">${flow.quarantine_action}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">RECONSTRUCTION MSE</div>
        <div class="forensic-val">${flow.reconstruction_error.toFixed(6)}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">THRESHOLD (τ)</div>
        <div class="forensic-val">${flow.anomaly_threshold} (${departureRatio}x departure)</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">PROTOTYPE DISTANCE</div>
        <div class="forensic-val">${flow.prototype_distance ? flow.prototype_distance.toFixed(4) : "N/A"}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">INFERENCE ROUNDTRIP</div>
        <div class="forensic-val">${flow.latency_ms ? flow.latency_ms.toFixed(2) : "0.80"} ms</div>
      </div>
    </div>
    <div class="forensic-reason">
      <strong>AI Rationale:</strong> ${flow.reason}
    </div>
    ${shapBlock}
  `;

  modal.classList.add("open");
}

