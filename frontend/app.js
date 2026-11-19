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
  flowReceiveTimes: [],
  isStreamPaused: false,
  topologyZoom: 1,
  selectedEndpoint: null,
  topologyPositions: new Map(),
  topologyDefaultPositions: new Map(),
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
  renderTopology();
  await fetchModelMetadata();
  initWebSocket();
  bindEventHandlers();
  updatePerformanceMetrics();
  setInterval(updatePerformanceMetrics, 1000);
});

async function fetchModelMetadata() {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/v1/models/info`);
    if (!res.ok) throw new Error("Backend unavailable");
    const data = await res.json();

    state.threshold = data.operating_threshold_tau || 0.0687;
    const tauFormatted = `τ = ${state.threshold.toFixed(4)}`;
    
    const cardTau = document.getElementById("card-tau-stat");
    if (cardTau) cardTau.textContent = tauFormatted;
    const sumThresh = document.getElementById("summary-threshold");
    if (sumThresh) sumThresh.textContent = state.threshold.toFixed(4);

    state.prototypes = data.metadata?.prototypes || [];

  } catch (err) {
    console.warn("Notice: /api/v1/models/info running in standard fallback mode:", err);
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Chart.js Setup (Splunk Enterprise Clean Dark Theme)
// ─────────────────────────────────────────────────────────────────────────────
function initCharts() {
  const gridColor = "#202822";
  const tickColor = "#788e90";
  const monoFont = { family: "DM Sans", size: 10.5 };
  const sansFont = { family: "DM Sans", size: 11 };

  // 1. Autoencoder Anomaly Departure vs Threshold Curve (Splunk SLA curve)
  const ctxMSE = document.getElementById("chart-mse-timeline").getContext("2d");
  state.charts.mseTimeline = new Chart(ctxMSE, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Model error",
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
          borderWidth: 1.8,
          pointRadius: 2,
          pointHoverRadius: 4,
          tension: 0.3
        },
        {
          label: "DoS",
          data: [],
          borderColor: "#e55353",
          borderWidth: 1.8,
          pointRadius: 2,
          pointHoverRadius: 4,
          tension: 0.3
        },
        {
          label: "PortScan",
          data: [],
          borderColor: "#f59e0b",
          borderWidth: 1.8,
          pointRadius: 2,
          pointHoverRadius: 4,
          tension: 0.3
        },
        {
          label: "Zero-Day",
          data: [],
          borderColor: "#a78bfa",
          borderWidth: 1.8,
          pointRadius: 2,
          pointHoverRadius: 4,
          tension: 0.3
        },
        {
          label: "Brute Force",
          data: [],
          borderColor: "#9bb9bb",
          borderWidth: 1.8,
          pointRadius: 2,
          pointHoverRadius: 4,
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
          ticks: { color: tickColor, font: monoFont, maxTicksLimit: 8 }
        },
        y: {
          grid: { color: gridColor },
          ticks: { color: tickColor, font: monoFont, stepSize: 1 },
          beginAtZero: true
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
    // A backend restart can change the inference path; don't mix old timings into new percentiles.
    state.latencies = [];
    state.flowReceiveTimes = [];
    updatePerformanceMetrics();
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
  const receivedAt = Date.now();
  state.flowReceiveTimes.push(receivedAt);
  state.flowReceiveTimes = state.flowReceiveTimes.filter(time => receivedAt - time <= 10000);

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
  renderTopology();
}

function renderTopology() {
  const graph = document.getElementById("topology-graph");
  const empty = document.getElementById("topology-empty");
  if (!graph || !empty) return;

  const endpoints = new Map();
  const connections = new Map();
  for (const flow of state.flowsHistory) {
    const source = String(flow.source_ip || "").trim();
    const destination = String(flow.destination_ip || "").trim();
    if (!source || !destination) continue;
    const classification = flow.classification || "BENIGN";

    for (const [ip, peer] of [[source, destination], [destination, source]]) {
      if (!endpoints.has(ip)) {
        endpoints.set(ip, {
          ip, flows: 0, peers: new Set(), ports: new Map(),
          lastSeen: flow.timestamp || "", classification
        });
      }
      const endpoint = endpoints.get(ip);
      endpoint.flows++;
      endpoint.peers.add(peer);
      if (endpoint.classification === "BENIGN" && classification !== "BENIGN") {
        endpoint.classification = classification;
      }
    }

    const edgeKey = `${source}\u0000${destination}`;
    if (!connections.has(edgeKey)) {
      connections.set(edgeKey, {
        source, destination, port: flow.destination_port, count: 0, classification
      });
    }
    const edge = connections.get(edgeKey);
    edge.count++;
    if (edge.classification === "BENIGN" && classification !== "BENIGN") {
      edge.classification = classification;
    }

    if (flow.destination_port !== undefined && flow.destination_port !== null) {
      const ports = endpoints.get(source).ports;
      const port = String(flow.destination_port);
      ports.set(port, (ports.get(port) || 0) + 1);
    }
  }

  const nodes = [...endpoints.values()].sort((a, b) => b.flows - a.flows).slice(0, 14);
  const visible = new Set(nodes.map(node => node.ip));
  const edges = [...connections.values()]
    .filter(edge => visible.has(edge.source) && visible.has(edge.destination))
    .sort((a, b) => b.count - a.count)
    .slice(0, 22);

  empty.hidden = nodes.length > 0;
  graph.replaceChildren();
  const viewGroup = svgElement("g", {
    transform: `translate(300 150) scale(${state.topologyZoom}) translate(-300 -150)`
  });
  graph.appendChild(viewGroup);

  const defs = svgElement("defs");
  const marker = svgElement("marker", {
    id: "topology-arrow", viewBox: "0 0 10 10", refX: "8", refY: "5",
    markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse"
  });
  marker.appendChild(svgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#788e90" }));
  defs.appendChild(marker);
  viewGroup.appendChild(defs);

  const positions = new Map();
  for (const node of nodes) {
    const position = state.topologyPositions.get(node.ip)
      || state.topologyDefaultPositions.get(node.ip);
    if (position) positions.set(node.ip, position);
  }

  const circleSlots = Array.from({ length: 56 }, (_, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index / 56);
    return { x: 300 + Math.cos(angle) * 190, y: 150 + Math.sin(angle) * 90 };
  });
  for (const node of nodes) {
    if (positions.has(node.ip)) continue;
    let bestPosition = null;
    let bestSpacing = -1;
    for (const candidate of circleSlots) {
      const nearest = positions.size
        ? Math.min(...[...positions.values()].map(point => Math.hypot(point.x - candidate.x, point.y - candidate.y)))
        : Infinity;
      if (nearest > bestSpacing) {
        bestSpacing = nearest;
        bestPosition = candidate;
      }
    }
    positions.set(node.ip, bestPosition);
    state.topologyDefaultPositions.set(node.ip, bestPosition);
  }

  if (nodes.length && !visible.has(state.selectedEndpoint)) {
    state.selectedEndpoint = nodes[0].ip;
  }

  for (const edge of edges) {
    const from = positions.get(edge.source);
    const to = positions.get(edge.destination);
    if (!from || !to) continue;
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const length = Math.max(1, Math.hypot(dx, dy));
    const inset = Math.min(25, length / 3);
    const line = svgElement("line", {
      x1: from.x + dx / length * inset,
      y1: from.y + dy / length * inset,
      x2: to.x - dx / length * inset,
      y2: to.y - dy / length * inset,
      class: "topology-edge", stroke: topologyColor(edge.classification),
      "marker-end": "url(#topology-arrow)"
    });
    const title = svgElement("title");
    title.textContent = `${edge.source} → ${edge.destination}${edge.port ? `:${edge.port}` : ""} · ${edge.count} flow${edge.count === 1 ? "" : "s"}`;
    line.appendChild(title);
    viewGroup.appendChild(line);

    if (edge.port) {
      const label = svgElement("text", {
        x: (from.x + to.x) / 2 + (dy / length) * 9,
        y: (from.y + to.y) / 2 - (dx / length) * 9,
        class: "topology-edge-label"
      });
      label.textContent = `:${edge.port}`;
      viewGroup.appendChild(label);
    }
  }

  nodes.forEach(node => {
    const point = positions.get(node.ip);
    const color = topologyColor(node.classification);
    const radius = Math.min(23, 16 + Math.log2(node.flows + 1) * 2);
    const group = svgElement("g", {
      class: `topology-node${node.ip === state.selectedEndpoint ? " selected" : ""}`,
      "data-endpoint": node.ip,
      tabindex: "0",
      role: "button",
      "aria-label": `Endpoint ${node.ip}, ${node.flows} flows`
    });
    group.appendChild(svgElement("circle", {
      cx: point.x, cy: point.y, r: radius, fill: "#07100a", stroke: color
    }));
    const count = svgElement("text", {
      x: point.x, y: point.y, class: "topology-node-count"
    });
    count.textContent = String(node.flows);
    group.appendChild(count);
    const label = svgElement("text", {
      x: point.x, y: point.y + radius + 13, class: "topology-node-label"
    });
    label.textContent = shortenEndpoint(node.ip);
    group.appendChild(label);
    const title = svgElement("title");
    title.textContent = `${node.ip} · ${node.peers.size} connected endpoint${node.peers.size === 1 ? "" : "s"}`;
    group.appendChild(title);
    viewGroup.appendChild(group);
  });

  if (!nodes.length) {
    state.selectedEndpoint = null;
    updateTopologyInfo(null);
    return;
  }
  updateTopologyInfo(endpoints.get(state.selectedEndpoint));
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

function topologyColor(classification) {
  if (classification === "DoS") return "#e55353";
  if (classification === "PortScan") return "#f59e0b";
  if (classification === "Infiltration_ZeroDay") return "#a78bfa";
  return "#22c55e";
}

function shortenEndpoint(ip) {
  return ip.length > 20 ? `${ip.slice(0, 9)}…${ip.slice(-8)}` : ip;
}

function updateTopologyInfo(endpoint) {
  const info = document.getElementById("topology-info");
  if (!info) return;
  info.replaceChildren();

  const kicker = document.createElement("div");
  kicker.className = "topology-info-kicker";
  kicker.textContent = endpoint ? "OBSERVED ENDPOINT" : "SELECT AN ENDPOINT";
  info.appendChild(kicker);
  const title = document.createElement("div");
  title.className = "topology-info-title";
  title.textContent = endpoint ? endpoint.ip : "No endpoint selected";
  info.appendChild(title);

  if (endpoint) {
    const facts = document.createElement("div");
    facts.className = "topology-facts";
    const topPort = [...endpoint.ports.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "—";
    for (const [label, value] of [
      ["Flows", endpoint.flows],
      ["Connections", endpoint.peers.size],
      ["Top port", topPort],
      ["Last seen", endpoint.lastSeen ? new Date(endpoint.lastSeen).toLocaleTimeString() : "Now"]
    ]) {
      const row = document.createElement("div");
      row.className = "topology-fact";
      const name = document.createElement("span");
      name.textContent = label;
      const valueNode = document.createElement("span");
      valueNode.textContent = String(value);
      row.append(name, valueNode);
      facts.appendChild(row);
    }
    info.appendChild(facts);
  } else {
    const message = document.createElement("p");
    message.textContent = "Endpoints appear here as traffic arrives.";
    info.appendChild(message);
  }

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
  updatePerformanceMetrics();

  // Enterprise Cybersecurity ROI Valuation (Ponemon / IBM Security Metrics):
  // Tier-1 Known Blocked Attacks prevent ~$850 in remediation/analyst hours
  // Tier-2 Zero-Day Quarantined Threats prevent ~$4,200 in critical breach/downtime damages
  const rawDollars = (state.knownCount * 850) + (state.zeroDayCount * 4200);

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
        sumVerdict.textContent = "Above threshold";
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

function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * fraction;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function updatePerformanceMetrics() {
  const percentiles = [
    ["perf-p50", 0.50],
    ["perf-p95", 0.95],
    ["perf-p99", 0.99]
  ];
  for (const [id, fraction] of percentiles) {
    const element = document.getElementById(id);
    const value = percentile(state.latencies, fraction);
    if (element && value !== null) {
      element.innerHTML = `${value.toFixed(1)}<span class="unit">ms</span>`;
    }
  }

  const now = Date.now();
  const recent = state.flowReceiveTimes.filter(time => now - time <= 10000);
  const elapsed = recent.length > 1 ? Math.max(1, (now - recent[0]) / 1000) : 1;
  const throughput = recent.length / elapsed;
  const throughputElement = document.getElementById("perf-throughput");
  if (throughputElement) {
    throughputElement.innerHTML = `${throughput.toFixed(1)}<span class="unit">flows/s</span>`;
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

  // 2. Update event-type timeline.
  const chartEvents = state.charts.eventsTimeline;
  chartEvents.data.labels.push(timeLabel);

  chartEvents.data.datasets[0].data.push(flow.classification === "BENIGN" ? 1 : 0);
  chartEvents.data.datasets[1].data.push(flow.classification === "DoS" ? 1 : 0);
  chartEvents.data.datasets[2].data.push(flow.classification === "PortScan" ? 1 : 0);
  chartEvents.data.datasets[3].data.push(flow.classification === "Infiltration_ZeroDay" ? 1 : 0);
  chartEvents.data.datasets[4].data.push(flow.classification === "BruteForce" ? 1 : 0);

  if (chartEvents.data.labels.length > 20) {
    chartEvents.data.labels.shift();
    chartEvents.data.datasets.forEach(dataset => dataset.data.shift());
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
  // Modal close handlers (Button, Backdrop click, and ESC key)
  const modal = document.getElementById("forensics-modal");
  const closeBtn = document.getElementById("btn-close-modal");

  if (closeBtn && modal) {
    closeBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      modal.classList.remove("open");
    });
  }

  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        modal.classList.remove("open");
      }
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal && modal.classList.contains("open")) {
      modal.classList.remove("open");
    }
  });

  const topologySvg = document.getElementById("topology-svg");
  if (topologySvg) {
    let draggedEndpoint = null;
    topologySvg.addEventListener("click", (event) => {
      const node = event.target.closest("[data-endpoint]");
      if (!node) return;
      state.selectedEndpoint = node.getAttribute("data-endpoint");
      renderTopology();
    });
    topologySvg.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      const node = event.target.closest("[data-endpoint]");
      if (!node) return;
      draggedEndpoint = node.getAttribute("data-endpoint");
      state.selectedEndpoint = draggedEndpoint;
      topologySvg.setPointerCapture(event.pointerId);
      renderTopology();
      event.preventDefault();
    });
    topologySvg.addEventListener("pointermove", (event) => {
      if (!draggedEndpoint) return;
      const matrix = topologySvg.getScreenCTM();
      if (!matrix) return;
      const point = topologySvg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const viewPoint = point.matrixTransform(matrix.inverse());
      const worldPoint = {
        x: Math.max(28, Math.min(572, 300 + (viewPoint.x - 300) / state.topologyZoom)),
        y: Math.max(28, Math.min(272, 150 + (viewPoint.y - 150) / state.topologyZoom))
      };
      state.topologyPositions.set(draggedEndpoint, worldPoint);
      renderTopology();
    });
    const stopDragging = (event) => {
      draggedEndpoint = null;
      if (topologySvg.hasPointerCapture(event.pointerId)) {
        topologySvg.releasePointerCapture(event.pointerId);
      }
    };
    topologySvg.addEventListener("pointerup", stopDragging);
    topologySvg.addEventListener("pointercancel", stopDragging);
    topologySvg.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const node = event.target.closest("[data-endpoint]");
      if (!node) return;
      event.preventDefault();
      state.selectedEndpoint = node.getAttribute("data-endpoint");
      renderTopology();
    });
  }

  const zoomIn = document.getElementById("topology-zoom-in");
  const zoomOut = document.getElementById("topology-zoom-out");
  const zoomReset = document.getElementById("topology-zoom-reset");
  if (zoomIn) zoomIn.addEventListener("click", () => {
    state.topologyZoom = Math.min(1.8, state.topologyZoom + 0.15);
    renderTopology();
  });
  if (zoomOut) zoomOut.addEventListener("click", () => {
    state.topologyZoom = Math.max(0.7, state.topologyZoom - 0.15);
    renderTopology();
  });
  if (zoomReset) zoomReset.addEventListener("click", () => {
    state.topologyZoom = 1;
    renderTopology();
  });

  // Safe binding for filter dropdown
  const filterElem = document.getElementById("filter-classification");
  if (filterElem) {
    filterElem.addEventListener("change", () => {
      const tbody = document.getElementById("flow-tbody");
      if (tbody) {
        tbody.innerHTML = "";
        state.flowsHistory.forEach(f => insertTableRow(f));
      }
    });
  }

  // Safe binding for optional injection buttons if present
  const btnBenign = document.getElementById("btn-inject-benign");
  if (btnBenign) btnBenign.addEventListener("click", () => injectThreat("BENIGN"));
  const btnDos = document.getElementById("btn-inject-dos");
  if (btnDos) btnDos.addEventListener("click", () => injectThreat("DoS"));
  const btnPort = document.getElementById("btn-inject-portscan");
  if (btnPort) btnPort.addEventListener("click", () => injectThreat("PortScan"));
  const btnZero = document.getElementById("btn-inject-zeroday");
  if (btnZero) btnZero.addEventListener("click", () => injectThreat("ZeroDay"));

  const toggleBtn = document.getElementById("btn-toggle-stream");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      state.isStreamPaused = !state.isStreamPaused;
      const streamText = document.getElementById("stream-btn-text");
      if (streamText) streamText.textContent = state.isStreamPaused ? "Resume Stream" : "Pause Stream";
      toggleBtn.style.color = state.isStreamPaused ? "var(--color-amber)" : "var(--text-secondary)";
    });
  }
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
function renderMitreMapping(classification) {
  const techniques = {
    DoS: {
      id: "T1498",
      name: "Network Denial of Service",
      url: "https://attack.mitre.org/techniques/T1498/"
    },
    PortScan: {
      id: "T1046",
      name: "Network Service Discovery",
      url: "https://attack.mitre.org/techniques/T1046/"
    },
    BruteForce: {
      id: "T1110",
      name: "Brute Force",
      url: "https://attack.mitre.org/techniques/T1110/"
    }
  };
  const technique = techniques[classification];
  const matchContent = technique
    ? `<a class="mitre-technique" href="${technique.url}" target="_blank" rel="noopener noreferrer">
         <span class="mitre-id">${technique.id}</span>
         <span class="mitre-name">${technique.name}</span>
         <span class="mitre-link-icon" aria-hidden="true">↗</span>
       </a>
       <p class="mitre-caption">Suggested from the event type</p>`
    : `<div class="mitre-unmapped">No direct technique match for this event type.</div>`;

  return `<section class="mitre-mapping" aria-label="MITRE ATT&CK mapping">
    <div class="mitre-heading">MITRE ATT&amp;CK</div>
    ${matchContent}
  </section>`;
}

function openForensicsModal(flow) {
  const modal = document.getElementById("forensics-modal");
  const content = document.getElementById("modal-content");

  const departureRatio = (flow.reconstruction_error / Math.max(1e-5, flow.anomaly_threshold)).toFixed(2);
  const mitreMapping = renderMitreMapping(flow.classification);

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
      ? "WHAT MADE THIS LOOK UNUSUAL"
      : "WHAT AFFECTED THE DETECTION";
    const subtitle = isZeroDay
      ? `Main signals — model error: ${flow.reconstruction_error.toFixed(4)} above threshold ${flow.anomaly_threshold}`
      : "Main signals behind this result";
    const legend = isZeroDay
      ? `<span style="color:var(--color-purple, #a855f7)">▮ Difference from normal traffic</span>`
      : `<span style="color:var(--color-red)">▮ Points to a threat</span>
         <span style="color:var(--color-green)">▮ Points to normal traffic</span>`;

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
        <div class="forensic-label">MODEL ERROR</div>
        <div class="forensic-val">${flow.reconstruction_error.toFixed(6)}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">ALERT THRESHOLD</div>
        <div class="forensic-val">${flow.anomaly_threshold} (${departureRatio}x above)</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">PATTERN DISTANCE</div>
        <div class="forensic-val">${flow.prototype_distance ? flow.prototype_distance.toFixed(4) : "N/A"}</div>
      </div>
      <div class="forensic-item">
        <div class="forensic-label">PROCESSING TIME</div>
        <div class="forensic-val">${flow.latency_ms ? flow.latency_ms.toFixed(2) : "0.80"} ms</div>
      </div>
    </div>
    <div class="forensic-reason">
      <strong>Why it was flagged:</strong> ${flow.reason}
    </div>
    ${mitreMapping}
    ${shapBlock}
  `;

  modal.classList.add("open");
}

