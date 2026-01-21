const state = {
  payload: null,
  runId: null,
};

const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

const setActiveTab = (tabId) => {
  qsa(".tab").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabId));
  qsa(".tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== tabId));
};

const updateSignalCard = (payload) => {
  if (!payload) return;
  const signal = payload.signal || {};
  qs("#signal-direction").textContent = signal.action || "N/A";
  qs("#signal-score").textContent = signal.score ?? "N/A";
  qs("#signal-confidence").textContent = signal.confidence ?? "N/A";
  qs("#signal-horizon").textContent = signal.horizon || "N/A";
  qs("#signal-risk").textContent = signal.risk_mode || "N/A";

  const confidencePct = Math.round((signal.confidence || 0) * 100);
  qs(".confidence-bar span").style.width = `${confidencePct}%`;
  qs("#confidence-value").textContent = `${confidencePct}%`;

  const reasonList = qs("#reason-codes");
  reasonList.innerHTML = "";
  (payload.evidence?.reason_codes || []).forEach((code) => {
    const detail = payload.evidence?.reason_details?.[code] || "";
    const li = document.createElement("li");
    li.textContent = detail ? `${code}: ${detail}` : code;
    reasonList.appendChild(li);
  });
};

const appendChat = (role, text) => {
  const log = qs("#chat-log");
  const msg = document.createElement("div");
  msg.className = "chat-message";
  msg.innerHTML = `<strong>${role}:</strong> ${text}`;
  log.appendChild(msg);
  log.scrollTop = log.scrollHeight;
};

const renderNarration = (data) => {
  qs("#narration-headline").textContent = data.headline || "";
  qs("#narration-summary").textContent = data.summary || "";
  const bullets = qs("#narration-bullets");
  bullets.innerHTML = "";
  (data.bullets || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    bullets.appendChild(li);
  });
  qs("#narration-disclaimer").textContent = data.disclaimer || "";
  const followups = qs("#narration-followups");
  followups.innerHTML = "";
  (data.followups || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    followups.appendChild(li);
  });
};

const drawEquityCurve = (points) => {
  const canvas = qs("#equity-canvas");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b1220";
  ctx.fillRect(0, 0, width, height);

  if (!points || points.length < 2) return;
  const equities = points.map((p) => p.equity);
  const benchmark = points.map((p) => p.benchmark_equity);
  const minVal = Math.min(...equities, ...benchmark);
  const maxVal = Math.max(...equities, ...benchmark);

  const scaleX = width / (points.length - 1);
  const scaleY = maxVal === minVal ? 1 : (height - 20) / (maxVal - minVal);

  const drawLine = (series, color) => {
    ctx.strokeStyle = color;
    ctx.beginPath();
    series.forEach((val, idx) => {
      const x = idx * scaleX;
      const y = height - 10 - (val - minVal) * scaleY;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  drawLine(equities, "#22c55e");
  drawLine(benchmark, "#60a5fa");
};

const updateMetrics = (metrics) => {
  if (!metrics) return;
  qs("#metric-cagr").textContent = (metrics.cagr ?? 0).toFixed(3);
  qs("#metric-sharpe").textContent = (metrics.sharpe ?? 0).toFixed(2);
  qs("#metric-sortino").textContent = (metrics.sortino ?? 0).toFixed(2);
  qs("#metric-maxdd").textContent = (metrics.maxdd ?? 0).toFixed(3);
  qs("#metric-turnover").textContent = (metrics.turnover ?? 0).toFixed(2);
};

const updateRegimes = (regimes) => {
  const table = qs("#regime-table");
  table.innerHTML = "";
  (regimes || []).forEach((reg) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${reg.regime_name}</td>
      <td>${(reg.cagr ?? 0).toFixed(3)}</td>
      <td>${(reg.sharpe ?? 0).toFixed(2)}</td>
      <td>${(reg.maxdd ?? 0).toFixed(3)}</td>
      <td>${(reg.winrate ?? 0).toFixed(2)}</td>
      <td>${reg.trades ?? 0}</td>
    `;
    table.appendChild(row);
  });
};

qs("#analyze-btn").addEventListener("click", async () => {
  const symbol = qs("#symbol-input").value.trim();
  const horizon = qs("#horizon-input").value;
  const riskMode = qs("#risk-input").value;
  if (!symbol) return;

  const resp = await fetch("/assistant/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol, horizon, risk_mode: riskMode }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    appendChat("system", data.error?.message || "Analyze failed");
    return;
  }
  state.payload = data;
  updateSignalCard(data);
  appendChat("system", `Analyzed ${data.symbol} (${data.as_of_date}).`);
});

qs("#chat-send").addEventListener("click", async () => {
  const message = qs("#chat-message").value.trim();
  if (!message) return;
  if (!state.payload) {
    appendChat("system", "Run Analyze first to generate a payload.");
    return;
  }
  appendChat("user", message);
  qs("#chat-message").value = "";
  const resp = await fetch("/assistant/narrate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ payload: state.payload, user_message: message }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    appendChat("system", data.error?.message || "Narrator failed");
    return;
  }
  renderNarration(data);
  appendChat("assistant", data.summary || "Narration ready.");
});

qs("#backtest-run").addEventListener("click", async () => {
  const symbol = qs("#backtest-symbol").value.trim() || null;
  const universe = qs("#backtest-universe").value;
  const dateStart = qs("#backtest-start").value;
  const dateEnd = qs("#backtest-end").value;
  const horizon = qs("#backtest-horizon").value;
  const riskMode = qs("#backtest-risk").value;

  const resp = await fetch("/backtest/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol,
      universe,
      date_start: dateStart,
      date_end: dateEnd,
      horizon,
      risk_mode: riskMode,
      signal_version_hash: "auto",
      cost_model: { fee_bps: 1, slippage_bps: 5 },
    }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    appendChat("system", data.error?.message || "Backtest run failed");
    return;
  }
  state.runId = data.run_id;

  const resultResp = await fetch(`/backtest/results?run_id=${state.runId}`);
  const resultData = await resultResp.json();
  updateMetrics(resultData.metrics);
  updateRegimes(resultData.regime_metrics);

  const curveResp = await fetch(`/backtest/equity-curve?run_id=${state.runId}`);
  const curveData = await curveResp.json();
  drawEquityCurve(curveData.points || []);
  appendChat("system", `Backtest ${state.runId} complete.`);
});

qsa(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
});

setActiveTab("overview");
