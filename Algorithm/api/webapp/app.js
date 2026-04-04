const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  apiKey: "",
  assistantChatSessionId: "",
  assistantChatTranscript: [],
  assistantActiveAnalysis: null,
  assistantLatestReport: null,
};

const ASSISTANT_CHAT_SESSION_STORAGE_KEY = "ftip.assistant.chat.session_id";
const ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY = "ftip.assistant.active_analysis";

const isoDate = (date) => date.toISOString().slice(0, 10);

const formatJson = (value) => JSON.stringify(value ?? {}, null, 2);
const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

const setDefaults = () => {
  const now = new Date();
  const toDate = isoDate(now);
  const from = new Date(now);
  from.setDate(from.getDate() - 365);
  qs("#to-date-input").value = toDate;
  qs("#as-of-date-input").value = toDate;
  qs("#from-date-input").value = isoDate(from);
  qs("#assistant-analyze-symbol").value = qs("#symbol-input").value.trim().toUpperCase() || "NVDA";
};

const setStatus = (text, isError = false) => {
  const el = qs("#status-line");
  el.textContent = text;
  el.classList.toggle("error", !!isError);
};

const setActiveTab = (tabId) => {
  qsa(".tab").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabId));
  qsa(".tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== tabId));
};

const getHeaders = () => {
  const headers = { "Content-Type": "application/json" };
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  return headers;
};

const getInputs = () => ({
  symbol: qs("#symbol-input").value.trim().toUpperCase(),
  lookback: Number(qs("#lookback-input").value || 252),
  from_date: qs("#from-date-input").value,
  to_date: qs("#to-date-input").value,
  as_of_date: qs("#as-of-date-input").value,
});

const parseJsonSafe = async (resp) => {
  try {
    return await resp.json();
  } catch {
    return null;
  }
};

const callJson = async (url, options = {}) => {
  const resp = await fetch(url, options);
  const data = await parseJsonSafe(resp);
  if (!resp.ok) {
    const msg = data?.detail || data?.error?.message || `Request failed: ${resp.status}`;
    throw new Error(msg);
  }
  return data;
};

const renderSignal = (signal) => {
  qs("#signal-symbol").textContent = signal?.symbol || "N/A";
  qs("#signal-asof").textContent = signal?.as_of || "N/A";
  qs("#signal-direction").textContent = signal?.signal || "N/A";
  qs("#signal-score").textContent = signal?.score ?? "N/A";
  qs("#signal-confidence").textContent = signal?.confidence ?? "N/A";
  qs("#signal-regime").textContent = signal?.regime || "N/A";
  qs("#signal-json").textContent = formatJson(signal || {});
};

const renderFeatures = (features) => {
  qs("#features-symbol").textContent = features?.symbol || "N/A";
  qs("#features-asof").textContent = features?.as_of || "N/A";
  qs("#features-lookback").textContent = features?.lookback ?? "N/A";
  qs("#features-json").textContent = formatJson(features || {});
};

const setLegacyStatus = (selector, text, tone = "muted") => {
  const el = qs(selector);
  el.textContent = text;
  el.classList.toggle("error", tone === "error");
  el.classList.toggle("success", tone === "success");
};

const setButtonLoading = (selector, loading, loadingText) => {
  const button = qs(selector);
  if (!button.dataset.defaultText) {
    button.dataset.defaultText = button.textContent;
  }
  button.disabled = loading;
  button.textContent = loading ? loadingText : button.dataset.defaultText;
};

const generateUuid = () => {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }

  if (window.crypto?.getRandomValues) {
    const bytes = new Uint8Array(16);
    window.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    return [
      hex.slice(0, 8),
      hex.slice(8, 12),
      hex.slice(12, 16),
      hex.slice(16, 20),
      hex.slice(20),
    ].join("-");
  }

  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
    const rand = Math.floor(Math.random() * 16);
    const value = char === "x" ? rand : (rand & 0x3) | 0x8;
    return value.toString(16);
  });
};

const persistAssistantSessionId = (sessionId) => {
  state.assistantChatSessionId = sessionId;
  qs("#assistant-chat-session-id").value = sessionId;
  window.localStorage.setItem(ASSISTANT_CHAT_SESSION_STORAGE_KEY, sessionId);
};

const formatActiveAnalysisLabel = (analysis) => {
  if (!analysis?.symbol) {
    return "Active analysis: none yet.";
  }
  return `Active analysis: ${analysis.symbol} / ${analysis.horizon || "n/a"} / ${
    analysis.risk_mode || "n/a"
  } / ${analysis.as_of_date || "n/a"}`;
};

const renderActiveAnalysisLabels = () => {
  const label = formatActiveAnalysisLabel(state.assistantActiveAnalysis);
  qs("#assistant-analyze-active-label").textContent = label;
  qs("#assistant-chat-active-label").textContent = label;
};

const persistActiveAnalysis = (analysis) => {
  state.assistantActiveAnalysis = analysis || null;
  if (analysis) {
    window.localStorage.setItem(
      ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY,
      JSON.stringify(analysis)
    );
  } else {
    window.localStorage.removeItem(ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY);
  }
  renderActiveAnalysisLabels();
};

const renderAssistantReport = (report) => {
  const container = qs("#assistant-analyze-report");
  state.assistantLatestReport = report || null;

  if (!report) {
    container.classList.add("empty");
    container.innerHTML = `
      <div class="analysis-report-empty">
        Run Assistant Analyze to generate a grounded system report.
      </div>
    `;
    return;
  }

  container.classList.remove("empty");
  const metrics = [
    { label: "Signal", value: report.signal?.action || "N/A" },
    { label: "Score", value: report.signal?.score ?? "N/A" },
    { label: "Confidence", value: report.signal?.confidence ?? "N/A" },
    { label: "As Of", value: report.as_of_date || "N/A" },
    { label: "Horizon", value: report.horizon || "N/A" },
    { label: "Risk Mode", value: report.risk_mode || "N/A" },
  ];
  const sections = [
    ["Signal Summary", report.signal_summary],
    ["Technical Analysis", report.technical_analysis],
    ["Fundamental Analysis", report.fundamental_analysis],
    ["Statistical Analysis", report.statistical_analysis],
    ["Sentiment Analysis", report.sentiment_analysis],
    ["Risk & Quality Analysis", report.risk_quality_analysis],
    ["Overall Analysis", report.overall_analysis],
    ["Strategy View", report.strategy_view],
  ];

  container.innerHTML = `
    <div class="report-hero">
      <div class="report-kicker">Grounded Analysis Report</div>
      <h5 class="report-title">${escapeHtml(report.symbol || "Unknown Symbol")}</h5>
      <p class="report-subtitle">${escapeHtml(
        `${report.as_of_date || "n/a"} · ${report.horizon || "n/a"} horizon · ${
          report.risk_mode || "n/a"
        } risk mode`
      )}</p>
    </div>
    <div class="report-metrics">
      ${metrics
        .map(
          (metric) => `
            <div class="report-metric">
              <div class="report-metric-label">${escapeHtml(metric.label)}</div>
              <div class="report-metric-value">${escapeHtml(metric.value)}</div>
            </div>
          `
        )
        .join("")}
    </div>
    <div class="report-sections">
      ${sections
        .map(
          ([title, body]) => `
            <section class="report-section">
              <h5>${escapeHtml(title)}</h5>
              <p>${escapeHtml(body || "No section text available.")}</p>
            </section>
          `
        )
        .join("")}
    </div>
  `;
};

const renderAssistantChatTranscript = () => {
  if (!state.assistantChatTranscript.length) {
    qs("#assistant-chat-response").textContent = "No messages yet.";
    return;
  }

  const blocks = state.assistantChatTranscript.map((entry) => {
    const lines = [`${entry.role}:`, entry.content];
    if (entry.citations?.length) {
      lines.push(`Citations: ${entry.citations.join(", ")}`);
    }
    return lines.join("\n");
  });
  qs("#assistant-chat-response").textContent = blocks.join("\n\n");
};

const resetAssistantChatSession = (message = "New local session prepared.") => {
  persistAssistantSessionId(generateUuid());
  state.assistantChatTranscript = [];
  persistActiveAnalysis(null);
  renderAssistantReport(null);
  renderAssistantChatTranscript();
  setLegacyStatus("#assistant-chat-status", message, "success");
};

const getAnalyzeInputs = () => ({
  symbol: qs("#assistant-analyze-symbol").value.trim().toUpperCase(),
  horizon: qs("#assistant-analyze-horizon").value.trim(),
  risk_mode: qs("#assistant-analyze-risk-mode").value.trim(),
});

const getAssistantChatMessage = () => qs("#assistant-chat-message").value.trim();

const runAssistantAnalyze = async () => {
  const { symbol, horizon, risk_mode } = getAnalyzeInputs();
  if (!symbol || !horizon || !risk_mode) {
    setLegacyStatus(
      "#assistant-analyze-status",
      "symbol, horizon, and risk_mode are required.",
      "error"
    );
    return;
  }

  const pendingSessionId = state.assistantChatSessionId || generateUuid();
  persistAssistantSessionId(pendingSessionId);
  setButtonLoading("#assistant-analyze-btn", true, "Running...");
  setLegacyStatus("#assistant-analyze-status", `Running assistant analyze for ${symbol}...`);
  try {
    const data = await callJson("/assistant/analyze", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        session_id: pendingSessionId,
        symbol,
        horizon,
        risk_mode,
      }),
    });
    if (data.session_id) {
      persistAssistantSessionId(data.session_id);
    }
    persistActiveAnalysis(data.active_analysis || null);
    renderAssistantReport(data);
    qs("#assistant-analyze-response").textContent = formatJson(data);
    setLegacyStatus(
      "#assistant-analyze-status",
      `Assistant analyze complete for ${data.symbol || symbol}.`,
      "success"
    );
  } catch (err) {
    qs("#assistant-analyze-response").textContent = formatJson({
      error: err.message,
      symbol,
      horizon,
      risk_mode,
    });
    setLegacyStatus("#assistant-analyze-status", `Analyze failed: ${err.message}`, "error");
  } finally {
    setButtonLoading("#assistant-analyze-btn", false, "Running...");
  }
};

const sendAssistantChat = async () => {
  const message = getAssistantChatMessage();
  if (!message) {
    setLegacyStatus("#assistant-chat-status", "Message is required.", "error");
    return;
  }

  const pendingSessionId = state.assistantChatSessionId || generateUuid();
  persistAssistantSessionId(pendingSessionId);

  setButtonLoading("#assistant-chat-btn", true, "Sending...");
  setLegacyStatus("#assistant-chat-status", "Sending assistant chat request...");
  try {
    const data = await callJson("/assistant/chat", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        session_id: pendingSessionId,
        message,
        context: {
          active_analysis: state.assistantActiveAnalysis,
        },
      }),
    });

    if (data.session_id) {
      persistAssistantSessionId(data.session_id);
    }
    if (data.active_analysis) {
      persistActiveAnalysis(data.active_analysis);
    }

    state.assistantChatTranscript.push({ role: "User", content: message });
    state.assistantChatTranscript.push({
      role: "Assistant",
      content: data.reply || "",
      citations: data.citations || [],
    });
    renderAssistantChatTranscript();
    qs("#assistant-chat-message").value = "";
    setLegacyStatus("#assistant-chat-status", "Assistant reply received.", "success");
  } catch (err) {
    setLegacyStatus("#assistant-chat-status", `Chat failed: ${err.message}`, "error");
  } finally {
    setButtonLoading("#assistant-chat-btn", false, "Sending...");
  }
};

const refreshOfficialLatest = async () => {
  const { symbol, lookback } = getInputs();
  if (!symbol) {
    setStatus("Symbol is required for latest signal/features.", true);
    return;
  }
  setStatus(`Refreshing latest official v1 rows for ${symbol}...`);
  try {
    const [signal, features] = await Promise.all([
      callJson(`/prosperity/latest/signal?symbol=${encodeURIComponent(symbol)}&lookback=${lookback}`),
      callJson(`/prosperity/latest/features?symbol=${encodeURIComponent(symbol)}&lookback=${lookback}`),
    ]);
    renderSignal(signal);
    renderFeatures(features);
    setStatus(`Loaded latest signal/features for ${symbol}.`);
  } catch (err) {
    setStatus(`Refresh failed: ${err.message}`, true);
  }
};

qs("#bootstrap-btn").addEventListener("click", async () => {
  setStatus("Running POST /prosperity/bootstrap...");
  try {
    await callJson("/prosperity/bootstrap", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({}),
    });
    setStatus("Bootstrap complete.");
  } catch (err) {
    setStatus(`Bootstrap failed: ${err.message}`, true);
  }
});

qs("#snapshot-btn").addEventListener("click", async () => {
  const { symbol, lookback, from_date, to_date, as_of_date } = getInputs();
  if (!symbol || !from_date || !to_date || !as_of_date) {
    setStatus("symbol/from_date/to_date/as_of_date are required.", true);
    return;
  }

  setStatus(`Running POST /prosperity/snapshot/run for ${symbol}...`);
  try {
    await callJson("/prosperity/snapshot/run", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        symbols: [symbol],
        from_date,
        to_date,
        as_of_date,
        lookback,
      }),
    });
    setStatus(`Snapshot run complete for ${symbol}. Refreshing latest signal/features...`);
    await refreshOfficialLatest();
  } catch (err) {
    setStatus(`Snapshot failed: ${err.message}`, true);
  }
});

qs("#refresh-btn").addEventListener("click", refreshOfficialLatest);

qs("#daily-snapshot-btn").addEventListener("click", async () => {
  setStatus("Running optional POST /jobs/prosperity/daily-snapshot...");
  try {
    await callJson("/jobs/prosperity/daily-snapshot", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({}),
    });
    setStatus("Daily snapshot job request accepted.");
  } catch (err) {
    setStatus(`Daily snapshot job failed: ${err.message}`, true);
  }
});

qs("#api-key-input").addEventListener("input", (e) => {
  state.apiKey = e.target.value.trim();
});

qs("#assistant-analyze-btn").addEventListener("click", runAssistantAnalyze);

qs("#assistant-chat-btn").addEventListener("click", sendAssistantChat);

qs("#assistant-chat-reset-btn").addEventListener("click", () => {
  resetAssistantChatSession("Started a fresh assistant chat session.");
});

qs("#assistant-chat-message").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    sendAssistantChat();
  }
});

qs("#symbol-input").addEventListener("input", (event) => {
  const legacySymbol = qs("#assistant-analyze-symbol");
  if (!legacySymbol.value.trim()) {
    legacySymbol.value = event.target.value.trim().toUpperCase();
  }
});

qsa(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
});

setDefaults();
persistAssistantSessionId(
  window.localStorage.getItem(ASSISTANT_CHAT_SESSION_STORAGE_KEY) || generateUuid()
);
try {
  persistActiveAnalysis(
    JSON.parse(window.localStorage.getItem(ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY) || "null")
  );
} catch {
  persistActiveAnalysis(null);
}
renderAssistantReport(null);
renderAssistantChatTranscript();
setActiveTab("signal");
