const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  apiKey: "",
  assistantChatSessionId: "",
  assistantChatTranscript: [],
  assistantActiveAnalysis: null,
  assistantLatestReport: null,
  assistantHealth: null,
  researchTab: "dashboard",
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

const emptyStateCard = (text) => `<div class="empty-state-card">${escapeHtml(text)}</div>`;

const setDefaults = () => {
  const now = new Date();
  const toDate = isoDate(now);
  const from = new Date(now);
  from.setDate(from.getDate() - 365);
  qs("#to-date-input").value = toDate;
  qs("#as-of-date-input").value = toDate;
  qs("#from-date-input").value = isoDate(from);
  qs("#assistant-analyze-symbol").value =
    qs("#symbol-input").value.trim().toUpperCase() || "NVDA";
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

const setResearchTab = (tabId) => {
  state.researchTab = tabId;
  qsa(".research-tab").forEach((btn) =>
    btn.classList.toggle("active", btn.dataset.researchTab === tabId)
  );
  qsa("[data-research-panel]").forEach((panel) =>
    panel.classList.toggle("hidden", panel.dataset.researchPanel !== tabId)
  );
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

const activeSignalLabel = () =>
  state.assistantLatestReport?.strategy?.final_signal ||
  state.assistantLatestReport?.signal?.final_action ||
  state.assistantLatestReport?.signal?.action ||
  state.assistantActiveAnalysis?.signal ||
  "n/a";

const activeFreshnessLabel = () =>
  state.assistantLatestReport?.freshness_summary?.overall_status ||
  state.assistantActiveAnalysis?.freshness_status ||
  "unknown";

const activeConvictionLabel = () =>
  state.assistantLatestReport?.strategy?.conviction_tier ||
  state.assistantActiveAnalysis?.conviction_tier ||
  "unknown";

const activeStrategyPostureLabel = () =>
  state.assistantLatestReport?.strategy?.strategy_posture ||
  state.assistantActiveAnalysis?.strategy_posture ||
  "n/a";

const activeReportVersionLabel = () =>
  state.assistantActiveAnalysis?.report_version ||
  state.assistantLatestReport?.report_version ||
  "n/a";

const buildNarratorPromptSet = () => {
  const symbol = state.assistantActiveAnalysis?.symbol || "this setup";
  return [
    `Why is ${symbol} ${activeSignalLabel()}?`,
    `Explain the strategy view for ${symbol}.`,
    `What is the bear case for ${symbol}?`,
    `What are the invalidators here?`,
    `What is weakest in the setup for ${symbol}?`,
    `What would improve conviction on ${symbol}?`,
  ];
};

const renderNarratorPromptChips = () => {
  const container = qs("#assistant-chat-suggested-prompts");
  const prompts = buildNarratorPromptSet();
  container.innerHTML = prompts
    .map(
      (prompt) =>
        `<button type="button" class="prompt-chip" data-prompt="${escapeHtml(prompt)}">${escapeHtml(prompt)}</button>`
    )
    .join("");
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
  const analysis = state.assistantActiveAnalysis;
  const label = formatActiveAnalysisLabel(analysis);
  qs("#assistant-analyze-active-label").textContent = label;
  qs("#assistant-chat-active-label").textContent = label;
  qs("#assistant-active-analysis-title").textContent = analysis?.symbol
    ? `${analysis.symbol} · ${analysis.horizon || "n/a"} · ${analysis.risk_mode || "n/a"}`
    : "No report loaded";
  const metaChips = [
    `Signal: ${activeSignalLabel()}`,
    `Freshness: ${activeFreshnessLabel()}`,
    `Conviction: ${activeConvictionLabel()}`,
    `Strategy: ${activeStrategyPostureLabel()}`,
    `Report: ${activeReportVersionLabel()}`,
  ];
  qs("#assistant-active-analysis-meta").innerHTML = metaChips
    .map((item) => `<div class="active-chip">${escapeHtml(item)}</div>`)
    .join("");
  const chatMeta = [
    `Signal: ${activeSignalLabel()}`,
    `Conviction: ${activeConvictionLabel()}`,
    `Strategy: ${activeStrategyPostureLabel()}`,
    `Freshness: ${activeFreshnessLabel()}`,
    `Report: ${activeReportVersionLabel()}`,
    `Session: ${state.assistantChatSessionId ? "active" : "local"}`,
  ];
  qs("#assistant-chat-analysis-meta").innerHTML = chatMeta
    .map((item) => `<div class="active-chip">${escapeHtml(item)}</div>`)
    .join("");
  qs("#assistant-chat-grounding-note").textContent = analysis?.symbol
    ? `Narrator is grounded to the active ${analysis.symbol} analysis artifact and will answer follow-up questions from the stored report and strategy.`
    : "Run Assistant Analyze to establish the active artifact the narrator should use.";
  renderNarratorPromptChips();
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

const renderMetricCards = (selector, metrics) => {
  const el = qs(selector);
  if (!metrics?.length) {
    el.innerHTML = emptyStateCard("No active report metrics yet.");
    return;
  }
  el.innerHTML = metrics
    .map(
      (metric) => `
        <div class="summary-card">
          <div class="summary-card-label">${escapeHtml(metric.label)}</div>
          <div class="summary-card-value">${escapeHtml(metric.value)}</div>
          <div class="summary-card-note">${escapeHtml(metric.note || "")}</div>
        </div>
      `
    )
    .join("");
};

const renderBullets = (items, emptyText = "No items available.") => {
  if (!items?.length) {
    return `<ul class="bullet-list"><li>${escapeHtml(emptyText)}</li></ul>`;
  }
  return `<ul class="bullet-list">${items
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("")}</ul>`;
};

const renderDriverBlock = (title, items, emptyText) => `
  <section class="drilldown-card">
    <h5>${escapeHtml(title)}</h5>
    ${
      items?.length
        ? `<ul class="bullet-list">${items
            .map(
              (item) => `
                <li>
                  <strong>${escapeHtml(item.label || "driver")}</strong>
                  ${item.score == null ? "" : ` · ${escapeHtml(item.score)}`}
                  <div class="drilldown-note">${escapeHtml(item.detail || "")}</div>
                </li>
              `
            )
            .join("")}</ul>`
        : `<div class="empty-state-card">${escapeHtml(emptyText)}</div>`
    }
  </section>
`;

const renderTextSection = (selector, title, body) => {
  const el = qs(selector);
  el.innerHTML = body
    ? `
      <div class="section-heading">${escapeHtml(title)}</div>
      <p>${escapeHtml(body)}</p>
    `
    : emptyStateCard(`${title} will render here.`);
};

const renderDomainStatusGrid = (report) => {
  const container = qs("#assistant-domain-status-grid");
  const freshnessDomains = report?.freshness_summary?.domains || {};
  const bundle = report?.data_bundle || {};
  const cards = [
    ["Bars", freshnessDomains.bars?.status, freshnessDomains.bars?.updated_at],
    ["News", freshnessDomains.news?.status, freshnessDomains.news?.updated_at],
    ["Sentiment", freshnessDomains.sentiment?.status, freshnessDomains.sentiment?.updated_at],
    [
      "Fundamentals",
      bundle.fundamental_filing?.meta?.status,
      bundle.fundamental_filing?.meta?.latest_report_date,
    ],
    ["Macro", bundle.macro_cross_asset?.meta?.status, bundle.macro_cross_asset?.benchmark_proxy],
    ["Relative", bundle.relative_context?.meta?.status, bundle.relative_context?.peer_count],
  ].filter((item) => item[1] != null || item[2] != null);

  if (!cards.length) {
    container.innerHTML = emptyStateCard("Domain freshness and coverage will render here.");
    return;
  }
  container.innerHTML = cards
    .map(
      ([label, status, note]) => `
        <div class="summary-card">
          <div class="summary-card-label">${escapeHtml(label)}</div>
          <div class="summary-card-value">${escapeHtml(status || "n/a")}</div>
          <div class="summary-card-note">${escapeHtml(note || "no detail")}</div>
        </div>
      `
    )
    .join("");
};

const renderFactorGrid = (report) => {
  const container = qs("#assistant-factor-grid");
  const composites = report?.feature_factor_bundle?.composite_intelligence;
  const strategyComponents = report?.strategy?.component_scores;
  if (!composites && !strategyComponents) {
    container.innerHTML = emptyStateCard("Composite scores and factor bundles render here.");
    return;
  }

  const compositeCards = Object.entries(composites || {}).map(
    ([label, value]) => `
      <div class="factor-card">
        <div class="factor-card-label">${escapeHtml(label)}</div>
        <div class="factor-card-value">${escapeHtml(
          value == null ? "n/a" : Number(value).toFixed(1)
        )}</div>
      </div>
    `
  );
  const componentCards = Object.entries(strategyComponents || {}).map(
    ([label, item]) => `
      <div class="factor-card">
        <div class="factor-card-label">${escapeHtml(label.replaceAll("_", " "))}</div>
        <div class="factor-card-value">${escapeHtml(Number(item.score || 0).toFixed(2))}</div>
        <div class="factor-card-note">weight ${escapeHtml(Number(item.weight || 0).toFixed(2))}</div>
      </div>
    `
  );

  container.innerHTML = [...compositeCards, ...componentCards].join("");
};

const renderDashboardSummary = (report) => {
  const container = qs("#assistant-dashboard-summary");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Run Assistant Analyze to populate the executive summary, conviction stack, and report freshness."
    );
    return;
  }
  const strategy = report.strategy || {};
  const cards = [
    {
      label: "Final Signal",
      value: strategy.final_signal || report.signal?.final_action || report.signal?.action || "n/a",
      note: `${report.symbol || "n/a"} · ${report.as_of_date || "n/a"}`,
    },
    {
      label: "Confidence",
      value:
        strategy.confidence == null ? "n/a" : Number(strategy.confidence).toFixed(2),
      note: `conviction ${strategy.conviction_tier || "unknown"}`,
    },
    {
      label: "Actionability",
      value:
        strategy.actionability_score == null
          ? "n/a"
          : Number(strategy.actionability_score).toFixed(1),
      note: strategy.strategy_posture || "posture unknown",
    },
    {
      label: "Fragility",
      value: strategy.fragility_tier || "unknown",
      note: `freshness ${report.freshness_summary?.overall_status || "unknown"}`,
    },
    {
      label: "Participant Fit",
      value: strategy.primary_participant_fit || "n/a",
      note: strategy.time_horizon_fit || "time-horizon fit unknown",
    },
    {
      label: "Scenario",
      value: report.scenario || "base",
      note: `execution ${strategy.execution_posture?.preferred_posture || "staged_watch"}`,
    },
  ];
  container.innerHTML = `
    <div class="summary-grid">
      ${cards
        .map(
          (card) => `
            <div class="summary-card">
              <div class="summary-card-label">${escapeHtml(card.label)}</div>
              <div class="summary-card-value">${escapeHtml(card.value)}</div>
              <div class="summary-card-note">${escapeHtml(card.note)}</div>
            </div>
          `
        )
        .join("")}
    </div>
    <div class="section-copy">
      <div class="section-heading">Executive Summary</div>
      <p>${escapeHtml(report.overall_analysis || report.signal_summary || "")}</p>
    </div>
  `;
};

const renderWhySignal = (report) => {
  const container = qs("#assistant-signal-drilldown");
  const why = report?.why_this_signal;
  if (!why) {
    container.innerHTML = emptyStateCard(
      "The driver panel will list top positives, top negatives, confidence modifiers, missing-data warnings, and freshness warnings for the active report."
    );
    return;
  }
  container.innerHTML = [
    renderDriverBlock(
      "Top Positive Drivers",
      why.top_positive_drivers,
      "No positive drivers were surfaced."
    ),
    renderDriverBlock(
      "Top Negative Drivers",
      why.top_negative_drivers,
      "No negative drivers were surfaced."
    ),
    `<section class="drilldown-card">
      <h5>Confidence / Risk Modifiers</h5>
      ${renderBullets(why.confidence_modifiers, "No explicit confidence degraders.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Missing-Data Warnings</h5>
      ${renderBullets(why.missing_data_warnings, "No missing-data warnings.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Freshness Warnings</h5>
      ${renderBullets(why.freshness_warnings, "No freshness warnings.")}
    </section>`,
  ].join("");
};

const renderStrategyMetrics = (report) => {
  const strategy = report?.strategy;
  if (!strategy) {
    renderMetricCards("#assistant-strategy-metrics", null);
    return;
  }
  renderMetricCards("#assistant-strategy-metrics", [
    {
      label: "Posture",
      value: strategy.strategy_posture || strategy.final_signal || "n/a",
      note: `public ${strategy.final_signal || report.signal?.final_action || "n/a"}`,
    },
    {
      label: "Actionability",
      value:
        strategy.actionability_score == null
          ? "n/a"
          : Number(strategy.actionability_score).toFixed(1),
      note: strategy.quality_of_setup || "quality unknown",
    },
    {
      label: "Confidence",
      value:
        strategy.confidence_score == null
          ? "n/a"
          : Number(strategy.confidence_score).toFixed(1),
      note: strategy.conviction_tier || "unknown conviction",
    },
    {
      label: "Participant",
      value: strategy.primary_participant_fit || "n/a",
      note: strategy.time_horizon_fit || "time-horizon fit unknown",
    },
    {
      label: "Execution",
      value: strategy.execution_posture?.preferred_posture || "n/a",
      note: `urgency ${strategy.execution_posture?.urgency_level || "unknown"}`,
    },
    {
      label: "Fragility Veto",
      value: strategy.hard_veto ? "active" : "none",
      note: `${(strategy.fragility_vetoes || []).length} dampener(s)`,
    },
  ]);
};

const renderStrategyScenarios = (report) => {
  const container = qs("#assistant-strategy-scenarios");
  const scenarios = report?.strategy?.scenario_matrix;
  if (!scenarios) {
    container.innerHTML = emptyStateCard(
      "Base / bull / bear / stress scenarios will render here."
    );
    return;
  }
  const order = [
    ["base", "Base Case"],
    ["bull", "Bull Case"],
    ["bear", "Bear Case"],
    ["stress", "Stress / Adverse"],
  ];
  container.innerHTML = order
    .map(([key, label]) => {
      const scenario = scenarios[key] || {};
      const bullets = [
        `Expected shift: ${scenario.expected_posture_shift || "n/a"}`,
        `Confidence: ${
          scenario.confidence_level == null ? "n/a" : Number(scenario.confidence_level).toFixed(1)
        } / 100 (${scenario.confidence_tier || "unknown"})`,
        `Support: ${(scenario.supporting_conditions || []).slice(0, 2).join(" | ") || "n/a"}`,
        `Risk: ${(scenario.risk_conditions || []).slice(0, 2).join(" | ") || "n/a"}`,
      ];
      return `
        <section class="drilldown-card">
          <h5>${escapeHtml(label)}</h5>
          <div class="drilldown-note">${escapeHtml(scenario.summary || "No scenario summary available.")}</div>
          ${renderBullets(bullets, "No scenario detail available.")}
        </section>
      `;
    })
    .join("");
};

const renderRiskTriggers = (report) => {
  const container = qs("#assistant-risk-triggers");
  const strategy = report?.strategy;
  if (!strategy) {
    container.innerHTML = emptyStateCard(
      "Invalidators, confirmation triggers, deterioration triggers, and vetoes will render here."
    );
    return;
  }
  const invalidators = strategy.invalidators || {};
  const vetoLines = (strategy.fragility_vetoes || []).map(
    (item) => `${item.name || "veto"}: ${item.reason || ""}`
  );
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Top Invalidators</h5>
      ${renderBullets(invalidators.top_invalidators, "No top invalidators surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Confirmation Triggers</h5>
      ${renderBullets(strategy.confirmation_triggers, "No confirmation triggers surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Deterioration Triggers</h5>
      ${renderBullets(strategy.deterioration_triggers, "No deterioration triggers surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Fragility / Vetoes</h5>
      ${renderBullets(vetoLines, "No active dampeners or hard vetoes.")}
    </section>`,
  ].join("");
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
    qs("#assistant-analyze-response").textContent = formatJson({});
    renderDashboardSummary(null);
    renderWhySignal(null);
    renderMetricCards("#assistant-signal-summary-cards", null);
    renderTextSection("#assistant-signal-section", "Signal Summary", "");
    renderTextSection("#assistant-technical-section", "Technical Analysis", "");
    renderTextSection("#assistant-statistical-section", "Statistical / Quant Analysis", "");
    renderTextSection("#assistant-strategy-section", "Strategy View", "");
    renderTextSection("#assistant-risk-section", "Risks / Weaknesses / Invalidators", "");
    renderStrategyMetrics(null);
    renderStrategyScenarios(null);
    renderRiskTriggers(null);
    renderTextSection("#assistant-fundamental-section", "Fundamental Analysis", "");
    renderTextSection("#assistant-sentiment-section", "Sentiment / Narrative / Flow Analysis", "");
    renderTextSection(
      "#assistant-macro-section",
      "Macro / Geopolitical / Cross-Asset Analysis",
      ""
    );
    renderTextSection("#assistant-evidence-section", "Evidence / Provenance", "");
    renderDomainStatusGrid(null);
    renderFactorGrid(null);
    renderActiveAnalysisLabels();
    return;
  }

  container.classList.remove("empty");
  const heroMetrics = [
    { label: "Final Signal", value: report.strategy?.final_signal || report.signal?.final_action || report.signal?.action || "N/A" },
    { label: "Raw Signal", value: report.signal?.action || "N/A" },
    { label: "Posture", value: report.strategy?.strategy_posture || "N/A" },
    {
      label: "Confidence",
      value:
        report.strategy?.confidence == null
          ? "N/A"
          : Number(report.strategy.confidence).toFixed(2),
    },
    { label: "Conviction", value: report.strategy?.conviction_tier || "N/A" },
    {
      label: "Actionability",
      value:
        report.strategy?.actionability_score == null
          ? "N/A"
          : Number(report.strategy.actionability_score).toFixed(1),
    },
    { label: "Fragility", value: report.strategy?.fragility_tier || "N/A" },
    { label: "Participant", value: report.strategy?.primary_participant_fit || "N/A" },
    { label: "Freshness", value: report.freshness_summary?.overall_status || "N/A" },
    { label: "As Of", value: report.as_of_date || "N/A" },
    { label: "Scenario", value: report.scenario || "N/A" },
  ];
  const sections = [
    ["Signal Summary", report.signal_summary],
    ["Technical Analysis", report.technical_analysis],
    ["Fundamental Analysis", report.fundamental_analysis],
    ["Statistical / Quant Analysis", report.statistical_analysis],
    ["Sentiment / Narrative / Flow Analysis", report.sentiment_analysis],
    ["Macro / Geopolitical / Cross-Asset Context", report.macro_geopolitical_analysis],
    ["Strategy View", report.strategy_view],
    ["Risks / Weaknesses / Invalidators", report.risks_weaknesses_invalidators],
    ["Evidence / Provenance", report.evidence_provenance],
  ];

  container.innerHTML = `
    <div class="report-hero">
      <div class="report-kicker">Grounded Analysis Report</div>
      <h5 class="report-title">${escapeHtml(report.symbol || "Unknown Symbol")}</h5>
      <p class="report-subtitle">${escapeHtml(
        `${report.as_of_date || "n/a"} · ${report.horizon || "n/a"} horizon · ${
          report.risk_mode || "n/a"
        } risk · ${report.analysis_depth || "standard"} depth`
      )}</p>
    </div>
    <div class="report-metrics">
      ${heroMetrics
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

  const signalMetrics = [
    {
      label: "Current Signal",
      value: report.strategy?.final_signal || report.signal?.final_action || report.signal?.action || "n/a",
      note: `raw ${report.signal?.action || "n/a"}`,
    },
    {
      label: "Confidence",
      value:
        report.strategy?.confidence == null
          ? "n/a"
          : Number(report.strategy.confidence).toFixed(2),
      note: report.strategy?.conviction_tier || "unknown conviction",
    },
    {
      label: "Actionability",
      value:
        report.strategy?.actionability_score == null
          ? "n/a"
          : Number(report.strategy.actionability_score).toFixed(1),
      note: report.strategy?.strategy_posture || "posture unknown",
    },
    {
      label: "Regime",
      value:
        report.feature_factor_bundle?.regime_engine?.regime_label ||
        report.key_features?.regime_label ||
        "n/a",
      note: `scenario ${report.scenario || "base"}`,
    },
    {
      label: "Opportunity Quality",
      value:
        report.feature_factor_bundle?.composite_intelligence?.["Opportunity Quality Score"] == null
          ? "n/a"
          : Number(
              report.feature_factor_bundle.composite_intelligence["Opportunity Quality Score"]
            ).toFixed(1),
      note: `freshness ${report.freshness_summary?.overall_status || "unknown"}`,
    },
  ];
  qs("#assistant-analyze-response").textContent = formatJson(report);
  renderDashboardSummary(report);
  renderWhySignal(report);
  renderMetricCards("#assistant-signal-summary-cards", signalMetrics);
  renderTextSection("#assistant-signal-section", "Signal Summary", report.signal_summary);
  renderTextSection("#assistant-technical-section", "Technical Analysis", report.technical_analysis);
  renderTextSection(
    "#assistant-statistical-section",
    "Statistical / Quant Analysis",
    report.statistical_analysis
  );
  renderStrategyMetrics(report);
  renderTextSection("#assistant-strategy-section", "Strategy View", report.strategy_view);
  renderStrategyScenarios(report);
  renderTextSection(
    "#assistant-risk-section",
    "Risks / Weaknesses / Invalidators",
    report.risks_weaknesses_invalidators
  );
  renderRiskTriggers(report);
  renderTextSection(
    "#assistant-fundamental-section",
    "Fundamental Analysis",
    report.fundamental_analysis
  );
  renderTextSection(
    "#assistant-sentiment-section",
    "Sentiment / Narrative / Flow Analysis",
    report.sentiment_analysis
  );
  renderTextSection(
    "#assistant-macro-section",
    "Macro / Geopolitical / Cross-Asset Analysis",
    report.macro_geopolitical_analysis
  );
  renderTextSection(
    "#assistant-evidence-section",
    "Evidence / Provenance",
    report.evidence_provenance
  );
  renderDomainStatusGrid(report);
  renderFactorGrid(report);
  renderActiveAnalysisLabels();
};

const renderAssistantChatTranscript = () => {
  if (!state.assistantChatTranscript.length) {
    qs("#assistant-chat-response").textContent = "No grounded narrator messages yet.";
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

const renderSystemHealth = (health) => {
  const container = qs("#assistant-system-health");
  if (!health) {
    container.innerHTML = emptyStateCard("Assistant and provider health will render here.");
    return;
  }
  const providerCards = Object.entries(health.providers || {}).map(
    ([name, provider]) => `
      <div class="summary-card">
        <div class="summary-card-label">${escapeHtml(name)}</div>
        <div class="summary-card-value">${escapeHtml(provider.status || "unknown")}</div>
        <div class="summary-card-note">${escapeHtml(provider.message || "")}</div>
      </div>
    `
  );
  const assistantCards = [
    {
      label: "assistant",
      value: health.assistant?.status || "unknown",
      note: `llm_enabled=${health.assistant?.llm_enabled} db_enabled=${health.assistant?.db_enabled}`,
    },
  ];
  container.innerHTML = [
    ...assistantCards.map(
      (item) => `
        <div class="summary-card">
          <div class="summary-card-label">${escapeHtml(item.label)}</div>
          <div class="summary-card-value">${escapeHtml(item.value)}</div>
          <div class="summary-card-note">${escapeHtml(item.note)}</div>
        </div>
      `
    ),
    ...providerCards,
  ].join("");
};

const refreshAssistantSystemHealth = async () => {
  setLegacyStatus("#assistant-health-status", "Refreshing assistant and provider health...");
  setButtonLoading("#assistant-health-refresh-btn", true, "Refreshing...");
  try {
    const [assistantHealth, providersHealth] = await Promise.all([
      callJson("/assistant/health", { headers: getHeaders() }),
      callJson("/providers/health", { headers: getHeaders() }),
    ]);
    state.assistantHealth = {
      assistant: assistantHealth,
      providers: providersHealth.providers || {},
    };
    renderSystemHealth(state.assistantHealth);
    setLegacyStatus("#assistant-health-status", "System health refreshed.", "success");
  } catch (err) {
    renderSystemHealth(null);
    setLegacyStatus(
      "#assistant-health-status",
      `Health refresh failed: ${err.message}`,
      "error"
    );
  } finally {
    setButtonLoading("#assistant-health-refresh-btn", false, "Refreshing...");
  }
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
  market_regime: qs("#assistant-analyze-market-regime").value.trim(),
  analysis_depth: qs("#assistant-analyze-depth").value.trim(),
  refresh_mode: qs("#assistant-analyze-refresh-mode").value.trim(),
  scenario_mode: qs("#assistant-analyze-scenario-mode").value.trim(),
});

const getAssistantChatMessage = () => qs("#assistant-chat-message").value.trim();

const runAssistantAnalyze = async () => {
  const { symbol, horizon, risk_mode, market_regime, analysis_depth, refresh_mode, scenario_mode } =
    getAnalyzeInputs();
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
        market_regime,
        analysis_depth,
        refresh_mode,
        scenario_mode,
      }),
    });
    if (data.session_id) {
      persistAssistantSessionId(data.session_id);
    }
    persistActiveAnalysis(data.active_analysis || null);
    renderAssistantReport(data);
    setLegacyStatus(
      "#assistant-analyze-status",
      `Assistant analyze complete for ${data.symbol || symbol}.`,
      "success"
    );
    if (state.researchTab === "dashboard") {
      setResearchTab("dashboard");
    }
  } catch (err) {
    qs("#assistant-analyze-response").textContent = formatJson({
      error: err.message,
      symbol,
      horizon,
      risk_mode,
      market_regime,
      analysis_depth,
      refresh_mode,
      scenario_mode,
    });
    setLegacyStatus("#assistant-analyze-status", `Analyze failed: ${err.message}`, "error");
  } finally {
    setButtonLoading("#assistant-analyze-btn", false, "Running...");
    renderActiveAnalysisLabels();
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
  setLegacyStatus("#assistant-chat-status", "Sending grounded narrator request...");
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
    renderActiveAnalysisLabels();
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
qs("#assistant-health-refresh-btn").addEventListener("click", refreshAssistantSystemHealth);
qs("#assistant-chat-suggested-prompts").addEventListener("click", (event) => {
  const button = event.target.closest("[data-prompt]");
  if (!button) {
    return;
  }
  qs("#assistant-chat-message").value = button.dataset.prompt || "";
  qs("#assistant-chat-message").focus();
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
qsa(".research-tab").forEach((tab) => {
  tab.addEventListener("click", () => setResearchTab(tab.dataset.researchTab));
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
renderSystemHealth(null);
setResearchTab("dashboard");
setActiveTab("signal");
refreshAssistantSystemHealth();
