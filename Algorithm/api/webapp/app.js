const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  apiKey: "",
  assistantChatSessionId: "",
  assistantChatTranscript: [],
  assistantActiveAnalysis: null,
  assistantLatestReport: null,
  assistantRecentReports: [],
  assistantWatchlist: [],
  assistantCompareSymbol: "",
  assistantHealth: null,
  demoMode: false,
  researchTab: "dashboard",
};

const ASSISTANT_CHAT_SESSION_STORAGE_KEY = "ftip.assistant.chat.session_id";
const ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY = "ftip.assistant.active_analysis";
const ASSISTANT_RECENT_REPORTS_STORAGE_KEY = "ftip.assistant.recent_reports";
const ASSISTANT_WATCHLIST_STORAGE_KEY = "ftip.assistant.watchlist";
const ASSISTANT_COMPARE_SYMBOL_STORAGE_KEY = "ftip.assistant.compare_symbol";
const FTIP_DEMO_MODE_STORAGE_KEY = "ftip.demo_mode";

const isoDate = (date) => date.toISOString().slice(0, 10);
const formatJson = (value) => JSON.stringify(value ?? {}, null, 2);
const normalizeSymbol = (value) => String(value ?? "").trim().toUpperCase();
const formatScore = (value, digits = 1) =>
  value == null || Number.isNaN(Number(value)) ? "n/a" : Number(value).toFixed(digits);
const formatPercent = (value, digits = 1) =>
  value == null || Number.isNaN(Number(value))
    ? "n/a"
    : `${Number(value * 100).toFixed(digits)}%`;
const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

const emptyStateCard = (text) => `<div class="empty-state-card">${escapeHtml(text)}</div>`;
const readStorageJson = (key, fallback) => {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
};
const writeStorageJson = (key, value) =>
  window.localStorage.setItem(key, JSON.stringify(value));

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

const compositeScore = (report, label) =>
  report?.feature_factor_bundle?.composite_intelligence?.[label];

const topDriverLabel = (report, polarity) => {
  const drivers =
    polarity === "positive"
      ? report?.why_this_signal?.top_positive_drivers
      : report?.why_this_signal?.top_negative_drivers;
  return drivers?.[0]?.label || drivers?.[0]?.detail || "n/a";
};

const conditionLabel = (item) => {
  if (!item) {
    return "n/a";
  }
  if (item.dimension && item.label) {
    return `${item.dimension}=${item.label}`;
  }
  return item.label || item.dimension || "n/a";
};

const buildActiveAnalysisFromReport = (report) => ({
  symbol: report?.symbol || "n/a",
  as_of_date: report?.as_of_date || "n/a",
  horizon: report?.horizon || "n/a",
  risk_mode: report?.risk_mode || "n/a",
  signal:
    report?.strategy?.final_signal ||
    report?.signal?.final_action ||
    report?.signal?.action ||
    "n/a",
  conviction_tier: report?.strategy?.conviction_tier || "unknown",
  strategy_posture: report?.strategy?.strategy_posture || "n/a",
  freshness_status: report?.freshness_summary?.overall_status || "unknown",
  report_version: report?.report_version || "n/a",
  strategy_version: report?.strategy?.strategy_version || "n/a",
  deployment_mode: report?.deployment_mode || "research_only",
  deployment_permission: report?.deployment_permission || "analysis_only",
  trust_tier: report?.trust_tier || "blocked",
  live_readiness_status: report?.model_readiness_status || "unknown",
  live_readiness_score: report?.live_readiness_score,
  rollout_stage: report?.rollout_stage || "historical_validation",
  candidate_classification: report?.candidate_classification || "watchlist_candidate",
  ranked_opportunity_score: report?.ranked_opportunity_score,
  portfolio_fit_quality: report?.portfolio_fit_quality,
  size_band: report?.size_band || "watchlist only",
  setup_archetype: report?.setup_archetype?.archetype_name || "n/a",
  research_version: report?.research_version || "n/a",
  learning_priority: report?.learning_priority || "observe",
  validation_version: report?.validation_version || report?.canonical_validation?.validation_version || "n/a",
});

const buildStoredReportEntry = (report, activeAnalysis) => {
  const analysis = activeAnalysis || buildActiveAnalysisFromReport(report);
  return {
    symbol: normalizeSymbol(report?.symbol || analysis?.symbol),
    stored_at: new Date().toISOString(),
    active_analysis: analysis,
    snapshot: {
      symbol: normalizeSymbol(report?.symbol || analysis?.symbol),
      as_of_date: report?.as_of_date || analysis?.as_of_date || "n/a",
      horizon: report?.horizon || analysis?.horizon || "n/a",
      risk_mode: report?.risk_mode || analysis?.risk_mode || "n/a",
      final_signal:
        report?.strategy?.final_signal ||
        report?.signal?.final_action ||
        report?.signal?.action ||
        "n/a",
      strategy_posture: report?.strategy?.strategy_posture || analysis?.strategy_posture || "n/a",
      conviction_tier: report?.strategy?.conviction_tier || analysis?.conviction_tier || "unknown",
      actionability_score: report?.strategy?.actionability_score,
      confidence: report?.strategy?.confidence,
      confidence_score: report?.strategy?.confidence_score,
      fragility_tier: report?.strategy?.fragility_tier || "unknown",
      participant_fit: report?.strategy?.primary_participant_fit || "n/a",
      freshness_status:
        report?.freshness_summary?.overall_status || analysis?.freshness_status || "unknown",
      report_version: report?.report_version || analysis?.report_version || "n/a",
      strategy_version: report?.strategy?.strategy_version || analysis?.strategy_version || "n/a",
      evaluation_version: report?.evaluation?.evaluation_version || "phase6",
      opportunity_quality: compositeScore(report, "Opportunity Quality Score"),
      cross_domain_conviction: compositeScore(report, "Cross-Domain Conviction Score"),
      signal_fragility: compositeScore(report, "Signal Fragility Index"),
      fundamental_durability: compositeScore(report, "Fundamental Durability Score"),
      narrative_crowding: compositeScore(report, "Narrative Crowding Index"),
      macro_alignment: compositeScore(report, "Macro Alignment Score"),
      market_structure_integrity: compositeScore(report, "Market Structure Integrity Score"),
      regime_stability: compositeScore(report, "Regime Stability Score"),
      evaluation_hit_rate:
        report?.evaluation?.signal_scorecard?.final_signal_overall?.hit_rate ?? null,
      evaluation_reliability:
        report?.evaluation?.calibration_summary?.confidence_reliability_score ?? null,
      deployment_mode: report?.deployment_mode || analysis?.deployment_mode || "research_only",
      deployment_permission:
        report?.deployment_permission || analysis?.deployment_permission || "analysis_only",
      trust_tier: report?.trust_tier || analysis?.trust_tier || "blocked",
      live_readiness_status:
        report?.model_readiness_status || analysis?.live_readiness_status || "unknown",
      live_readiness_score:
        report?.live_readiness_score ?? analysis?.live_readiness_score ?? null,
      rollout_stage: report?.rollout_stage || analysis?.rollout_stage || "historical_validation",
      risk_budget_tier: report?.risk_budget_tier || "none",
      candidate_classification:
        report?.candidate_classification || analysis?.candidate_classification || "watchlist_candidate",
      ranked_opportunity_score:
        report?.ranked_opportunity_score ?? analysis?.ranked_opportunity_score ?? null,
      portfolio_candidate_score: report?.portfolio_candidate_score,
      portfolio_fit_quality:
        report?.portfolio_fit_quality ?? analysis?.portfolio_fit_quality ?? null,
      watchlist_priority_score: report?.watchlist_priority_score,
      deployability_rank: report?.deployability_rank,
      size_band: report?.size_band || analysis?.size_band || "watchlist only",
      weight_band: report?.weight_band || "n/a",
      risk_budget_band: report?.risk_budget_band || "n/a",
      overlap_score: report?.overlap_score,
      redundancy_score: report?.redundancy_score,
      diversification_contribution_score: report?.diversification_contribution_score,
      execution_quality_score: report?.execution_quality_score,
      friction_penalty: report?.friction_penalty,
      turnover_penalty: report?.turnover_penalty,
      setup_archetype:
        report?.setup_archetype?.archetype_name || analysis?.setup_archetype || "n/a",
      learning_priority: report?.learning_priority || analysis?.learning_priority || "observe",
      research_version: report?.research_version || analysis?.research_version || "n/a",
      strongest_condition: conditionLabel(report?.evaluation?.strongest_conditions?.[0]),
      weakest_condition: conditionLabel(report?.evaluation?.weakest_conditions?.[0]),
      validation_version:
        report?.validation_version || report?.canonical_validation?.validation_version || "phase10",
      walkforward_windows:
        report?.canonical_validation?.walkforward_summary?.window_count ?? null,
      validation_net_edge:
        report?.canonical_validation?.net_return_summary?.average_edge_return ?? null,
      validation_cost_drag:
        report?.canonical_validation?.friction_cost_summary?.average_cost_drag ?? null,
      executive_summary: report?.overall_analysis || report?.signal_summary || "",
      strategy_summary: report?.strategy_view || "",
      portfolio_summary: report?.portfolio_context_summary || "",
      top_driver: topDriverLabel(report, "positive"),
      top_risk:
        report?.strategy?.invalidators?.top_invalidators?.[0] ||
        topDriverLabel(report, "negative"),
    },
    report,
  };
};

const findStoredReport = (symbol) =>
  state.assistantRecentReports.find((item) => item.symbol === normalizeSymbol(symbol)) || null;

const persistRecentReports = (entries) => {
  const normalized = Array.isArray(entries) ? entries : [];
  state.assistantRecentReports = normalized.slice(0, 8);
  writeStorageJson(ASSISTANT_RECENT_REPORTS_STORAGE_KEY, state.assistantRecentReports);
};

const upsertRecentReport = (report, activeAnalysis) => {
  const entry = buildStoredReportEntry(report, activeAnalysis);
  const existing = state.assistantRecentReports.filter((item) => item.symbol !== entry.symbol);
  persistRecentReports([entry, ...existing]);
  renderRecentAnalyses();
  renderCompareWorkspace();
  renderWatchlistWorkspace();
  refreshCompareOptions();
};

const persistWatchlist = (symbols) => {
  const normalized = Array.isArray(symbols) ? symbols : [];
  state.assistantWatchlist = [...new Set(normalized.map(normalizeSymbol).filter(Boolean))].slice(0, 24);
  writeStorageJson(ASSISTANT_WATCHLIST_STORAGE_KEY, state.assistantWatchlist);
};

const persistCompareSymbol = (symbol) => {
  state.assistantCompareSymbol = normalizeSymbol(symbol);
  window.localStorage.setItem(
    ASSISTANT_COMPARE_SYMBOL_STORAGE_KEY,
    state.assistantCompareSymbol
  );
  if (qs("#assistant-compare-symbol")) {
    qs("#assistant-compare-symbol").value = state.assistantCompareSymbol;
  }
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

const applyDemoMode = (enabled) => {
  state.demoMode = !!enabled;
  document.body.classList.toggle("demo-mode", state.demoMode);
  qs("#demo-mode-status").textContent = state.demoMode ? "Demo mode on" : "Demo mode off";
  qs("#demo-mode-toggle").textContent = state.demoMode ? "Disable Demo Mode" : "Enable Demo Mode";
  window.localStorage.setItem(FTIP_DEMO_MODE_STORAGE_KEY, state.demoMode ? "1" : "0");
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

const activeDeploymentPermissionLabel = () =>
  state.assistantLatestReport?.deployment_permission ||
  state.assistantActiveAnalysis?.deployment_permission ||
  "analysis_only";

const activeTrustTierLabel = () =>
  state.assistantLatestReport?.trust_tier ||
  state.assistantActiveAnalysis?.trust_tier ||
  "blocked";

const activeReadinessLabel = () => {
  const score =
    state.assistantLatestReport?.live_readiness_score ??
    state.assistantActiveAnalysis?.live_readiness_score;
  const status =
    state.assistantLatestReport?.model_readiness_status ||
    state.assistantActiveAnalysis?.live_readiness_status ||
    "unknown";
  return score == null ? status : `${status} (${Number(score).toFixed(1)})`;
};

const activeReportVersionLabel = () =>
  state.assistantActiveAnalysis?.report_version ||
  state.assistantLatestReport?.report_version ||
  "n/a";

const buildNarratorPromptSet = () => {
  const symbol = state.assistantActiveAnalysis?.symbol || "this setup";
  return [
    `Why is ${symbol} ${activeSignalLabel()}?`,
    `Explain the strategy view for ${symbol}.`,
    `Which idea fits the portfolio better than ${symbol} right now?`,
    `What is the bear case for ${symbol}?`,
    `What are the invalidators here?`,
    `Is ${symbol} ready for live capital or only paper/shadow mode?`,
    `Why is ${symbol} watchlist-only or blocked in portfolio terms?`,
    `What is weakest in the setup for ${symbol}?`,
    `What would improve conviction on ${symbol}?`,
    `What does evaluation history say about setups like ${symbol}?`,
    `What is the platform learning lately about ${symbol}?`,
    `Where is the model drifting right now on ${symbol}?`,
    `What experiments should be run next for ${symbol}?`,
    `What setup archetype is ${symbol} right now?`,
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
    `Candidate: ${analysis?.candidate_classification || "watchlist_candidate"}`,
    `Archetype: ${analysis?.setup_archetype || "n/a"}`,
    `Learning: ${analysis?.learning_priority || "observe"}`,
    `Permission: ${activeDeploymentPermissionLabel()}`,
    `Size: ${analysis?.size_band || "watchlist only"}`,
    `Trust: ${activeTrustTierLabel()}`,
    `Report: ${activeReportVersionLabel()}`,
  ];
  qs("#assistant-active-analysis-meta").innerHTML = metaChips
    .map((item) => `<div class="active-chip">${escapeHtml(item)}</div>`)
    .join("");
  const chatMeta = [
    `Signal: ${activeSignalLabel()}`,
    `Conviction: ${activeConvictionLabel()}`,
    `Strategy: ${activeStrategyPostureLabel()}`,
    `Candidate: ${analysis?.candidate_classification || "watchlist_candidate"}`,
    `Archetype: ${analysis?.setup_archetype || "n/a"}`,
    `Learning: ${analysis?.learning_priority || "observe"}`,
    `Permission: ${activeDeploymentPermissionLabel()}`,
    `Size: ${analysis?.size_band || "watchlist only"}`,
    `Readiness: ${activeReadinessLabel()}`,
    `Freshness: ${activeFreshnessLabel()}`,
    `Report: ${activeReportVersionLabel()}`,
    `Session: ${state.assistantChatSessionId ? "active" : "local"}`,
  ];
  qs("#assistant-chat-analysis-meta").innerHTML = chatMeta
    .map((item) => `<div class="active-chip">${escapeHtml(item)}</div>`)
    .join("");
  qs("#assistant-chat-grounding-note").textContent = analysis?.symbol
    ? `Narrator is grounded to the active ${analysis.symbol} analysis artifact and will answer follow-up questions from the stored report, strategy, portfolio-construction, evaluation, deployment-readiness, and continuous-learning layers.`
    : "Run Assistant Analyze to establish the active artifact the narrator should use.";
  if (qs("#assistant-compare-active-symbol")) {
    qs("#assistant-compare-active-symbol").value = analysis?.symbol || "";
  }
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
  const containers = [
    qs("#assistant-domain-status-grid"),
    qs("#assistant-domain-status-grid-evidence"),
  ].filter(Boolean);
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
    containers.forEach((container) => {
      container.innerHTML = emptyStateCard("Domain freshness and coverage will render here.");
    });
    return;
  }
  const html = cards
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
  containers.forEach((container) => {
    container.innerHTML = html;
  });
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
    {
      label: "Deployment",
      value: report.deployment_permission || "analysis_only",
      note: `${report.deployment_mode || "research_only"} · trust ${report.trust_tier || "blocked"}`,
    },
    {
      label: "Portfolio Candidate",
      value: report.candidate_classification || "watchlist_candidate",
      note: `rank ${report.portfolio_rank || "n/a"} · fit ${formatScore(
        report.portfolio_fit_quality
      )}`,
    },
    {
      label: "Live Readiness",
      value:
        report.live_readiness_score == null
          ? report.model_readiness_status || "unknown"
          : Number(report.live_readiness_score).toFixed(1),
      note: report.model_readiness_status || "model readiness",
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

const renderDashboardWorkflow = (report) => {
  const container = qs("#assistant-dashboard-workflow");
  if (!report) {
    const placeholders = [
      ["Executive Read", "Start with the dashboard for signal, posture, conviction, and top risks.", "dashboard"],
      ["Decision Layer", "Open strategy to inspect actionability, scenarios, invalidators, and execution posture.", "strategy"],
      ["Portfolio Fit", "Use the portfolio workspace to inspect ranking, overlap, size-band logic, and rotation pressure.", "portfolio"],
      ["Trust Layer", "Use evaluation and evidence to show scorecards, calibration, provenance, and freshness.", "evaluation"],
      ["Learning Loop", "Open the learning lab for drift alerts, regime learnings, and governed improvement candidates.", "research"],
    ];
    container.innerHTML = placeholders
      .map(
        ([title, note, tab]) => `
          <section class="workflow-step demo-callout">
            <div class="workflow-step-title">${escapeHtml(title)}</div>
            <div class="workflow-step-note">${escapeHtml(note)}</div>
            <div class="workflow-step-actions">
              <button type="button" class="secondary" data-open-research-tab="${escapeHtml(
                tab
              )}">Open ${escapeHtml(tab)}</button>
            </div>
          </section>
        `
      )
      .join("");
    return;
  }

  const steps = [
    ["Signal and Thesis", report.signal_summary, "analyze"],
    ["Strategy Console", report.strategy_view, "strategy"],
    ["Portfolio Construction", report.portfolio_context_summary || report.portfolio_fit_analysis, "portfolio"],
    [
      "Deployment Readiness",
      report.deployment_readiness_summary || report.deployment_permission_analysis,
      "strategy",
    ],
    ["Proof and Trust", report.evaluation_research_analysis || report.evidence_provenance, "evaluation"],
    ["Learning and Improvement", report.learning_summary || report.adaptation_queue_summary, "research"],
  ];
  container.innerHTML = steps
    .map(
      ([title, note, tab]) => `
        <section class="workflow-step">
          <div class="workflow-step-head">
            <div class="workflow-step-title">${escapeHtml(title)}</div>
            <span class="legacy-badge">${escapeHtml(report.symbol || "active")}</span>
          </div>
          <div class="workflow-step-note">${escapeHtml(note || "No workflow note available.")}</div>
          <div class="workflow-step-actions">
            <button type="button" class="secondary" data-open-research-tab="${escapeHtml(
              tab
            )}">Open ${escapeHtml(tab)}</button>
          </div>
        </section>
      `
    )
    .join("");
};

const renderDashboardTrustStrip = (report) => {
  if (!report) {
    qs("#assistant-dashboard-trust-strip").innerHTML = emptyStateCard(
      "Freshness, versions, coverage quality, and evaluation readiness render here."
    );
    return;
  }
  renderMetricCards("#assistant-dashboard-trust-strip", [
    {
      label: "Freshness",
      value: report.freshness_summary?.overall_status || "unknown",
      note: report.freshness_summary?.summary || "active recency",
    },
    {
      label: "Coverage",
      value: report.data_bundle?.meta?.overall_quality || "mixed",
      note: report.data_bundle?.meta?.coverage_status || "domain completeness",
    },
    {
      label: "Version Stack",
      value: `${report.report_version || "n/a"} / ${report.strategy?.strategy_version || "n/a"}`,
      note: `evaluation ${report.evaluation?.evaluation_version || "phase6"}`,
    },
    {
      label: "Evaluation",
      value: report.evaluation?.status || "limited",
      note:
        report.evaluation?.prediction_linkage_summary?.total_predictions == null
          ? "sample pending"
          : `${report.evaluation.prediction_linkage_summary.total_predictions} tracked predictions`,
    },
    {
      label: "Deployment Mode",
      value: report.deployment_mode || "research_only",
      note: report.rollout_stage || "historical_validation",
    },
    {
      label: "Trust Tier",
      value: report.trust_tier || "blocked",
      note: report.deployment_permission || "analysis_only",
    },
    {
      label: "Readiness",
      value:
        report.live_readiness_score == null
          ? report.model_readiness_status || "unknown"
          : Number(report.live_readiness_score).toFixed(1),
      note:
        report.human_review_required == null
          ? "review policy pending"
          : `human review ${report.human_review_required ? "required" : "still advised"}`,
    },
    {
      label: "Portfolio Fit",
      value:
        report.portfolio_fit_quality == null
          ? "n/a"
          : Number(report.portfolio_fit_quality).toFixed(1),
      note: `${report.candidate_classification || "watchlist_candidate"} · ${report.size_band || "watchlist only"}`,
    },
    {
      label: "Learning",
      value: report.setup_archetype?.archetype_name || "n/a",
      note: `${report.learning_priority || "observe"} / ${report.research_version || "phase10"}`,
    },
  ]);
};

const renderRecentAnalyses = () => {
  const container = qs("#assistant-dashboard-recent");
  if (!state.assistantRecentReports.length) {
    container.innerHTML = emptyStateCard(
      "Recent analyzed names will appear here for fast reload, compare, and watchlist actions."
    );
    return;
  }
  container.innerHTML = `
    <div class="recent-analysis-list">
      ${state.assistantRecentReports
        .map((entry) => {
          const snapshot = entry.snapshot || {};
          return `
            <section class="recent-analysis-card">
              <div class="recent-analysis-head">
                <div>
                  <div class="recent-analysis-title">${escapeHtml(entry.symbol)}</div>
                  <div class="table-subtitle">${escapeHtml(
                    `${snapshot.as_of_date || "n/a"} · ${snapshot.horizon || "n/a"} · ${
                      snapshot.risk_mode || "n/a"
                    }`
                  )}</div>
                </div>
                <span class="legacy-badge">${escapeHtml(snapshot.final_signal || "n/a")}</span>
              </div>
              <div class="recent-analysis-meta">
                <span class="micro-chip">posture ${escapeHtml(snapshot.strategy_posture || "n/a")}</span>
                <span class="micro-chip">conviction ${escapeHtml(
                  snapshot.conviction_tier || "unknown"
                )}</span>
                <span class="micro-chip">fragility ${escapeHtml(
                  snapshot.fragility_tier || "unknown"
                )}</span>
                <span class="micro-chip">candidate ${escapeHtml(
                  snapshot.candidate_classification || "watchlist_candidate"
                )}</span>
                <span class="micro-chip">archetype ${escapeHtml(
                  snapshot.setup_archetype || "n/a"
                )}</span>
                <span class="micro-chip">learning ${escapeHtml(
                  snapshot.learning_priority || "observe"
                )}</span>
                <span class="micro-chip">permission ${escapeHtml(
                  snapshot.deployment_permission || "analysis_only"
                )}</span>
                <span class="micro-chip">size ${escapeHtml(
                  snapshot.size_band || "watchlist only"
                )}</span>
                <span class="micro-chip">freshness ${escapeHtml(
                  snapshot.freshness_status || "unknown"
                )}</span>
              </div>
              <div class="table-note">${escapeHtml(
                snapshot.portfolio_summary ||
                  snapshot.executive_summary ||
                  snapshot.top_driver ||
                  "No summary available."
              )}</div>
              <div class="recent-analysis-actions">
                <button type="button" class="secondary" data-load-report="${escapeHtml(
                  entry.symbol
                )}">Load</button>
                <button type="button" class="secondary" data-compare-symbol="${escapeHtml(
                  entry.symbol
                )}">Compare</button>
                <button type="button" class="secondary" data-watchlist-symbol="${escapeHtml(
                  entry.symbol
                )}">Watchlist</button>
              </div>
            </section>
          `;
        })
        .join("")}
    </div>
  `;
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
    {
      label: "Deployability",
      value: report?.deployment_permission || "analysis_only",
      note: report?.trust_tier || "blocked",
    },
    {
      label: "Live Readiness",
      value:
        report?.live_readiness_score == null
          ? report?.model_readiness_status || "unknown"
          : Number(report.live_readiness_score).toFixed(1),
      note: report?.deployment_mode || "research_only",
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

const renderStrategyPlaybook = (report) => {
  const container = qs("#assistant-strategy-playbook");
  const strategy = report?.strategy;
  if (!strategy) {
    container.innerHTML = emptyStateCard(
      "Execution posture, participant fit, and what would improve or worsen the setup will render here."
    );
    return;
  }
  const execution = strategy.execution_posture || {};
  const fit = [
    `Participant fit: ${strategy.primary_participant_fit || "n/a"}`,
    `Time-horizon fit: ${strategy.time_horizon_fit || "n/a"}`,
    `Execution posture: ${execution.preferred_posture || "n/a"} / urgency ${
      execution.urgency_level || "unknown"
    }`,
    `Patience: ${execution.patience_level || "n/a"} / cleanliness ${
      execution.signal_cleanliness || "n/a"
    }`,
  ];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Execution Posture</h5>
      ${renderBullets(fit, "No execution posture available.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>What Would Improve It</h5>
      ${renderBullets(
        (strategy.confirmation_triggers || []).slice(0, 4),
        "No explicit confirmation trigger surfaced."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>What Would Worsen It</h5>
      ${renderBullets(
        (strategy.deterioration_triggers || []).slice(0, 4),
        "No deterioration trigger surfaced."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Risk Context</h5>
      ${renderBullets(
        [execution.risk_context_summary, strategy.quality_of_setup, strategy.confidence_quality].filter(
          Boolean
        ),
        "No additional risk context surfaced."
      )}
    </section>`,
  ].join("");
};

const renderDeploymentReadiness = (report) => {
  const summary = qs("#assistant-deployment-summary-section");
  const detail = qs("#assistant-deployment-readiness");
  if (!report) {
    renderTextSection(
      "#assistant-deployment-summary-section",
      "Deployment Readiness Summary",
      ""
    );
    detail.innerHTML = emptyStateCard(
      "Deployment permission, blockers, and review requirements will render here."
    );
    return;
  }
  renderTextSection(
    "#assistant-deployment-summary-section",
    "Deployment Readiness Summary",
    report.deployment_readiness_summary
  );
  detail.innerHTML = [
    `<section class="drilldown-card">
      <h5>Permission / Trust</h5>
      ${renderBullets(
        [
          `Deployment mode: ${report.deployment_mode || "research_only"}`,
          `Permission: ${report.deployment_permission || "analysis_only"}`,
          `Trust tier: ${report.trust_tier || "blocked"}`,
          `Live readiness: ${
            report.live_readiness_score == null
              ? report.model_readiness_status || "unknown"
              : `${Number(report.live_readiness_score).toFixed(1)} / 100`
          }`,
        ],
        "No deployment permission state available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Blockers / Review</h5>
      ${renderBullets(
        [
          ...(report.deployment_blockers || []),
          ...(report.live_readiness_blockers || []),
          report.minimum_required_review
            ? `Minimum review: ${report.minimum_required_review}`
            : null,
          report.human_review_required == null
            ? null
            : `Human review required: ${report.human_review_required ? "yes" : "still expected"}`,
        ].filter(Boolean),
        "No active blockers surfaced."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Drift / Pause Alerts</h5>
      ${renderBullets(
        [
          ...(report.recent_degradation_flags || []),
          ...(report.drift_alerts || []),
          ...(report.deployment_risk_alerts || []),
          report.pause_recommended ? "Pause recommended for live-use support." : null,
          report.degrade_to_paper_recommended
            ? "Degrade to paper/shadow recommended."
            : null,
        ].filter(Boolean),
        "No active drift or pause alert surfaced."
      )}
    </section>`,
  ].join("");
};

const renderDeploymentRiskBudget = (report) => {
  const container = qs("#assistant-deployment-risk-budget");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Risk-budget and exposure-caution guidance will render here."
    );
    return;
  }
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Risk Budget / Exposure</h5>
      ${renderBullets(
        [
          `Risk budget tier: ${report.risk_budget_tier || "none"}`,
          `Exposure caution: ${report.exposure_caution_level || "extreme"}`,
          `Fragility-adjusted size band: ${report.fragility_adjusted_size_band || "none"}`,
          `Confidence-adjusted size band: ${report.confidence_adjusted_size_band || "none"}`,
          `Maximum risk mode allowed: ${report.maximum_risk_mode_allowed || "research_only"}`,
        ],
        "No risk-budget guidance available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Rollout Discipline</h5>
      ${renderBullets(
        [
          `Rollout stage: ${report.rollout_stage || "historical_validation"}`,
          `Readiness checkpoint: ${report.readiness_checkpoint || "watch"}`,
          ...(report.promotion_criteria || []),
          ...(report.demotion_criteria || []),
          ...(report.stage_transition_notes || []),
        ].filter(Boolean),
        "No rollout criteria available."
      )}
    </section>`,
  ].join("");
};

const renderPortfolioMetrics = (report) => {
  if (!report) {
    renderMetricCards("#assistant-portfolio-metrics", null);
    return;
  }
  renderMetricCards("#assistant-portfolio-metrics", [
    {
      label: "Candidate Class",
      value: report.candidate_classification || "watchlist_candidate",
      note: `portfolio rank ${report.portfolio_rank || "n/a"}`,
    },
    {
      label: "Portfolio Score",
      value:
        report.portfolio_candidate_score == null
          ? "n/a"
          : Number(report.portfolio_candidate_score).toFixed(1),
      note: `ranked opportunity ${formatScore(report.ranked_opportunity_score)}`,
    },
    {
      label: "Portfolio Fit",
      value:
        report.portfolio_fit_quality == null
          ? "n/a"
          : Number(report.portfolio_fit_quality).toFixed(1),
      note: `diversification ${formatScore(report.diversification_contribution_score)}`,
    },
    {
      label: "Overlap / Redundancy",
      value: `${formatScore(report.overlap_score)} / ${formatScore(report.redundancy_score)}`,
      note: report.most_redundant_symbol
        ? `closest overlap ${report.most_redundant_symbol}`
        : "no major redundancy peer",
    },
    {
      label: "Size Band",
      value: report.size_band || "watchlist only",
      note: report.weight_band || report.risk_budget_band || "band pending",
    },
    {
      label: "Execution Quality",
      value:
        report.execution_quality_score == null
          ? "n/a"
          : Number(report.execution_quality_score).toFixed(1),
      note: `friction ${formatScore(report.friction_penalty)} · turnover ${formatScore(
        report.turnover_penalty
      )}`,
    },
  ]);
};

const renderPortfolioControls = (report) => {
  const container = qs("#assistant-portfolio-controls");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Size-band, overlap, exposure warnings, and caution logic will render here."
    );
    return;
  }
  const warnings = [
    report.concentration_warning,
    report.cluster_concentration_warning,
    report.sector_crowding_warning,
    report.fragility_cluster_warning,
    report.macro_exposure_warning,
    report.theme_exposure_warning,
  ].filter(Boolean);
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Size / Risk Bands</h5>
      ${renderBullets(
        [
          `Size band: ${report.size_band || "watchlist only"}`,
          `Weight band: ${report.weight_band || "n/a"}`,
          `Risk budget band: ${report.risk_budget_band || "n/a"}`,
          `Caution level: ${report.caution_level || "measured"}`,
          `Max priority allowed: ${report.max_priority_allowed || "watchlist_only"}`,
        ],
        "No size-band guidance available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Adjustments / Warnings</h5>
      ${renderBullets(
        [
          `Fragility adjustment: ${report.fragility_adjustment || "neutral"}`,
          `Confidence adjustment: ${report.confidence_adjustment || "neutral"}`,
          `Concentration adjustment: ${report.concentration_adjustment || "neutral"}`,
          `Overlap adjustment: ${report.overlap_adjustment || "neutral"}`,
          `Deployment adjustment: ${report.deployment_mode_adjustment || "neutral"}`,
          ...warnings,
        ],
        "No portfolio-control warning surfaced."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Execution Discipline</h5>
      ${renderBullets(
        [
          `Execution quality: ${formatScore(report.execution_quality_score)}`,
          `Friction penalty: ${formatScore(report.friction_penalty)}`,
          `Turnover penalty: ${formatScore(report.turnover_penalty)}`,
          `Wait for better entry: ${report.wait_for_better_entry_flag ? "yes" : "no"}`,
          `Confirmation preferred: ${report.confirmation_preferred_flag ? "yes" : "no"}`,
        ],
        "No execution-quality note surfaced."
      )}
    </section>`,
  ].join("");
};

const renderPortfolioRanking = (report) => {
  const container = qs("#assistant-portfolio-ranking");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Cohort ranking and portfolio-adjusted candidate ordering will render here."
    );
    return;
  }
  const rows = report.cohort_ranking || [];
  if (!rows.length) {
    container.innerHTML = emptyStateCard(
      "Portfolio ranking will populate once there is an active cohort of stored analyses."
    );
    return;
  }
  container.innerHTML = `
    <div class="comparison-matrix">
      <table class="workspace-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Symbol</th>
            <th>Classification</th>
            <th>Portfolio Score</th>
            <th>Fit</th>
            <th>Permission</th>
            <th>Size Band</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .slice(0, 8)
            .map(
              (row) => `
                <tr>
                  <td>${escapeHtml(row.portfolio_rank || "n/a")}</td>
                  <td>
                    <div class="table-symbol">${escapeHtml(row.symbol || "n/a")}</div>
                    <div class="table-subtitle">${escapeHtml(
                      `${row.strategy_posture || "n/a"} · ${row.conviction_tier || "unknown"}`
                    )}</div>
                  </td>
                  <td>${escapeHtml(row.candidate_classification || "watchlist_candidate")}</td>
                  <td>${escapeHtml(formatScore(row.portfolio_candidate_score))}</td>
                  <td>${escapeHtml(formatScore(row.portfolio_fit_quality))}</td>
                  <td>${escapeHtml(row.deployment_permission || "analysis_only")}</td>
                  <td>${escapeHtml(row.size_band || "watchlist only")}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
};

const renderPortfolioWorkflow = (report) => {
  const container = qs("#assistant-portfolio-workflow");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Watchlist, blocked reasons, stale review needs, and rotation pressure will render here."
    );
    return;
  }
  const blocked = (report.portfolio_construction?.workflow?.blocked_candidates || []).map(
    (item) =>
      `${item.symbol || "n/a"} · ${item.classification || "blocked_candidate"}${
        item.reasons?.length ? ` · ${item.reasons.join(" | ")}` : ""
      }`
  );
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Workflow Summary</h5>
      ${renderBullets(
        [
          report.portfolio_workflow_summary,
          report.candidate_upgrade_reason,
          report.candidate_downgrade_reason,
          report.replacement_candidate_notes,
          `Priority shift flag: ${report.priority_shift_flag ? "yes" : "no"}`,
          `Rebalance attention: ${report.rebalance_attention_flag ? "yes" : "no"}`,
          `Rotation pressure: ${formatScore(report.rotation_pressure_score)}`,
        ].filter(Boolean),
        "No workflow summary available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Watchlists / Active Candidates</h5>
      ${renderBullets(
        [
          `Prioritized watchlist: ${(report.portfolio_construction?.workflow?.prioritized_watchlist || []).join(", ") || "none"}`,
          `Active candidates: ${(report.portfolio_construction?.workflow?.active_portfolio_candidates || []).join(", ") || "none"}`,
          `Stale review needed: ${(report.portfolio_construction?.workflow?.stale_review_needed || []).join(", ") || "none"}`,
        ],
        "No watchlist workflow surfaced."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Blocked / Redundant</h5>
      ${renderBullets(blocked, "No blocked or redundant cohort candidate surfaced.")}
    </section>`,
  ].join("");
};

const renderEvaluationMetrics = (report) => {
  const evaluation = report?.evaluation;
  const validation = report?.canonical_validation;
  if ((!evaluation || Object.keys(evaluation).length === 0) && (!validation || Object.keys(validation).length === 0)) {
    renderMetricCards("#assistant-evaluation-metrics", null);
    return;
  }
  const signal = evaluation.signal_scorecard || {};
  const strategy = evaluation.strategy_scorecard || {};
  const calibration = evaluation.calibration_summary || {};
  const walkforward = validation?.walkforward_summary || {};
  const net = validation?.net_return_summary || {};
  const friction = validation?.friction_cost_summary || {};
  renderMetricCards("#assistant-evaluation-metrics", [
    {
      label: "Matured Signals",
      value: signal.final_signal_overall?.matured_count ?? "n/a",
      note: `${evaluation.prediction_linkage_summary?.total_predictions ?? "n/a"} total predictions`,
    },
    {
      label: "Hit Rate",
      value:
        signal.final_signal_overall?.hit_rate == null
          ? "n/a"
          : Number(signal.final_signal_overall.hit_rate * 100).toFixed(1) + "%",
      note: "final strategy posture",
    },
    {
      label: "Avg Fwd Return",
      value:
        signal.final_signal_overall?.average_forward_return == null
          ? "n/a"
          : Number(signal.final_signal_overall.average_forward_return * 100).toFixed(2) + "%",
      note: report.horizon || "n/a",
    },
    {
      label: "Actionable Spread",
      value:
        strategy.actionable_vs_watchlist_return_spread == null
          ? "n/a"
          : Number(strategy.actionable_vs_watchlist_return_spread * 100).toFixed(2) + "%",
      note: "actionable vs wait/watch",
    },
    {
      label: "Reliability",
      value:
        calibration.confidence_reliability_score == null
          ? "n/a"
          : Number(calibration.confidence_reliability_score).toFixed(1),
      note: calibration.confidence_monotonicity || "sample-limited",
    },
    {
      label: "Walk-Forward",
      value: walkforward.window_count ?? "n/a",
      note: walkforward.status || validation?.status || "limited",
    },
    {
      label: "Net Edge",
      value:
        net.average_edge_return == null
          ? "n/a"
          : Number(net.average_edge_return * 100).toFixed(2) + "%",
      note:
        friction.average_cost_drag == null
          ? "net of friction"
          : `cost drag ${Number(friction.average_cost_drag * 100).toFixed(2)}%`,
    },
    {
      label: "Status",
      value: validation?.status || evaluation?.status || "unknown",
      note: validation?.validation_version || evaluation?.evaluation_version || "phase6",
    },
  ]);
};

const renderEvaluationRegimeGrid = (report) => {
  const container = qs("#assistant-regime-grid");
  const evaluation = report?.evaluation;
  const breakdown = evaluation?.regime_breakdown || {};
  const strongest = evaluation?.strongest_conditions || [];
  const weakest = evaluation?.weakest_conditions || [];
  if (!evaluation || Object.keys(evaluation).length === 0) {
    container.innerHTML = emptyStateCard(
      "Regime and cohort breakdowns will render here."
    );
    return;
  }
  const regimeRows = (breakdown.regime_label || []).slice(0, 3).map(
    (item) =>
      `${item.label}: avg ${
        item.average_forward_return == null
          ? "n/a"
          : Number(item.average_forward_return * 100).toFixed(2) + "%"
      } · hit ${
        item.hit_rate == null ? "n/a" : Number(item.hit_rate * 100).toFixed(1) + "%"
      }`
  );
  const fragilityRows = (breakdown.fragility_tier || []).slice(0, 3).map(
    (item) =>
      `${item.label}: avg ${
        item.average_forward_return == null
          ? "n/a"
          : Number(item.average_forward_return * 100).toFixed(2) + "%"
      } · hit ${
        item.hit_rate == null ? "n/a" : Number(item.hit_rate * 100).toFixed(1) + "%"
      }`
  );
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Regime Labels</h5>
      ${renderBullets(regimeRows, "No mature regime cohort yet.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Fragility Tiers</h5>
      ${renderBullets(fragilityRows, "No mature fragility cohort yet.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Strongest Conditions</h5>
      ${renderBullets(
        strongest.map(
          (item) =>
            `${item.dimension}=${item.label} · avg ${
              item.average_forward_return == null
                ? "n/a"
                : Number(item.average_forward_return * 100).toFixed(2) + "%"
            }`
        ),
        "No standout conditions yet."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Weakest Conditions</h5>
      ${renderBullets(
        weakest.map(
          (item) =>
            `${item.dimension}=${item.label} · avg ${
              item.average_forward_return == null
                ? "n/a"
                : Number(item.average_forward_return * 100).toFixed(2) + "%"
            }`
        ),
        "No weak-condition pattern yet."
      )}
    </section>`,
  ].join("");
};

const renderEvaluationFailureModes = (report) => {
  const container = qs("#assistant-failure-modes");
  const evaluation = report?.evaluation;
  const validation = report?.canonical_validation;
  const calibration = evaluation?.calibration_summary || {};
  const ranking = evaluation?.bucket_results || [];
  if ((!evaluation || Object.keys(evaluation).length === 0) && (!validation || Object.keys(validation).length === 0)) {
    container.innerHTML = emptyStateCard(
      "Failure modes, strongest conditions, and bucket results will render here."
    );
    return;
  }
  const driftNotes = calibration.calibration_drift_notes || [];
  const bucketSummaries = ranking
    .filter((item) => item.status === "available")
    .slice(0, 3)
    .map(
      (item) =>
        `${item.score_name}: spread ${
          item.favorable_vs_unfavorable_return_spread == null
            ? "n/a"
            : Number(item.favorable_vs_unfavorable_return_spread * 100).toFixed(2) + "%"
        } · ${item.monotonicity || "mixed"}`
    );
  const failureModes = ((validation?.failure_modes || evaluation?.failure_modes || [])).map(
    (item) =>
      `${item.dimension}=${item.label} · avg ${
        item.average_forward_return == null
          ? "n/a"
          : Number(item.average_forward_return * 100).toFixed(2) + "%"
      } · hit ${
        item.hit_rate == null ? "n/a" : Number(item.hit_rate * 100).toFixed(1) + "%"
      }`
  );
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Calibration Drift Notes</h5>
      ${renderBullets(driftNotes, "No material drift note surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Bucket Scorecards</h5>
      ${renderBullets(bucketSummaries, "No ranking bucket result available yet.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Failure Modes</h5>
      ${renderBullets(failureModes, "No repeated failure mode is isolated yet.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Research Layer</h5>
      ${renderBullets(
        [
          report.evaluation_summary,
          report.confidence_reliability_summary,
          report.regime_usefulness_summary,
          report.canonical_validation_summary,
          report.walkforward_validation_summary,
          report.net_of_friction_validation_summary,
        ].filter(Boolean),
        "No research summary available."
      )}
    </section>`,
  ].join("");
};

const renderEvaluationFactors = (report) => {
  const container = qs("#assistant-evaluation-factors");
  const attribution = report?.evaluation?.factor_attribution_summary || {};
  const scoreLeaders = attribution.proprietary_score_attribution || [];
  const componentLeaders = attribution.strategy_component_attribution || [];
  if (!scoreLeaders.length && !componentLeaders.length) {
    container.innerHTML = emptyStateCard(
      "Factor and strategy component usefulness will render here."
    );
    return;
  }
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Proprietary Score Leaders</h5>
      ${renderBullets(
        scoreLeaders.slice(0, 5).map(
          (item) =>
            `${item.score_name}: spread ${formatPercent(
              item.favorable_vs_unfavorable_return_spread,
              2
            )} · ${item.monotonicity || "mixed"}`
        ),
        "No proprietary score attribution yet."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Strategy Component Leaders</h5>
      ${renderBullets(
        componentLeaders.slice(0, 5).map(
          (item) =>
            `${item.score_name}: spread ${formatPercent(
              item.favorable_vs_unfavorable_return_spread,
              2
            )} · ${item.monotonicity || "mixed"}`
        ),
        "No component attribution yet."
      )}
    </section>`,
  ].join("");
};

const renderEvidenceSupport = (report) => {
  const support = qs("#assistant-evidence-supporting");
  const conflict = qs("#assistant-evidence-conflicts");
  if (!report) {
    support.innerHTML = emptyStateCard("Key supporting evidence will render here.");
    conflict.innerHTML = emptyStateCard(
      "Conflicting evidence, missingness, and freshness caveats will render here."
    );
    return;
  }

  const positiveDrivers = (report.why_this_signal?.top_positive_drivers || [])
    .slice(0, 4)
    .map((item) => `${item.label || "driver"}${item.detail ? `: ${item.detail}` : ""}`);
  const negatives = (report.why_this_signal?.top_negative_drivers || [])
    .slice(0, 4)
    .map((item) => `${item.label || "risk"}${item.detail ? `: ${item.detail}` : ""}`);
  const caveats = [
    ...(report.why_this_signal?.missing_data_warnings || []),
    ...(report.why_this_signal?.freshness_warnings || []),
  ];

  support.innerHTML = [
    `<section class="drilldown-card">
      <h5>Supporting Evidence</h5>
      ${renderBullets(positiveDrivers, "No explicit supporting evidence surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Provenance Notes</h5>
      ${renderBullets(
        [
          report.evidence_provenance,
          report.evaluation_summary,
          report.regime_usefulness_summary,
          report.deployment_readiness_summary,
          report.deployment_permission_analysis,
          report.portfolio_context_summary,
          report.portfolio_fit_analysis,
          report.learning_summary,
          report.regime_learning_summary,
          report.adaptation_queue_summary,
          report.experiment_registry_summary,
          report.archetype_motif_summary,
        ].filter(Boolean),
        "No provenance note available."
      )}
    </section>`,
  ].join("");

  conflict.innerHTML = [
    `<section class="drilldown-card">
      <h5>Conflicting Evidence</h5>
      ${renderBullets(negatives, "No explicit conflicting evidence surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Missingness / Freshness Caveats</h5>
      ${renderBullets(caveats, "No missing-data or freshness caveat surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Deployment Blockers</h5>
      ${renderBullets(
        [
          ...(report.deployment_blockers || []),
          ...(report.live_readiness_blockers || []),
          ...(report.drift_alerts || []),
          ...((report.learning_drift_alerts || []).map(
            (item) => `${item.affected_component || "drift"}: ${item.evidence || "drift alert"}`
          ) || []),
        ],
        "No explicit deployment blocker surfaced."
      )}
    </section>`,
  ].join("");
};

const renderArtifactLedger = (report) => {
  const container = qs("#assistant-artifact-ledger");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Report, strategy, evaluation, and provenance metadata will render here."
    );
    return;
  }
  const lines = [
    `Report version: ${report.report_version || "n/a"}`,
    `Strategy version: ${report.strategy?.strategy_version || "n/a"}`,
    `Evaluation version: ${report.evaluation?.evaluation_version || "phase6"}`,
    `Deployment mode: ${report.deployment_mode || "research_only"}`,
    `Deployment permission: ${report.deployment_permission || "analysis_only"}`,
    `Trust tier: ${report.trust_tier || "blocked"}`,
    `Deployment readiness artifact: ${report.deployment_readiness_artifact_id || "n/a"}`,
    `Deployment audit artifact: ${report.deployment_audit_artifact_id || "n/a"}`,
    `Portfolio artifact: ${report.portfolio_construction_artifact_id || "n/a"}`,
    `Prediction artifact: ${
      report.prediction_record_id || report.prediction_record_artifact_id || "n/a"
    }`,
    `Evaluation artifact: ${report.evaluation_artifact_id || "n/a"}`,
    `Learning artifact: ${report.learning_artifact_id || "n/a"}`,
    `Research version: ${report.research_version || "n/a"}`,
    `Setup archetype: ${report.setup_archetype?.archetype_name || "n/a"} / priority ${
      report.learning_priority || "observe"
    }`,
    `Portfolio rank: ${report.portfolio_rank || "n/a"} / class ${
      report.candidate_classification || "watchlist_candidate"
    } / size ${report.size_band || "watchlist only"}`,
    `Freshness status: ${report.freshness_summary?.overall_status || "unknown"}`,
    `Scenario: ${report.scenario || "base"} / depth ${report.analysis_depth || "standard"}`,
  ];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Artifact Metadata</h5>
      ${renderBullets(lines, "No artifact metadata available.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Traceability</h5>
      ${renderBullets(
        [
          report.signal_summary,
          report.strategy_view,
          report.evaluation_research_analysis,
          report.deployment_readiness_summary,
          report.rollout_stage_summary,
          report.learning_summary,
          report.experiment_registry_summary,
        ].filter(Boolean),
        "No traceability notes available."
      )}
    </section>`,
  ].join("");
};

const renderResearchDrilldown = (report) => {
  const container = qs("#assistant-research-drilldown");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Strategy components, factor attribution, and diagnostic drilldowns will render here."
    );
    return;
  }
  const components = Object.entries(report.strategy?.component_scores || {})
    .sort((left, right) => Number(right[1]?.weight || 0) - Number(left[1]?.weight || 0))
    .slice(0, 4)
    .map(
      ([name, payload]) =>
        `${name.replaceAll("_", " ")}: score ${formatScore(payload.score, 2)} · weight ${formatScore(
          payload.weight,
          2
        )}`
    );
  const scores = [
    `Opportunity Quality: ${formatScore(compositeScore(report, "Opportunity Quality Score"))}`,
    `Cross-Domain Conviction: ${formatScore(
      compositeScore(report, "Cross-Domain Conviction Score")
    )}`,
    `Signal Fragility: ${formatScore(compositeScore(report, "Signal Fragility Index"))}`,
    `Narrative Crowding: ${formatScore(compositeScore(report, "Narrative Crowding Index"))}`,
  ];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Strategy Components</h5>
      ${renderBullets(components, "No strategy component stack available.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Composite Scores</h5>
      ${renderBullets(scores, "No composite score detail available.")}
    </section>`,
  ].join("");
};

const renderLearningMetrics = (report) => {
  if (!report) {
    renderMetricCards("#assistant-learning-metrics", null);
    return;
  }
  const learning = report.continuous_learning || {};
  const cohort = learning.cohort_summary || {};
  const registry = report.experiment_registry || {};
  const activeMotif = (report.active_motifs || [])[0] || {};
  renderMetricCards("#assistant-learning-metrics", [
    {
      label: "Setup Archetype",
      value: report.setup_archetype?.archetype_name || "n/a",
      note: report.setup_archetype?.deployment_caution_level || "caution unknown",
    },
    {
      label: "Research Version",
      value: report.research_version || "phase10",
      note: `priority ${report.learning_priority || "observe"}`,
    },
    {
      label: "Cohort",
      value:
        cohort.tracked_reports == null ? "n/a" : `${cohort.tracked_reports} reports`,
      note:
        cohort.peer_reports == null
          ? "peer set unavailable"
          : `${cohort.peer_reports} peers / ${cohort.unique_symbols || 0} symbols`,
    },
    {
      label: "Drift Alerts",
      value: `${(report.learning_drift_alerts || []).length}`,
      note: (report.learning_drift_alerts || [])[0]?.severity || "none active",
    },
    {
      label: "Experiments",
      value: `${(registry.open_experiments || []).length}`,
      note: `${(registry.approved_improvements || []).length} approved / ${
        (registry.rejected_improvements || []).length
      } rejected`,
    },
    {
      label: "Active Motif",
      value: activeMotif.motif_id || "n/a",
      note: activeMotif.motif_summary || "no active motif highlighted",
    },
  ]);
};

const renderLearningRegimes = (report) => {
  const container = qs("#assistant-learning-regimes");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Regime-conditioned learnings and strongest or weakest environments render here."
    );
    return;
  }
  const learnings = report.regime_conditioned_learnings || [];
  const strongest = [...learnings]
    .sort((left, right) => Number(right.average_reliability || 0) - Number(left.average_reliability || 0))
    .slice(0, 2)
    .map(
      (item) =>
        `${item.regime_label || "unknown"}: reliability ${formatScore(
          item.average_reliability,
          1
        )} / hit rate ${formatScore(item.average_hit_rate, 2)}`
    );
  const weakest = [...learnings]
    .sort((left, right) => Number(left.average_reliability || 0) - Number(right.average_reliability || 0))
    .slice(0, 2)
    .map(
      (item) =>
        `${item.regime_label || "unknown"}: reliability ${formatScore(
          item.average_reliability,
          1
        )} / hit rate ${formatScore(item.average_hit_rate, 2)}`
    );
  const notes = learnings.slice(0, 5).map((item) => {
    const summary = item.decision_quality_summary || "No summary available.";
    const suggestion = item.adaptation_suggestion || "No adaptation suggestion.";
    return `${item.regime_label || "unknown"}: ${summary} Adaptation: ${suggestion}`;
  });
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Regime Learning Summary</h5>
      <p>${escapeHtml(report.regime_learning_summary || "No regime learning summary available.")}</p>
      ${renderBullets(notes, "No regime-conditioned learnings yet.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Strongest Regimes</h5>
      ${renderBullets(strongest, "No strongest regime surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Weakest Regimes</h5>
      ${renderBullets(weakest, "No weakest regime surfaced.")}
    </section>`,
  ].join("");
};

const renderLearningAdaptations = (report) => {
  const container = qs("#assistant-learning-adaptations");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Reweighting candidates, interaction learnings, and research hypotheses render here."
    );
    return;
  }
  const reweights = (report.reweighting_candidates || []).slice(0, 4).map((item) => {
    const change = (item.suggested_weight_changes || [])[0] || {};
    return `${item.target_family || "candidate"}: ${change.direction || "observe"} ${
      change.target || ""
    } (${formatScore(item.confidence_in_recommendation, 2)} confidence / sample ${
      item.sample_size || 0
    })`;
  });
  const hypotheses = (report.research_hypotheses || []).slice(0, 4).map(
    (item) =>
      `${item.hypothesis_title || "hypothesis"}: ${item.observed_pattern || "No observed pattern."}`
  );
  const interactions = (report.interaction_candidates || []).slice(0, 4).map(
    (item) =>
      `${item.interaction_candidate || "interaction"}: ${
        item.description || item.conditional_usefulness || "No interaction note."
      }`
  );
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Adaptation Queue</h5>
      <p>${escapeHtml(report.adaptation_queue_summary || "No adaptation queue summary available.")}</p>
      ${renderBullets(reweights, "No reweighting candidates surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Research Hypotheses</h5>
      ${renderBullets(hypotheses, "No research hypotheses surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Interaction Candidates</h5>
      ${renderBullets(interactions, "No interaction candidates surfaced.")}
    </section>`,
  ].join("");
};

const renderLearningExperiments = (report) => {
  const container = qs("#assistant-learning-experiments");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Proposed, approved, and under-review experiments render here."
    );
    return;
  }
  const registry = report.experiment_registry || {};
  const openItems = (registry.open_experiments || []).slice(0, 4).map(
    (item) =>
      `${item.title || item.experiment_id || "experiment"}: ${item.validation_status || "pending"} / ${
        item.approval_status || "review"
      }`
  );
  const approved = (registry.approved_improvements || []).slice(0, 3).map(
    (item) => `${item.title || item.experiment_id || "approved"}`
  );
  const queue = (report.improvement_queue || []).slice(0, 4).map(
    (item) =>
      `${item.title || item.target_family || item.hypothesis_title || "queue item"}: ${
        item.priority || item.severity || "review"
      }`
  );
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Experiment Registry</h5>
      <p>${escapeHtml(report.experiment_registry_summary || "No experiment registry summary available.")}</p>
      ${renderBullets(openItems, "No open experiments currently tracked.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Improvement Queue</h5>
      ${renderBullets(queue, "No improvement queue item surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Approved Improvements</h5>
      ${renderBullets(approved, "No approved improvements yet.")}
    </section>`,
  ].join("");
};

const renderLearningArchetypes = (report) => {
  const container = qs("#assistant-learning-archetypes");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Setup archetypes, active motifs, and failure modes render here."
    );
    return;
  }
  const active = report.setup_archetype || {};
  const motifs = (report.active_motifs || []).slice(0, 4).map(
    (item) => `${item.motif_id || "motif"}: ${item.motif_summary || "No motif summary."}`
  );
  const families = ((report.signal_family_library || {}).archetype_cohorts || [])
    .slice(0, 4)
    .map(
      (item) =>
        `${item.archetype_name || "archetype"}: ${item.sample_count || 0} samples / reliability ${formatScore(
          item.average_reliability,
          1
        )}`
    );
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Active Archetype</h5>
      <p>${escapeHtml(report.archetype_motif_summary || "No archetype summary available.")}</p>
      ${renderBullets(
        [
          active.summary,
          active.defining_characteristics?.length
            ? `Characteristics: ${active.defining_characteristics.join(", ")}`
            : null,
          active.common_failure_modes?.length
            ? `Failure modes: ${active.common_failure_modes.join(", ")}`
            : null,
          active.best_regimes?.length ? `Best regimes: ${active.best_regimes.join(", ")}` : null,
        ].filter(Boolean),
        "No active archetype detail available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Active Motifs</h5>
      ${renderBullets(motifs, "No active motifs surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Signal Family Library</h5>
      ${renderBullets(families, "No signal family cohorts available.")}
    </section>`,
  ].join("");
};

const renderSystemAudit = (report) => {
  const container = qs("#assistant-system-audit");
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Report, strategy, evaluation, freshness, and coverage audit details render here."
    );
    return;
  }
  const domains = report.freshness_summary?.domains || {};
  const domainNotes = Object.entries(domains)
    .slice(0, 6)
    .map(([name, payload]) => `${name}: ${payload?.status || "n/a"} / ${payload?.updated_at || "n/a"}`);
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Versioning</h5>
      ${renderBullets(
        [
          `Report ${report.report_version || "n/a"}`,
          `Strategy ${report.strategy?.strategy_version || "n/a"}`,
          `Evaluation ${report.evaluation?.evaluation_version || "phase6"}`,
          `Research ${report.research_version || "phase10"}`,
        ],
        "No version data available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Freshness by Domain</h5>
      ${renderBullets(domainNotes, "No domain freshness breakdown available.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Deployment Trust Layer</h5>
      ${renderBullets(
        [
          `Mode ${report.deployment_mode || "research_only"} / permission ${report.deployment_permission || "analysis_only"}`,
          `Trust ${report.trust_tier || "blocked"} / readiness ${
            report.live_readiness_score == null
              ? report.model_readiness_status || "unknown"
              : `${Number(report.live_readiness_score).toFixed(1)} / 100`
          }`,
          `Portfolio ${report.candidate_classification || "watchlist_candidate"} / fit ${formatScore(
            report.portfolio_fit_quality
          )} / size ${report.size_band || "watchlist only"}`,
          `Learning ${report.learning_priority || "observe"} / archetype ${
            report.setup_archetype?.archetype_name || "n/a"
          }`,
          ...(report.deployment_risk_alerts || []),
        ].filter(Boolean),
        "No deployment trust layer data available."
      )}
    </section>`,
  ].join("");
};

const renderDeploymentAudit = (report) => {
  const evidenceContainer = qs("#assistant-deployment-audit");
  const governanceContainer = qs("#assistant-deployment-governance");
  const containers = [evidenceContainer, governanceContainer].filter(Boolean);
  if (!containers.length) {
    return;
  }
  if (!report) {
    containers.forEach((container) => {
      container.innerHTML = emptyStateCard(
        "Deployment permission, rollout stage, audit snapshot, and pause conditions will render here."
      );
    });
    return;
  }
  const audit = report.live_use_audit_snapshot || {};
  const html = [
    `<section class="drilldown-card">
      <h5>Audit Snapshot</h5>
      ${renderBullets(
        [
          `Mode ${report.deployment_mode || "research_only"} / ${report.deployment_permission || "analysis_only"}`,
          `Trust ${report.trust_tier || "blocked"} / readiness ${
            report.live_readiness_score == null
              ? report.model_readiness_status || "unknown"
              : `${Number(report.live_readiness_score).toFixed(1)} / 100`
          }`,
          `Rollout stage ${report.rollout_stage || "historical_validation"} / checkpoint ${
            report.readiness_checkpoint || "watch"
          }`,
          `Human review required: ${report.human_review_required ? "yes" : "still advised"}`,
        ],
        "No audit snapshot available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Alerts / Rollback</h5>
      ${renderBullets(
        [
          ...(report.drift_alerts || []),
          ...(report.deployment_risk_alerts || []),
          report.rollback_reason,
          report.pause_recommended ? "Pause recommended." : null,
          report.degrade_to_paper_recommended ? "Degrade to paper/shadow recommended." : null,
          audit.rationale_summary,
        ].filter(Boolean),
        "No deployment alert or rollback note surfaced."
      )}
    </section>`,
  ].join("");
  containers.forEach((container) => {
    container.innerHTML = html;
  });
};

const refreshCompareOptions = () => {
  const options = [
    ...state.assistantWatchlist,
    ...state.assistantRecentReports.map((item) => item.symbol),
  ];
  qs("#assistant-compare-options").innerHTML = [...new Set(options.filter(Boolean))]
    .map((symbol) => `<option value="${escapeHtml(symbol)}"></option>`)
    .join("");
};

const loadStoredReportAsActive = (symbol, statusSelector = "#assistant-analyze-status") => {
  const entry = findStoredReport(symbol);
  if (!entry) {
    if (statusSelector) {
      setLegacyStatus(
        statusSelector,
        `No stored report is available for ${normalizeSymbol(symbol)} yet. Run Analyze first.`,
        "error"
      );
    }
    return false;
  }
  qs("#assistant-analyze-symbol").value = entry.symbol;
  persistActiveAnalysis(entry.active_analysis || buildActiveAnalysisFromReport(entry.report));
  renderAssistantReport(entry.report);
  if (statusSelector) {
    setLegacyStatus(statusSelector, `Loaded stored analysis for ${entry.symbol}.`, "success");
  }
  return true;
};

const addSymbolToWatchlist = (symbol, statusSelector = "#assistant-watchlist-status") => {
  const normalized = normalizeSymbol(symbol);
  if (!normalized) {
    if (statusSelector) {
      setLegacyStatus(statusSelector, "Symbol is required.", "error");
    }
    return;
  }
  persistWatchlist([...state.assistantWatchlist, normalized]);
  renderWatchlistWorkspace();
  refreshCompareOptions();
  if (statusSelector) {
    setLegacyStatus(statusSelector, `${normalized} added to the watchlist.`, "success");
  }
};

const removeSymbolFromWatchlist = (symbol) => {
  persistWatchlist(state.assistantWatchlist.filter((item) => item !== normalizeSymbol(symbol)));
  renderWatchlistWorkspace();
  refreshCompareOptions();
};

const renderWatchlistWorkspace = () => {
  const container = qs("#assistant-watchlist-grid");
  if (!state.assistantWatchlist.length) {
    container.innerHTML = emptyStateCard("Watchlist monitoring rows will render here.");
    return;
  }
  container.innerHTML = `
    <div class="comparison-matrix">
      <table class="workspace-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Signal</th>
            <th>Candidate</th>
            <th>Portfolio Score</th>
            <th>Fit</th>
            <th>Fragility</th>
            <th>Permission</th>
            <th>Size Band</th>
            <th>Freshness</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${state.assistantWatchlist
            .map((symbol) => {
              const entry = findStoredReport(symbol);
              const snapshot = entry?.snapshot || {};
              return `
                <tr>
                  <td>
                    <div class="table-symbol">${escapeHtml(symbol)}</div>
                    <div class="table-subtitle">${escapeHtml(
                      entry ? `${snapshot.as_of_date || "n/a"} · ${snapshot.horizon || "n/a"}` : "No stored artifact yet"
                    )}</div>
                  </td>
                  <td>${escapeHtml(snapshot.final_signal || "pending")}</td>
                  <td>${escapeHtml(snapshot.candidate_classification || "watchlist_candidate")}</td>
                  <td>${escapeHtml(formatScore(snapshot.portfolio_candidate_score))}</td>
                  <td>${escapeHtml(formatScore(snapshot.portfolio_fit_quality))}</td>
                  <td>${escapeHtml(snapshot.fragility_tier || "unknown")}</td>
                  <td>${escapeHtml(snapshot.deployment_permission || "analysis_only")}</td>
                  <td>${escapeHtml(snapshot.size_band || "watchlist only")}</td>
                  <td>${escapeHtml(snapshot.freshness_status || "unknown")}</td>
                  <td>
                    <div class="table-actions">
                      <button type="button" class="secondary" data-load-report="${escapeHtml(
                        symbol
                      )}">Load</button>
                      <button type="button" class="secondary" data-compare-symbol="${escapeHtml(
                        symbol
                      )}">Compare</button>
                      <button type="button" class="secondary" data-watchlist-remove="${escapeHtml(
                        symbol
                      )}">Remove</button>
                    </div>
                  </td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;
};

const compareMetricLeader = (left, right, higherBetter = true) => {
  if (left == null && right == null) {
    return "n/a";
  }
  if (left == null) {
    return "peer";
  }
  if (right == null) {
    return "active";
  }
  if (Math.abs(Number(left) - Number(right)) < 0.1) {
    return "balanced";
  }
  const activeBetter = higherBetter ? Number(left) > Number(right) : Number(left) < Number(right);
  return activeBetter ? "active" : "peer";
};

const renderCompareWorkspace = () => {
  const summary = qs("#assistant-compare-summary");
  const details = qs("#assistant-compare-details");
  const active = state.assistantLatestReport
    ? buildStoredReportEntry(state.assistantLatestReport, state.assistantActiveAnalysis)
    : null;
  const peer = findStoredReport(state.assistantCompareSymbol);

  if (!active) {
    summary.innerHTML = emptyStateCard(
      "Run or load an active analysis before using the comparison workspace."
    );
    details.innerHTML = emptyStateCard(
      "Compare signal, strategy, fragility, conviction, and evaluation snapshots here."
    );
    return;
  }

  if (!state.assistantCompareSymbol) {
    summary.innerHTML = emptyStateCard(
      "Choose a second symbol from recent analyses or the watchlist to compare against the active report."
    );
    details.innerHTML = emptyStateCard(
      "The active report will be compared once a peer symbol is selected."
    );
    return;
  }

  if (!peer) {
    summary.innerHTML = emptyStateCard(
      `${state.assistantCompareSymbol} is not stored yet. Run Analyze on that symbol to compare it.`
    );
    details.innerHTML = emptyStateCard(
      "Only stored assistant reports can be compared in the buyer-facing matrix."
    );
    return;
  }

  const left = active.snapshot || {};
  const right = peer.snapshot || {};
  renderMetricCards("#assistant-compare-summary", [
    {
      label: "Stronger Portfolio Score",
      value:
        compareMetricLeader(left.portfolio_candidate_score, right.portfolio_candidate_score, true) === "active"
          ? active.symbol
          : compareMetricLeader(left.portfolio_candidate_score, right.portfolio_candidate_score, true) === "peer"
            ? peer.symbol
            : "balanced",
      note: `${active.symbol} ${formatScore(left.portfolio_candidate_score)} vs ${peer.symbol} ${formatScore(
        right.portfolio_candidate_score
      )}`,
    },
    {
      label: "Better Portfolio Fit",
      value:
        compareMetricLeader(left.portfolio_fit_quality, right.portfolio_fit_quality, true) === "active"
          ? active.symbol
          : compareMetricLeader(left.portfolio_fit_quality, right.portfolio_fit_quality, true) === "peer"
            ? peer.symbol
            : "balanced",
      note: `${active.symbol} ${formatScore(left.portfolio_fit_quality)} vs ${peer.symbol} ${formatScore(
        right.portfolio_fit_quality
      )}`,
    },
    {
      label: "Lower Fragility",
      value:
        compareMetricLeader(left.signal_fragility, right.signal_fragility, false) === "active"
          ? active.symbol
          : compareMetricLeader(left.signal_fragility, right.signal_fragility, false) === "peer"
            ? peer.symbol
            : "balanced",
      note: `${active.symbol} ${formatScore(left.signal_fragility)} vs ${peer.symbol} ${formatScore(
        right.signal_fragility
      )}`,
    },
    {
      label: "Less Redundant",
      value:
        compareMetricLeader(left.redundancy_score, right.redundancy_score, false) === "active"
          ? active.symbol
          : compareMetricLeader(left.redundancy_score, right.redundancy_score, false) === "peer"
            ? peer.symbol
            : "balanced",
      note: `${active.symbol} ${formatScore(left.redundancy_score)} vs ${peer.symbol} ${formatScore(
        right.redundancy_score
      )}`,
    },
    {
      label: "Cleaner Execution",
      value:
        compareMetricLeader(left.execution_quality_score, right.execution_quality_score, true) === "active"
          ? active.symbol
          : compareMetricLeader(left.execution_quality_score, right.execution_quality_score, true) === "peer"
            ? peer.symbol
            : "balanced",
      note: `${active.symbol} ${formatScore(left.execution_quality_score)} vs ${peer.symbol} ${formatScore(
        right.execution_quality_score
      )}`,
    },
    {
      label: "Better Reliability",
      value:
        compareMetricLeader(left.evaluation_reliability, right.evaluation_reliability, true) === "active"
          ? active.symbol
          : compareMetricLeader(left.evaluation_reliability, right.evaluation_reliability, true) === "peer"
            ? peer.symbol
            : "balanced",
      note: `${active.symbol} ${formatScore(left.evaluation_reliability)} vs ${peer.symbol} ${formatScore(
        right.evaluation_reliability
      )}`,
    },
    {
      label: "Higher Readiness",
      value:
        compareMetricLeader(left.live_readiness_score, right.live_readiness_score, true) === "active"
          ? active.symbol
          : compareMetricLeader(left.live_readiness_score, right.live_readiness_score, true) === "peer"
            ? peer.symbol
            : "balanced",
      note: `${active.symbol} ${formatScore(left.live_readiness_score)} vs ${peer.symbol} ${formatScore(
        right.live_readiness_score
      )}`,
    },
  ]);

  const metrics = [
    ["Final Signal", left.final_signal, right.final_signal],
    ["Strategy Posture", left.strategy_posture, right.strategy_posture],
    ["Candidate Class", left.candidate_classification, right.candidate_classification],
    ["Conviction Tier", left.conviction_tier, right.conviction_tier],
    ["Deployment Permission", left.deployment_permission, right.deployment_permission],
    ["Trust Tier", left.trust_tier, right.trust_tier],
    ["Size Band", left.size_band, right.size_band],
    ["Live Readiness", formatScore(left.live_readiness_score), formatScore(right.live_readiness_score)],
    ["Rollout Stage", left.rollout_stage, right.rollout_stage],
    ["Actionability", formatScore(left.actionability_score), formatScore(right.actionability_score)],
    ["Opportunity Quality", formatScore(left.opportunity_quality), formatScore(right.opportunity_quality)],
    ["Portfolio Score", formatScore(left.portfolio_candidate_score), formatScore(right.portfolio_candidate_score)],
    ["Portfolio Fit", formatScore(left.portfolio_fit_quality), formatScore(right.portfolio_fit_quality)],
    ["Overlap Score", formatScore(left.overlap_score), formatScore(right.overlap_score)],
    ["Redundancy Score", formatScore(left.redundancy_score), formatScore(right.redundancy_score)],
    ["Diversification", formatScore(left.diversification_contribution_score), formatScore(right.diversification_contribution_score)],
    ["Execution Quality", formatScore(left.execution_quality_score), formatScore(right.execution_quality_score)],
    ["Cross-Domain Conviction", formatScore(left.cross_domain_conviction), formatScore(right.cross_domain_conviction)],
    ["Signal Fragility", formatScore(left.signal_fragility), formatScore(right.signal_fragility)],
    ["Fundamental Durability", formatScore(left.fundamental_durability), formatScore(right.fundamental_durability)],
    ["Narrative Crowding", formatScore(left.narrative_crowding), formatScore(right.narrative_crowding)],
    ["Macro Alignment", formatScore(left.macro_alignment), formatScore(right.macro_alignment)],
    ["Eval Hit Rate", formatPercent(left.evaluation_hit_rate), formatPercent(right.evaluation_hit_rate)],
    ["Reliability", formatScore(left.evaluation_reliability), formatScore(right.evaluation_reliability)],
  ];

  details.innerHTML = `
    <div class="comparison-matrix">
      <table class="workspace-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>${escapeHtml(active.symbol)}</th>
            <th>${escapeHtml(peer.symbol)}</th>
          </tr>
        </thead>
        <tbody>
          ${metrics
            .map(
              ([label, leftValue, rightValue]) => `
                <tr>
                  <td>${escapeHtml(label)}</td>
                  <td>${escapeHtml(leftValue || "n/a")}</td>
                  <td>${escapeHtml(rightValue || "n/a")}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
    <div class="drilldown-grid">
      <section class="drilldown-card">
        <h5>${escapeHtml(active.symbol)} Snapshot</h5>
        ${renderBullets([left.top_driver, left.top_risk, left.executive_summary].filter(Boolean), "No snapshot note available.")}
      </section>
      <section class="drilldown-card">
        <h5>${escapeHtml(peer.symbol)} Snapshot</h5>
        ${renderBullets([right.top_driver, right.top_risk, right.executive_summary].filter(Boolean), "No snapshot note available.")}
      </section>
    </div>
  `;
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
    renderDashboardWorkflow(null);
    renderDashboardTrustStrip(null);
    renderMetricCards("#assistant-signal-summary-cards", null);
    renderTextSection("#assistant-signal-section", "Signal Summary", "");
    renderTextSection("#assistant-overall-analysis-section", "Overall Thesis", "");
    renderTextSection("#assistant-technical-section", "Technical Analysis", "");
    renderTextSection("#assistant-statistical-section", "Statistical / Quant Analysis", "");
    renderTextSection("#assistant-strategy-section", "Strategy View", "");
    renderTextSection("#assistant-risk-section", "Risks / Weaknesses / Invalidators", "");
    renderStrategyMetrics(null);
    renderStrategyScenarios(null);
    renderStrategyPlaybook(null);
    renderRiskTriggers(null);
    renderEvaluationMetrics(null);
    renderTextSection("#assistant-evaluation-section", "Evaluation Research", "");
    renderTextSection("#assistant-calibration-section", "Confidence / Calibration", "");
    renderEvaluationRegimeGrid(null);
    renderEvaluationFailureModes(null);
    renderEvaluationFactors(null);
    renderTextSection("#assistant-fundamental-section", "Fundamental Analysis", "");
    renderTextSection("#assistant-sentiment-section", "Sentiment / Narrative / Flow Analysis", "");
    renderTextSection(
      "#assistant-macro-section",
      "Macro / Geopolitical / Cross-Asset Analysis",
      ""
    );
    renderTextSection("#assistant-evidence-section", "Evidence / Provenance", "");
    renderTextSection("#assistant-deployment-summary-section", "Deployment Readiness Summary", "");
    renderMetricCards("#assistant-portfolio-metrics", null);
    renderTextSection("#assistant-portfolio-context-section", "Portfolio Context", "");
    renderTextSection("#assistant-portfolio-fit-section", "Portfolio Fit", "");
    renderTextSection("#assistant-portfolio-execution-section", "Execution Quality", "");
    renderPortfolioControls(null);
    renderPortfolioRanking(null);
    renderPortfolioWorkflow(null);
    renderEvidenceSupport(null);
    renderArtifactLedger(null);
    renderDomainStatusGrid(null);
    renderFactorGrid(null);
    renderLearningMetrics(null);
    renderLearningRegimes(null);
    renderLearningAdaptations(null);
    renderLearningExperiments(null);
    renderLearningArchetypes(null);
    renderResearchDrilldown(null);
    renderSystemAudit(null);
    renderDeploymentReadiness(null);
    renderDeploymentRiskBudget(null);
    renderDeploymentAudit(null);
    renderCompareWorkspace();
    renderWatchlistWorkspace();
    renderActiveAnalysisLabels();
    renderRecentAnalyses();
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
    { label: "Deployment", value: report.deployment_permission || "N/A" },
    { label: "Trust Tier", value: report.trust_tier || "N/A" },
    {
      label: "Readiness",
      value:
        report.live_readiness_score == null
          ? report.model_readiness_status || "N/A"
          : Number(report.live_readiness_score).toFixed(1),
    },
    { label: "Candidate", value: report.candidate_classification || "N/A" },
    {
      label: "Portfolio Fit",
      value:
        report.portfolio_fit_quality == null
          ? "N/A"
          : Number(report.portfolio_fit_quality).toFixed(1),
    },
    { label: "Size Band", value: report.size_band || "N/A" },
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
    ["Event & Catalyst Risk", report.event_catalyst_risk_analysis],
    ["Liquidity / Execution Fragility", report.liquidity_execution_fragility_analysis],
    ["Market Breadth / Internal State", report.market_breadth_internal_state_analysis],
    ["Cross-Asset Confirmation", report.cross_asset_confirmation_analysis],
    ["Stress / Spillover Conditions", report.stress_spillover_analysis],
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
    {
      label: "Deployment Permission",
      value: report.deployment_permission || "analysis_only",
      note: report.trust_tier || "blocked",
    },
    {
      label: "Portfolio Candidate",
      value: report.candidate_classification || "watchlist_candidate",
      note: `portfolio score ${formatScore(report.portfolio_candidate_score)}`,
    },
    {
      label: "Live Readiness",
      value:
        report.live_readiness_score == null
          ? report.model_readiness_status || "unknown"
          : Number(report.live_readiness_score).toFixed(1),
      note: report.deployment_mode || "research_only",
    },
  ];
  qs("#assistant-analyze-response").textContent = formatJson(report);
  renderDashboardSummary(report);
  renderWhySignal(report);
  renderDashboardWorkflow(report);
  renderDashboardTrustStrip(report);
  renderMetricCards("#assistant-signal-summary-cards", signalMetrics);
  renderTextSection("#assistant-signal-section", "Signal Summary", report.signal_summary);
  renderTextSection(
    "#assistant-overall-analysis-section",
    "Overall Thesis",
    report.overall_analysis
  );
  renderTextSection("#assistant-technical-section", "Technical Analysis", report.technical_analysis);
  renderTextSection(
    "#assistant-statistical-section",
    "Statistical / Quant Analysis",
    report.statistical_analysis
  );
  renderStrategyMetrics(report);
  renderTextSection("#assistant-strategy-section", "Strategy View", report.strategy_view);
  renderStrategyScenarios(report);
  renderStrategyPlaybook(report);
  renderTextSection(
    "#assistant-risk-section",
    "Risks / Weaknesses / Invalidators",
    report.risks_weaknesses_invalidators
  );
  renderRiskTriggers(report);
  renderEvaluationMetrics(report);
  renderTextSection(
    "#assistant-evaluation-section",
    "Evaluation Research",
    report.evaluation_research_analysis
  );
  renderTextSection(
    "#assistant-calibration-section",
    "Confidence / Calibration",
    report.confidence_reliability_summary
  );
  renderEvaluationRegimeGrid(report);
  renderEvaluationFailureModes(report);
  renderEvaluationFactors(report);
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
  renderTextSection(
    "#assistant-deployment-summary-section",
    "Deployment Readiness Summary",
    report.deployment_readiness_summary
  );
  renderPortfolioMetrics(report);
  renderTextSection(
    "#assistant-portfolio-context-section",
    "Portfolio Context",
    report.portfolio_context_summary
  );
  renderTextSection(
    "#assistant-portfolio-fit-section",
    "Portfolio Fit",
    report.portfolio_fit_analysis
  );
  renderTextSection(
    "#assistant-portfolio-execution-section",
    "Execution Quality",
    report.execution_quality_analysis
  );
  renderPortfolioControls(report);
  renderPortfolioRanking(report);
  renderPortfolioWorkflow(report);
  renderEvidenceSupport(report);
  renderArtifactLedger(report);
  renderDomainStatusGrid(report);
  renderFactorGrid(report);
  renderLearningMetrics(report);
  renderLearningRegimes(report);
  renderLearningAdaptations(report);
  renderLearningExperiments(report);
  renderLearningArchetypes(report);
  renderResearchDrilldown(report);
  renderSystemAudit(report);
  renderDeploymentReadiness(report);
  renderDeploymentRiskBudget(report);
  renderDeploymentAudit(report);
  renderCompareWorkspace();
  renderWatchlistWorkspace();
  renderRecentAnalyses();
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

const resetAssistantChatSession = (message = "Fresh narrator session prepared.") => {
  persistAssistantSessionId(generateUuid());
  state.assistantChatTranscript = [];
  renderAssistantChatTranscript();
  setLegacyStatus("#assistant-chat-status", message, "success");
  renderActiveAnalysisLabels();
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
    upsertRecentReport(data, data.active_analysis || null);
    persistCompareSymbol(state.assistantCompareSymbol);
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
qs("#demo-mode-toggle").addEventListener("click", () => {
  applyDemoMode(!state.demoMode);
  if (state.demoMode) {
    setActiveTab("legacy");
    setResearchTab("dashboard");
  }
});
qs("#assistant-compare-load-btn").addEventListener("click", () => {
  const symbol = normalizeSymbol(qs("#assistant-compare-symbol").value);
  if (!symbol) {
    setLegacyStatus("#assistant-compare-status", "Choose a comparison symbol.", "error");
    return;
  }
  persistCompareSymbol(symbol);
  renderCompareWorkspace();
  setLegacyStatus("#assistant-compare-status", `Comparator set to ${symbol}.`, "success");
});
qs("#assistant-compare-clear-btn").addEventListener("click", () => {
  persistCompareSymbol("");
  renderCompareWorkspace();
  setLegacyStatus("#assistant-compare-status", "Comparison cleared.", "success");
});
qs("#assistant-watchlist-add-btn").addEventListener("click", () => {
  addSymbolToWatchlist(qs("#assistant-watchlist-symbol").value);
  qs("#assistant-watchlist-symbol").value = "";
});
qs("#assistant-watchlist-add-active-btn").addEventListener("click", () => {
  addSymbolToWatchlist(state.assistantActiveAnalysis?.symbol);
});
qs("#assistant-watchlist-clear-btn").addEventListener("click", () => {
  persistWatchlist([]);
  renderWatchlistWorkspace();
  refreshCompareOptions();
  setLegacyStatus("#assistant-watchlist-status", "Watchlist cleared.", "success");
});
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

document.body.addEventListener("click", (event) => {
  const openTab = event.target.closest("[data-open-research-tab]");
  if (openTab) {
    setResearchTab(openTab.dataset.openResearchTab);
    setActiveTab("legacy");
    return;
  }

  const loadReportButton = event.target.closest("[data-load-report]");
  if (loadReportButton) {
    loadStoredReportAsActive(loadReportButton.dataset.loadReport, "#assistant-analyze-status");
    setActiveTab("legacy");
    setResearchTab("dashboard");
    return;
  }

  const compareButton = event.target.closest("[data-compare-symbol]");
  if (compareButton) {
    const symbol = normalizeSymbol(compareButton.dataset.compareSymbol);
    persistCompareSymbol(symbol);
    renderCompareWorkspace();
    setLegacyStatus("#assistant-compare-status", `Comparator set to ${symbol}.`, "success");
    setActiveTab("legacy");
    setResearchTab("compare");
    return;
  }

  const watchlistButton = event.target.closest("[data-watchlist-symbol]");
  if (watchlistButton) {
    addSymbolToWatchlist(watchlistButton.dataset.watchlistSymbol);
    return;
  }

  const watchlistRemoveButton = event.target.closest("[data-watchlist-remove]");
  if (watchlistRemoveButton) {
    removeSymbolFromWatchlist(watchlistRemoveButton.dataset.watchlistRemove);
    setLegacyStatus("#assistant-watchlist-status", "Watchlist updated.", "success");
  }
});

qsa(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
});
qsa(".research-tab").forEach((tab) => {
  tab.addEventListener("click", () => setResearchTab(tab.dataset.researchTab));
});

setDefaults();
applyDemoMode(window.localStorage.getItem(FTIP_DEMO_MODE_STORAGE_KEY) === "1");
persistAssistantSessionId(
  window.localStorage.getItem(ASSISTANT_CHAT_SESSION_STORAGE_KEY) || generateUuid()
);
persistRecentReports(readStorageJson(ASSISTANT_RECENT_REPORTS_STORAGE_KEY, []));
persistWatchlist(readStorageJson(ASSISTANT_WATCHLIST_STORAGE_KEY, []));
persistCompareSymbol(window.localStorage.getItem(ASSISTANT_COMPARE_SYMBOL_STORAGE_KEY) || "");
persistActiveAnalysis(readStorageJson(ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY, null));
refreshCompareOptions();
if (!state.assistantActiveAnalysis?.symbol || !loadStoredReportAsActive(state.assistantActiveAnalysis.symbol, null)) {
  renderAssistantReport(null);
}
renderRecentAnalyses();
renderWatchlistWorkspace();
renderCompareWorkspace();
renderAssistantChatTranscript();
renderSystemHealth(null);
setResearchTab("dashboard");
setActiveTab(state.demoMode ? "legacy" : "signal");
refreshAssistantSystemHealth();
