const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

// ---------------------------------------------------------------------------
// Chart.js helpers
// ---------------------------------------------------------------------------
const _charts = {};

const destroyChart = (id) => {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
};

const _chartBaseOptions = (axisMin, axisMax) => ({
  indexAxis: "y",
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 250 },
  plugins: { legend: { display: false }, tooltip: { enabled: true } },
  scales: {
    x: {
      min: axisMin, max: axisMax,
      ticks: { color: "#aab7cb", font: { size: 10 } },
      grid: { color: "#26364b" },
    },
    y: {
      ticks: { color: "#edf3ff", font: { size: 10 } },
      grid: { display: false },
    },
  },
});

const makeChart = (id, labels, values, colors, axisMin, axisMax) => {
  destroyChart(id);
  const canvas = document.getElementById(id);
  if (!canvas) return;
  _charts[id] = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderRadius: 3,
        barThickness: 14,
      }],
    },
    options: _chartBaseOptions(axisMin, axisMax),
  });
};

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
  workspaceContinuityStore: {},
  demoMode: false,
  activeTab: "legacy",
  researchTab: "dashboard",
  copilotCollapsed: true,
};

const ASSISTANT_CHAT_SESSION_STORAGE_KEY = "ftip.assistant.chat.session_id";
const ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY = "ftip.assistant.active_analysis";
const ASSISTANT_RECENT_REPORTS_STORAGE_KEY = "ftip.assistant.recent_reports";
const ASSISTANT_WATCHLIST_STORAGE_KEY = "ftip.assistant.watchlist";
const ASSISTANT_COMPARE_SYMBOL_STORAGE_KEY = "ftip.assistant.compare_symbol";
const FTIP_DEMO_MODE_STORAGE_KEY = "ftip.demo_mode";
const ASSISTANT_COPILOT_COLLAPSED_STORAGE_KEY = "ftip.assistant.copilot.collapsed";
const ASSISTANT_WORKSPACE_CONTINUITY_STORAGE_KEY = "ftip.assistant.workspace_continuity";

const isoDate = (date) => date.toISOString().slice(0, 10);
const formatJson = (value) => JSON.stringify(value ?? {}, null, 2);
const normalizeSymbol = (value) => String(value ?? "").trim().toUpperCase();
const toFiniteNumber = (value) => {
  if (value == null || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};
const formatNumber = (value, digits = 1) => {
  const numeric = toFiniteNumber(value);
  return numeric == null ? "n/a" : numeric.toFixed(digits);
};
const formatScore = (value, digits = 1) => formatNumber(value, digits);
const formatPercent = (value, digits = 1) => {
  const numeric = toFiniteNumber(value);
  if (numeric == null) {
    return "n/a";
  }
  const scaled = Math.abs(numeric) <= 1.5 ? numeric * 100 : numeric;
  return `${scaled.toFixed(digits)}%`;
};
const formatPct = (value, digits = 1) => formatPercent(value, digits);
const formatDate = (value) => {
  if (value == null || value === "") {
    return "n/a";
  }
  const text = String(value);
  return text.length >= 10 ? text.slice(0, 10) : text;
};
const formatTier = (value) =>
  value == null || value === "" ? "n/a" : String(value).replaceAll("_", " ");
const formatCoverage = (value, digits = 0) => formatPercent(value, digits);
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
const writeStorageJson = (key, value) => {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
};
const writeStorageValue = (key, value) => {
  try {
    window.localStorage.setItem(key, String(value));
    return true;
  } catch {
    return false;
  }
};
const removeStorageValue = (key) => {
  try {
    window.localStorage.removeItem(key);
  } catch {}
};
const uniqueStringList = (values, limit = 6) =>
  [...new Set((Array.isArray(values) ? values : []).map((item) => String(item || "").trim()).filter(Boolean))].slice(
    0,
    limit
  );
const resolveCopilotContinuityScope = (pageContext = null) => {
  const context = pageContext || {};
  const workspaceId =
    context.workspace_id || state.assistantActiveAnalysis?.platform_workspace_id || null;
  const workspaceName =
    context.workspace_name ||
    state.assistantLatestReport?.platform_workspace?.name ||
    null;
  const symbol =
    context.symbol ||
    state.assistantActiveAnalysis?.symbol ||
    state.assistantLatestReport?.symbol ||
    null;
  if (workspaceId) {
    return `workspace:${workspaceId}`;
  }
  if (workspaceName) {
    return `workspace_name:${String(workspaceName).trim().toLowerCase()}`;
  }
  if (symbol) {
    return `symbol:${normalizeSymbol(symbol)}`;
  }
  return "global";
};
const getWorkspaceContinuityRecord = (pageContext = null) => {
  const scopeId =
    (pageContext && pageContext.continuity_scope_id) || resolveCopilotContinuityScope(pageContext);
  return state.workspaceContinuityStore?.[scopeId] || {
    scope_id: scopeId,
    recent_symbols: [],
    recent_dossiers: [],
    recent_workflows: [],
    recent_export_pack_types: [],
    recent_pages: [],
    recent_prompts: [],
  };
};
const persistWorkspaceContinuityStore = (store) => {
  state.workspaceContinuityStore = store || {};
  writeStorageJson(
    ASSISTANT_WORKSPACE_CONTINUITY_STORAGE_KEY,
    state.workspaceContinuityStore
  );
};
const updateWorkspaceContinuity = (partial = {}, pageContext = null) => {
  const context = pageContext || buildCopilotPageContext();
  const scopeId =
    context.continuity_scope_id || resolveCopilotContinuityScope(context);
  const current = getWorkspaceContinuityRecord({ continuity_scope_id: scopeId });
  const next = {
    ...current,
    scope_id: scopeId,
    workspace_id: context.workspace_id || current.workspace_id || null,
    workspace_name: context.workspace_name || current.workspace_name || null,
    recommendation_state:
      partial.recommendation_state ??
      context.recommendation_state ??
      current.recommendation_state ??
      null,
    committee_state:
      partial.committee_state ?? context.committee_state ?? current.committee_state ?? null,
    provider_health_status:
      partial.provider_health_status ??
      context.provider_health_status ??
      current.provider_health_status ??
      null,
    proof_maturity_level:
      partial.proof_maturity_level ??
      context.proof_maturity_level ??
      current.proof_maturity_level ??
      null,
    calibration_status:
      partial.calibration_status ??
      context.calibration_status ??
      current.calibration_status ??
      null,
    export_count:
      partial.export_count ?? context.export_count ?? current.export_count ?? 0,
    current_view_summary:
      partial.current_view_summary ??
      context.current_view_summary ??
      current.current_view_summary ??
      null,
    recent_symbols: uniqueStringList(
      [
        partial.symbol,
        context.symbol,
        ...(partial.recent_symbols || []),
        ...(current.recent_symbols || []),
      ],
      6
    ),
    recent_dossiers: uniqueStringList(
      [
        partial.dossier_id,
        context.dossier_id,
        ...(partial.recent_dossiers || []),
        ...(current.recent_dossiers || []),
      ],
      6
    ),
    recent_workflows: uniqueStringList(
      [
        partial.workflow_id,
        context.workflow_id,
        ...(partial.recent_workflows || []),
        ...(current.recent_workflows || []),
      ],
      6
    ),
    recent_export_pack_types: uniqueStringList(
      [
        ...(partial.recent_export_pack_types || []),
        ...(context.active_pack_types || []),
        ...(current.recent_export_pack_types || []),
      ],
      6
    ),
    recent_pages: uniqueStringList(
      [
        partial.page_label,
        context.page_label,
        ...(partial.recent_pages || []),
        ...(current.recent_pages || []),
      ],
      6
    ),
    recent_prompts: uniqueStringList(
      [
        ...(partial.recent_prompts || []),
        ...(current.recent_prompts || []),
      ],
      5
    ),
    updated_at: new Date().toISOString(),
  };
  persistWorkspaceContinuityStore({
    ...state.workspaceContinuityStore,
    [scopeId]: next,
  });
  return next;
};
const recordCopilotPrompt = (message, pageContext = null) => {
  const prompt = String(message || "").trim();
  if (!prompt) {
    return;
  }
  updateWorkspaceContinuity({ recent_prompts: [prompt] }, pageContext);
};

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
  state.activeTab = tabId;
  qsa(".tab").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabId));
  qsa(".tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== tabId));
  renderCopilotShell();
};

const setResearchTab = (tabId) => {
  state.researchTab = tabId;
  qsa(".research-tab").forEach((btn) =>
    btn.classList.toggle("active", btn.dataset.researchTab === tabId)
  );
  qsa("[data-research-panel]").forEach((panel) =>
    panel.classList.toggle("hidden", panel.dataset.researchPanel !== tabId)
  );
  renderCopilotShell();
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
  portfolio_risk_model_version: report?.portfolio_risk_model_version || "n/a",
  hidden_overlap_score: report?.hidden_overlap_score,
  portfolio_stress_score: report?.portfolio_stress_score,
  replacement_candidate: report?.replacement_candidate || null,
  size_band: report?.size_band || "watchlist only",
  setup_archetype: report?.setup_archetype?.archetype_name || "n/a",
  research_version: report?.research_version || "n/a",
  learning_priority: report?.learning_priority || "observe",
  validation_version: report?.validation_version || report?.canonical_validation?.validation_version || "n/a",
  operational_guardrails_version: report?.operational_guardrails_version || "n/a",
  system_health_status: report?.system_health_status || "unknown",
  shadow_mode_status: report?.shadow_mode_status || "unknown",
  current_operating_mode: report?.current_operating_mode || "normal",
  pause_required: Boolean(report?.pause_required),
  model_drift_score: report?.model_drift_score,
  data_reliability_score: report?.data_reliability_score,
  source_governance_version: report?.source_governance_version || "n/a",
  source_profile: report?.source_profile || "internal_research",
  buyer_demo_suitability: report?.buyer_demo_suitability || "unknown",
  commercialization_risk_score: report?.commercialization_risk_score,
  axiom_framework_version: report?.axiom_framework_version || "n/a",
  axiom_regime_label: report?.axiom_regime_label || "indeterminate",
  axiom_trade_family: report?.axiom_trade_family || "none",
  axiom_deployability_tier: report?.axiom_deployability_tier || "unknown",
  axiom_evidence_backed_deployability_tier:
    report?.axiom_evidence_backed_deployability_tier || "unknown",
  axiom_deployable_alpha_utility: report?.axiom_deployable_alpha_utility,
  axiom_validated_edge: report?.axiom_validated_edge,
  axiom_portfolio_fit_label: report?.axiom_portfolio_fit_label || "watchlist_only",
  axiom_portfolio_rank_score: report?.axiom_portfolio_rank_score,
  axiom_calibration_status: report?.axiom_calibration_status || "partial",
  axiom_audience_type: report?.axiom_audience_type || "general",
  axiom_report_profile: report?.axiom_report_profile || "trading_focused",
  axiom_lineage_summary: report?.axiom_lineage_summary || "",
  platform_profile: report?.platform_profile || null,
  platform_workspace_id: report?.platform_workspace?.workspace_id || null,
  platform_workflow_id: report?.platform_workflow?.workflow_id || null,
  platform_workflow_stage: report?.platform_workflow?.stage || null,
  platform_dossier_id: report?.platform_dossier?.dossier_id || null,
  platform_dossier_type: report?.platform_dossier?.dossier_type || null,
  platform_dossier_status: report?.platform_dossier?.evidence_status || null,
  platform_access_role:
    report?.platform_access_summary?.effective_role || null,
  platform_pending_approval_count:
    report?.platform_summary_view?.pending_approval_count ?? null,
  platform_export_count:
    report?.platform_summary_view?.export_count ?? null,
  platform_recommendation_locked:
    report?.platform_dossier?.metadata?.recommendation_locked ?? null,
  operating_workflow_version: report?.operating_workflow_version || "n/a",
  daily_operating_summary: report?.daily_operating_summary || "",
  weekly_operating_summary: report?.weekly_operating_summary || "",
  monthly_operating_summary: report?.monthly_operating_summary || "",
  trust_maintenance_summary: report?.trust_maintenance_summary || "",
  postmortem_summary: report?.postmortem_summary || "",
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
      portfolio_fit_rank: report?.portfolio_fit_rank ?? null,
      marginal_portfolio_utility: report?.marginal_portfolio_utility ?? null,
      portfolio_contribution_score: report?.portfolio_contribution_score ?? null,
      watchlist_priority_score: report?.watchlist_priority_score,
      deployability_rank: report?.deployability_rank,
      size_band: report?.size_band || analysis?.size_band || "watchlist only",
      weight_band: report?.weight_band || "n/a",
      risk_budget_band: report?.risk_budget_band || "n/a",
      overlap_score: report?.overlap_score,
      redundancy_score: report?.redundancy_score,
      hidden_overlap_score: report?.hidden_overlap_score,
      complementarity_score: report?.complementarity_score,
      diversification_contribution_score: report?.diversification_contribution_score,
      execution_quality_score: report?.execution_quality_score,
      friction_penalty: report?.friction_penalty,
      turnover_penalty: report?.turnover_penalty,
      portfolio_stress_score: report?.portfolio_stress_score,
      portfolio_fragility_score: report?.portfolio_fragility_score,
      correlation_breakdown_risk: report?.correlation_breakdown_risk,
      exposure_cluster: report?.exposure_cluster || "n/a",
      replacement_candidate: report?.replacement_candidate || null,
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
      axiom_framework_version:
        report?.axiom_framework_version || analysis?.axiom_framework_version || "n/a",
      axiom_regime_label:
        report?.axiom_regime_label || analysis?.axiom_regime_label || "indeterminate",
      axiom_trade_family:
        report?.axiom_trade_family || analysis?.axiom_trade_family || "none",
      axiom_deployability_tier:
        report?.axiom_deployability_tier || analysis?.axiom_deployability_tier || "unknown",
      axiom_evidence_backed_deployability_tier:
        report?.axiom_evidence_backed_deployability_tier ||
        analysis?.axiom_evidence_backed_deployability_tier ||
        "unknown",
      axiom_deployable_alpha_utility:
        report?.axiom_deployable_alpha_utility ??
        analysis?.axiom_deployable_alpha_utility ??
        null,
      axiom_validated_edge:
        report?.axiom_validated_edge ?? analysis?.axiom_validated_edge ?? null,
      axiom_portfolio_fit_label:
        report?.axiom_portfolio_fit_label ||
        analysis?.axiom_portfolio_fit_label ||
        "watchlist_only",
      axiom_portfolio_rank_score:
        report?.axiom_portfolio_rank_score ?? analysis?.axiom_portfolio_rank_score ?? null,
      axiom_calibration_status:
        report?.axiom_calibration_status || analysis?.axiom_calibration_status || "partial",
      axiom_audience_type:
        report?.axiom_audience_type || analysis?.axiom_audience_type || "general",
      axiom_report_profile:
        report?.axiom_report_profile || analysis?.axiom_report_profile || "trading_focused",
      axiom_lineage_summary:
        report?.axiom_lineage_summary || analysis?.axiom_lineage_summary || "",
      platform_profile:
        report?.platform_profile || analysis?.platform_profile || null,
      platform_workspace_id:
        report?.platform_workspace?.workspace_id || analysis?.platform_workspace_id || null,
      platform_workflow_id:
        report?.platform_workflow?.workflow_id || analysis?.platform_workflow_id || null,
      platform_workflow_stage:
        report?.platform_workflow?.stage || analysis?.platform_workflow_stage || null,
      platform_dossier_id:
        report?.platform_dossier?.dossier_id || analysis?.platform_dossier_id || null,
      platform_dossier_type:
        report?.platform_dossier?.dossier_type || analysis?.platform_dossier_type || null,
      platform_dossier_status:
        report?.platform_dossier?.evidence_status || analysis?.platform_dossier_status || null,
      platform_access_role:
        report?.platform_access_summary?.effective_role || analysis?.platform_access_role || null,
      platform_pending_approval_count:
        report?.platform_summary_view?.pending_approval_count ??
        analysis?.platform_pending_approval_count ??
        null,
      platform_export_count:
        report?.platform_summary_view?.export_count ??
        analysis?.platform_export_count ??
        null,
      platform_recommendation_locked:
        report?.platform_dossier?.metadata?.recommendation_locked ??
        analysis?.platform_recommendation_locked ??
        null,
      platform_overview_summary: report?.platform_overview_summary || "",
      platform_dossier_summary: report?.platform_dossier_summary || "",
      platform_monitoring_summary: report?.platform_monitoring_summary || "",
      portfolio_risk_model_version:
        report?.portfolio_risk_model_version || analysis?.portfolio_risk_model_version || "n/a",
      operational_guardrails_version:
        report?.operational_guardrails_version ||
        analysis?.operational_guardrails_version ||
        "phase12",
      system_health_status:
        report?.system_health_status || analysis?.system_health_status || "unknown",
      shadow_mode_status:
        report?.shadow_mode_status || analysis?.shadow_mode_status || "unknown",
      current_operating_mode:
        report?.current_operating_mode || analysis?.current_operating_mode || "normal",
      pause_required:
        report?.pause_required ?? analysis?.pause_required ?? false,
      model_drift_score:
        report?.model_drift_score ?? analysis?.model_drift_score ?? null,
      data_reliability_score:
        report?.data_reliability_score ?? analysis?.data_reliability_score ?? null,
      operating_workflow_version:
        report?.operating_workflow_version ||
        analysis?.operating_workflow_version ||
        "phase14",
      executive_summary: report?.overall_analysis || report?.signal_summary || "",
      strategy_summary: report?.strategy_view || "",
      portfolio_summary: report?.portfolio_context_summary || "",
      operational_summary:
        report?.system_health_summary || report?.drift_control_summary || "",
      axiom_summary: report?.axiom_summary || "",
      axiom_summary_card_text: report?.axiom_summary_card_text || "",
      axiom_ic_memo_summary: report?.axiom_ic_memo_summary || "",
      axiom_risk_deployability_memo_summary:
        report?.axiom_risk_deployability_memo_summary || "",
      axiom_lineage_summary: report?.axiom_lineage_summary || "",
      workflow_summary:
        report?.daily_operating_summary ||
        report?.weekly_operating_summary ||
        report?.monthly_operating_summary ||
        "",
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
  writeStorageValue(
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

  const wrap = qs("#signal-chart-wrap");
  const score = signal?.score ?? null;
  const conf = signal?.confidence ?? null;
  if (score == null && conf == null) {
    if (wrap) wrap.style.display = "none";
    destroyChart("signal-score-chart");
    return;
  }
  if (wrap) wrap.style.display = "";
  const labels = [];
  const values = [];
  const colors = [];
  if (score != null) {
    labels.push("Score");
    values.push(Number(score));
    colors.push(Number(score) >= 0 ? "#93e6b0" : "#ff9d9d");
  }
  if (conf != null) {
    labels.push("Confidence");
    values.push(Number(conf));
    colors.push("#4ea1ff");
  }
  makeChart("signal-score-chart", labels, values, colors, -1, 1);
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
  writeStorageValue(ASSISTANT_CHAT_SESSION_STORAGE_KEY, sessionId);
};

const applyDemoMode = (enabled) => {
  state.demoMode = !!enabled;
  document.body.classList.toggle("demo-mode", state.demoMode);
  qs("#demo-mode-status").textContent = state.demoMode ? "Demo mode on" : "Demo mode off";
  qs("#demo-mode-toggle").textContent = state.demoMode ? "Disable Demo Mode" : "Enable Demo Mode";
  writeStorageValue(FTIP_DEMO_MODE_STORAGE_KEY, state.demoMode ? "1" : "0");
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
    `Show me the IC memo version of ${symbol}.`,
    `What direct sources support the AXIOM fragility score for ${symbol}?`,
    `What is the weakest evidence in the AXIOM stack for ${symbol}?`,
    `How would ${symbol} look for a family office versus a hedge fund?`,
    `Summarize the workflow stage and dossier status for ${symbol}.`,
    `What would improve conviction on ${symbol}?`,
    `What does evaluation history say about setups like ${symbol}?`,
    `What is the platform learning lately about ${symbol}?`,
    `Where is the model drifting right now on ${symbol}?`,
    `Is the system healthy enough to trust right now for ${symbol}?`,
    `Is the current source stack buyer-demo safe for ${symbol}?`,
    `What changed today for ${symbol} that deserves review first?`,
    `What should be in this week's review for ${symbol}?`,
    `What failed recently around ${symbol} and why?`,
    `What should be prioritized next month for ${symbol}?`,
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

const researchTabLabel = () => {
  const labels = {
    dashboard: "dashboard",
    strategy: "strategy",
    portfolio: "portfolio",
    evaluation: "evaluation",
    compare: "compare",
    watchlist: "watchlist",
    chat: "narrator",
    platform: "platform console",
    research: "learning lab",
    health: "system health",
  };
  return labels[state.researchTab] || state.researchTab || "dashboard";
};

const buildCopilotPageContext = () => {
  const analysis = state.assistantActiveAnalysis || {};
  const report = state.assistantLatestReport || {};
  const workspace = report.platform_workspace || {};
  const dossier = report.platform_dossier || {};
  const workflow = report.platform_workflow || {};
  const axiomCard = report.axiom_summary_card || {};
  const storedExports = report.platform_stored_exports || [];
  const recommendationState = report.platform_recommendation_state || {};
  const committeeDecision = report.platform_committee_decision || {};
  const reviewSummary = report.platform_review_summary || {};
  const healthSummary = report.platform_health_summary || {};
  const proofSummary = report.platform_proof_summary || {};
  const calibrationHardening = report.platform_calibration_hardening || {};
  const unresolvedConcerns = reviewSummary.unresolved_concern_count || 0;
  const providerHealthStatus =
    report.provider_health_status ||
    healthSummary.integration_health_summary?.overall_status ||
    ((healthSummary.warnings || []).length ? "warning" : null);
  const contextWarning =
    (healthSummary.warnings || [])[0] ||
    (report.live_readiness_blockers || [])[0] ||
    (report.deployment_blockers || [])[0] ||
    (report.axiom_weakest_evidence_areas || [])[0] ||
    null;
  const baseContext = {
    app_tab: state.activeTab || "legacy",
    research_tab: state.researchTab || "dashboard",
    page_label: researchTabLabel(),
    page_focus:
      state.researchTab === "platform"
        ? "workflow_and_committee_console"
        : state.researchTab === "portfolio"
        ? "portfolio_fit_and_sizing"
        : state.researchTab === "evaluation"
        ? "proof_cycle_and_calibration"
        : state.researchTab === "health"
        ? "system_health_and_provider_integrity"
        : state.researchTab === "chat"
        ? "narrator_and_copilot"
        : "analysis_and_monitoring",
    workspace_id: analysis.platform_workspace_id || workspace.workspace_id || null,
    workspace_name: workspace.name || null,
    workflow_id: analysis.platform_workflow_id || workflow.workflow_id || null,
    workflow_stage: analysis.platform_workflow_stage || workflow.stage || null,
    dossier_id: analysis.platform_dossier_id || dossier.dossier_id || null,
    dossier_type: analysis.platform_dossier_type || dossier.dossier_type || null,
    dossier_status: analysis.platform_dossier_status || dossier.evidence_status || null,
    symbol: analysis.symbol || report.symbol || null,
    axiom_artifact_id: report.axiom_artifact_id || null,
    report_id: report.report_id || null,
    export_count: storedExports.length,
    active_pack_types: storedExports.slice(0, 3).map((item) => item.pack_type).filter(Boolean),
    deployability_tier:
      analysis.axiom_evidence_backed_deployability_tier ||
      analysis.axiom_deployability_tier ||
      axiomCard.deployability_tier ||
      null,
    recommendation_state: recommendationState.state || report.platform_dossier?.current_recommendation_state || null,
    committee_state: committeeDecision.decision_status || null,
    review_state:
      unresolvedConcerns > 0 ? "concerns_open" : reviewSummary.thread_summary ? "reviewed" : null,
    unresolved_concern_count: unresolvedConcerns,
    pending_approval_count:
      report.platform_summary_view?.pending_approval_count ||
      report.platform_dashboard?.executive_metrics?.pending_approval_count ||
      0,
    provider_health_status: providerHealthStatus,
    proof_maturity_level: proofSummary.evidence_maturity_level || null,
    calibration_status:
      calibrationHardening.status || report.platform_model_credibility_snapshot?.status || null,
    report_format_date: formatDate(report.as_of_date),
    current_view_summary:
      report.platform_dashboard_summary ||
      report.platform_dossier_summary ||
      report.axiom_proprietary_synthesis ||
      report.signal_summary ||
      null,
    context_warning: contextWarning,
  };
  const continuityScopeId = resolveCopilotContinuityScope(baseContext);
  return {
    ...baseContext,
    continuity_scope_id: continuityScopeId,
    workspace_continuity: state.workspaceContinuityStore?.[continuityScopeId] || {},
  };
};

const buildCopilotPromptSet = () => {
  const pageContext = buildCopilotPageContext();
  const continuity = pageContext.workspace_continuity || {};
  const symbol =
    state.assistantActiveAnalysis?.symbol ||
    state.assistantLatestReport?.symbol ||
    "this setup";
  const recommendationState = pageContext.recommendation_state || "draft";
  const base = [
    `What matters most on the ${researchTabLabel()} page right now?`,
    `Summarize the current workspace, dossier, and recommendation state for ${symbol}.`,
    `What is uniquely mispriced in ${symbol} right now?`,
    `What is the weakest evidence or biggest concern for ${symbol}?`,
    `What would have to improve to move ${symbol} beyond ${recommendationState.replaceAll("_", " ")}?`,
  ];
  if (pageContext.unresolved_concern_count > 0) {
    base.unshift(
      `What unresolved concern is most likely to block ${symbol} right now?`
    );
  }
  if (pageContext.committee_state) {
    base.push(
      `Explain the current committee state for ${symbol} and what it really means.`
    );
  }
  if (pageContext.pending_approval_count > 0) {
    base.push(`Which approval or workflow gate needs attention first?`);
  }
  if (pageContext.proof_maturity_level) {
    base.push(
      `How mature is the proof cycle behind ${symbol} and where is it still weak?`
    );
  }
  if (pageContext.provider_health_status) {
    base.push(
      `Which provider or data-quality issue is most likely to distort the current view?`
    );
  }
  if ((continuity.recent_symbols || []).length > 1) {
    const comparisonPeer = (continuity.recent_symbols || []).find(
      (item) => normalizeSymbol(item) !== normalizeSymbol(symbol)
    );
    if (comparisonPeer) {
      base.push(
        `How does ${symbol} compare with recently viewed ${comparisonPeer} on deployability and proof quality?`
      );
    }
  }
  if ((continuity.recent_prompts || []).length) {
    base.push(`What changed since the last workspace question on this setup?`);
  }
  if ((continuity.recent_export_pack_types || []).length) {
    base.push(
      `Which recent export pack is most presentation-ready and what is still weak in it?`
    );
  }
  if (state.researchTab === "platform") {
    base.push(
      `Which workflow or committee item needs review first?`,
      `Which export pack is most presentation-ready for ${symbol}?`
    );
  } else if (state.researchTab === "portfolio") {
    base.push(
      `How should ${symbol} be sized versus the rest of the book?`,
      `Where is overlap or hidden redundancy highest right now?`
    );
  } else if (state.researchTab === "evaluation") {
    base.push(
      `What is calibration saying about setups like ${symbol}?`,
      `Is the current proof cycle supportive, mixed, or weak?`
    );
  } else if (state.researchTab === "health") {
    base.push(
      `What is the biggest system-health blocker right now?`,
      `Which provider or pipeline path looks weakest today?`
    );
  }
  return base;
};

const buildCopilotDockSummary = (pageContext) => {
  const symbol = pageContext.symbol || "No symbol";
  const page = pageContext.page_label || "dashboard";
  const workflow = formatTier(pageContext.workflow_stage || "no_workflow");
  const recommendation = formatTier(pageContext.recommendation_state || "no_recommendation");
  const unresolved = Number(pageContext.unresolved_concern_count || 0);
  const continuity = pageContext.workspace_continuity || {};
  const memoryState =
    (continuity.recent_symbols || []).length || (continuity.recent_prompts || []).length
      ? "memory live"
      : "memory light";
  return {
    title: pageContext.symbol
      ? `${symbol} · ${formatTier(pageContext.deployability_tier || "watch_only")}`
      : "FTIP Platform Copilot",
    subtitle: `${page} · ${workflow} · ${recommendation}${unresolved ? ` · ${unresolved} concern${unresolved === 1 ? "" : "s"}` : ""} · ${memoryState}`,
    status: pageContext.context_warning
      ? "Warning visible"
      : pageContext.workspace_name || pageContext.workspace_id
      ? "Context live"
      : "Awaiting context",
  };
};

const renderCopilotShell = () => {
  const shell = qs("#assistant-copilot-shell");
  const contextEl = qs("#assistant-copilot-context");
  const promptsEl = qs("#assistant-copilot-prompts");
  const toggle = qs("#assistant-copilot-toggle");
  const titleEl = qs("#assistant-copilot-title");
  const inlineEl = qs("#assistant-copilot-inline-context");
  const stateEl = qs("#assistant-copilot-state");
  if (!shell || !contextEl || !promptsEl || !toggle || !titleEl || !inlineEl || !stateEl) {
    return;
  }

  shell.classList.toggle("collapsed", !!state.copilotCollapsed);
  toggle.textContent = state.copilotCollapsed ? "Open" : "Minimize";

  const pageContext = buildCopilotPageContext();
  updateWorkspaceContinuity({}, pageContext);
  const dockSummary = buildCopilotDockSummary(pageContext);
  titleEl.textContent = dockSummary.title;
  inlineEl.textContent = dockSummary.subtitle;
  stateEl.textContent = dockSummary.status;
  const continuity = pageContext.workspace_continuity || getWorkspaceContinuityRecord(pageContext);

  const contextChips = [
    `Page ${pageContext.page_label}`,
    `Workspace ${pageContext.workspace_name || pageContext.workspace_id || "n/a"}`,
    `Dossier ${pageContext.dossier_id || "n/a"}`,
    `Workflow ${formatTier(pageContext.workflow_stage || "n/a")}`,
    `Symbol ${pageContext.symbol || "n/a"}`,
    `Recommendation ${formatTier(pageContext.recommendation_state || "n/a")}`,
    `Committee ${formatTier(pageContext.committee_state || "n/a")}`,
    `Review ${formatTier(pageContext.review_state || "n/a")}`,
    `Approvals ${pageContext.pending_approval_count || 0}`,
    `Exports ${pageContext.export_count || 0}`,
    `Recent symbols ${(continuity.recent_symbols || []).slice(0, 2).join(" / ") || "n/a"}`,
    `Proof ${pageContext.proof_maturity_level || continuity.proof_maturity_level || "n/a"}`,
  ];
  const visibleChips = state.copilotCollapsed ? contextChips.slice(0, 3) : contextChips;
  contextEl.innerHTML = visibleChips
    .map((item) => `<div class="active-chip">${escapeHtml(item)}</div>`)
    .join("");
  promptsEl.innerHTML = buildCopilotPromptSet()
    .map(
      (prompt) =>
        `<button type="button" class="prompt-chip" data-copilot-prompt="${escapeHtml(prompt)}">${escapeHtml(prompt)}</button>`
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
    `AXIOM: ${analysis?.axiom_evidence_backed_deployability_tier || analysis?.axiom_deployability_tier || "unknown"}`,
    `Size: ${analysis?.size_band || "watchlist only"}`,
    `Workflow: ${analysis?.platform_workflow_stage || "n/a"}`,
    `Dossier: ${analysis?.platform_dossier_status || "n/a"}`,
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
    `Workflow: ${analysis?.platform_workflow_stage || "n/a"}`,
    `Dossier: ${analysis?.platform_dossier_status || "n/a"}`,
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
  renderCopilotShell();
};

const persistActiveAnalysis = (analysis) => {
  state.assistantActiveAnalysis = analysis || null;
  if (analysis) {
    writeStorageJson(
      ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY,
      analysis
    );
    updateWorkspaceContinuity(
      {
        symbol: analysis.symbol || null,
        workflow_id: analysis.platform_workflow_id || null,
        dossier_id: analysis.platform_dossier_id || null,
        recommendation_state:
          analysis.platform_recommendation_state ||
          analysis.axiom_evidence_backed_deployability_tier ||
          null,
        page_label: researchTabLabel(),
      },
      {
        symbol: analysis.symbol || null,
        workspace_id: analysis.platform_workspace_id || null,
        workflow_id: analysis.platform_workflow_id || null,
        workflow_stage: analysis.platform_workflow_stage || null,
        dossier_id: analysis.platform_dossier_id || null,
        recommendation_state:
          analysis.platform_recommendation_state ||
          analysis.axiom_evidence_backed_deployability_tier ||
          null,
        page_label: researchTabLabel(),
      }
    );
  } else {
    removeStorageValue(ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY);
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
      <p>${escapeHtml(
        report.axiom_summary_card_text ||
          report.axiom_summary ||
          report.overall_analysis ||
          report.signal_summary ||
          ""
      )}</p>
    </div>
  `;
};

const renderAxiomOverview = (report) => {
  const container = qs("#assistant-axiom-overview");
  if (!container) {
    return;
  }
  if (!report) {
    destroyChart("axiom-engine-chart");
    container.innerHTML = emptyStateCard(
      "AXIOM regime, deployability, evidence status, and engine scorecard will render here."
    );
    return;
  }
  const explanation = report.axiom_explanation || {};
  const engineScores = report.axiom_engine_scores || {};
  const metrics = [
    {
      label: "DAU",
      value: formatScore(report.axiom_deployable_alpha_utility),
      note: `validated edge ${formatScore(report.axiom_validated_edge)}`,
    },
    {
      label: "Regime",
      value: report.axiom_regime_label || "indeterminate",
      note: `family ${report.axiom_trade_family || "none"}`,
    },
    {
      label: "Deployability",
      value:
        report.axiom_evidence_backed_deployability_tier ||
        report.axiom_deployability_tier ||
        "unknown",
      note: `size ${report.axiom_final_size_band || report.axiom_size_band_recommendation || "none"}`,
    },
    {
      label: "Calibration",
      value: report.axiom_calibration_status || "partial",
      note: `${(report.axiom_historical_evidence_report || {}).matured_count || 0} matured`,
    },
    {
      label: "Strongest Engine",
      value: explanation.strongest_engine?.engine || "n/a",
      note: formatScore(explanation.strongest_engine?.score),
    },
    {
      label: "Weakest Engine",
      value: explanation.weakest_engine?.engine || "n/a",
      note: formatScore(explanation.weakest_engine?.score),
    },
    {
      label: "Audience",
      value: report.axiom_audience_type || "general",
      note: report.axiom_report_profile || "trading_focused",
    },
    {
      label: "Evidence",
      value: (report.axiom_summary_card || {}).evidence_status || "partial",
      note: report.axiom_lineage_summary || "lineage pending",
    },
  ];
  const engineEntries = Object.entries(engineScores).slice(0, 8);
  const hasEngineScores = engineEntries.some(([, p]) => p?.score != null);
  const engineChartId = "axiom-engine-chart";
  container.innerHTML = `
    <section class="drilldown-card">
      <h5>AXIOM Overview</h5>
      <div class="summary-grid">
        ${metrics
          .map(
            (metric) => `
              <div class="summary-card">
                <div class="summary-card-label">${escapeHtml(metric.label)}</div>
                <div class="summary-card-value">${escapeHtml(metric.value || "n/a")}</div>
                <div class="summary-card-note">${escapeHtml(metric.note || "n/a")}</div>
              </div>
            `
          )
          .join("")}
      </div>
    </section>
    <section class="drilldown-card">
      <h5>Engine Scorecard</h5>
      ${hasEngineScores
        ? `<div class="chart-wrap" style="height:${Math.max(80, engineEntries.length * 26)}px"><canvas id="${engineChartId}"></canvas></div>`
        : renderBullets([], "No AXIOM engine scores available.")}
    </section>
    <section class="drilldown-card">
      <h5>Proprietary Readout</h5>
      ${renderBullets(
        [
          report.axiom_proprietary_synthesis,
          report.axiom_support_vs_drag_summary,
          report.axiom_why_now_summary,
          report.axiom_unique_mispricing_summary,
          report.axiom_setup_character_summary,
          report.axiom_false_positive_risk_summary,
          report.axiom_decision_hierarchy_summary,
        ].filter(Boolean),
        "No proprietary AXIOM synthesis is available."
      )}
    </section>
  `;
  if (hasEngineScores) {
    const labels = engineEntries.map(([n]) => n.replaceAll("_", " "));
    const values = engineEntries.map(([, p]) => p?.score != null ? Number(p.score) : null);
    const colors = values.map((v) => v == null ? "#26364b" : v >= 0.6 ? "#93e6b0" : v >= 0.4 ? "#f5d17c" : "#ff9d9d");
    makeChart(engineChartId, labels, values.map((v) => v ?? 0), colors, 0, 1);
  }
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

const renderAxiomReportPack = (report) => {
  const container = qs("#assistant-axiom-report-pack");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "AXIOM summary card, IC memo, and risk/deployability memo will render here."
    );
    return;
  }
  const summaryCard = report.axiom_summary_card || {};
  const icMemo = report.axiom_ic_memo || {};
  const riskMemo = report.axiom_risk_deployability_memo || {};
  const historical = report.axiom_historical_evidence_report || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>AXIOM Summary Card</h5>
      ${renderBullets(
        [
          summaryCard.summary,
          `Regime ${summaryCard.regime_label || "indeterminate"} / family ${summaryCard.trade_family || "none"}`,
          `Deployability ${summaryCard.deployability_tier || "unknown"} / size ${summaryCard.size_band || "none"}`,
          `Strongest ${summaryCard.strongest_engine?.engine || "n/a"} / weakest ${summaryCard.weakest_engine?.engine || "n/a"}`,
          report.axiom_exceptionality_summary,
          report.axiom_decision_hierarchy_summary,
        ],
        "No AXIOM summary card is available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>IC Memo</h5>
      ${renderBullets(
        [
          report.axiom_ic_memo_summary,
          icMemo.thesis,
          icMemo.market_pricing_view,
          report.axiom_why_now_summary,
          report.axiom_unique_mispricing_summary,
          (icMemo.recommended_action || {}).rationale,
        ].filter(Boolean),
        "No IC memo is available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Risk & Deployability Memo</h5>
      ${renderBullets(
        [
          report.axiom_risk_deployability_memo_summary,
          riskMemo.memo_summary,
          report.axiom_false_positive_risk_summary,
          ...(riskMemo.downgrade_triggers || []).slice(0, 3),
        ].filter(Boolean),
        "No risk and deployability memo is available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Historical Evidence Summary</h5>
      ${renderBullets(
        [
          report.axiom_historical_evidence_summary_text,
          ...(historical.evidence_notes || []),
          ...(historical.recent_symbol_evidence || []).slice(0, 2),
        ].filter(Boolean),
        "No historical evidence summary is available."
      )}
    </section>`,
  ].join("");
};

const renderAxiomLineage = (report) => {
  const container = qs("#assistant-axiom-lineage");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "AXIOM evidence lineage, weakest coverage areas, and source-confidence notes will render here."
    );
    return;
  }
  const lineage = report.axiom_lineage || {};
  const weakest = (lineage.weakest_evidence_areas || []).slice(0, 5).map(
    (item) =>
      `${(item.engine || "unknown").replaceAll("_", " ")} / ${(item.component || "unknown").replaceAll(
        "_",
        " "
      )}: ${item.coverage_status || "partial"} via ${item.evidence_type || "unknown"}`
  );
  const engineBlocks = Object.values(lineage.engine_lineage || {})
    .slice(0, 4)
    .map((item) => {
      const block = (item.blocks || [])[0] || {};
      return `${(item.engine || "unknown").replaceAll("_", " ")}: ${item.engine_status || "unknown"} / ${
        block.evidence_type || "unknown"
      } / conf ${formatScore(item.confidence)}`;
    });
  const profile = report.axiom_workspace_profile || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Lineage Summary</h5>
      ${renderBullets(
        [
          report.axiom_lineage_summary,
          `Audience ${profile.audience_type || "general"} / report profile ${profile.report_profile || "trading_focused"}`,
          `Framework ${report.axiom_framework_version || "n/a"} / report ${report.report_version || "n/a"}`,
        ].filter(Boolean),
        "No AXIOM lineage summary is available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Weakest Evidence Areas</h5>
      ${renderBullets(weakest, "No weak evidence area is currently surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Engine Provenance Mix</h5>
      ${renderBullets(engineBlocks, "No engine provenance snapshot is available.")}
    </section>`,
  ].join("");
};

const renderDashboardWorkflow = (report) => {
  const container = qs("#assistant-dashboard-workflow");
  if (!report) {
    const placeholders = [
      ["Daily Triage", "Start with the dashboard for changed signals, active trust state, and priority candidates.", "dashboard"],
      ["Decision Layer", "Open strategy to inspect actionability, scenarios, invalidators, and execution posture.", "strategy"],
      ["Portfolio Fit", "Use the portfolio workspace to inspect ranking, overlap, size-band logic, and rotation pressure.", "portfolio"],
      ["Weekly Review", "Use evaluation and trust surfaces to inspect scorecards, calibration, provenance, and operational warnings.", "evaluation"],
      ["Monthly Refinement", "Open the learning lab for drift alerts, governed improvement candidates, and research priorities.", "research"],
      ["Runbook", "Use the health workspace for operator workflow, post-mortems, trust maintenance, and runbook guidance.", "health"],
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
    ["Daily Triage", report.daily_operating_summary || report.what_changed_panel, "dashboard"],
    ["Strategy Console", report.strategy_view, "strategy"],
    ["Portfolio Construction", report.portfolio_context_summary || report.portfolio_fit_analysis, "portfolio"],
    [
      "Deployment Readiness",
      report.deployment_readiness_summary || report.deployment_permission_analysis,
      "strategy",
    ],
    ["Weekly Review", report.weekly_operating_summary || report.evaluation_research_analysis, "evaluation"],
    ["Monthly Refinement", report.monthly_operating_summary || report.learning_summary, "research"],
    ["Trust Maintenance", report.trust_maintenance_summary || report.deployment_readiness_summary, "health"],
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
    {
      label: "Workflow",
      value: report.current_operating_mode || "normal",
      note:
        (report.operator_attention_items || [])[0] ||
        report.daily_operating_summary ||
        "operator loop",
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
                <span class="micro-chip">axiom ${escapeHtml(
                  snapshot.axiom_deployability_tier || "unknown"
                )}</span>
                <span class="micro-chip">regime ${escapeHtml(
                  snapshot.axiom_regime_label || "indeterminate"
                )}</span>
                <span class="micro-chip">workflow ${escapeHtml(
                  snapshot.platform_workflow_stage || "n/a"
                )}</span>
                <span class="micro-chip">dossier ${escapeHtml(
                  snapshot.platform_dossier_status || "n/a"
                )}</span>
                <span class="micro-chip">size ${escapeHtml(
                  snapshot.size_band || "watchlist only"
                )}</span>
                <span class="micro-chip">freshness ${escapeHtml(
                  snapshot.freshness_status || "unknown"
                )}</span>
              </div>
              <div class="table-note">${escapeHtml(
                snapshot.workflow_summary ||
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
      note: `diversification ${formatScore(report.diversification_contribution_score)} · fit rank ${report.portfolio_fit_rank || "n/a"}`,
    },
    {
      label: "Marginal Utility",
      value:
        report.marginal_portfolio_utility == null
          ? "n/a"
          : Number(report.marginal_portfolio_utility).toFixed(1),
      note: report.replacement_candidate
        ? `replacement ${report.replacement_candidate}`
        : `contribution ${formatScore(report.portfolio_contribution_score)}`,
    },
    {
      label: "Hidden Overlap",
      value: `${formatScore(report.overlap_score)} / ${formatScore(report.hidden_overlap_score)}`,
      note: report.most_redundant_symbol
        ? `redundancy ${formatScore(report.redundancy_score)} · peer ${report.most_redundant_symbol}`
        : `redundancy ${formatScore(report.redundancy_score)}`,
    },
    {
      label: "Size Band",
      value: report.size_band || "watchlist only",
      note: `${report.weight_band || report.risk_budget_band || "band pending"} · stress ${formatScore(
        report.portfolio_stress_score
      )}`,
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
      <h5>Risk Model</h5>
      ${renderBullets(
        [
          `Portfolio risk model: ${report.portfolio_risk_model_version || "n/a"}`,
          `Hidden overlap score: ${formatScore(report.hidden_overlap_score)}`,
          `Complementarity score: ${formatScore(report.complementarity_score)}`,
          `Portfolio stress: ${formatScore(report.portfolio_stress_score)}`,
          `Portfolio fragility: ${formatScore(report.portfolio_fragility_score)}`,
          `Correlation breakdown risk: ${formatScore(report.correlation_breakdown_risk)}`,
          `Exposure cluster: ${report.exposure_cluster || "n/a"}`,
          `Style affinity: ${report.style_affinity || "n/a"}`,
          `Top loadings: ${(report.factor_loading_summary || []).join(", ") || "none"}`,
        ],
        "No portfolio risk-model detail surfaced."
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
          report.portfolio_risk_model_summary,
          report.hidden_overlap_redundancy_analysis,
          report.replacement_diversification_analysis,
          report.candidate_upgrade_reason,
          report.candidate_downgrade_reason,
          report.replacement_candidate_notes,
          report.portfolio_quality_upgrade_reason,
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
          report.axiom_lineage_summary,
          report.data_provider_quality_summary,
          report.axiom_ic_memo_summary,
          report.axiom_risk_deployability_memo_summary,
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
    `AXIOM artifact: ${report.axiom_artifact_id || "n/a"}`,
    `AXIOM history artifact: ${report.axiom_history_artifact_id || "n/a"}`,
    `AXIOM calibration artifact: ${report.axiom_calibration_artifact_id || "n/a"}`,
    `AXIOM portfolio governance artifact: ${report.axiom_portfolio_governance_artifact_id || "n/a"}`,
    `AXIOM lineage artifact: ${report.axiom_lineage_artifact_id || "n/a"}`,
    `AXIOM report pack artifact: ${report.axiom_report_pack_artifact_id || "n/a"}`,
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
    `AXIOM audience/profile: ${report.axiom_audience_type || "general"} / ${
      report.axiom_report_profile || "trading_focused"
    }`,
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
          report.axiom_summary_card_text,
          report.axiom_lineage_summary,
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
    destroyChart("research-composite-chart");
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
  const compositeLabels = [
    "Opportunity Quality Score",
    "Cross-Domain Conviction Score",
    "Signal Fragility Index",
    "Narrative Crowding Index",
    "Macro Alignment Score",
    "Regime Stability Score",
  ];
  const compositeValues = compositeLabels.map((lbl) => compositeScore(report, lbl) ?? null);
  const hasComposites = compositeValues.some((v) => v != null);
  const compositeChartId = "research-composite-chart";
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Strategy Components</h5>
      ${renderBullets(components, "No strategy component stack available.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Composite Scores</h5>
      ${hasComposites
        ? `<div class="chart-wrap" style="height:${Math.max(80, compositeLabels.length * 26)}px"><canvas id="${compositeChartId}"></canvas></div>`
        : renderBullets([], "No composite score detail available.")}
    </section>`,
  ].join("");
  if (hasComposites) {
    const colors = compositeValues.map((v) =>
      v == null ? "#26364b" : v >= 0.6 ? "#93e6b0" : v >= 0.35 ? "#4ea1ff" : "#ff9d9d"
    );
    makeChart(
      compositeChartId,
      compositeLabels.map((l) => l.replace(" Score", "").replace(" Index", "")),
      compositeValues.map((v) => v ?? 0),
      colors,
      0, 1
    );
  }
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
  const operationalLines = [
    `System health ${report.system_health_status || "unknown"} / provider ${
      report.provider_health_status || "unknown"
    }`,
    `Operating mode ${report.current_operating_mode || "normal"} / shadow ${
      report.shadow_mode_status || "unknown"
    }`,
    `Data reliability ${formatScore(report.data_reliability_score, 1)} / 100 / drift ${formatScore(
      report.model_drift_score,
      1
    )} / 100`,
    ...(report.degraded_domain_list || []).slice(0, 4).map((item) => `Degraded domain: ${item}`),
  ];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Versioning</h5>
      ${renderBullets(
        [
          `Report ${report.report_version || "n/a"}`,
          `Strategy ${report.strategy?.strategy_version || "n/a"}`,
          `AXIOM ${report.axiom_framework_version || "n/a"} / ${report.axiom_report_profile || "trading_focused"}`,
          `Evaluation ${report.evaluation?.evaluation_version || "phase6"}`,
          `Research ${report.research_version || "phase10"}`,
          `Operational ${report.operational_guardrails_version || "phase12"}`,
          `Source governance ${report.source_governance_version || "phase13"}`,
          `Operator workflow ${report.operating_workflow_version || "phase14"}`,
        ],
        "No version data available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Freshness by Domain</h5>
      ${renderBullets(domainNotes, "No domain freshness breakdown available.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Operational Guardrails</h5>
      ${renderBullets(
        [
          report.system_health_summary,
          report.shadow_mode_summary,
          report.drift_control_summary,
          report.incident_history_summary,
          ...operationalLines,
        ].filter(Boolean),
        "No operational guardrail data available."
      )}
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

const renderCommercialReadiness = (report) => {
  const container = qs("#assistant-commercial-readiness");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Source profile, commercialization blockers, and clean-stack guidance render here."
    );
    return;
  }
  const cleanupQueue = (report.commercial_cleanup_queue || [])
    .slice(0, 5)
    .map(
      (item) =>
        `${item.source_name || "source"} (${item.priority || "review"}): ${
          item.cleanup_reason || "cleanup required"
        }`
    );
  const gatedDomains = (report.degraded_due_to_profile || []).map(
    (item) => `Profile-gated domain: ${item}`
  );
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Profile</h5>
      ${renderBullets(
        [
          `Source profile ${report.source_profile || "internal_research"}`,
          `Buyer-demo suitability ${report.buyer_demo_suitability || "unknown"}`,
          `Buyer-safe status ${report.buyer_safe_profile_status || "unknown"}`,
          `Commercialization risk ${formatScore(report.commercialization_risk_score, 1)} / 100`,
          `Licensing risk tier ${report.licensing_risk_tier || "unknown"}`,
        ],
        "No source-profile data available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Governance</h5>
      ${renderBullets(
        [
          report.commercialization_readiness_summary,
          report.source_governance_summary,
          ...(report.commercial_blockers || []),
          ...(report.disallowed_sources || []).map((item) => `Disallowed source: ${item}`),
          ...gatedDomains,
        ].filter(Boolean),
        "No commercialization blocker surfaced."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Clean Stack Path</h5>
      ${renderBullets(
        [
          report.buyer_diligence_summary,
          ...cleanupQueue,
        ].filter(Boolean),
        "No cleanup queue surfaced."
      )}
    </section>`,
  ].join("");
};

const renderPlatformOverview = (report) => {
  const container = qs("#assistant-platform-overview");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Workspace, workflow, dossier counts, and institutional profile context render here."
    );
    return;
  }
  const summary = report.platform_summary_view || {};
  const workspace = report.platform_workspace || {};
  const workflow = report.platform_workflow || {};
  const dossier = report.platform_dossier || {};
  const metrics = [
    {
      label: "Platform Profile",
      value: report.platform_profile || "none",
      note: `${workspace.audience_type || report.axiom_audience_type || "general"} / ${
        workspace.report_profile || report.axiom_report_profile || "trading_focused"
      }`,
    },
    {
      label: "Workspace",
      value: workspace.name || "none",
      note: workspace.workspace_id || "workspace not attached",
    },
    {
      label: "Workflow",
      value: workflow.title || "none",
      note: `${workflow.stage || "n/a"} / ${workflow.status || "n/a"}`,
    },
    {
      label: "Dossier",
      value: dossier.title || "none",
      note: `${dossier.dossier_type || "n/a"} / ${dossier.evidence_status || "n/a"}`,
    },
    {
      label: "Dossier Count",
      value: String(summary.dossier_count ?? 0),
      note: `${summary.workflow_count ?? 0} workflow(s) / ${summary.workspace_count ?? 0} workspace(s)`,
    },
    {
      label: "Latest Tier",
      value: dossier.latest_deployability_tier || "n/a",
      note: `${dossier.latest_regime_label || "n/a"} / ${dossier.latest_size_band || "n/a"}`,
    },
  ];
  container.innerHTML = `
    <div class="summary-grid">
      ${metrics
        .map(
          (metric) => `
            <div class="summary-card">
              <div class="summary-card-label">${escapeHtml(metric.label)}</div>
              <div class="summary-card-value">${escapeHtml(metric.value || "n/a")}</div>
              <div class="summary-card-note">${escapeHtml(metric.note || "n/a")}</div>
            </div>
          `
        )
        .join("")}
    </div>
  `;
};

const formatDistributionSummary = (distribution) => {
  const entries = Object.entries(distribution || {}).filter(([, value]) => value != null);
  if (!entries.length) {
    return "none";
  }
  return entries
    .map(([key, value]) => `${String(key).replaceAll("_", " ")} ${value}`)
    .join(", ");
};

const platformFilterValue = (selector) =>
  String(qs(selector)?.value || "")
    .trim()
    .toLowerCase();

const dossierMatchesPlatformFilters = (dossier) => {
  const filters = {
    workspace: platformFilterValue("#assistant-platform-filter-workspace"),
    template: platformFilterValue("#assistant-platform-filter-workflow-template"),
    tier: platformFilterValue("#assistant-platform-filter-tier"),
    regime: platformFilterValue("#assistant-platform-filter-regime"),
    stage: platformFilterValue("#assistant-platform-filter-stage"),
    evidence: platformFilterValue("#assistant-platform-filter-evidence"),
  };
  const fields = {
    workspace: String(dossier.workspace_name || dossier.workspace || "").toLowerCase(),
    template: String(dossier.workflow_template_id || dossier.workflow_template || "").toLowerCase(),
    tier: String(dossier.latest_deployability_tier || "").toLowerCase(),
    regime: String(dossier.latest_regime_label || "").toLowerCase(),
    stage: String(dossier.stage || "").toLowerCase(),
    evidence: String(dossier.evidence_status || "").toLowerCase(),
  };
  return Object.entries(filters).every(([key, value]) => !value || fields[key].includes(value));
};

const renderPlatformDashboard = (report) => {
  const container = qs("#assistant-platform-dashboard");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Executive dashboard metrics, high-DAU dossiers, export activity, and integration health render here."
    );
    return;
  }
  const dashboard = report.platform_dashboard || {};
  const metrics = dashboard.executive_metrics || {};
  const pending = dashboard.pending_approvals || [];
  const exports = dashboard.recent_exports || [];
  const highDau = dashboard.high_dau_dossiers || [];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Executive Metrics</h5>
      ${renderBullets(
        [
          `Workspace ${metrics.workspace_name || "n/a"}`,
          `Dossiers ${metrics.dossier_count ?? 0} / workflows ${metrics.workflow_count ?? 0}`,
          `Pending approvals ${metrics.pending_approval_count ?? 0} / exports ${metrics.export_count ?? 0}`,
          `Integration bindings ${metrics.integration_binding_count ?? 0}`,
          report.platform_dashboard_summary,
        ].filter(Boolean),
        "No executive metrics are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Pending Approvals</h5>
      ${renderBullets(
        pending.slice(0, 5).map(
          (item) =>
            `${item.requested_role || "review"} / ${item.status || "pending"} / stage ${item.stage || "n/a"}`
        ),
        "No pending approvals are open."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Recent High-DAU Dossiers</h5>
      ${renderBullets(
        highDau.slice(0, 5).map(
          (item) =>
            `${item.title || item.symbol || "Dossier"} / DAU ${formatScore(
              item.deployable_alpha_utility,
              1
            )} / ${item.latest_deployability_tier || "n/a"}`
        ),
        "No high-DAU dossier list is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Export / Integration Snapshot</h5>
      ${renderBullets(
        [
          `Recent exports ${exports.length}`,
          report.platform_export_rendering_summary,
          report.platform_integration_health_summary,
          report.platform_demo_readiness_summary,
        ].filter(Boolean),
        "No export or integration snapshot is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformTemplateProfile = (report) => {
  const container = qs("#assistant-platform-template-profile");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Workflow template, platform profile, and preferred dossier/report emphasis render here."
    );
    return;
  }
  const profile = report.platform_profile_details || {};
  const template = report.platform_workflow_template || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Workflow Template</h5>
      ${renderBullets(
        [
          template.title,
          template.description,
          `Orientation: ${template.orientation || "research"}`,
          `Stages: ${(template.stage_sequence || []).join(" -> ") || "n/a"}`,
          `Report emphasis: ${(template.preferred_report_pack_emphasis || []).join(", ") || "n/a"}`,
          `AXIOM emphasis: ${(template.expected_axiom_emphasis || []).join(", ") || "n/a"}`,
        ].filter(Boolean),
        "No workflow template is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Platform Profile</h5>
      ${renderBullets(
        [
          `Profile ${profile.profile_id || report.platform_profile || "n/a"}`,
          `Audience ${profile.audience_type || report.axiom_audience_type || "general"}`,
          `Default workflow ${(profile.default_workflow_template || "n/a").replaceAll("_", " ")}`,
          `Report profile ${profile.default_report_profile || report.axiom_report_profile || "trading_focused"}`,
          `Preferred AXIOM sections: ${(profile.preferred_axiom_sections || []).join(", ") || "n/a"}`,
          `Preferred dossier sections: ${(profile.preferred_dossier_sections || []).join(", ") || "n/a"}`,
        ],
        "No platform profile is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformBootstrap = (report) => {
  const container = qs("#assistant-platform-bootstrap");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Workspace bootstrap defaults, seeded demo summary, and profile-driven provisioning context render here."
    );
    return;
  }
  const summary = report.platform_bootstrap_summary || {};
  const defaults = report.platform_bootstrap_profile_defaults || {};
  const bundles = report.platform_demo_bundles || [];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Bootstrap Summary</h5>
      ${renderBullets(
        [
          report.platform_bootstrap_summary_text,
          `Profile ${summary.platform_profile || report.platform_profile || "n/a"}`,
          `Workflow template ${summary.workflow_template_id || report.platform_workflow?.workflow_template_id || "n/a"}`,
          `Seeded workflows ${summary.seeded_workflow_count ?? 0} / dossiers ${summary.seeded_dossier_count ?? 0}`,
          `Seeded exports ${summary.seeded_export_count ?? 0} / stored exports ${
            summary.seeded_stored_export_count ?? 0
          } / integrations ${summary.seeded_integration_count ?? 0}`,
        ].filter(Boolean),
        "No bootstrap summary is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Profile Defaults</h5>
      ${renderBullets(
        [
          `Workspace pattern ${defaults.default_workspace_name_pattern || "n/a"}`,
          `Dashboard emphasis ${(defaults.default_dashboard_emphasis || []).join(", ") || "n/a"}`,
          `Export emphasis ${(defaults.default_export_pack_emphasis || []).join(", ") || "n/a"}`,
          `Preferred AXIOM sections ${(defaults.preferred_axiom_sections || []).join(", ") || "n/a"}`,
          `Preferred dossier sections ${(defaults.preferred_dossier_sections || []).join(", ") || "n/a"}`,
        ],
        "No bootstrap profile defaults are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Walkthrough Hints</h5>
      ${renderBullets(
        summary.walkthrough_hints || [],
        "No walkthrough hints are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Available Seed Bundles</h5>
      ${renderBullets(
        bundles.slice(0, 6).map(
          (item) =>
            `${item.bundle_id || "bundle"} / ${(item.workflow_template_id || "template").replaceAll("_", " ")} / ${
              item.seeded_entities?.length || 0
            } seeded entity(s)`
        ),
        "No demo bundle catalog is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformReadiness = (report) => {
  const container = qs("#assistant-platform-readiness");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Deployment readiness categories, warnings, and remediation notes render here."
    );
    return;
  }
  const readiness = report.platform_readiness_report || {};
  const snapshot = report.platform_readiness_snapshot || {};
  const categories = readiness.categories || [];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Overall Readiness</h5>
      ${renderBullets(
        [
          report.platform_readiness_report_summary,
          `Pilot ready ${String(readiness.pilot_ready ?? snapshot.pilot_ready ?? false)}`,
          `Status ${readiness.overall_status || "partial"} / score ${formatScore(readiness.overall_score, 1)}`,
          `Legacy readiness ${snapshot.analysis_readiness || "partial"} / ${
            snapshot.workflow_readiness || "partial"
          } / ${snapshot.export_readiness || "partial"} / ${
            snapshot.integration_readiness || "partial"
          }`,
        ].filter(Boolean),
        "No readiness report is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Category Status</h5>
      ${renderBullets(
        categories.slice(0, 8).map(
          (item) =>
            `${item.category || "category"} / ${item.status || "partial"} / score ${formatScore(
              item.score,
              1
            )} / ${item.summary || "no summary"}`
        ),
        "No readiness categories are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Warnings / Remediation</h5>
      ${renderBullets(
        [
          ...((readiness.warnings || []).slice(0, 5).map(
            (item) => `${item.category || "warning"} / ${item.severity || "warning"} / ${item.message || "n/a"}`
          )),
          ...((readiness.remediation_notes || []).slice(0, 4)),
        ],
        "No readiness warnings are attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformPilotPackage = (report) => {
  const container = qs("#assistant-platform-pilot-package");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Pilot package profile, dossier counts, export counts, and walkthrough hints render here."
    );
    return;
  }
  const pilotPackage = report.platform_pilot_package || {};
  const workspaceSummary = pilotPackage.workspace_summary || {};
  const collaborationSummary = pilotPackage.collaboration_summary || {};
  const exportSummary = pilotPackage.export_summary || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Pilot Package Summary</h5>
      ${renderBullets(
        [
          report.platform_pilot_package_summary,
          `Workspace ${(pilotPackage.workspace || {}).name || report.platform_workspace?.name || "n/a"}`,
          `Profile ${(pilotPackage.platform_profile || {}).profile_id || report.platform_profile || "n/a"}`,
          `Workflow template ${(pilotPackage.workflow_template || {}).template_id || report.platform_workflow?.workflow_template_id || "n/a"}`,
        ].filter(Boolean),
        "No pilot package is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Operational Counts</h5>
      ${renderBullets(
        [
          `Dossiers ${((workspaceSummary.summary_view || {}).dossier_count) ?? 0}`,
          `Workflows ${((workspaceSummary.summary_view || {}).workflow_count) ?? 0}`,
          `Stored exports ${exportSummary.stored_export_count ?? 0}`,
          `Unresolved concerns ${collaborationSummary.unresolved_concern_count ?? 0}`,
          `Locked recommendations ${collaborationSummary.locked_recommendation_count ?? 0}`,
          `Committee snapshots ${collaborationSummary.committee_snapshot_count ?? 0}`,
        ],
        "No pilot-package operational counts are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Top Warnings / Walkthrough</h5>
      ${renderBullets(
        [
          ...((pilotPackage.top_warnings || []).slice(0, 4).map(
            (item) => `${item.category || "warning"} / ${item.severity || "warning"} / ${item.message || "n/a"}`
          )),
          ...((pilotPackage.walkthrough_hints || []).slice(0, 4)),
        ],
        "No pilot-package warnings or walkthrough hints are attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformDemoBundles = (report) => {
  const container = qs("#assistant-platform-demo-bundles");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Available demo bundles, seeded objects, and apply status render here."
    );
    return;
  }
  const bundles = report.platform_demo_bundles || [];
  const summary = report.platform_bootstrap_summary || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Bundle Catalog</h5>
      ${renderBullets(
        bundles.slice(0, 8).map(
          (item) =>
            `${item.bundle_id || "bundle"} / ${item.audience_type || "general"} / ${
              item.title || "Untitled Bundle"
            }`
        ),
        "No demo bundle catalog is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Seeded Scope</h5>
      ${renderBullets(
        bundles.slice(0, 6).map(
          (item) =>
            `${item.bundle_id || "bundle"} seeds ${(item.seeded_entities || []).length} dossier blueprint(s) and ${
              (item.default_export_packs || []).length
            } default pack(s)`
        ),
        "No seeded-scope summary is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Apply State</h5>
      ${renderBullets(
        [
          report.platform_demo_bundle_summary,
          `Applied bundle ${summary.demo_bundle_id || "n/a"}`,
          `Demo seeded ${String(summary.demo_seeded || false)}`,
        ].filter(Boolean),
        "No demo bundle apply state is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformDossiers = (report) => {
  const container = qs("#assistant-platform-dossiers");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Dossier cards with stage, deployability, regime, and evidence state render here."
    );
    return;
  }
  const workspaceAnalytics = report.platform_workspace_analytics || {};
  const dossiers = (
    workspaceAnalytics.dossier_records ||
    report.platform_recent_dossiers ||
    []
  ).filter(dossierMatchesPlatformFilters);
  if (!dossiers.length) {
    container.innerHTML = emptyStateCard(
      "No dossier matches the current filters. Run Analyze with dossier creation enabled or attach to an existing dossier."
    );
    return;
  }
  container.innerHTML = dossiers
    .slice(0, 12)
    .map(
      (item) => `
        <section class="recent-analysis-card">
          <div class="recent-analysis-head">
            <div>
              <div class="recent-analysis-title">${escapeHtml(item.title || item.symbol || "Dossier")}</div>
              <div class="table-subtitle">${escapeHtml(
                `${item.symbol || "n/a"} · ${item.dossier_type || "coverage"} · ${
                  item.as_of_date || "n/a"
                }`
              )}</div>
            </div>
            <span class="legacy-badge">${escapeHtml(item.latest_deployability_tier || "n/a")}</span>
          </div>
          <div class="recent-analysis-meta">
            <span class="micro-chip">workspace ${escapeHtml(item.workspace_name || report.platform_workspace?.name || "n/a")}</span>
            <span class="micro-chip">template ${escapeHtml(item.workflow_template_id || report.platform_workflow?.workflow_template_id || "n/a")}</span>
            <span class="micro-chip">stage ${escapeHtml(item.stage || "n/a")}</span>
            <span class="micro-chip">status ${escapeHtml(item.status || "n/a")}</span>
            <span class="micro-chip">regime ${escapeHtml(item.latest_regime_label || "n/a")}</span>
            <span class="micro-chip">family ${escapeHtml(item.latest_trade_family || "n/a")}</span>
            <span class="micro-chip">size ${escapeHtml(item.latest_size_band || "n/a")}</span>
            <span class="micro-chip">evidence ${escapeHtml(item.evidence_status || "partial")}</span>
            <span class="micro-chip">DAU ${escapeHtml(formatScore(item.deployable_alpha_utility, 1))}</span>
          </div>
          <div class="table-note">${escapeHtml(
            item.dossier_id || "No dossier identifier available."
          )}</div>
        </section>
      `
    )
    .join("");
};

const renderPlatformDossierDetail = (report) => {
  const container = qs("#assistant-platform-dossier-detail");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "A dossier-linked AXIOM summary, section families, and analysis linkage render here."
    );
    return;
  }
  const dossier = report.platform_dossier || {};
  const sections = dossier.sections || [];
  const analysisLink = report.platform_analysis_link || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Dossier Summary</h5>
      ${renderBullets(
        [
          report.platform_dossier_summary,
          `Latest deployability ${dossier.latest_deployability_tier || "n/a"} / regime ${
            dossier.latest_regime_label || "n/a"
          } / family ${dossier.latest_trade_family || "n/a"}`,
          `Latest size ${dossier.latest_size_band || "n/a"} / evidence ${
            dossier.evidence_status || "partial"
          }`,
          `Linked AXIOM artifact ${
            dossier.latest_axiom_analysis_id || analysisLink.axiom_artifact_id || "n/a"
          }`,
        ].filter(Boolean),
        "No dossier summary is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Dossier Sections</h5>
      ${renderBullets(
        sections.map(
          (item) => `${item.title || item.section_key || "section"}: ${item.status || "available"}`
        ),
        "No dossier sections are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Analysis Link</h5>
      ${renderBullets(
        [
          `Report ${analysisLink.report_id || "n/a"} / session ${analysisLink.session_id || "n/a"}`,
          `History artifact ${analysisLink.axiom_history_artifact_id || "n/a"}`,
          `Calibration artifact ${analysisLink.axiom_calibration_artifact_id || "n/a"}`,
          report.axiom_summary_card_text,
        ].filter(Boolean),
        "No dossier-linked analysis reference is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformMonitoring = (report) => {
  const container = qs("#assistant-platform-monitoring");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Monitoring triggers, invalidation flags, and workflow-stage watch items render here."
    );
    return;
  }
  const dossier = report.platform_dossier || {};
  const monitoring = dossier.monitoring_state || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Monitoring State</h5>
      ${renderBullets(
        [
          report.platform_monitoring_summary,
          `Operating mode ${monitoring.current_operating_mode || report.current_operating_mode || "normal"}`,
          ...((monitoring.monitoring_triggers || []).slice(0, 5)),
        ].filter(Boolean),
        "No dossier monitoring state is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Invalidation / Downgrade</h5>
      ${renderBullets(
        [
          ...((monitoring.invalidation_flags || []).slice(0, 5)),
          ...((monitoring.downgrade_triggers || []).slice(0, 5)),
        ],
        "No invalidation or downgrade trigger is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Platform Overview</h5>
      ${renderBullets(
        [
          report.platform_overview_summary,
          `Dossiers by tier: ${formatJson((report.platform_summary_view || {}).dossiers_by_deployability_tier || {})}`,
          `Dossiers by regime: ${formatJson((report.platform_summary_view || {}).dossiers_by_regime || {})}`,
        ],
        "No platform summary aggregation is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformControls = (report) => {
  const container = qs("#assistant-platform-controls");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Workflow actions, approvals, audit timeline, export packs, access context, and integrations render here."
    );
    return;
  }
  const access = report.platform_access_summary || {};
  const actions = report.platform_allowed_actions || [];
  const approvals = report.platform_approvals || [];
  const timeline = report.platform_timeline || [];
  const exports = report.platform_exports || [];
  const integrationSummary = report.platform_integration_summary || {};
  const health = report.platform_health_summary || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Workflow Actions</h5>
      ${renderBullets(
        [
          report.platform_workflow_actions_summary,
          ...actions.slice(0, 6).map(
            (item) =>
              `${item.action_type || "action"}: ${item.allowed ? "allowed" : "blocked"}${
                item.next_stage ? ` / next ${item.next_stage}` : ""
              }${item.note ? ` / ${item.note}` : ""}`
          ),
        ].filter(Boolean),
        "No workflow action state is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Audit &amp; Approvals</h5>
      ${renderBullets(
        [
          report.platform_audit_timeline_summary,
          ...approvals.slice(0, 4).map(
            (item) =>
              `${item.requested_role || "review"} approval: ${item.status || "pending"} / stage ${
                item.stage || "n/a"
              }`
          ),
          ...timeline.slice(0, 4).map(
            (item) =>
              `${item.title || item.event_type || "event"}: ${item.summary || "no summary"}`
          ),
        ].filter(Boolean),
        "No audit or approval activity is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Export Packs</h5>
      ${renderBullets(
        [
          report.platform_export_summary,
          ...exports.slice(0, 4).map(
            (item) =>
              `${item.pack_type || "pack"}: ${item.status || "generated"} / ${
                item.approval_status || "no approval gate"
              } / ${item.content_hash || "no hash"}`
          ),
        ].filter(Boolean),
        "No export pack metadata is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Access / Integration / Health</h5>
      ${renderBullets(
        [
          report.platform_access_control_summary,
          `Role ${access.effective_role || "service_account"} / permissions ${
            (access.effective_permissions || []).length
          }`,
          report.platform_integration_health_summary,
          `Integrations ${(integrationSummary.binding_count ?? 0)} / warnings ${
            (integrationSummary.warnings || []).length
          }`,
          `Platform warnings ${(health.warnings || []).length} / pending approvals ${
            health.pending_approval_count ?? 0
          }`,
        ].filter(Boolean),
        "No access or health context is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformApprovals = (report) => {
  const container = qs("#assistant-platform-approvals");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Pending approvals, recent decisions, and stage progression visibility render here."
    );
    return;
  }
  const approvals = report.platform_approvals || [];
  const timeline = report.platform_timeline || [];
  const dossier = report.platform_dossier || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Pending / Recent Decisions</h5>
      ${renderBullets(
        approvals.slice(0, 6).map(
          (item) =>
            `${item.requested_role || "review"} / ${item.status || "pending"} / stage ${item.stage || "n/a"}`
        ),
        "No approvals are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Recommendation Lock</h5>
      ${renderBullets(
        [
          `Locked ${String((dossier.metadata || {}).recommendation_locked || false)}`,
          report.platform_workflow_actions_summary,
        ].filter(Boolean),
        "No lock state is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Stage Progression</h5>
      ${renderBullets(
        timeline.slice(0, 6).map(
          (item) =>
            `${item.stage || "n/a"} / ${item.title || item.event_type || "event"} / ${
              item.status || "n/a"
            }`
        ),
        "No workflow timeline is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformProof = (report) => {
  const container = qs("#assistant-platform-proof");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Tracked recommendation counts, supportive vs mixed evidence, and buyer-grade proof summaries render here."
    );
    return;
  }
  const proof = report.platform_proof_summary || {};
  const credibility = report.platform_model_credibility_snapshot || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Proof Summary</h5>
      ${renderBullets(
        [
          report.platform_proof_cycle_summary,
          `Tracked ${proof.tracked_recommendation_count ?? 0}`,
          `Matured ${proof.matured_tracking_count ?? 0}`,
          `Supportive ${proof.supportive_count ?? 0} / mixed ${proof.mixed_count ?? 0} / weak ${
            proof.weak_count ?? 0
          }`,
          `Evidence maturity ${proof.evidence_maturity_level || "limited"}`,
        ].filter(Boolean),
        "No proof summary is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Replay Consistency / Credibility</h5>
      ${renderBullets(
        [
          `Replay consistency ${proof.replay_consistency_label || "partial"}`,
          `Credibility ${credibility.status || "partial"}`,
          credibility.buyer_summary,
          report.platform_model_credibility_summary,
        ].filter(Boolean),
        "No model credibility snapshot is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Top Evidence Rows</h5>
      ${renderBullets(
        [
          ...((proof.top_regime_rows || []).slice(0, 3).map(
            (item) => `Regime ${item.label || "unknown"} / avg edge ${formatPct(item.average_net_edge_return, 2)}`
          )),
          ...((proof.top_tier_rows || []).slice(0, 3).map(
            (item) => `Tier ${item.label || "unknown"} / avg edge ${formatPct(item.average_net_edge_return, 2)}`
          )),
        ],
        "No matured proof rows are attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformOutcomes = (report) => {
  const container = qs("#assistant-platform-outcomes");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Horizon outcomes, favorable/adverse excursion, and recommendation assessments render here."
    );
    return;
  }
  const snapshot = report.platform_outcome_snapshot || {};
  const attribution = report.platform_outcome_attribution || {};
  const evidence = report.platform_recommendation_evidence_summary || {};
  const windows = snapshot.windows || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Assessment</h5>
      ${renderBullets(
        [
          report.platform_tracking_summary,
          report.platform_outcome_summary,
          `Evidence ${evidence.evidence_status || "pending"}`,
          `Assessment ${((snapshot.assessment || {}).assessment_status) || "pending"}`,
          attribution.summary,
        ].filter(Boolean),
        "No outcome attribution is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Horizon Windows</h5>
      ${renderBullets(
        Object.entries(windows)
          .slice(0, 6)
          .map(
            ([label, item]) =>
              `${label} / ${item.status || "pending"} / net edge ${formatPct(
                item.net_edge_return,
                2
              )} / MAE ${formatPct(item.mae, 2)} / MFE ${formatPct(item.mfe, 2)}`
          ),
        "No tracked horizon windows are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Supportive vs Contradictory</h5>
      ${renderBullets(
        [
          `Supportive ${(snapshot.assessment || {}).supportive_window_count ?? 0}`,
          `Contradictory ${(snapshot.assessment || {}).contradictory_window_count ?? 0}`,
          ...((evidence.supportive_horizons || []).slice(0, 4).map((item) => `Supportive: ${item}`)),
          ...((evidence.contradicted_horizons || []).slice(0, 4).map((item) => `Contradicted: ${item}`)),
        ],
        "No supportive/contradictory horizon breakdown is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformCalibration = (report) => {
  const container = qs("#assistant-platform-calibration");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Calibration hardening, drift warnings, and sample-quality notes render here."
    );
    return;
  }
  const calibration = report.platform_calibration_hardening || {};
  const drift = report.platform_drift_summary || {};
  const dauBuckets = (calibration.dau_bucket_realized_outcomes || {}).buckets || [];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Calibration Hardening</h5>
      ${renderBullets(
        [
          report.platform_calibration_hardening_summary,
          `Tracked ${calibration.tracked_recommendation_count ?? 0}`,
          `Matured ${calibration.matured_count ?? 0}`,
          `Paper hit rate ${formatPct(calibration.paper_trade_hit_rate, 1)}`,
          `Downside rate ${formatPct(calibration.downside_rate, 1)}`,
        ].filter(Boolean),
        "No calibration hardening summary is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>DAU Buckets</h5>
      ${renderBullets(
        dauBuckets.slice(0, 5).map(
          (item) =>
            `${item.label || "bucket"} / avg edge ${formatPct(
              item.average_net_edge_return,
              2
            )} / hit ${formatPct(item.hit_rate, 1)}`
        ),
        "No DAU bucket behavior is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Drift / Warnings</h5>
      ${renderBullets(
        [
          report.platform_drift_summary_text,
          `Drift ${drift.status || "partial"}`,
          ...(drift.warnings || []).slice(0, 4),
          ...(calibration.warnings || []).slice(0, 4),
        ].filter(Boolean),
        "No drift or calibration warning state is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformBenchmarks = (report) => {
  const container = qs("#assistant-platform-benchmarks");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Regime, trade-family, tier, and recommendation-state cohort comparisons render here."
    );
    return;
  }
  const benchmarks = report.platform_benchmarks || {};
  const strongest = benchmarks.strongest_cohorts || [];
  const weakest = benchmarks.weakest_cohorts || [];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Benchmark Summary</h5>
      ${renderBullets(
        [
          report.platform_benchmark_summary,
          `Status ${benchmarks.status || "partial"}`,
          `Horizon ${benchmarks.horizon_label || "21d"}`,
        ].filter(Boolean),
        "No benchmark summary is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Strongest Cohorts</h5>
      ${renderBullets(
        strongest.slice(0, 6).map(
          (item) =>
            `${item.dimension || "dimension"} / ${item.label || "label"} / avg edge ${formatPct(
              item.average_net_edge_return,
              2
            )}`
        ),
        "No strongest cohort rows are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Weakest Cohorts</h5>
      ${renderBullets(
        weakest.slice(0, 6).map(
          (item) =>
            `${item.dimension || "dimension"} / ${item.label || "label"} / avg edge ${formatPct(
              item.average_net_edge_return,
              2
            )}`
        ),
        "No weakest cohort rows are attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformReviews = (report) => {
  const container = qs("#assistant-platform-reviews");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Structured review comments, concern severity, and resolution state render here."
    );
    return;
  }
  const reviewSummary = report.platform_review_summary || {};
  const comments = report.platform_review_comments || [];
  const flags = reviewSummary.concern_flags || [];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Review Summary</h5>
      ${renderBullets(
        [
          report.platform_collaboration_summary,
          `Comments ${((reviewSummary.thread_summary || {}).total_comments) ?? 0}`,
          `Unresolved concerns ${reviewSummary.unresolved_concern_count ?? 0}`,
        ].filter(Boolean),
        "No structured review summary is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Open Concerns</h5>
      ${renderBullets(
        flags.slice(0, 6).map(
          (item) =>
            `${item.concern_type || "concern"} / ${item.severity || "watch"} / ${
              item.summary || "no summary"
            }`
        ),
        "No unresolved concern is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Recent Comments</h5>
      ${renderBullets(
        comments.slice(0, 6).map(
          (item) =>
            `${item.comment_type || "general"} / ${item.severity || "info"} / ${
              item.status || "open"
            } / ${item.body || "no comment"}`
        ),
        "No review comment is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformAssignments = (report) => {
  const container = qs("#assistant-platform-assignments");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Owner, reviewer slots, and assignment coverage render here."
    );
    return;
  }
  const summary = report.platform_assignment_summary || {};
  const assignments = report.platform_assignments || [];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Assignment Summary</h5>
      ${renderBullets(
        [
          report.platform_assignment_summary_text,
          `Owner ${summary?.owner?.assignee_placeholder || "unassigned"}`,
          `Primary reviewer ${summary?.primary_reviewer?.assignee_placeholder || "unassigned"}`,
          `Risk reviewer ${summary?.risk_reviewer?.assignee_placeholder || "unassigned"}`,
          `Committee reviewer ${summary?.committee_reviewer?.assignee_placeholder || "unassigned"}`,
        ].filter(Boolean),
        "No assignment summary is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Assignment Records</h5>
      ${renderBullets(
        assignments.slice(0, 6).map(
          (item) =>
            `${item.slot_type || "slot"} / ${item.assignee_placeholder || "unassigned"} / ${
              item.status || "assigned"
            }`
        ),
        "No assignment record is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformCommitteeDecision = (report) => {
  const container = qs("#assistant-platform-committee");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Committee outcomes, conditions, evidence gaps, and risk notes render here."
    );
    return;
  }
  const decision = report.platform_committee_decision || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Latest Decision</h5>
      ${renderBullets(
        [
          report.platform_committee_summary,
          `Decision ${decision.decision_status || "not recorded"}`,
          `Recommendation ${decision.recommendation_state || "draft"}`,
          decision.summary,
        ].filter(Boolean),
        "No committee decision snapshot is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Conditions / Risks / Gaps</h5>
      ${renderBullets(
        [
          ...((decision.conditions || []).slice(0, 3).map(
            (item) => `Condition: ${item.label || item}`
          )),
          ...((decision.key_risks || []).slice(0, 3).map((item) => `Risk: ${item}`)),
          ...((decision.key_evidence_gaps || []).slice(0, 3).map((item) => `Gap: ${item}`)),
        ],
        "No committee conditions, risks, or evidence gaps are attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformRecommendationState = (report) => {
  const container = qs("#assistant-platform-recommendation-state");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Recommendation state, lock status, and revision history render here."
    );
    return;
  }
  const state = report.platform_recommendation_state || {};
  const history = report.platform_recommendation_history || [];
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Current State</h5>
      ${renderBullets(
        [
          report.platform_collaboration_summary,
          `State ${state.state || "draft"}`,
          `Locked ${String(state.locked || false)}`,
          state.summary,
          state.rationale,
        ].filter(Boolean),
        "No recommendation state is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>State History</h5>
      ${renderBullets(
        history.slice(0, 6).map(
          (item) =>
            `${item.action_type || "change"} / ${item.previous_state || "n/a"} -> ${
              item.new_state || "n/a"
            } / locked ${String(item.locked || false)}`
        ),
        "No recommendation history is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformExports = (report) => {
  const container = qs("#assistant-platform-exports");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Available packs, rendered export formats, and preview snippets render here."
    );
    return;
  }
  const exports = report.platform_exports || [];
  const rendered = report.platform_rendered_exports || [];
  const stored = report.platform_stored_exports || [];
  const capabilities = report.platform_export_capabilities || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Available Packs</h5>
      ${renderBullets(
        [
          ...((report.platform_supported_export_packs || []).slice(0, 8).map(
            (item) => `${item.replaceAll("_", " ")}`
          )),
          report.platform_export_summary,
        ].filter(Boolean),
        "No export pack types are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Generated Exports</h5>
      ${renderBullets(
        exports.slice(0, 5).map(
          (item) =>
            `${item.pack_type || "pack"} / ${item.approval_status || "no gate"} / ${
              item.content_hash || "no hash"
            }`
        ),
        "No generated export manifests are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Rendered Output</h5>
      ${renderBullets(
        rendered.slice(0, 5).map(
          (item) =>
            `${item.export_format || "format"} / ${item.file_name_hint || "file"} / checksum ${
              item.checksum || "n/a"
            }`
        ),
        "No rendered export output is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Stored Export History</h5>
      ${renderBullets(
        stored.slice(0, 6).map(
          (item) =>
            `${item.pack_type || "pack"} / ${item.export_format || "format"} / ${
              item.version_label || "v?"
            } / ${item.approval_status || "no approval"} / ${item.checksum || "no checksum"}`
        ),
        "No durable stored export history is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Export Detail</h5>
      ${renderBullets(
        stored.slice(0, 3).map(
          (item) =>
            `${item.file_name_hint || "file"} / ${item.storage_backend || "backend"} / ${
              item.storage_key || "no storage key"
            } / integrity ${item.status || "unknown"}`
        ),
        "No stored export detail is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Export Capabilities</h5>
      ${renderBullets(
        [
          `html ${String(capabilities.html_supported ?? true)}`,
          `markdown ${String(capabilities.markdown_supported ?? true)}`,
          `json ${String(capabilities.json_supported ?? true)}`,
          `pdf ready ${String(capabilities.pdf_ready ?? false)}`,
          `docx ready ${String(capabilities.docx_ready ?? false)}`,
          report.platform_export_capability_summary,
          report.platform_export_storage_summary,
        ].filter(Boolean),
        "No export capability summary is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Preview Snippets</h5>
      ${renderBullets(
        (stored.length ? stored : rendered).slice(0, 3).map((item) =>
          String(
            item.rendered_content || item?.metadata?.content_preview || item?.metadata?.render_metadata?.content_preview || ""
          )
            .replace(/\s+/g, " ")
            .trim()
            .slice(0, 180)
        ),
        "No rendered preview snippet is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformIntegrations = (report) => {
  const container = qs("#assistant-platform-integrations");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Integration bindings, health states, last execution, and capabilities render here."
    );
    return;
  }
  const bindings = report.platform_integration_bindings || [];
  const health = report.platform_health_summary || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Bindings</h5>
      ${renderBullets(
        bindings.slice(0, 6).map(
          (item) =>
            `${item.integration_type || "custom"} / ${item.status || "configured"} / ${
              ((item.health || {}).status) || "unknown"
            }`
        ),
        "No integration binding is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Capabilities / Last Execution</h5>
      ${renderBullets(
        bindings.slice(0, 5).flatMap((item) => {
          const definition = item.definition || {};
          const capabilityList = (definition.capabilities || [])
            .slice(0, 2)
            .map((cap) => `${item.integration_type}: ${cap.title || cap.capability_id}`);
          const lastExecution = item.last_execution
            ? `${item.integration_type} last execution ${item.last_execution.status || "n/a"} / ${
                item.last_execution.completed_at || "n/a"
              }`
            : `${item.integration_type} has no execution history`;
          return [lastExecution, ...capabilityList];
        }),
        "No connector capability or execution history is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Health / Warnings</h5>
      ${renderBullets(
        [
          report.platform_integration_health_summary,
          `Platform warnings ${(health.warnings || []).length}`,
          ...((health.warnings || []).slice(0, 4)),
        ].filter(Boolean),
        "No integration warning state is attached."
      )}
    </section>`,
  ].join("");
};

const renderPlatformAnalytics = (report) => {
  const container = qs("#assistant-platform-analytics");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Cross-workspace analytics, tier distribution, regime distribution, and evidence support render here."
    );
    return;
  }
  const analytics = report.platform_cross_workspace_analytics || {};
  const workspaceAnalytics = report.platform_workspace_analytics || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Workspace Analytics</h5>
      ${renderBullets(
        [
          `Average DAU ${formatScore(workspaceAnalytics.average_dau, 1)}`,
          `Live-candidate ratio ${formatPct(workspaceAnalytics.live_candidate_ratio, 1)}`,
          `Evidence distribution ${formatDistributionSummary(
            workspaceAnalytics.evidence_status_distribution
          )}`,
          report.platform_dashboard_summary,
        ].filter(Boolean),
        "No workspace analytics are attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Cross-Workspace Distribution</h5>
      ${renderBullets(
        [
          `Deployability ${formatDistributionSummary(analytics.deployability_distribution)}`,
          `Regime ${formatDistributionSummary(analytics.regime_distribution)}`,
          `Trade family ${formatDistributionSummary(analytics.trade_family_distribution)}`,
          `Audience ${formatDistributionSummary(analytics.counts_by_audience_type)}`,
          report.platform_analytics_summary,
        ],
        "No cross-workspace distribution is attached."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Throughput / Evidence Support</h5>
      ${renderBullets(
        [
          `Approval throughput ${formatDistributionSummary(analytics.approval_throughput)}`,
          `Export throughput ${formatDistributionSummary(analytics.export_throughput)}`,
          `Supportive evidence ratio ${formatPct(analytics.supportive_evidence_ratio, 1)}`,
          `Average DAU across workspaces ${formatScore(analytics.average_dau_across_workspaces, 1)}`,
          report.platform_demo_readiness_summary,
        ].filter(Boolean),
        "No throughput or evidence support analytics are attached."
      )}
    </section>`,
  ].join("");
};

const renderOperatingWorkflow = (report) => {
  const container = qs("#assistant-operating-workflow");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Daily triage, weekly review, monthly refinement, post-mortem focus, and trust maintenance render here."
    );
    return;
  }
  const triage = (report.todays_candidate_triage || [])
    .slice(0, 4)
    .map(
      (item) =>
        `${item.symbol || "symbol"}: ${item.signal || "n/a"} / ${
          item.deployment_permission || "analysis_only"
        } / ${item.candidate_classification || "watchlist_candidate"}`
    );
  const weeklyItems = (report.weekly_operator_attention_items || []).slice(0, 4);
  const monthlyItems = (report.research_priority_queue || [])
    .slice(0, 4)
    .map(
      (item) =>
        `${item.title || "priority"}: ${item.priority || "review"} / ${
          item.reason || "follow-up review required"
        }`
    );
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Daily Workflow</h5>
      <p>${escapeHtml(report.daily_operating_summary || "No daily workflow summary available.")}</p>
      ${renderBullets(
        [
          report.what_changed_panel,
          ...(report.new_warnings_downgrades || []),
          ...triage,
        ].filter(Boolean),
        "No daily triage items surfaced."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Weekly Review</h5>
      <p>${escapeHtml(report.weekly_operating_summary || "No weekly review summary available.")}</p>
      ${renderBullets(weeklyItems, "No weekly operator attention item surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Monthly Refinement</h5>
      <p>${escapeHtml(report.monthly_operating_summary || "No monthly refinement summary available.")}</p>
      ${renderBullets(monthlyItems, "No monthly research priority surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Post-Mortem / Trust</h5>
      ${renderBullets(
        [
          report.postmortem_summary,
          report.trust_maintenance_summary,
          ...(report.postmortem_queue || []).slice(0, 3),
          ...(report.required_evidence_for_promotion || []).slice(0, 3),
        ].filter(Boolean),
        "No post-mortem or trust item surfaced."
      )}
    </section>`,
  ].join("");
};

const renderOperatorRunbook = (report) => {
  const container = qs("#assistant-operator-runbook");
  if (!container) {
    return;
  }
  if (!report) {
    container.innerHTML = emptyStateCard(
      "Daily, weekly, monthly, and incident-response runbook guidance renders here."
    );
    return;
  }
  const runbook = report.operator_runbook || {};
  container.innerHTML = [
    `<section class="drilldown-card">
      <h5>Runbook Summary</h5>
      <p>${escapeHtml(report.operator_runbook_summary || "No operator runbook summary available.")}</p>
      ${renderBullets(report.runbook_attention_notes || [], "No runbook attention note surfaced.")}
    </section>`,
    `<section class="drilldown-card">
      <h5>Daily / Weekly</h5>
      ${renderBullets(
        [...(runbook.daily_workflow || []), ...(runbook.weekly_workflow || [])],
        "No daily or weekly workflow guidance available."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Monthly / Incident Response</h5>
      ${renderBullets(
        [...(runbook.monthly_workflow || []), ...(runbook.incident_response || [])],
        "No monthly or incident-response guidance available."
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
  const operationalAlerts = (report.operational_alerts || [])
    .slice(0, 5)
    .map((item) => `${item.alert_domain || "alert"}: ${item.alert_summary || "No summary."}`);
  const incidentNotes = (report.incident_history || [])
    .slice(0, 4)
    .map((item) => `${item.alert_severity || "info"} / ${item.alert_domain || "incident"}: ${item.summary || "No summary."}`);
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
          `Operating mode ${report.current_operating_mode || "normal"} / shadow ${
            report.shadow_mode_status || "unknown"
          }`,
          `Data reliability ${formatScore(report.data_reliability_score, 1)} / 100 / model drift ${formatScore(
            report.model_drift_score,
            1
          )} / 100`,
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
          ...(report.provider_degradation_notes || []),
          ...(report.degraded_domain_list || []),
          ...operationalAlerts,
          report.rollback_reason,
          report.downgrade_reason,
          report.pause_recommended ? "Pause recommended." : null,
          report.pause_required ? "Pause required." : null,
          report.degrade_to_paper_recommended ? "Degrade to paper/shadow recommended." : null,
          report.downgrade_to_shadow_recommended
            ? "Operational downgrade to shadow recommended."
            : null,
          audit.rationale_summary,
        ].filter(Boolean),
        "No deployment alert or rollback note surfaced."
      )}
    </section>`,
    `<section class="drilldown-card">
      <h5>Incident History</h5>
      ${renderBullets(
        [report.incident_history_summary, ...incidentNotes].filter(Boolean),
        "No operational incident history surfaced."
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
  if (report) {
    updateWorkspaceContinuity({
      symbol: report.symbol || null,
      workflow_id: report.platform_workflow?.workflow_id || null,
      dossier_id: report.platform_dossier?.dossier_id || null,
      recent_export_pack_types: (report.platform_stored_exports || [])
        .map((item) => item.pack_type)
        .filter(Boolean),
      recommendation_state:
        report.platform_recommendation_state?.state ||
        report.platform_dossier?.current_recommendation_state ||
        null,
      committee_state: report.platform_committee_decision?.decision_status || null,
      provider_health_status:
        report.provider_health_status ||
        report.platform_health_summary?.integration_health_summary?.overall_status ||
        null,
      proof_maturity_level:
        report.platform_proof_summary?.evidence_maturity_level || null,
      calibration_status:
        report.platform_calibration_hardening?.status ||
        report.platform_model_credibility_snapshot?.status ||
        null,
      current_view_summary:
        report.platform_dashboard_summary ||
        report.platform_dossier_summary ||
        report.axiom_proprietary_synthesis ||
        report.signal_summary ||
        null,
      page_label: researchTabLabel(),
    });
  }

  if (!report) {
    container.classList.add("empty");
    container.innerHTML = `
      <div class="analysis-report-empty">
        Run Assistant Analyze to generate a grounded system report.
      </div>
    `;
    qs("#assistant-analyze-response").textContent = formatJson({});
    renderDashboardSummary(null);
    renderAxiomOverview(null);
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
    renderAxiomReportPack(null);
    renderEvidenceSupport(null);
    renderAxiomLineage(null);
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
    renderCommercialReadiness(null);
    renderPlatformDashboard(null);
    renderPlatformOverview(null);
    renderPlatformTemplateProfile(null);
    renderPlatformBootstrap(null);
    renderPlatformReadiness(null);
    renderPlatformPilotPackage(null);
    renderPlatformDemoBundles(null);
    renderPlatformDossiers(null);
    renderPlatformDossierDetail(null);
    renderPlatformMonitoring(null);
    renderPlatformProof(null);
    renderPlatformOutcomes(null);
    renderPlatformCalibration(null);
    renderPlatformBenchmarks(null);
    renderPlatformControls(null);
    renderPlatformApprovals(null);
    renderPlatformReviews(null);
    renderPlatformAssignments(null);
    renderPlatformCommitteeDecision(null);
    renderPlatformRecommendationState(null);
    renderPlatformExports(null);
    renderPlatformIntegrations(null);
    renderPlatformAnalytics(null);
    renderOperatingWorkflow(null);
    renderOperatorRunbook(null);
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
    {
      label: "AXIOM DAU",
      value:
        report.axiom_deployable_alpha_utility == null
          ? "N/A"
          : Number(report.axiom_deployable_alpha_utility).toFixed(1),
    },
    { label: "AXIOM Regime", value: report.axiom_regime_label || "N/A" },
    {
      label: "AXIOM Tier",
      value:
        report.axiom_evidence_backed_deployability_tier ||
        report.axiom_deployability_tier ||
        "N/A",
    },
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
    ["AXIOM Summary", report.axiom_summary],
    ["AXIOM Institutional Memo", report.axiom_ic_memo_summary],
    ["AXIOM Evidence Lineage", report.axiom_lineage_summary],
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
  renderAxiomOverview(report);
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
  renderAxiomReportPack(report);
  renderEvidenceSupport(report);
  renderAxiomLineage(report);
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
  renderCommercialReadiness(report);
  renderPlatformDashboard(report);
  renderPlatformOverview(report);
  renderPlatformTemplateProfile(report);
  renderPlatformBootstrap(report);
  renderPlatformReadiness(report);
  renderPlatformPilotPackage(report);
  renderPlatformDemoBundles(report);
  renderPlatformDossiers(report);
  renderPlatformDossierDetail(report);
  renderPlatformMonitoring(report);
  renderPlatformProof(report);
  renderPlatformOutcomes(report);
  renderPlatformCalibration(report);
  renderPlatformBenchmarks(report);
  renderPlatformControls(report);
  renderPlatformApprovals(report);
  renderPlatformReviews(report);
  renderPlatformAssignments(report);
  renderPlatformCommitteeDecision(report);
  renderPlatformRecommendationState(report);
  renderPlatformExports(report);
  renderPlatformIntegrations(report);
  renderPlatformAnalytics(report);
  renderOperatingWorkflow(report);
  renderOperatorRunbook(report);
  renderSystemHealth(state.assistantHealth, report);
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
    if (qs("#assistant-copilot-response")) {
      qs("#assistant-copilot-response").textContent =
        "Persistent copilot is ready. Run Assistant Analyze or open a dossier-rich platform view for deeper grounded answers.";
    }
    return;
  }

  const blocks = state.assistantChatTranscript.map((entry) => {
    const lines = [`${entry.role}:`, entry.content];
    if (entry.citations?.length) {
      lines.push(`Citations: ${entry.citations.join(", ")}`);
    }
    return lines.join("\n");
  });
  const transcript = blocks.join("\n\n");
  qs("#assistant-chat-response").textContent = transcript;
  if (qs("#assistant-copilot-response")) {
    qs("#assistant-copilot-response").textContent = transcript;
  }
};

const renderSystemHealth = (health, report = state.assistantReport) => {
  const container = qs("#assistant-system-health");
  const operationalCards = report
    ? [
        {
          label: "system",
          value: report.system_health_status || "unknown",
          note: `mode ${report.current_operating_mode || "normal"} / shadow ${
            report.shadow_mode_status || "unknown"
          }`,
        },
        {
          label: "operational",
          value:
            report.data_reliability_score == null
              ? "n/a"
              : `${formatScore(report.data_reliability_score, 1)} / 100`,
          note: `drift ${formatScore(report.model_drift_score, 1)} / 100 · ${
            report.pause_required ? "pause required" : "no forced pause"
          }`,
        },
      ]
    : [];
  if (!health && !report) {
    container.innerHTML = emptyStateCard("Assistant and provider health will render here.");
    return;
  }
  const providerCards = Object.entries((health || {}).providers || {}).map(
    ([name, provider]) => `
      <div class="summary-card">
        <div class="summary-card-label">${escapeHtml(name)}</div>
        <div class="summary-card-value">${escapeHtml(provider.status || "unknown")}</div>
        <div class="summary-card-note">${escapeHtml(provider.message || "")}</div>
      </div>
    `
  );
  const assistantCards = health
    ? [
        {
          label: "assistant",
          value: health.assistant?.status || "unknown",
          note: `llm_enabled=${health.assistant?.llm_enabled} db_enabled=${health.assistant?.db_enabled}`,
        },
      ]
    : [];
  container.innerHTML = [
    ...[...assistantCards, ...operationalCards].map(
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
    renderSystemHealth(state.assistantHealth, state.assistantReport);
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
  audience_type: qs("#assistant-analyze-audience").value.trim() || "general",
  report_profile:
    qs("#assistant-analyze-report-profile").value.trim() || "trading_focused",
  platform_profile: qs("#assistant-analyze-platform-profile").value.trim() || null,
  workspace_id: qs("#assistant-analyze-workspace-id").value.trim() || null,
  workflow_template_id: qs("#assistant-analyze-workflow-template").value.trim() || null,
  dossier_id: qs("#assistant-analyze-dossier-id").value.trim() || null,
  create_dossier: Boolean(qs("#assistant-analyze-create-dossier")?.checked),
});

const getAssistantChatMessage = () => qs("#assistant-chat-message").value.trim();

const runAssistantAnalyze = async () => {
  const {
    symbol,
    horizon,
    risk_mode,
    market_regime,
    analysis_depth,
    refresh_mode,
    scenario_mode,
    audience_type,
    report_profile,
    platform_profile,
    workspace_id,
    workflow_template_id,
    dossier_id,
    create_dossier,
  } =
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
        audience_type,
        report_profile,
        platform_profile,
        workspace_id,
        workflow_template_id,
        dossier_id,
        create_dossier,
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
      audience_type,
      report_profile,
      platform_profile,
      workspace_id,
      workflow_template_id,
      dossier_id,
      create_dossier,
    });
    setLegacyStatus("#assistant-analyze-status", `Analyze failed: ${err.message}`, "error");
  } finally {
    setButtonLoading("#assistant-analyze-btn", false, "Running...");
    renderActiveAnalysisLabels();
  }
};

const submitAssistantChatMessage = async ({
  message,
  inputSelector,
  buttonSelector,
  statusSelector,
}) => {
  if (!message) {
    setLegacyStatus(statusSelector, "Message is required.", "error");
    return;
  }

  const pendingSessionId = state.assistantChatSessionId || generateUuid();
  persistAssistantSessionId(pendingSessionId);

  setButtonLoading(buttonSelector, true, "Sending...");
  setLegacyStatus(statusSelector, "Sending grounded narrator request...");
  try {
    const pageContext = buildCopilotPageContext();
    recordCopilotPrompt(message, pageContext);
    const data = await callJson("/assistant/chat", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        session_id: pendingSessionId,
        message,
        context: {
          active_analysis: state.assistantActiveAnalysis,
          page_context: pageContext,
          report_context: {
            report_id: state.assistantLatestReport?.report_id || null,
            axiom_artifact_id: state.assistantLatestReport?.axiom_artifact_id || null,
            export_count: (state.assistantLatestReport?.platform_stored_exports || []).length,
            workflow_stage: state.assistantLatestReport?.platform_workflow?.stage || null,
            dossier_id: state.assistantLatestReport?.platform_dossier?.dossier_id || null,
          },
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
    if (qs(inputSelector)) {
      qs(inputSelector).value = "";
    }
    setLegacyStatus(statusSelector, "Assistant reply received.", "success");
  } catch (err) {
    setLegacyStatus(statusSelector, `Chat failed: ${err.message}`, "error");
  } finally {
    setButtonLoading(buttonSelector, false, "Sending...");
    renderActiveAnalysisLabels();
  }
};

const sendAssistantChat = async () =>
  submitAssistantChatMessage({
    message: getAssistantChatMessage(),
    inputSelector: "#assistant-chat-message",
    buttonSelector: "#assistant-chat-btn",
    statusSelector: "#assistant-chat-status",
  });

const sendPersistentCopilotChat = async () =>
  submitAssistantChatMessage({
    message: qs("#assistant-copilot-message").value.trim(),
    inputSelector: "#assistant-copilot-message",
    buttonSelector: "#assistant-copilot-send",
    statusSelector: "#assistant-copilot-status",
  });

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

qs("#assistant-copilot-send").addEventListener("click", sendPersistentCopilotChat);
qs("#assistant-copilot-toggle").addEventListener("click", () => {
  state.copilotCollapsed = !state.copilotCollapsed;
  writeStorageValue(
    ASSISTANT_COPILOT_COLLAPSED_STORAGE_KEY,
    state.copilotCollapsed ? "1" : "0"
  );
  renderCopilotShell();
});
qs("#assistant-copilot-prompts").addEventListener("click", (event) => {
  const button = event.target.closest("[data-copilot-prompt]");
  if (!button) {
    return;
  }
  state.copilotCollapsed = false;
  qs("#assistant-copilot-message").value = button.dataset.copilotPrompt || "";
  writeStorageValue(ASSISTANT_COPILOT_COLLAPSED_STORAGE_KEY, "0");
  renderCopilotShell();
  qs("#assistant-copilot-message").focus();
});
qs("#assistant-copilot-message").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    sendPersistentCopilotChat();
  }
});

qs("#symbol-input").addEventListener("input", (event) => {
  const legacySymbol = qs("#assistant-analyze-symbol");
  if (!legacySymbol.value.trim()) {
    legacySymbol.value = event.target.value.trim().toUpperCase();
  }
});

[
  "#assistant-platform-filter-workspace",
  "#assistant-platform-filter-workflow-template",
  "#assistant-platform-filter-tier",
  "#assistant-platform-filter-regime",
  "#assistant-platform-filter-stage",
  "#assistant-platform-filter-evidence",
].forEach((selector) => {
  const element = qs(selector);
  if (!element) {
    return;
  }
  element.addEventListener("input", () => {
    if (state.assistantLatestReport) {
      renderPlatformDossiers(state.assistantLatestReport);
    }
  });
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
{
  const storedCopilotState = window.localStorage.getItem(
    ASSISTANT_COPILOT_COLLAPSED_STORAGE_KEY
  );
  state.copilotCollapsed = storedCopilotState == null ? true : storedCopilotState === "1";
}
persistAssistantSessionId(
  window.localStorage.getItem(ASSISTANT_CHAT_SESSION_STORAGE_KEY) || generateUuid()
);
persistWorkspaceContinuityStore(
  readStorageJson(ASSISTANT_WORKSPACE_CONTINUITY_STORAGE_KEY, {})
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
renderCopilotShell();
setResearchTab("dashboard");
setActiveTab(state.demoMode ? "legacy" : "signal");
refreshAssistantSystemHealth();
