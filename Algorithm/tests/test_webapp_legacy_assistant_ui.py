import subprocess
import textwrap
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


def test_legacy_assistant_ui_preserves_active_analysis_context() -> None:
    with TestClient(app) as client:
        html = client.get("/app").text
        js = client.get("/app/static/app.js").text

    assert 'id="assistant-analyze-active-label"' in html
    assert 'id="assistant-chat-active-label"' in html
    assert 'id="assistant-chat-grounding-note"' in html
    assert 'id="assistant-chat-suggested-prompts"' in html
    assert 'id="assistant-copilot-shell"' in html
    assert 'id="assistant-copilot-context"' in html
    assert 'id="assistant-copilot-prompts"' in html
    assert 'id="assistant-copilot-message"' in html
    assert 'id="assistant-copilot-send"' in html
    assert 'id="assistant-copilot-response"' in html
    assert 'id="assistant-copilot-title"' in html
    assert 'id="assistant-copilot-inline-context"' in html
    assert 'id="assistant-copilot-state"' in html
    assert 'id="assistant-analyze-report"' in html
    assert 'id="demo-mode-toggle"' in html
    assert 'id="assistant-dashboard-workflow"' in html
    assert 'id="assistant-dashboard-trust-strip"' in html
    assert 'id="assistant-dashboard-recent"' in html
    assert 'id="assistant-axiom-overview"' in html
    assert 'id="assistant-strategy-metrics"' in html
    assert 'id="assistant-strategy-scenarios"' in html
    assert 'id="assistant-strategy-playbook"' in html
    assert 'id="assistant-risk-triggers"' in html
    assert 'id="assistant-deployment-summary-section"' in html
    assert 'id="assistant-deployment-readiness"' in html
    assert 'id="assistant-deployment-risk-budget"' in html
    assert 'data-research-tab="portfolio"' in html
    assert 'id="assistant-portfolio-metrics"' in html
    assert 'id="assistant-portfolio-context-section"' in html
    assert 'id="assistant-portfolio-fit-section"' in html
    assert 'id="assistant-portfolio-execution-section"' in html
    assert 'id="assistant-portfolio-controls"' in html
    assert 'id="assistant-portfolio-ranking"' in html
    assert 'id="assistant-portfolio-workflow"' in html
    assert 'data-research-tab="evaluation"' in html
    assert 'data-research-tab="compare"' in html
    assert 'data-research-tab="watchlist"' in html
    assert 'id="assistant-evaluation-metrics"' in html
    assert 'id="assistant-evaluation-section"' in html
    assert 'id="assistant-calibration-section"' in html
    assert 'id="assistant-regime-grid"' in html
    assert 'id="assistant-failure-modes"' in html
    assert 'id="assistant-evaluation-factors"' in html
    assert 'data-research-tab="dashboard"' in html
    assert 'data-research-tab="chat"' in html
    assert 'id="assistant-signal-drilldown"' in html
    assert 'id="assistant-evidence-supporting"' in html
    assert 'id="assistant-evidence-conflicts"' in html
    assert 'id="assistant-axiom-lineage"' in html
    assert 'id="assistant-artifact-ledger"' in html
    assert 'id="assistant-axiom-report-pack"' in html
    assert 'id="assistant-analyze-audience"' in html
    assert 'id="assistant-analyze-report-profile"' in html
    assert 'id="assistant-analyze-platform-profile"' in html
    assert 'id="assistant-analyze-workspace-id"' in html
    assert 'id="assistant-analyze-workflow-template"' in html
    assert 'id="assistant-analyze-dossier-id"' in html
    assert 'id="assistant-analyze-create-dossier"' in html
    assert 'id="assistant-deployment-audit"' in html
    assert 'id="assistant-compare-summary"' in html
    assert 'id="assistant-compare-details"' in html
    assert 'id="assistant-watchlist-grid"' in html
    assert 'id="assistant-system-health"' in html
    assert 'id="assistant-system-audit"' in html
    assert 'id="assistant-commercial-readiness"' in html
    assert 'id="assistant-deployment-governance"' in html
    assert 'id="assistant-operating-workflow"' in html
    assert 'id="assistant-operator-runbook"' in html
    assert 'data-research-tab="platform"' in html
    assert 'id="assistant-platform-overview"' in html
    assert 'id="assistant-platform-dashboard"' in html
    assert 'id="assistant-platform-template-profile"' in html
    assert 'id="assistant-platform-bootstrap"' in html
    assert 'id="assistant-platform-readiness"' in html
    assert 'id="assistant-platform-pilot-package"' in html
    assert 'id="assistant-platform-demo-bundles"' in html
    assert 'id="assistant-platform-dossier-filters"' in html
    assert 'id="assistant-platform-dossiers"' in html
    assert 'id="assistant-platform-dossier-detail"' in html
    assert 'id="assistant-platform-monitoring"' in html
    assert 'id="assistant-platform-controls"' in html
    assert 'id="assistant-platform-approvals"' in html
    assert 'id="assistant-platform-reviews"' in html
    assert 'id="assistant-platform-assignments"' in html
    assert 'id="assistant-platform-committee"' in html
    assert 'id="assistant-platform-recommendation-state"' in html
    assert 'id="assistant-platform-exports"' in html
    assert 'id="assistant-platform-integrations"' in html
    assert 'id="assistant-platform-analytics"' in html
    assert 'id="assistant-platform-proof"' in html
    assert 'id="assistant-platform-outcomes"' in html
    assert 'id="assistant-platform-calibration"' in html
    assert 'id="assistant-platform-benchmarks"' in html
    assert 'id="assistant-learning-metrics"' in html
    assert 'id="assistant-learning-regimes"' in html
    assert 'id="assistant-learning-adaptations"' in html
    assert 'id="assistant-learning-experiments"' in html
    assert 'id="assistant-learning-archetypes"' in html
    assert "Learning Lab" in html
    assert "assistantActiveAnalysis" in js
    assert "ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY" in js
    assert "ASSISTANT_COPILOT_COLLAPSED_STORAGE_KEY" in js
    assert "ASSISTANT_RECENT_REPORTS_STORAGE_KEY" in js
    assert "ASSISTANT_WATCHLIST_STORAGE_KEY" in js
    assert "FTIP_DEMO_MODE_STORAGE_KEY" in js
    assert "session_id: pendingSessionId" in js
    assert "active_analysis: state.assistantActiveAnalysis" in js
    assert "page_context: buildCopilotPageContext()" in js
    assert "setResearchTab" in js
    assert "applyDemoMode" in js
    assert "buildNarratorPromptSet" in js
    assert "renderNarratorPromptChips" in js
    assert "renderCopilotShell" in js
    assert "sendPersistentCopilotChat" in js
    assert "Platform Copilot" in html
    assert "What matters most on the" in js
    assert "const formatPct" in js
    assert "copilotCollapsed: true" in js
    assert "renderDashboardWorkflow" in js
    assert "renderDashboardTrustStrip" in js
    assert "renderRecentAnalyses" in js
    assert "renderAxiomOverview" in js
    assert "renderAxiomReportPack" in js
    assert "renderAxiomLineage" in js
    assert "renderEvaluationMetrics" in js
    assert "renderEvaluationRegimeGrid" in js
    assert "renderEvaluationFailureModes" in js
    assert "renderEvaluationFactors" in js
    assert "canonical_validation" in js
    assert "Walk-Forward" in js
    assert "renderEvidenceSupport" in js
    assert "renderArtifactLedger" in js
    assert "renderCompareWorkspace" in js
    assert "renderWatchlistWorkspace" in js
    assert "renderStrategyMetrics" in js
    assert "renderStrategyScenarios" in js
    assert "renderStrategyPlaybook" in js
    assert "renderRiskTriggers" in js
    assert "renderDeploymentReadiness" in js
    assert "renderDeploymentRiskBudget" in js
    assert "renderDeploymentAudit" in js
    assert "renderPortfolioMetrics" in js
    assert "renderPortfolioControls" in js
    assert "renderPortfolioRanking" in js
    assert "renderPortfolioWorkflow" in js
    assert "portfolio_risk_model" in js
    assert "Hidden Overlap" in js
    assert "renderLearningMetrics" in js
    assert "renderLearningRegimes" in js
    assert "renderLearningAdaptations" in js
    assert "renderLearningExperiments" in js
    assert "renderLearningArchetypes" in js
    assert "portfolio_construction" in js
    assert "Which idea fits the portfolio better" in js
    assert "What is the platform learning lately" in js
    assert "operational_guardrails" in js
    assert "Operational Guardrails" in js
    assert "shadow_mode_status" in js
    assert "current_operating_mode" in js
    assert "data_reliability_score" in js
    assert "renderCommercialReadiness" in js
    assert "renderPlatformOverview" in js
    assert "renderPlatformDashboard" in js
    assert "renderPlatformTemplateProfile" in js
    assert "renderPlatformBootstrap" in js
    assert "renderPlatformReadiness" in js
    assert "renderPlatformPilotPackage" in js
    assert "renderPlatformDemoBundles" in js
    assert "renderPlatformDossiers" in js
    assert "renderPlatformDossierDetail" in js
    assert "renderPlatformMonitoring" in js
    assert "renderPlatformControls" in js
    assert "renderPlatformApprovals" in js
    assert "renderPlatformReviews" in js
    assert "renderPlatformAssignments" in js
    assert "renderPlatformCommitteeDecision" in js
    assert "renderPlatformRecommendationState" in js
    assert "renderPlatformExports" in js
    assert "renderPlatformIntegrations" in js
    assert "renderPlatformAnalytics" in js
    assert "renderPlatformProof" in js
    assert "renderPlatformOutcomes" in js
    assert "renderPlatformCalibration" in js
    assert "renderPlatformBenchmarks" in js
    assert "platform_access_control_summary" in js
    assert "platform_workflow_actions_summary" in js
    assert "platform_audit_timeline_summary" in js
    assert "platform_export_summary" in js
    assert "platform_export_rendering_summary" in js
    assert "platform_export_storage_summary" in js
    assert "platform_export_capability_summary" in js
    assert "platform_stored_exports" in js
    assert "platform_integration_health_summary" in js
    assert "platform_dashboard_summary" in js
    assert "platform_analytics_summary" in js
    assert "platform_proof_cycle_summary" in js
    assert "platform_tracking_summary" in js
    assert "platform_outcome_summary" in js
    assert "platform_calibration_hardening_summary" in js
    assert "platform_drift_summary_text" in js
    assert "platform_benchmark_summary" in js
    assert "platform_model_credibility_summary" in js
    assert "platform_demo_readiness_summary" in js
    assert "platform_bootstrap_summary_text" in js
    assert "platform_readiness_report_summary" in js
    assert "platform_pilot_package_summary" in js
    assert "platform_demo_bundle_summary" in js
    assert "platform_collaboration_summary" in js
    assert "platform_committee_summary" in js
    assert "platform_assignment_summary_text" in js
    assert "assistant-platform-filter-tier" in html
    assert "assistant-platform-filter-regime" in html
    assert "assistant-platform-filter-stage" in html
    assert "assistant-platform-filter-evidence" in html
    assert "renderOperatingWorkflow" in js
    assert "renderOperatorRunbook" in js
    assert "commercialization_readiness_summary" in js
    assert "source_governance_summary" in js
    assert "buyer_diligence_summary" in js
    assert "daily_operating_summary" in js
    assert "weekly_operating_summary" in js
    assert "monthly_operating_summary" in js
    assert "postmortem_summary" in js
    assert "trust_maintenance_summary" in js
    assert "operator_runbook_summary" in js
    assert "Show me the IC memo version of" in js
    assert "What direct sources support the AXIOM fragility score" in js
    assert "AXIOM Summary Card" in js
    assert "AXIOM Evidence & Lineage" in html
    assert "Is the current source stack buyer-demo safe" in js
    assert "What changed today for" in js
    assert "Summarize the workflow stage and dossier status for" in js
    assert "Is the system healthy enough to trust right now" in js
    assert "refreshAssistantSystemHealth" in js
    assert "platform_profile" in js
    assert "create_dossier" in js
    assert "Stored Export History" in js
    assert "Export Capabilities" in js
    assert "Pilot Package" in html
    assert "Bootstrap / Provisioning" in html
    assert "Demo Bundles" in html


def test_platform_proof_renderer_has_percent_formatter_alias() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        const fs = require("fs");
        const path = require("path");
        const vm = require("vm");

        const source = fs.readFileSync(
          path.join(process.cwd(), "api/webapp/app.js"),
          "utf8"
        ) + "\\n;globalThis.__testExports = { renderPlatformProof, state };";

        const elements = new Map();
        const makeElement = () => ({
          value: "",
          textContent: "",
          innerHTML: "",
          disabled: false,
          dataset: {},
          style: {},
          addEventListener() {},
          removeEventListener() {},
          focus() {},
          closest() { return null; },
          classList: { toggle() {}, add() {}, remove() {} },
        });
        const querySelector = (selector) => {
          if (!elements.has(selector)) {
            elements.set(selector, makeElement());
          }
          return elements.get(selector);
        };

        const context = {
          console,
          Date,
          JSON,
          Math,
          Number,
          String,
          Boolean,
          Array,
          Object,
          Promise,
          Map,
          Set,
          window: {
            localStorage: {
              getItem() { return null; },
              setItem() {},
              removeItem() {},
            },
            crypto: { randomUUID: () => "00000000-0000-4000-8000-000000000000" },
          },
          document: {
            body: makeElement(),
            querySelector,
            querySelectorAll() { return []; },
          },
          fetch: async () => ({
            ok: true,
            async json() {
              return { status: "ok", llm_enabled: true, db_enabled: true };
            },
          }),
          setTimeout,
          clearTimeout,
        };
        context.document.body.addEventListener = () => {};
        context.globalThis = context;

        vm.createContext(context);
        vm.runInContext(source, context);

        context.__testExports.renderPlatformProof({
          platform_proof_summary: {
            tracked_recommendation_count: 8,
            matured_tracking_count: 6,
            supportive_count: 4,
            mixed_count: 1,
            weak_count: 1,
            evidence_maturity_level: "emerging",
            replay_consistency_label: "mixed",
            top_regime_rows: [{ label: "trend", average_net_edge_return: 0.084 }],
            top_tier_rows: [{ label: "paper_trade_only", average_net_edge_return: 0.031 }],
          },
          platform_proof_cycle_summary: "Proof summary attached.",
          platform_model_credibility_snapshot: { status: "partial", buyer_summary: "Use with pilot caution." },
          platform_model_credibility_summary: "Credibility remains sample constrained.",
        });

        const html = querySelector("#assistant-platform-proof").innerHTML;
        if (!html.includes("8.40%") || !html.includes("3.10%")) {
          throw new Error("formatPct alias did not render proof percentages");
        }
        console.log("ok");
        """
    )

    result = subprocess.run(
        ["node", "-e", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout


def test_copilot_shell_is_compact_by_default_and_page_context_is_system_aware() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        const fs = require("fs");
        const path = require("path");
        const vm = require("vm");

        const source = fs.readFileSync(
          path.join(process.cwd(), "api/webapp/app.js"),
          "utf8"
        ) + "\\n;globalThis.__testExports = { buildCopilotPageContext, renderCopilotShell, state };";

        const elements = new Map();
        const makeElement = () => {
          const classes = new Set();
          return {
            value: "",
            textContent: "",
            innerHTML: "",
            disabled: false,
            dataset: {},
            style: {},
            addEventListener() {},
            removeEventListener() {},
            focus() {},
            closest() { return null; },
            classList: {
              toggle(name, force) {
                if (force === undefined) {
                  if (classes.has(name)) {
                    classes.delete(name);
                    return false;
                  }
                  classes.add(name);
                  return true;
                }
                if (force) {
                  classes.add(name);
                } else {
                  classes.delete(name);
                }
                return !!force;
              },
              add(name) { classes.add(name); },
              remove(name) { classes.delete(name); },
              contains(name) { return classes.has(name); },
            },
          };
        };
        const querySelector = (selector) => {
          if (!elements.has(selector)) {
            elements.set(selector, makeElement());
          }
          return elements.get(selector);
        };

        const context = {
          console,
          Date,
          JSON,
          Math,
          Number,
          String,
          Boolean,
          Array,
          Object,
          Promise,
          Map,
          Set,
          window: {
            localStorage: {
              getItem() { return null; },
              setItem() {},
              removeItem() {},
            },
            crypto: { randomUUID: () => "00000000-0000-4000-8000-000000000000" },
          },
          document: {
            body: makeElement(),
            querySelector,
            querySelectorAll() { return []; },
          },
          fetch: async () => ({
            ok: true,
            async json() {
              return { status: "ok", llm_enabled: true, db_enabled: true };
            },
          }),
          setTimeout,
          clearTimeout,
        };
        context.document.body.addEventListener = () => {};
        context.globalThis = context;

        vm.createContext(context);
        vm.runInContext(source, context);

        const { buildCopilotPageContext, renderCopilotShell, state } = context.__testExports;
        state.activeTab = "legacy";
        state.researchTab = "platform";
        state.assistantActiveAnalysis = {
          symbol: "NVDA",
          platform_workspace_id: "workspace-1",
          platform_workflow_id: "workflow-1",
          platform_workflow_stage: "decision",
          platform_dossier_id: "dossier-1",
          axiom_evidence_backed_deployability_tier: "paper_trade_only",
        };
        state.assistantLatestReport = {
          symbol: "NVDA",
          as_of_date: "2026-05-24",
          platform_workspace: { workspace_id: "workspace-1", name: "Pilot HF Workspace" },
          platform_workflow: { workflow_id: "workflow-1", stage: "decision" },
          platform_dossier: {
            dossier_id: "dossier-1",
            dossier_type: "coverage",
            evidence_status: "limited",
            current_recommendation_state: "approved_paper",
          },
          platform_recommendation_state: { state: "approved_paper" },
          platform_committee_decision: { decision_status: "approved_with_conditions" },
          platform_review_summary: {
            unresolved_concern_count: 2,
            thread_summary: { total_comments: 3 },
          },
          platform_summary_view: { pending_approval_count: 1 },
          platform_stored_exports: [{ pack_type: "ic_memo_pack" }, { pack_type: "dossier_pack" }],
          platform_health_summary: { warnings: ["Provider confidence is mixed."] },
          platform_proof_summary: { evidence_maturity_level: "emerging" },
          platform_calibration_hardening: { status: "partial" },
        };

        const pageContext = buildCopilotPageContext();
        renderCopilotShell();

        const shell = querySelector("#assistant-copilot-shell");
        const chipsHtml = querySelector("#assistant-copilot-context").innerHTML;
        const promptsHtml = querySelector("#assistant-copilot-prompts").innerHTML;
        const title = querySelector("#assistant-copilot-title").textContent;
        const subtitle = querySelector("#assistant-copilot-inline-context").textContent;
        const status = querySelector("#assistant-copilot-state").textContent;

        if (pageContext.page_focus !== "workflow_and_committee_console") {
          throw new Error("copilot page context missing platform focus");
        }
        if (pageContext.committee_state !== "approved_with_conditions") {
          throw new Error("committee state missing from copilot page context");
        }
        if (pageContext.unresolved_concern_count !== 2) {
          throw new Error("review concern count missing from copilot page context");
        }
        if (!shell.classList.contains("collapsed")) {
          throw new Error("copilot shell should start collapsed");
        }
        if ((chipsHtml.match(/active-chip/g) || []).length !== 3) {
          throw new Error("collapsed copilot shell should render only three chips");
        }
        if (!promptsHtml.includes("committee state") || !promptsHtml.includes("provider or data-quality issue")) {
          throw new Error("copilot prompt set is missing system-aware suggestions");
        }
        if (!title.includes("NVDA") || !subtitle.includes("2 concerns") || status !== "Warning visible") {
          throw new Error("copilot dock summary did not render expected compact state");
        }
        console.log("ok");
        """
    )

    result = subprocess.run(
        ["node", "-e", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout


def test_recent_report_persistence_quota_failure_does_not_break_analyze_flow() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        const fs = require("fs");
        const path = require("path");
        const vm = require("vm");

        const source = fs.readFileSync(
          path.join(process.cwd(), "api/webapp/app.js"),
          "utf8"
        ) + "\\n;globalThis.__testExports = { upsertRecentReport, state };";

        const elements = new Map();
        const makeElement = () => ({
          value: "",
          textContent: "",
          innerHTML: "",
          disabled: false,
          dataset: {},
          style: {},
          addEventListener() {},
          removeEventListener() {},
          focus() {},
          closest() { return null; },
          classList: { toggle() {}, add() {}, remove() {} },
        });
        const querySelector = (selector) => {
          if (!elements.has(selector)) {
            elements.set(selector, makeElement());
          }
          return elements.get(selector);
        };

        const localStorage = {
          _values: new Map(),
          getItem(key) {
            return this._values.has(key) ? this._values.get(key) : null;
          },
          setItem(key, value) {
            if (key === "ftip.assistant.recent_reports" && String(value).length > 5000) {
              throw new Error("QuotaExceededError");
            }
            this._values.set(key, String(value));
          },
          removeItem(key) {
            this._values.delete(key);
          },
        };

        const context = {
          console,
          Date,
          JSON,
          Math,
          Number,
          String,
          Boolean,
          Array,
          Object,
          Promise,
          Map,
          Set,
          window: {
            localStorage,
            crypto: { randomUUID: () => "00000000-0000-4000-8000-000000000000" },
          },
          document: {
            body: makeElement(),
            querySelector,
            querySelectorAll() { return []; },
          },
          fetch: async (url) => ({
            ok: true,
            async json() {
              if (String(url).includes("/providers/health")) {
                return { providers: {} };
              }
              return { status: "ok", llm_enabled: true, db_enabled: true };
            },
          }),
          setTimeout,
          clearTimeout,
        };
        context.document.body.addEventListener = () => {};
        context.globalThis = context;

        vm.createContext(context);
        vm.runInContext(source, context);

        const bigText = "x".repeat(12000);
        const report = {
          symbol: "NVDA",
          as_of_date: "2026-05-09",
          horizon: "swing",
          risk_mode: "balanced",
          signal: { action: "BUY" },
          strategy: {
            final_signal: "BUY",
            conviction_tier: "high",
            strategy_posture: "actionable_long",
            confidence: 0.81,
            actionability_score: 72,
          },
          freshness_summary: { overall_status: "fresh", domains: {} },
          deployment_permission: "paper_shadow_only",
          trust_tier: "paper_only",
          candidate_classification: "watchlist_candidate",
          size_band: "paper / shadow band",
          report_version: "phase-test",
          overall_analysis: bigText,
          signal_summary: bigText,
          technical_analysis: bigText,
          fundamental_analysis: bigText,
          statistical_analysis: bigText,
          sentiment_analysis: bigText,
          macro_geopolitical_analysis: bigText,
          strategy_view: bigText,
          risks_weaknesses_invalidators: bigText,
          evidence_provenance: bigText,
          deployment_readiness_summary: bigText,
          portfolio_context_summary: bigText,
          portfolio_fit_analysis: bigText,
          execution_quality_analysis: bigText,
          learning_summary: bigText,
        };

        context.__testExports.upsertRecentReport(report, null);

        if (context.__testExports.state.assistantRecentReports.length !== 1) {
          throw new Error("recent report was not retained in memory");
        }
        if (context.__testExports.state.assistantRecentReports[0].symbol !== "NVDA") {
          throw new Error("recent report symbol was not retained");
        }
        console.log("ok");
        """
    )

    result = subprocess.run(
        ["node", "-e", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout
