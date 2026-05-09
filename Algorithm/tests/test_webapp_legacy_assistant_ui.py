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
    assert 'id="assistant-analyze-report"' in html
    assert 'id="demo-mode-toggle"' in html
    assert 'id="assistant-dashboard-workflow"' in html
    assert 'id="assistant-dashboard-trust-strip"' in html
    assert 'id="assistant-dashboard-recent"' in html
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
    assert 'id="assistant-artifact-ledger"' in html
    assert 'id="assistant-deployment-audit"' in html
    assert 'id="assistant-compare-summary"' in html
    assert 'id="assistant-compare-details"' in html
    assert 'id="assistant-watchlist-grid"' in html
    assert 'id="assistant-system-health"' in html
    assert 'id="assistant-system-audit"' in html
    assert 'id="assistant-deployment-governance"' in html
    assert 'id="assistant-learning-metrics"' in html
    assert 'id="assistant-learning-regimes"' in html
    assert 'id="assistant-learning-adaptations"' in html
    assert 'id="assistant-learning-experiments"' in html
    assert 'id="assistant-learning-archetypes"' in html
    assert "Learning Lab" in html
    assert "assistantActiveAnalysis" in js
    assert "ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY" in js
    assert "ASSISTANT_RECENT_REPORTS_STORAGE_KEY" in js
    assert "ASSISTANT_WATCHLIST_STORAGE_KEY" in js
    assert "FTIP_DEMO_MODE_STORAGE_KEY" in js
    assert "session_id: pendingSessionId" in js
    assert "active_analysis: state.assistantActiveAnalysis" in js
    assert "setResearchTab" in js
    assert "applyDemoMode" in js
    assert "buildNarratorPromptSet" in js
    assert "renderNarratorPromptChips" in js
    assert "renderDashboardWorkflow" in js
    assert "renderDashboardTrustStrip" in js
    assert "renderRecentAnalyses" in js
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
    assert "Is the system healthy enough to trust right now" in js
    assert "refreshAssistantSystemHealth" in js


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
