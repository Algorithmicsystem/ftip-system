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
    assert "refreshAssistantSystemHealth" in js
