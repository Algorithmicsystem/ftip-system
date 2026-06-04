"""Phase 19.1: API Versioning and Documentation registry."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

V1_ENDPOINTS: List[Dict[str, Any]] = [
    # Prosperity
    {"path": "/prosperity/signal/{symbol}", "method": "GET", "category": "signals",
     "tier_required": "free", "rate_limit_per_minute": 30,
     "description": "Retrieve the latest AXIOM signal for a symbol.",
     "response_schema": "SignalResponse", "deprecated": False},
    {"path": "/prosperity/snapshot/{symbol}", "method": "GET", "category": "signals",
     "tier_required": "free", "rate_limit_per_minute": 30,
     "description": "Full AXIOM snapshot with all engine scores for a symbol.",
     "response_schema": "SnapshotResponse", "deprecated": False},
    {"path": "/prosperity/universe", "method": "GET", "category": "signals",
     "tier_required": "free", "rate_limit_per_minute": 30,
     "description": "List all symbols in the universe.",
     "response_schema": "UniverseResponse", "deprecated": False},
    {"path": "/prosperity/batch", "method": "POST", "category": "signals",
     "tier_required": "pro", "rate_limit_per_minute": 60,
     "description": "Batch signal retrieval for multiple symbols.",
     "response_schema": "BatchSignalResponse", "deprecated": False},
    # AXIOM
    {"path": "/axiom/scores/{symbol}", "method": "GET", "category": "axiom",
     "tier_required": "free", "rate_limit_per_minute": 30,
     "description": "AXIOM composite scores for a symbol.",
     "response_schema": "AxiomScoresResponse", "deprecated": False},
    {"path": "/axiom/history/{symbol}", "method": "GET", "category": "axiom",
     "tier_required": "pro", "rate_limit_per_minute": 60,
     "description": "Historical AXIOM scores for a symbol.",
     "response_schema": "AxiomHistoryResponse", "deprecated": False},
    {"path": "/axiom/factors", "method": "GET", "category": "axiom",
     "tier_required": "pro", "rate_limit_per_minute": 60,
     "description": "Current factor loadings and exposures.",
     "response_schema": "FactorResponse", "deprecated": False},
    # Risk
    {"path": "/axiom/risk/sri", "method": "GET", "category": "risk",
     "tier_required": "pro", "rate_limit_per_minute": 60,
     "description": "Systemic Risk Index (SRI) for today.",
     "response_schema": "SRIResponse", "deprecated": False},
    {"path": "/axiom/risk/sri/history", "method": "GET", "category": "risk",
     "tier_required": "pro", "rate_limit_per_minute": 30,
     "description": "Historical SRI values for charting.",
     "response_schema": "SRIHistoryResponse", "deprecated": False},
    # Intelligence
    {"path": "/intelligence/universal/{symbol}", "method": "GET", "category": "intelligence",
     "tier_required": "enterprise", "rate_limit_per_minute": 120,
     "description": "Universal intelligence package for a symbol (all signals, all context).",
     "response_schema": "UniversalIntelligenceResponse", "deprecated": False},
    # Competitive
    {"path": "/competitive/{symbol}", "method": "GET", "category": "competitive",
     "tier_required": "enterprise", "rate_limit_per_minute": 60,
     "description": "Full competitive intelligence report for a symbol.",
     "response_schema": "CompetitiveReport", "deprecated": False},
    {"path": "/competitive/{symbol}/vs/{competitor}", "method": "GET", "category": "competitive",
     "tier_required": "enterprise", "rate_limit_per_minute": 60,
     "description": "Head-to-head competitive profile comparison.",
     "response_schema": "CompetitorProfile", "deprecated": False},
    {"path": "/competitive/{symbol}/management-quality", "method": "GET", "category": "competitive",
     "tier_required": "enterprise", "rate_limit_per_minute": 60,
     "description": "Management quality scores for a symbol.",
     "response_schema": "ManagementQualityResponse", "deprecated": False},
    # Macro
    {"path": "/macro/regime", "method": "GET", "category": "macro",
     "tier_required": "enterprise", "rate_limit_per_minute": 60,
     "description": "Current global macro regime classification.",
     "response_schema": "MacroRegimeResponse", "deprecated": False},
    {"path": "/macro/cross-asset", "method": "GET", "category": "macro",
     "tier_required": "enterprise", "rate_limit_per_minute": 60,
     "description": "Cross-asset confirmation snapshot.",
     "response_schema": "CrossAssetSnapshot", "deprecated": False},
    # Explainability
    {"path": "/explain/{symbol}", "method": "GET", "category": "explain",
     "tier_required": "pro", "rate_limit_per_minute": 30,
     "description": "Natural language explanation of AXIOM signal.",
     "response_schema": "ExplainResponse", "deprecated": False},
    {"path": "/explain/report/{symbol}", "method": "GET", "category": "explain",
     "tier_required": "enterprise", "rate_limit_per_minute": 20,
     "description": "Full research report with reasoning chains.",
     "response_schema": "ResearchReport", "deprecated": False},
    # Briefing
    {"path": "/jobs/briefing/morning", "method": "GET", "category": "briefing",
     "tier_required": "pro", "rate_limit_per_minute": 30,
     "description": "Morning intelligence briefing package.",
     "response_schema": "MorningBriefing", "deprecated": False},
    # Linkage
    {"path": "/linkage/graph/{symbol}", "method": "GET", "category": "linkage",
     "tier_required": "pro", "rate_limit_per_minute": 30,
     "description": "Linkage graph for a symbol (supply chain, sector peers).",
     "response_schema": "LinkageGraph", "deprecated": False},
    # PE / SMB
    {"path": "/pe/entity/{entity_id}/health", "method": "GET", "category": "pe",
     "tier_required": "enterprise", "rate_limit_per_minute": 60,
     "description": "PE entity financial health assessment.",
     "response_schema": "PEHealthResponse", "deprecated": False},
    {"path": "/smb/entity/{entity_id}/health", "method": "GET", "category": "smb",
     "tier_required": "enterprise", "rate_limit_per_minute": 60,
     "description": "SMB entity financial health assessment.",
     "response_schema": "SMBHealthResponse", "deprecated": False},
    # ML
    {"path": "/axiom/ml/model-status", "method": "GET", "category": "ml",
     "tier_required": "enterprise", "rate_limit_per_minute": 30,
     "description": "Current ML model version, drift status, and performance.",
     "response_schema": "MLModelStatus", "deprecated": False},
    # Orchestration
    {"path": "/orchestration/pipeline/run", "method": "POST", "category": "orchestration",
     "tier_required": "enterprise", "rate_limit_per_minute": 10,
     "description": "Trigger a full daily pipeline run.",
     "response_schema": "PipelineRunResult", "deprecated": False},
    {"path": "/orchestration/health", "method": "GET", "category": "orchestration",
     "tier_required": "pro", "rate_limit_per_minute": 60,
     "description": "System health status across all subsystems.",
     "response_schema": "SystemHealth", "deprecated": False},
    # Developer
    {"path": "/developer/api-docs", "method": "GET", "category": "developer",
     "tier_required": "free", "rate_limit_per_minute": 60,
     "description": "Full API documentation and endpoint registry.",
     "response_schema": "APIDocumentation", "deprecated": False},
    {"path": "/developer/sdk/python", "method": "GET", "category": "developer",
     "tier_required": "free", "rate_limit_per_minute": 30,
     "description": "Download the AXIOM Python SDK.",
     "response_schema": "PythonSDK", "deprecated": False},
    {"path": "/developer/sdk/javascript", "method": "GET", "category": "developer",
     "tier_required": "free", "rate_limit_per_minute": 30,
     "description": "Download the AXIOM JavaScript SDK.",
     "response_schema": "JavaScriptSDK", "deprecated": False},
    {"path": "/developer/webhooks", "method": "POST", "category": "developer",
     "tier_required": "pro", "rate_limit_per_minute": 30,
     "description": "Subscribe to AXIOM webhook events.",
     "response_schema": "WebhookSubscription", "deprecated": False},
    {"path": "/developer/usage", "method": "GET", "category": "developer",
     "tier_required": "pro", "rate_limit_per_minute": 30,
     "description": "API usage metrics for your tenant.",
     "response_schema": "UsageMetrics", "deprecated": False},
    {"path": "/developer/analytics", "method": "GET", "category": "developer",
     "tier_required": "enterprise", "rate_limit_per_minute": 30,
     "description": "Platform-wide analytics (enterprise only).",
     "response_schema": "PlatformAnalytics", "deprecated": False},
]


def get_api_documentation() -> Dict[str, Any]:
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for ep in V1_ENDPOINTS:
        cat = ep["category"]
        categories.setdefault(cat, []).append(ep)
    total = len(V1_ENDPOINTS)
    deprecated = sum(1 for ep in V1_ENDPOINTS if ep.get("deprecated"))
    return {
        "version": "v1",
        "total_endpoints": total,
        "deprecated_endpoints": deprecated,
        "active_endpoints": total - deprecated,
        "categories": list(categories.keys()),
        "endpoints_by_category": categories,
        "endpoints": V1_ENDPOINTS,
    }


def get_endpoint_info(path: str, method: str = "GET") -> Dict[str, Any]:
    method_upper = method.upper()
    for ep in V1_ENDPOINTS:
        if ep["path"] == path and ep["method"] == method_upper:
            return ep
    return {}


def get_endpoints_by_category(category: str) -> List[Dict[str, Any]]:
    return [ep for ep in V1_ENDPOINTS if ep["category"] == category]
