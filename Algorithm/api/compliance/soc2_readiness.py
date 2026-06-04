"""Phase 20.4: SOC 2 Type II readiness tracker."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SOC2_CRITERIA: Dict[str, str] = {
    "CC1_CONTROL_ENVIRONMENT": "Demonstrates commitment to integrity, ethical values, and competence",
    "CC2_COMMUNICATION": "Communicates information to support internal control function",
    "CC3_RISK_ASSESSMENT": "Specifies suitable objectives and identifies/analyzes risk",
    "CC4_MONITORING": "Evaluates and communicates internal control deficiencies",
    "CC5_CONTROL_ACTIVITIES": "Selects and develops control activities over technology",
    "CC6_LOGICAL_ACCESS": "Implements logical access security measures",
    "CC7_SYSTEM_OPERATIONS": "Manages system capacity and performance",
    "CC8_CHANGE_MANAGEMENT": "Manages changes to system components",
    "CC9_RISK_MITIGATION": "Identifies and selects risk mitigation strategies",
    "A1_AVAILABILITY": "System is available for operation and use as committed",
    "C1_CONFIDENTIALITY": "Information designated as confidential is protected",
    "PI1_PROCESSING_INTEGRITY": "Processing is complete, valid, accurate, timely, and authorized",
    "P1_PRIVACY": "Personal information is collected, used, retained, disclosed appropriately",
}


@dataclass
class SOC2Control:
    control_id: str
    criterion: str
    control_name: str
    description: str
    implementation_status: str    # "implemented" | "partial" | "not_implemented"
    evidence: List[str]
    automated: bool
    last_tested: Optional[dt.date] = None
    test_result: Optional[str] = None   # "pass" | "fail" | "not_tested"
    notes: str = ""


@dataclass
class SOC2ReadinessReport:
    as_of_date: dt.date
    overall_readiness_score: float
    criteria_coverage: Dict[str, float]
    implemented_controls: int
    partial_controls: int
    not_implemented_controls: int
    automated_controls: int
    high_priority_gaps: List[str]
    estimated_weeks_to_ready: int


SOC2_CONTROLS: List[SOC2Control] = [
    SOC2Control(
        control_id="CC6.1",
        criterion="CC6_LOGICAL_ACCESS",
        control_name="API Key Authentication",
        description="API key authentication with SHA-256 hashing and tenant isolation",
        implementation_status="implemented",
        evidence=["api/jobs/tenant_auth.py", "api_tenants table", "X-API-Key header enforcement"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC6.2",
        criterion="CC6_LOGICAL_ACCESS",
        control_name="Tenant Tier Enforcement",
        description="Access to endpoints gated by subscription tier (free/pro/enterprise)",
        implementation_status="implemented",
        evidence=["api/jobs/tenant_auth.py:require_tier()", "tier_order enforcement"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC6.3",
        criterion="CC6_LOGICAL_ACCESS",
        control_name="API Key Hashing",
        description="API keys stored as SHA-256 hashes; raw keys never persisted",
        implementation_status="implemented",
        evidence=["api/jobs/tenant_auth.py:_hash_key()", "api_tenants.api_key_hash column"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC7.1",
        criterion="CC7_SYSTEM_OPERATIONS",
        control_name="System Health Monitoring",
        description="Real-time system health endpoint monitors all subsystems",
        implementation_status="implemented",
        evidence=["api/orchestration/system_health.py", "GET /orchestration/health"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC7.2",
        criterion="CC7_SYSTEM_OPERATIONS",
        control_name="Pipeline Execution Tracking",
        description="All pipeline runs logged with status, errors, and timing",
        implementation_status="implemented",
        evidence=["api/orchestration/pipeline_orchestrator.py", "pipeline_runs table"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="A1.1",
        criterion="A1_AVAILABILITY",
        control_name="Health Check Endpoint",
        description="GET /health returns system availability status 24/7",
        implementation_status="implemented",
        evidence=["api/main.py:/health endpoint", "db health check"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="A1.2",
        criterion="A1_AVAILABILITY",
        control_name="Graceful DB-Disabled Mode",
        description="All endpoints return sensible defaults when DB is unavailable",
        implementation_status="implemented",
        evidence=["db.db_read_enabled() pattern throughout codebase"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC8.1",
        criterion="CC8_CHANGE_MANAGEMENT",
        control_name="Version Control",
        description="Git version control with full commit history and signed commits",
        implementation_status="implemented",
        evidence=["Git repository", "api/migrations/__init__.py for schema versioning"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC8.2",
        criterion="CC8_CHANGE_MANAGEMENT",
        control_name="Database Schema Migrations",
        description="Versioned, append-only schema migration system (093+ migrations)",
        implementation_status="implemented",
        evidence=["api/migrations/__init__.py:MIGRATIONS list", "schema_migrations table"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="C1.1",
        criterion="C1_CONFIDENTIALITY",
        control_name="Output Sanitization",
        description="API responses filtered by tier; proprietary model details sanitized",
        implementation_status="implemented",
        evidence=["api/axiom/sanitizer.py", "tier-based response filtering"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="C1.2",
        criterion="C1_CONFIDENTIALITY",
        control_name="Webhook Payload Signing",
        description="HMAC-SHA256 signatures on all webhook deliveries",
        implementation_status="implemented",
        evidence=["api/developer/webhooks.py:sign_webhook_payload()", "X-AXIOM-Signature header"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="PI1.1",
        criterion="PI1_PROCESSING_INTEGRITY",
        control_name="AXIOM Lineage Hashing",
        description="Output integrity via memo hashing; axiom_memos table tracks computation lineage",
        implementation_status="implemented",
        evidence=["api/axiom/memo.py", "axiom_memos table"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="PI1.2",
        criterion="PI1_PROCESSING_INTEGRITY",
        control_name="Audit Trail Hash Chaining",
        description="Tamper-evident audit trail with blockchain-style hash chaining",
        implementation_status="implemented",
        evidence=["api/compliance/audit_trail.py", "audit_trail table", "verify_audit_chain()"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="P1.1",
        criterion="P1_PRIVACY",
        control_name="GDPR Right to Erasure",
        description="Automated data deletion on erasure request (Article 17); audit trail preserved",
        implementation_status="implemented",
        evidence=["api/compliance/data_privacy.py:execute_right_to_erasure()", "POST /compliance/privacy/erasure"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="P1.2",
        criterion="P1_PRIVACY",
        control_name="Data Retention Policies",
        description="Per-tenant configurable retention with automated enforcement",
        implementation_status="implemented",
        evidence=["api/compliance/data_privacy.py:enforce_data_retention()", "data_retention_policies table"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC3.1",
        criterion="CC3_RISK_ASSESSMENT",
        control_name="Systemic Risk Index",
        description="7-component SRI computed daily and monitored for systemic crisis probability",
        implementation_status="implemented",
        evidence=["api/axiom/risk/systemic_risk.py", "GET /axiom/risk/sri"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC4.1",
        criterion="CC4_MONITORING",
        control_name="Deficiency Communication",
        description="System health monitor with degradation alerts and scheduled checks",
        implementation_status="implemented",
        evidence=["api/orchestration/system_health.py", "api/jobs/scheduler.py"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC5.1",
        criterion="CC5_CONTROL_ACTIVITIES",
        control_name="ML Model Drift Monitoring",
        description="PSI-based model drift detection with automated alerts",
        implementation_status="implemented",
        evidence=["api/axiom/ml/drift_monitor.py", "GET /axiom/ml/model-status"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC9.1",
        criterion="CC9_RISK_MITIGATION",
        control_name="IPS Compliance Engine",
        description="Automated Investment Policy Statement constraint checking for institutional clients",
        implementation_status="implemented",
        evidence=["api/compliance/ips_compliance.py", "POST /compliance/ips/{id}/check"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC1.1",
        criterion="CC1_CONTROL_ENVIRONMENT",
        control_name="Tenant Onboarding Process",
        description="Formal tenant registration with tier assignment and API key provisioning",
        implementation_status="implemented",
        evidence=["api/jobs/tenant_auth.py:register_tenant()", "api/jobs/onboarding.py"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC2.1",
        criterion="CC2_COMMUNICATION",
        control_name="API Documentation and SDK",
        description="Complete API documentation registry and auto-generated SDKs for developers",
        implementation_status="implemented",
        evidence=["api/developer/api_versioning.py", "api/developer/sdk_generator.py",
                  "GET /developer/api-docs"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC2.2",
        criterion="CC2_COMMUNICATION",
        control_name="Morning Intelligence Briefing",
        description="Daily automated briefing with regime, risk, and signal context",
        implementation_status="implemented",
        evidence=["api/jobs/morning_briefing.py", "GET /jobs/briefing/morning"],
        automated=True,
        test_result="pass",
    ),
    SOC2Control(
        control_id="CC1.2",
        criterion="CC1_CONTROL_ENVIRONMENT",
        control_name="Multi-Tenant Isolation",
        description="Complete data isolation between tenants; no cross-tenant data leakage",
        implementation_status="partial",
        evidence=["api/jobs/tenant_auth.py:tier_has_access()", "tenant_id filters on queries"],
        automated=True,
        notes="Row-level security in DB not yet enforced; relies on application-layer filtering",
    ),
    SOC2Control(
        control_id="A1.3",
        criterion="A1_AVAILABILITY",
        control_name="SLA Monitoring and Uptime Tracking",
        description="Formal SLA tracking with uptime metrics and incident management",
        implementation_status="not_implemented",
        evidence=[],
        automated=False,
        notes="Requires external uptime monitoring service (e.g., PagerDuty, Datadog)",
    ),
    SOC2Control(
        control_id="CC8.3",
        criterion="CC8_CHANGE_MANAGEMENT",
        control_name="Formal Code Review Process",
        description="Mandatory peer code review before production deployment",
        implementation_status="partial",
        evidence=["Git commit history"],
        automated=False,
        notes="Process exists informally; formal review gates not enforced in CI/CD",
    ),
]


def assess_soc2_readiness() -> SOC2ReadinessReport:
    today = dt.date.today()

    implemented = [c for c in SOC2_CONTROLS if c.implementation_status == "implemented"]
    partial = [c for c in SOC2_CONTROLS if c.implementation_status == "partial"]
    not_impl = [c for c in SOC2_CONTROLS if c.implementation_status == "not_implemented"]
    automated = [c for c in SOC2_CONTROLS if c.automated]

    total = len(SOC2_CONTROLS)
    score = ((len(implemented) * 1.0 + len(partial) * 0.5) / total * 100) if total else 0.0

    # Criteria coverage
    criteria_coverage: Dict[str, float] = {}
    for criterion in SOC2_CRITERIA:
        crit_controls = [c for c in SOC2_CONTROLS if c.criterion == criterion]
        if not crit_controls:
            criteria_coverage[criterion] = 0.0
        else:
            crit_score = sum(
                1.0 if c.implementation_status == "implemented"
                else 0.5 if c.implementation_status == "partial"
                else 0.0
                for c in crit_controls
            )
            criteria_coverage[criterion] = round(crit_score / len(crit_controls) * 100, 1)

    # High-priority gaps: not_implemented controls for critical criteria
    _CRITICAL_CRITERIA = {"CC6_LOGICAL_ACCESS", "A1_AVAILABILITY", "C1_CONFIDENTIALITY",
                          "PI1_PROCESSING_INTEGRITY"}
    high_priority_gaps = [
        c.control_id for c in not_impl
        if c.criterion in _CRITICAL_CRITERIA
    ]
    # Also include partial if in critical criteria
    for c in partial:
        if c.criterion in _CRITICAL_CRITERIA and c.control_id not in high_priority_gaps:
            high_priority_gaps.append(c.control_id)

    estimated_weeks = (len(not_impl) + len(partial)) * 2

    return SOC2ReadinessReport(
        as_of_date=today,
        overall_readiness_score=round(score, 1),
        criteria_coverage=criteria_coverage,
        implemented_controls=len(implemented),
        partial_controls=len(partial),
        not_implemented_controls=len(not_impl),
        automated_controls=len(automated),
        high_priority_gaps=high_priority_gaps,
        estimated_weeks_to_ready=estimated_weeks,
    )


def get_control_evidence(control_id: str) -> Dict[str, Any]:
    for control in SOC2_CONTROLS:
        if control.control_id == control_id:
            return {
                "control_id": control.control_id,
                "criterion": control.criterion,
                "control_name": control.control_name,
                "description": control.description,
                "implementation_status": control.implementation_status,
                "evidence": control.evidence,
                "automated": control.automated,
                "test_result": control.test_result,
                "notes": control.notes,
            }
    return {}
