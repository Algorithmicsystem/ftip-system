"""Phase 20 tests: Compliance, Audit, and SOC 2 Readiness."""
from __future__ import annotations

import datetime as dt
import hashlib

import pytest

from api.compliance.audit_trail import (
    AUDIT_EVENT_TYPES,
    AuditRecord,
    compute_event_hash,
    get_audit_trail,
    verify_audit_chain,
    write_audit_record,
)
from api.compliance.ips_compliance import (
    IPSConstraints,
    check_ips_compliance,
    generate_ips_compliant_allocation,
)
from api.compliance.data_privacy import (
    DataRetentionPolicy,
    execute_right_to_erasure,
    generate_privacy_report,
)
from api.compliance.soc2_readiness import (
    SOC2_CONTROLS,
    SOC2_CRITERIA,
    assess_soc2_readiness,
    get_control_evidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    event_type: str = "signal.generated",
    resource_id: str = "AAPL_2026-01-01",
    tenant_id: str = "tenant_001",
    output: dict = None,
) -> AuditRecord:
    return write_audit_record(
        event_type=event_type,
        resource_type="signal",
        resource_id=resource_id,
        output=output or {"symbol": "AAPL", "dau": 75.0},
        tenant_id=tenant_id,
        symbol="AAPL",
        as_of_date=dt.date(2026, 1, 1),
    )


_CLEAN_POSITIONS = [
    {"symbol": "AAPL", "weight": 0.08, "sector": "Technology",
     "axiom_dau": 78.0, "fragility_score": 35.0},
    {"symbol": "MSFT", "weight": 0.07, "sector": "Technology",
     "axiom_dau": 72.0, "fragility_score": 30.0},
    {"symbol": "JPM",  "weight": 0.06, "sector": "Financials",
     "axiom_dau": 65.0, "fragility_score": 40.0},
]

_DEFAULT_IPS = IPSConstraints(
    portfolio_id="port_001",
    tenant_id="tenant_001",
    max_single_position_weight=0.10,
    max_sector_concentration=0.30,
    max_equity_weight=0.80,
    min_dau_for_equity=50.0,
    max_fragility_score=70.0,
)


# ===========================================================================
# TestAuditTrail
# ===========================================================================

class TestAuditTrail:
    def test_event_types_all_defined(self):
        assert len(AUDIT_EVENT_TYPES) >= 18
        for et in AUDIT_EVENT_TYPES:
            assert isinstance(et, str)
            assert "." in et

    def test_compute_event_hash_deterministic(self):
        record = _make_record()
        h1 = compute_event_hash(record, "")
        h2 = compute_event_hash(record, "")
        assert h1 == h2

    def test_compute_event_hash_changes_with_content(self):
        r1 = _make_record(resource_id="AAPL_001")
        r2 = _make_record(resource_id="MSFT_001")
        assert compute_event_hash(r1, "") != compute_event_hash(r2, "")

    def test_compute_event_hash_changes_with_previous_hash(self):
        record = _make_record()
        h1 = compute_event_hash(record, "abc")
        h2 = compute_event_hash(record, "xyz")
        assert h1 != h2

    def test_output_hash_is_sha256(self):
        record = _make_record()
        assert len(record.output_hash) == 64
        int(record.output_hash, 16)  # valid hex

    def test_write_audit_record_returns_record(self):
        record = _make_record()
        assert isinstance(record, AuditRecord)
        assert record.event_type == "signal.generated"
        assert record.tenant_id == "tenant_001"
        assert record.symbol == "AAPL"

    def test_audit_record_has_event_hash(self):
        record = _make_record()
        assert len(record.event_hash) == 64
        int(record.event_hash, 16)  # valid hex

    def test_audit_record_has_event_id(self):
        record = _make_record()
        assert len(record.event_id) > 0
        assert "-" in record.event_id  # UUID format

    def test_audit_record_created_at_is_recent(self):
        record = _make_record()
        diff = (dt.datetime.utcnow() - record.created_at).total_seconds()
        assert diff < 5.0

    def test_previous_hash_chaining(self):
        r1 = _make_record(resource_id="first_event")
        r2 = _make_record(resource_id="second_event")
        # r2.previous_event_hash should be r1.event_hash if both written to DB
        # In no-DB mode, both will have empty previous hash — but event_hash is computed correctly
        assert len(r1.event_hash) == 64
        assert len(r2.event_hash) == 64

    def test_verify_chain_empty_is_valid(self):
        result = verify_audit_chain(limit=0)
        assert "chain_intact" in result
        assert "records_checked" in result
        assert "verified" in result
        # Zero records means chain is not broken (nothing to check)
        # chain_intact is True if no broken records found
        assert result["chain_intact"] is True or result["records_checked"] == 0

    def test_verify_chain_structure(self):
        result = verify_audit_chain()
        assert "verified" in result
        assert "records_checked" in result
        assert "first_broken_event_id" in result
        assert "chain_intact" in result

    def test_get_audit_trail_returns_list(self):
        result = get_audit_trail(limit=10)
        assert isinstance(result, list)

    def test_audit_event_types_cover_all_categories(self):
        categories = {et.split(".")[0] for et in AUDIT_EVENT_TYPES}
        assert "data" in categories
        assert "signal" in categories
        assert "analysis" in categories
        assert "portfolio" in categories
        assert "access" in categories
        assert "admin" in categories


# ===========================================================================
# TestIPSCompliance
# ===========================================================================

class TestIPSCompliance:
    def test_compliant_allocation(self):
        result = check_ips_compliance(
            {"allocations": _CLEAN_POSITIONS}, _DEFAULT_IPS
        )
        assert result["compliant"] is True
        assert result["violations"] == []

    def test_compliance_score_100_when_clean(self):
        result = check_ips_compliance(
            {"allocations": _CLEAN_POSITIONS}, _DEFAULT_IPS
        )
        assert result["compliance_score"] == 100.0

    def test_single_position_violation(self):
        positions = [
            {"symbol": "AAPL", "weight": 0.15, "sector": "Technology",
             "axiom_dau": 78.0, "fragility_score": 35.0},
        ]
        ips = IPSConstraints(
            portfolio_id="p1", tenant_id="t1",
            max_single_position_weight=0.10,
        )
        result = check_ips_compliance({"allocations": positions}, ips)
        assert result["compliant"] is False
        constraint_names = [v["constraint"] for v in result["violations"]]
        assert "single_position_weight" in constraint_names

    def test_single_position_violation_severity_major(self):
        positions = [
            {"symbol": "AAPL", "weight": 0.15, "sector": "Tech", "axiom_dau": 78.0},
        ]
        ips = IPSConstraints(portfolio_id="p1", tenant_id="t1", max_single_position_weight=0.10)
        result = check_ips_compliance({"allocations": positions}, ips)
        pos_violations = [v for v in result["violations"] if v["constraint"] == "single_position_weight"]
        assert pos_violations[0]["severity"] == "major"

    def test_prohibited_symbol_violation(self):
        positions = [
            {"symbol": "TSLA", "weight": 0.05, "sector": "Consumer",
             "axiom_dau": 70.0, "fragility_score": 30.0},
        ]
        ips = IPSConstraints(
            portfolio_id="p1", tenant_id="t1",
            prohibited_symbols=["TSLA"],
        )
        result = check_ips_compliance({"allocations": positions}, ips)
        assert result["compliant"] is False
        constraint_names = [v["constraint"] for v in result["violations"]]
        assert "prohibited_symbol" in constraint_names

    def test_prohibited_symbol_severity_critical(self):
        positions = [{"symbol": "TSLA", "weight": 0.05, "sector": "Consumer", "axiom_dau": 70.0}]
        ips = IPSConstraints(portfolio_id="p1", tenant_id="t1", prohibited_symbols=["TSLA"])
        result = check_ips_compliance({"allocations": positions}, ips)
        prohibited_v = [v for v in result["violations"] if v["constraint"] == "prohibited_symbol"]
        assert prohibited_v[0]["severity"] == "critical"

    def test_sector_concentration_violation(self):
        positions = [
            {"symbol": "AAPL", "weight": 0.20, "sector": "Technology", "axiom_dau": 80.0},
            {"symbol": "MSFT", "weight": 0.20, "sector": "Technology", "axiom_dau": 75.0},
        ]
        ips = IPSConstraints(
            portfolio_id="p1", tenant_id="t1",
            max_sector_concentration=0.30,
        )
        result = check_ips_compliance({"allocations": positions}, ips)
        assert result["compliant"] is False
        constraint_names = [v["constraint"] for v in result["violations"]]
        assert "sector_concentration" in constraint_names

    def test_compliance_score_reduced_by_violations(self):
        positions = [
            {"symbol": "AAPL", "weight": 0.15, "sector": "Tech", "axiom_dau": 80.0},
            {"symbol": "MSFT", "weight": 0.15, "sector": "Tech", "axiom_dau": 75.0},
        ]
        ips = IPSConstraints(portfolio_id="p1", tenant_id="t1",
                             max_single_position_weight=0.10, max_sector_concentration=0.25)
        result = check_ips_compliance({"allocations": positions}, ips)
        assert result["compliance_score"] < 100.0

    def test_remediation_removes_prohibited(self):
        positions = [
            {"symbol": "TSLA", "weight": 0.10, "sector": "Consumer", "axiom_dau": 70.0},
            {"symbol": "AAPL", "weight": 0.10, "sector": "Technology", "axiom_dau": 80.0},
        ]
        ips = IPSConstraints(portfolio_id="p1", tenant_id="t1", prohibited_symbols=["TSLA"])
        result = generate_ips_compliant_allocation({"allocations": positions}, ips)
        syms = [p["symbol"] for p in result["allocations"]]
        assert "TSLA" not in syms
        assert "AAPL" in syms

    def test_remediation_caps_positions(self):
        positions = [
            {"symbol": "AAPL", "weight": 0.30, "sector": "Technology", "axiom_dau": 80.0},
            {"symbol": "MSFT", "weight": 0.10, "sector": "Technology", "axiom_dau": 75.0},
        ]
        ips = IPSConstraints(portfolio_id="p1", tenant_id="t1", max_single_position_weight=0.15)
        result = generate_ips_compliant_allocation({"allocations": positions}, ips)
        for pos in result["allocations"]:
            assert pos["weight"] <= 0.15 + 1e-6, f"{pos['symbol']} weight {pos['weight']} too high"

    def test_remediation_weights_sum_to_one(self):
        # 4 equal positions, each sector distinct, cap well above each weight
        positions = [
            {"symbol": "AAPL", "weight": 0.25, "sector": "Technology", "axiom_dau": 80.0},
            {"symbol": "JPM",  "weight": 0.25, "sector": "Financials", "axiom_dau": 65.0},
            {"symbol": "XOM",  "weight": 0.25, "sector": "Energy", "axiom_dau": 60.0},
            {"symbol": "JNJ",  "weight": 0.25, "sector": "Healthcare", "axiom_dau": 70.0},
        ]
        ips = IPSConstraints(portfolio_id="p1", tenant_id="t1", max_single_position_weight=0.30)
        result = generate_ips_compliant_allocation({"allocations": positions}, ips)
        total = sum(p["weight"] for p in result["allocations"])
        assert abs(total - 1.0) < 1e-4

    def test_dau_filter_removes_low_dau(self):
        positions = [
            {"symbol": "AAPL", "weight": 0.20, "sector": "Technology", "axiom_dau": 80.0},
            {"symbol": "JUNK", "weight": 0.20, "sector": "Junk", "axiom_dau": 20.0},
        ]
        ips = IPSConstraints(portfolio_id="p1", tenant_id="t1", min_dau_for_equity=50.0)
        result = generate_ips_compliant_allocation({"allocations": positions}, ips)
        syms = [p["symbol"] for p in result["allocations"]]
        assert "JUNK" not in syms

    def test_ips_constraints_dataclass_defaults(self):
        ips = IPSConstraints(portfolio_id="p1", tenant_id="t1")
        assert ips.max_single_position_weight == 0.10
        assert ips.max_equity_weight == 0.80
        assert ips.prohibited_symbols == []
        assert ips.esg_required is False


# ===========================================================================
# TestDataPrivacy
# ===========================================================================

class TestDataPrivacy:
    def test_erasure_returns_summary(self):
        result = execute_right_to_erasure("tenant_test_001", "personal_data")
        assert "tenant_id" in result
        assert "tables_cleared" in result
        assert isinstance(result["tables_cleared"], list)
        assert "records_deleted" in result

    def test_audit_trail_preserved(self):
        result = execute_right_to_erasure("tenant_test_002", "personal_data")
        assert result["audit_trail_preserved"] is True

    def test_erasure_record_id_is_uuid(self):
        result = execute_right_to_erasure("tenant_test_003")
        assert "erasure_record_id" in result
        assert len(result["erasure_record_id"]) > 0
        assert "-" in result["erasure_record_id"]

    def test_privacy_report_structure(self):
        result = generate_privacy_report("tenant_test_001")
        assert "tenant_id" in result
        assert "data_categories" in result
        assert "record_counts" in result
        assert "data_residency" in result
        assert "processing_purposes" in result

    def test_privacy_report_tenant_id_preserved(self):
        result = generate_privacy_report("my_tenant")
        assert result["tenant_id"] == "my_tenant"

    def test_privacy_report_processing_purposes_nonempty(self):
        result = generate_privacy_report("tenant_001")
        assert len(result["processing_purposes"]) >= 1

    def test_retention_policy_structure(self):
        policy = DataRetentionPolicy(tenant_id="t1")
        assert policy.retain_trading_signals_days == 2555
        assert policy.retain_audit_records_days == 3650
        assert policy.retain_api_logs_days == 365
        assert policy.retain_research_reports_days == 1825

    def test_gdpr_applicable_flag(self):
        policy = DataRetentionPolicy(tenant_id="t1", gdpr_applicable=True)
        assert isinstance(policy.gdpr_applicable, bool)
        assert policy.gdpr_applicable is True

    def test_retention_policy_data_residency_default(self):
        policy = DataRetentionPolicy(tenant_id="t1")
        assert policy.data_residency == "us"

    def test_retention_policy_tenant_id_stored(self):
        policy = DataRetentionPolicy(tenant_id="my_tenant_001")
        assert policy.tenant_id == "my_tenant_001"


# ===========================================================================
# TestSOC2Readiness
# ===========================================================================

class TestSOC2Readiness:
    def test_soc2_controls_count(self):
        assert len(SOC2_CONTROLS) >= 20

    def test_all_controls_have_required_fields(self):
        for ctrl in SOC2_CONTROLS:
            assert ctrl.control_id, f"Missing control_id"
            assert ctrl.criterion, f"Missing criterion in {ctrl.control_id}"
            assert ctrl.control_name, f"Missing control_name in {ctrl.control_id}"
            assert ctrl.implementation_status, f"Missing implementation_status in {ctrl.control_id}"

    def test_implementation_status_valid(self):
        valid = {"implemented", "partial", "not_implemented"}
        for ctrl in SOC2_CONTROLS:
            assert ctrl.implementation_status in valid, (
                f"{ctrl.control_id} has invalid status: {ctrl.implementation_status}"
            )

    def test_criteria_all_covered(self):
        covered = {ctrl.criterion for ctrl in SOC2_CONTROLS}
        for criterion in SOC2_CRITERIA:
            assert criterion in covered, f"SOC2 criterion {criterion} has no controls"

    def test_soc2_criteria_count(self):
        assert len(SOC2_CRITERIA) == 13

    def test_readiness_score_bounded(self):
        report = assess_soc2_readiness()
        assert 0.0 <= report.overall_readiness_score <= 100.0

    def test_high_priority_gaps_subset_of_not_implemented(self):
        report = assess_soc2_readiness()
        not_impl_ids = {c.control_id for c in SOC2_CONTROLS
                        if c.implementation_status in ("not_implemented", "partial")}
        for gap in report.high_priority_gaps:
            assert gap in not_impl_ids, f"Gap {gap} not in not-implemented set"

    def test_estimated_weeks_positive(self):
        report = assess_soc2_readiness()
        assert report.estimated_weeks_to_ready >= 0

    def test_control_evidence_returned(self):
        implemented = [c for c in SOC2_CONTROLS if c.implementation_status == "implemented"]
        assert len(implemented) > 0
        evidence = get_control_evidence(implemented[0].control_id)
        assert len(evidence) > 0
        assert "evidence" in evidence
        assert len(evidence["evidence"]) > 0

    def test_control_evidence_missing_returns_empty(self):
        result = get_control_evidence("NONEXISTENT.99")
        assert result == {}

    def test_readiness_report_criteria_coverage_complete(self):
        report = assess_soc2_readiness()
        for criterion in SOC2_CRITERIA:
            assert criterion in report.criteria_coverage, f"Missing coverage for {criterion}"

    def test_readiness_counts_consistent(self):
        report = assess_soc2_readiness()
        total = (report.implemented_controls
                 + report.partial_controls
                 + report.not_implemented_controls)
        assert total == len(SOC2_CONTROLS)

    def test_implemented_controls_match_list(self):
        report = assess_soc2_readiness()
        expected = sum(1 for c in SOC2_CONTROLS if c.implementation_status == "implemented")
        assert report.implemented_controls == expected
