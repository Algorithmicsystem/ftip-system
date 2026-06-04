"""Phase 20.5: Compliance platform routes."""
from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.jobs.tenant_auth import require_tier

router = APIRouter(
    prefix="/compliance",
    tags=["compliance"],
    dependencies=[Depends(require_tier("enterprise"))],
)


# ===========================================================================
# Audit Trail
# ===========================================================================

@router.get("/audit")
def get_audit_trail(
    tenant_id: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    symbol: Optional[str] = Query(default=None),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=1000),
) -> Dict[str, Any]:
    from api.compliance.audit_trail import get_audit_trail as _get
    sd = dt.date.fromisoformat(start_date) if start_date else None
    ed = dt.date.fromisoformat(end_date) if end_date else None
    event_types = [event_type] if event_type else None
    records = _get(
        tenant_id=tenant_id, event_types=event_types,
        symbol=symbol, start_date=sd, end_date=ed, limit=limit,
    )
    return {
        "count": len(records),
        "records": [
            {
                "event_id": r.event_id,
                "event_type": r.event_type,
                "tenant_id": r.tenant_id,
                "actor_type": r.actor_type,
                "resource_type": r.resource_type,
                "resource_id": r.resource_id,
                "symbol": r.symbol,
                "as_of_date": str(r.as_of_date) if r.as_of_date else None,
                "output_summary": r.output_summary,
                "event_hash": r.event_hash,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ],
    }


@router.get("/audit/verify")
def verify_audit_chain(
    start_event_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=500),
) -> Dict[str, Any]:
    from api.compliance.audit_trail import verify_audit_chain as _verify
    return _verify(start_event_id=start_event_id, limit=limit)


@router.get("/audit/{event_id}")
def get_audit_event(event_id: str) -> Dict[str, Any]:
    from api.compliance.audit_trail import get_audit_event as _get
    record = _get(event_id)
    if not record:
        raise HTTPException(status_code=404, detail="Audit event not found")
    return {
        "event_id": record.event_id,
        "event_type": record.event_type,
        "tenant_id": record.tenant_id,
        "actor_type": record.actor_type,
        "resource_type": record.resource_type,
        "resource_id": record.resource_id,
        "symbol": record.symbol,
        "as_of_date": str(record.as_of_date) if record.as_of_date else None,
        "output_hash": record.output_hash,
        "output_summary": record.output_summary,
        "previous_event_hash": record.previous_event_hash,
        "event_hash": record.event_hash,
        "ip_address": record.ip_address,
        "api_version": record.api_version,
        "created_at": record.created_at.isoformat(),
    }


# ===========================================================================
# IPS Compliance
# ===========================================================================

class IPSConstraintsRequest(BaseModel):
    portfolio_id: str
    tenant_id: str
    max_single_position_weight: float = 0.10
    max_sector_concentration: float = 0.30
    max_equity_weight: float = 0.80
    max_alternatives_weight: float = 0.20
    min_cash_weight: float = 0.05
    min_credit_rating: Optional[str] = None
    min_dau_for_equity: float = 50.0
    max_fragility_score: float = 70.0
    max_portfolio_var_1d: float = 0.02
    max_tracking_error: float = 0.05
    prohibited_symbols: List[str] = []
    prohibited_sectors: List[str] = []
    esg_required: bool = False
    rebalancing_threshold: float = 0.05
    rebalancing_frequency: str = "quarterly"


@router.post("/ips")
def create_ips_constraints(body: IPSConstraintsRequest) -> Dict[str, Any]:
    from api.compliance.ips_compliance import IPSConstraints, save_ips_constraints
    ips = IPSConstraints(**body.model_dump())
    saved = save_ips_constraints(ips)
    return dataclasses.asdict(ips) | {"saved": saved}


@router.get("/ips/{portfolio_id}")
def get_ips_constraints(portfolio_id: str) -> Dict[str, Any]:
    from api.compliance.ips_compliance import load_ips_constraints
    ips = load_ips_constraints(portfolio_id)
    if not ips:
        raise HTTPException(status_code=404, detail="IPS constraints not found")
    return dataclasses.asdict(ips)


class AllocationCheckRequest(BaseModel):
    allocation: Dict[str, Any]
    portfolio_value_usd: float = 1_000_000


@router.post("/ips/{portfolio_id}/check")
def check_ips_compliance(portfolio_id: str, body: AllocationCheckRequest) -> Dict[str, Any]:
    from api.compliance.ips_compliance import IPSConstraints, check_ips_compliance as _check
    from api.compliance.ips_compliance import load_ips_constraints
    ips = load_ips_constraints(portfolio_id)
    if not ips:
        ips = IPSConstraints(portfolio_id=portfolio_id, tenant_id="unknown")
    return _check(body.allocation, ips, body.portfolio_value_usd)


@router.post("/ips/{portfolio_id}/remediate")
def remediate_allocation(portfolio_id: str, body: AllocationCheckRequest) -> Dict[str, Any]:
    from api.compliance.ips_compliance import IPSConstraints, generate_ips_compliant_allocation
    from api.compliance.ips_compliance import load_ips_constraints
    ips = load_ips_constraints(portfolio_id)
    if not ips:
        ips = IPSConstraints(portfolio_id=portfolio_id, tenant_id="unknown")
    return generate_ips_compliant_allocation(body.allocation, ips)


# ===========================================================================
# Data Privacy
# ===========================================================================

class ErasureRequest(BaseModel):
    tenant_id: str
    erasure_scope: str = "personal_data"


@router.post("/privacy/erasure")
def request_erasure(body: ErasureRequest) -> Dict[str, Any]:
    from api.compliance.data_privacy import execute_right_to_erasure
    return execute_right_to_erasure(body.tenant_id, body.erasure_scope)


@router.get("/privacy/report")
def get_privacy_report(tenant_id: str = Query(...)) -> Dict[str, Any]:
    from api.compliance.data_privacy import generate_privacy_report
    return generate_privacy_report(tenant_id)


class RetentionPolicyRequest(BaseModel):
    tenant_id: str
    retain_trading_signals_days: int = 2555
    retain_audit_records_days: int = 3650
    retain_api_logs_days: int = 365
    retain_research_reports_days: int = 1825
    data_residency: str = "us"
    gdpr_applicable: bool = False


@router.post("/privacy/retention")
def set_retention_policy(body: RetentionPolicyRequest) -> Dict[str, Any]:
    from api.compliance.data_privacy import DataRetentionPolicy, save_retention_policy
    policy = DataRetentionPolicy(
        tenant_id=body.tenant_id,
        retain_trading_signals_days=body.retain_trading_signals_days,
        retain_audit_records_days=body.retain_audit_records_days,
        retain_api_logs_days=body.retain_api_logs_days,
        retain_research_reports_days=body.retain_research_reports_days,
        data_residency=body.data_residency,
        gdpr_applicable=body.gdpr_applicable,
    )
    saved = save_retention_policy(policy)
    return dataclasses.asdict(policy) | {"saved": saved}


@router.get("/privacy/retention")
def get_retention_policy(tenant_id: str = Query(...)) -> Dict[str, Any]:
    from api.compliance.data_privacy import load_retention_policy
    policy = load_retention_policy(tenant_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Retention policy not found")
    return dataclasses.asdict(policy)


# ===========================================================================
# SOC 2
# ===========================================================================

@router.get("/soc2/readiness")
def get_soc2_readiness() -> Dict[str, Any]:
    from api.compliance.soc2_readiness import assess_soc2_readiness
    report = assess_soc2_readiness()
    return dataclasses.asdict(report) | {"as_of_date": str(report.as_of_date)}


@router.get("/soc2/controls")
def get_soc2_controls() -> Dict[str, Any]:
    from api.compliance.soc2_readiness import SOC2_CONTROLS
    return {
        "count": len(SOC2_CONTROLS),
        "controls": [
            {
                "control_id": c.control_id,
                "criterion": c.criterion,
                "control_name": c.control_name,
                "implementation_status": c.implementation_status,
                "automated": c.automated,
                "test_result": c.test_result,
            }
            for c in SOC2_CONTROLS
        ],
    }


@router.get("/soc2/controls/{control_id}")
def get_soc2_control(control_id: str) -> Dict[str, Any]:
    from api.compliance.soc2_readiness import get_control_evidence
    result = get_control_evidence(control_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Control {control_id} not found")
    return result
