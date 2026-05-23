from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.platform import service as platform_service
from api.platform.persistence import PlatformStore, platform_store
from tests.test_platform_phase5 import _reset_platform_store
from tests.test_platform_phase6 import _bootstrap_workspace, _headers
from tests.test_platform_phase7 import _attach_sample_analysis


def _prepare_workspace(store: PlatformStore) -> tuple[dict, dict, dict]:
    workspace, workflow, dossier = _bootstrap_workspace(store)
    platform_service.create_membership_service(
        {
            "user_id": "analyst-1",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "analyst",
        },
        store=store,
    )
    platform_service.create_membership_service(
        {
            "user_id": "reviewer-1",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "reviewer",
        },
        store=store,
    )
    platform_service.create_membership_service(
        {
            "user_id": "committee-1",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "committee",
        },
        store=store,
    )
    _attach_sample_analysis(
        store,
        dossier["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )
    return workspace, workflow, dossier


def test_phase8c_comment_assignment_decision_and_recommendation_flow() -> None:
    store = PlatformStore(use_memory=True)
    workspace, workflow, dossier = _prepare_workspace(store)

    comment = platform_service.create_dossier_comment_service(
        dossier["dossier_id"],
        {
            "comment_type": "liquidity_concern",
            "severity": "material",
            "body": "Execution depth still looks thin for live sizing.",
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )
    assert comment["review_summary"]["unresolved_concern_count"] == 1

    assignment = platform_service.update_workflow_assignment_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "slot_type": "primary_reviewer",
            "assignee_placeholder": "reviewer-1",
            "notes": ["Own the next evidence pass."],
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )
    assert assignment["assignment_summary"]["primary_reviewer"]["assignee_placeholder"] == "reviewer-1"

    decision = platform_service.record_committee_decision_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "decision_status": "approved_with_conditions",
            "recommendation_state": "approved_paper",
            "summary": "Approved for paper deployment pending liquidity follow-up.",
            "conditions": ["Recheck execution depth before live escalation."],
            "key_risks": ["Liquidity gap around event windows."],
            "key_evidence_strengths": ["Research integrity remains constructive."],
            "key_evidence_gaps": ["Need fresher execution evidence."],
            "rationale": "Committee wants a controlled deployment path first.",
        },
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )
    assert decision["committee_decision"]["decision_status"] == "approved_with_conditions"
    assert decision["recommendation_state"]["state"] == "approved_paper"

    frozen = platform_service.update_recommendation_state_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "recommendation_state": "approved_paper",
            "action_type": "freeze_recommendation",
            "lock_recommendation": True,
            "summary": "Paper-only recommendation frozen after committee review.",
            "rationale": "Freeze the paper recommendation until the liquidity concern is cleared.",
        },
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )
    assert frozen["recommendation_state"]["locked"] is True

    resolved = platform_service.resolve_dossier_comment_service(
        dossier["dossier_id"],
        comment["comment"]["comment_id"],
        {"rationale": "Liquidity review complete and acceptable for paper deployment."},
        user_context={"user_id": "reviewer-1", "role": "reviewer"},
        store=store,
    )
    assert resolved["comment"]["status"] == "resolved"
    assert resolved["review_summary"]["unresolved_concern_count"] == 0

    dossier_view = platform_service.get_dossier_view(
        dossier["dossier_id"],
        user_context={"user_id": "reviewer-1", "role": "reviewer"},
        store=store,
    )
    assert dossier_view["review_summary"]["unresolved_concern_count"] == 0
    assert dossier_view["assignment_summary"]["primary_reviewer"]["assignee_placeholder"] == "reviewer-1"
    assert dossier_view["committee_decision"]["decision_status"] == "approved_with_conditions"
    assert dossier_view["recommendation_state"]["state"] == "approved_paper"
    assert dossier_view["recommendation_state"]["locked"] is True

    export = platform_service.create_dossier_export_service(
        dossier["dossier_id"],
        {"pack_type": "ic_memo_pack"},
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )["export"]
    section_keys = [item["section_key"] for item in export["ordered_sections"]]
    assert "committee_decision" in section_keys
    assert "recommendation_state" in section_keys
    assert "reviewer_assignments" in section_keys


def test_phase8c_workflow_actions_handle_escalation_downgrade_and_concern_resolution() -> None:
    store = PlatformStore(use_memory=True)
    workspace, workflow, dossier = _prepare_workspace(store)

    platform_service.create_dossier_comment_service(
        dossier["dossier_id"],
        {
            "comment_type": "fragility_concern",
            "severity": "critical",
            "body": "Fragility rose after the latest event window.",
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )

    escalated = platform_service.execute_workflow_action_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "action_type": "escalate_to_committee",
            "rationale": "Need committee review before the next stage.",
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )
    assert escalated["workflow"]["status"] == "in_review"
    approvals = store.list_approval_requests(
        workflow_id=workflow["workflow_id"],
        dossier_id=dossier["dossier_id"],
    )
    assert any(item["requested_role"] == "committee" and item["status"] == "pending" for item in approvals)
    assert escalated["audit_event"]["event_type"] == "escalated_to_committee"

    downgraded = platform_service.execute_workflow_action_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "action_type": "downgrade_to_watch",
            "rationale": "Material fragility concern requires a watch-only downgrade.",
        },
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )
    assert downgraded["dossier"]["metadata"]["recommendation_state"]["state"] == "watch_only"

    resolved = platform_service.execute_workflow_action_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "action_type": "resolve_concerns",
            "rationale": "Concerns were addressed in follow-up review.",
        },
        user_context={"user_id": "reviewer-1", "role": "reviewer"},
        store=store,
    )
    assert resolved["dossier"]["metadata"]["review_summary"]["unresolved_concern_count"] == 0

    timeline = platform_service.list_workflow_timeline_service(
        workflow["workflow_id"],
        user_context={"user_id": "reviewer-1", "role": "reviewer"},
        store=store,
    )["timeline"]
    event_types = {item["event_type"] for item in timeline}
    assert "escalated_to_committee" in event_types
    assert "downgraded_to_watch" in event_types
    assert "workflow_resolve_concerns" in event_types


def test_phase8c_routes_enforce_tenant_safe_collaboration_access() -> None:
    _reset_platform_store(platform_store)
    workspace_one, workflow_one, dossier_one = _prepare_workspace(platform_store)
    workspace_two, workflow_two, dossier_two = _bootstrap_workspace(platform_store)
    platform_service.create_membership_service(
        {
            "user_id": "analyst-foreign",
            "workspace_id": workspace_two["workspace_id"],
            "organization_id": workspace_two["organization_id"],
            "role": "analyst",
        },
        store=platform_store,
    )

    with TestClient(app) as client:
        add_comment = client.post(
            f"/platform/dossiers/{dossier_one['dossier_id']}/comments",
            json={
                "comment_type": "evidence_gap",
                "severity": "watch",
                "body": "Need clearer evidence before escalation.",
            },
            headers=_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        assert add_comment.status_code == 200
        comment_id = add_comment.json()["comment"]["comment_id"]

        assignment = client.post(
            f"/platform/workflows/{workflow_one['workflow_id']}/assignments",
            json={
                "dossier_id": dossier_one["dossier_id"],
                "slot_type": "risk_reviewer",
                "assignee_placeholder": "reviewer-1",
            },
            headers=_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        assert assignment.status_code == 200

        decision = client.post(
            f"/platform/workflows/{workflow_one['workflow_id']}/committee-decision",
            json={
                "dossier_id": dossier_one["dossier_id"],
                "decision_status": "watch",
                "recommendation_state": "watch_only",
                "summary": "Keep this in watch-only mode for now.",
            },
            headers=_headers(
                "committee-1",
                "committee",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        assert decision.status_code == 200

        own_comments = client.get(
            f"/platform/dossiers/{dossier_one['dossier_id']}/comments",
            headers=_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        assert own_comments.status_code == 200
        assert own_comments.json()["review_summary"]["unresolved_concern_count"] == 1

        foreign_comments = client.get(
            f"/platform/dossiers/{dossier_one['dossier_id']}/comments",
            headers=_headers(
                "analyst-foreign",
                "analyst",
                workspace_id=workspace_two["workspace_id"],
            ),
        )
        assert foreign_comments.status_code == 403

        resolved = client.post(
            f"/platform/dossiers/{dossier_one['dossier_id']}/comments/{comment_id}/resolve",
            json={"rationale": "Addressed during review."},
            headers=_headers(
                "reviewer-1",
                "reviewer",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        assert resolved.status_code == 200

        own_review_summary = client.get(
            f"/platform/workflows/{workflow_one['workflow_id']}/review-summary",
            headers=_headers(
                "reviewer-1",
                "reviewer",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        assert own_review_summary.status_code == 200
        assert own_review_summary.json()["review_summary"]["thread_summary"]["total_comments"] >= 1

        committee_snapshot = client.get(
            f"/platform/workflows/{workflow_one['workflow_id']}/committee-decision",
            headers=_headers(
                "committee-1",
                "committee",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        assert committee_snapshot.status_code == 200
        assert committee_snapshot.json()["committee_decision"]["decision_status"] == "watch"

        recommendation = client.post(
            f"/platform/workflows/{workflow_one['workflow_id']}/recommendation-state",
            json={
                "dossier_id": dossier_one["dossier_id"],
                "recommendation_state": "watch_only",
                "action_type": "freeze_recommendation",
                "lock_recommendation": True,
                "summary": "Freeze watch-only until new evidence arrives.",
            },
            headers=_headers(
                "committee-1",
                "committee",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        assert recommendation.status_code == 200
        assert recommendation.json()["recommendation_state"]["locked"] is True

        audit = platform_service.list_workflow_timeline_service(
            workflow_one["workflow_id"],
            user_context={
                "user_id": "committee-1",
                "role": "committee",
                "workspace_id": workspace_one["workspace_id"],
            },
            store=platform_store,
        )["timeline"]
        event_types = {item["event_type"] for item in audit}
        assert "comment_added" in event_types
        assert "comment_resolved" in event_types
        assert "assignment_updated" in event_types
        assert "committee_decision_recorded" in event_types
        assert "recommendation_frozen" in event_types
