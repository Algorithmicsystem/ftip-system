from fastapi.testclient import TestClient

from api.main import app


def test_legacy_assistant_ui_preserves_active_analysis_context() -> None:
    with TestClient(app) as client:
        html = client.get("/app").text
        js = client.get("/app/static/app.js").text

    assert 'id="assistant-analyze-active-label"' in html
    assert 'id="assistant-chat-active-label"' in html
    assert 'id="assistant-analyze-report"' in html
    assert "assistantActiveAnalysis" in js
    assert "ASSISTANT_ACTIVE_ANALYSIS_STORAGE_KEY" in js
    assert "session_id: pendingSessionId" in js
    assert "active_analysis: state.assistantActiveAnalysis" in js
