"""Phase 18 — error envelope adoption tests.

Verifies that simple_error() is used consistently for locked-job and
computation-error responses across signals, features, prosperity, and
narrator helpers.
"""
from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# 1. simple_error() contract
# ---------------------------------------------------------------------------

def test_simple_error_shape():
    from api.errors import simple_error
    resp = simple_error("locked", "job is already in progress", status_code=409)
    body = json.loads(resp.body)
    assert body["error"] == "locked"
    assert body["detail"] == "job is already in progress"
    assert resp.status_code == 409


def test_simple_error_extra_merged():
    from api.errors import simple_error
    resp = simple_error("locked", "busy", status_code=409, extra={"run_id": "abc", "lock_owner": "host1"})
    body = json.loads(resp.body)
    assert body["run_id"] == "abc"
    assert body["lock_owner"] == "host1"


def test_err_response_shape():
    from api.errors import err_response
    resp = err_response("validation_error", "symbol required", status_code=400)
    body = json.loads(resp.body)
    assert body["error"]["type"] == "validation_error"
    assert body["error"]["message"] == "symbol required"
    assert "trace_id" in body["error"]
    assert body["trace_id"] == body["error"]["trace_id"]


def test_err_response_sets_trace_header():
    from api.errors import err_response
    resp = err_response("not_found", "missing", status_code=404)
    assert "X-Trace-Id" in resp.headers


# ---------------------------------------------------------------------------
# 2. signals.py imports simple_error (no raw JSONResponse locked pattern)
# ---------------------------------------------------------------------------

def test_signals_imports_simple_error():
    import importlib, ast, pathlib
    src = pathlib.Path("api/jobs/signals.py").read_text()
    assert "from api.errors import simple_error" in src


def test_signals_no_raw_locked_jsonresponse():
    import pathlib
    src = pathlib.Path("api/jobs/signals.py").read_text()
    assert 'content={"error": "locked"' not in src


def test_signals_uses_simple_error_for_lock():
    import pathlib
    src = pathlib.Path("api/jobs/signals.py").read_text()
    assert 'simple_error("locked"' in src


# ---------------------------------------------------------------------------
# 3. features.py imports simple_error (no raw JSONResponse locked pattern)
# ---------------------------------------------------------------------------

def test_features_imports_simple_error():
    import pathlib
    src = pathlib.Path("api/jobs/features.py").read_text()
    assert "from api.errors import simple_error" in src


def test_features_no_raw_locked_jsonresponse():
    import pathlib
    src = pathlib.Path("api/jobs/features.py").read_text()
    assert 'content={"error": "locked"' not in src


def test_features_uses_simple_error_for_lock():
    import pathlib
    src = pathlib.Path("api/jobs/features.py").read_text()
    assert 'simple_error("locked"' in src


# ---------------------------------------------------------------------------
# 4. prosperity.py imports simple_error (no raw JSONResponse locked pattern)
# ---------------------------------------------------------------------------

def test_prosperity_imports_simple_error():
    import pathlib
    src = pathlib.Path("api/jobs/prosperity.py").read_text()
    assert "from api.errors import simple_error" in src


def test_prosperity_no_raw_locked_jsonresponse():
    import pathlib
    src = pathlib.Path("api/jobs/prosperity.py").read_text()
    assert 'content={"error": "locked"' not in src


def test_prosperity_uses_simple_error_for_lock():
    import pathlib
    src = pathlib.Path("api/jobs/prosperity.py").read_text()
    assert 'simple_error("locked"' in src


def test_prosperity_computation_error_shape():
    """computation error dicts now carry 'error' and 'detail' keys."""
    import pathlib
    src = pathlib.Path("api/jobs/prosperity.py").read_text()
    assert '"computation_error"' in src
    assert '"detail"' in src


def test_prosperity_status_lowercase_error():
    """status value in per-symbol error dict is lowercase 'error'."""
    import pathlib
    src = pathlib.Path("api/jobs/prosperity.py").read_text()
    assert '"status": "error"' in src


# ---------------------------------------------------------------------------
# 5. narrator/routes.py internal helper error dicts are normalised
# ---------------------------------------------------------------------------

def test_narrator_routes_no_bare_error_str():
    import pathlib
    src = pathlib.Path("api/narrator/routes.py").read_text()
    assert '{"error": str(exc)}' not in src


def test_narrator_routes_uses_internal_error_key():
    import pathlib
    src = pathlib.Path("api/narrator/routes.py").read_text()
    assert '"internal_error"' in src


def test_narrator_routes_has_detail_key():
    import pathlib
    src = pathlib.Path("api/narrator/routes.py").read_text()
    assert '"detail": str(exc)' in src
