"""Regression tests for Phase 17 — migration registration and version column alignment."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 1. Migrations 041-044 are registered in MIGRATIONS list
# ---------------------------------------------------------------------------

def test_migrations_list_importable():
    from api.migrations import MIGRATIONS
    assert isinstance(MIGRATIONS, list)
    assert len(MIGRATIONS) > 0


def _migration_versions():
    from api.migrations import MIGRATIONS
    return [v for v, _ in MIGRATIONS]


def test_migration_041_registered():
    assert "041_provider_reliability_daily" in _migration_versions()


def test_migration_042_registered():
    assert "042_feature_provenance_daily" in _migration_versions()


def test_migration_043_registered():
    assert "043_symbol_linkage" in _migration_versions()


def test_migration_044_registered():
    assert "044_version_columns_text" in _migration_versions()


def test_migration_order_is_ascending():
    """Migration versions must appear in strictly ascending order."""
    versions = _migration_versions()
    tail = [v for v in versions if v.startswith("04")]
    assert tail == sorted(tail), f"Out-of-order: {tail}"


# ---------------------------------------------------------------------------
# 2. Migration SQL files exist on disk
# ---------------------------------------------------------------------------

def test_sql_file_041_exists():
    from pathlib import Path
    import api.migrations as m
    sql_path = Path(m.__file__).with_name("041_provider_reliability_daily.sql")
    assert sql_path.exists(), f"Missing: {sql_path}"


def test_sql_file_042_exists():
    from pathlib import Path
    import api.migrations as m
    sql_path = Path(m.__file__).with_name("042_feature_provenance_daily.sql")
    assert sql_path.exists()


def test_sql_file_043_exists():
    from pathlib import Path
    import api.migrations as m
    sql_path = Path(m.__file__).with_name("043_symbol_linkage.sql")
    assert sql_path.exists()


def test_sql_file_044_exists():
    from pathlib import Path
    import api.migrations as m
    sql_path = Path(m.__file__).with_name("044_version_columns_text.sql")
    assert sql_path.exists()


# ---------------------------------------------------------------------------
# 3. Migration 044 SQL content is correct
# ---------------------------------------------------------------------------

def test_migration_044_alters_signals_daily():
    from pathlib import Path
    import api.migrations as m
    sql = Path(m.__file__).with_name("044_version_columns_text.sql").read_text()
    assert "signals_daily" in sql
    assert "signal_version" in sql
    assert "TYPE TEXT" in sql


def test_migration_044_alters_features_daily():
    from pathlib import Path
    import api.migrations as m
    sql = Path(m.__file__).with_name("044_version_columns_text.sql").read_text()
    assert "features_daily" in sql
    assert "feature_version" in sql


def test_migration_044_covers_intraday_tables():
    from pathlib import Path
    import api.migrations as m
    sql = Path(m.__file__).with_name("044_version_columns_text.sql").read_text()
    assert "signals_intraday" in sql
    assert "features_intraday" in sql


def test_migration_044_uses_safe_cast():
    """Must use USING clause to avoid cast failure on existing integer values."""
    from pathlib import Path
    import api.migrations as m
    sql = Path(m.__file__).with_name("044_version_columns_text.sql").read_text()
    assert "USING" in sql


# ---------------------------------------------------------------------------
# 4. Migration 041–043 SQL content spot-checks
# ---------------------------------------------------------------------------

def test_migration_041_creates_provider_reliability_daily():
    from pathlib import Path
    import api.migrations as m
    sql = Path(m.__file__).with_name("041_provider_reliability_daily.sql").read_text()
    assert "provider_reliability_daily" in sql
    assert "CREATE TABLE" in sql.upper()


def test_migration_042_creates_feature_provenance_daily():
    from pathlib import Path
    import api.migrations as m
    sql = Path(m.__file__).with_name("042_feature_provenance_daily.sql").read_text()
    assert "feature_provenance_daily" in sql
    assert "CREATE TABLE" in sql.upper()


def test_migration_043_creates_symbol_linkage():
    from pathlib import Path
    import api.migrations as m
    sql = Path(m.__file__).with_name("043_symbol_linkage.sql").read_text()
    assert "symbol_linkage" in sql
    assert "CREATE TABLE" in sql.upper()
