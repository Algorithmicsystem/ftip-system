from api import migrations


class _FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql):
        self.executed.append(sql)


def test_job_lock_owner_migration_reads_sql_file():
    cur = _FakeCursor()
    migrations._migration_job_lock_owner(cur)

    assert cur.executed, "expected migration to run SQL"
    sql = cur.executed[0]
    assert "ftip_job_runs" in sql
    assert "prosperity_job_runs" in sql


def test_job_lock_owner_migration_registered():
    versions = [version for version, _ in migrations.MIGRATIONS]
    assert "004_job_lock_owner" in versions


def test_ftip_job_runs_columns_migration_registered():
    versions = [version for version, _ in migrations.MIGRATIONS]
    assert "005_ftip_job_runs_columns" in versions


def test_ftip_job_runs_lock_cleanup_migration_registered():
    versions = [version for version, _ in migrations.MIGRATIONS]
    assert "006_ftip_job_runs_lock_cleanup" in versions


def test_ftip_job_runs_columns_migration_adds_as_of_date():
    cur = _FakeCursor()
    migrations._migration_ftip_job_runs_columns(cur)

    assert cur.executed, "expected migration to run SQL"
    sql = cur.executed[0]
    assert "ftip_job_runs" in sql
    assert "as_of_date" in sql
