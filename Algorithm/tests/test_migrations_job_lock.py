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
