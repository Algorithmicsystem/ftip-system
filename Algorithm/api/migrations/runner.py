from __future__ import annotations

def apply_migrations() -> None:
    # Legacy shim to preserve existing imports; real work lives in api.db.apply_migrations
    from api import db

    db.apply_migrations()
