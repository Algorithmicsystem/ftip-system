from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    db_enabled = os.getenv("FTIP_DB_ENABLED")
    db_url = os.getenv("DATABASE_URL")

    print(f"FTIP_DB_ENABLED={db_enabled or ''}")
    print(f"DATABASE_URL={db_url or ''}")

    if not db_enabled or not db_url:
        print("Missing required env vars. Copy .env.example to .env.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
