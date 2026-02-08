from __future__ import annotations

import argparse
import os
import sys
import time

import psycopg


def _connect_once(db_url: str) -> None:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            row = cur.fetchone()
    if not row or row[0] != 1:
        raise RuntimeError("database connectivity check failed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check database connectivity.")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is required for db_check.", file=sys.stderr)
        return 1

    for attempt in range(1, args.retries + 1):
        try:
            _connect_once(db_url)
            print("Database connectivity OK.")
            return 0
        except Exception as exc:
            if attempt >= args.retries:
                print(f"Database connectivity failed: {exc}", file=sys.stderr)
                return 1
            time.sleep(args.sleep)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
