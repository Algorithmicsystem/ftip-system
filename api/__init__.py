from __future__ import annotations

from pathlib import Path
import pkgutil

# Prefer local application modules under Algorithm/api when resolving `api.*` imports
# (e.g. `uvicorn api.main:app` from the repository root).
__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
_algorithm_api_path = Path(__file__).resolve().parent.parent / "Algorithm" / "api"
if _algorithm_api_path.exists():
    __path__.append(str(_algorithm_api_path))
