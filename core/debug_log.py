"""Log runtime opsional SpaceAx (aktifkan dengan SPACEAX_DEBUG=1)."""
import json
import os
import time

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(_BASE, "data", "logs", "spaceax_runtime.ndjson")


def spaceax_log(location: str, message: str, data: dict | None = None) -> None:
    if os.environ.get("SPACEAX_DEBUG", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        payload = {
            "ts": int(time.time() * 1000),
            "where": location,
            "msg": message,
            "data": data or {},
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
