from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.env import load_project_env

load_project_env(ROOT)

from sports_prop_edge.integrations.soccer_client import _api_get, _api_key

out = ROOT / "data" / "cache" / "_api_football_check.txt"
lines = []
try:
    key = _api_key()
    lines.append(f"env_loaded: {bool(key)}")
    lines.append(f"key_len: {len(key)}")
    lines.append(f"key_prefix: {key[:6]}...")
    data = _api_get("status")
    resp = data.get("response") or {}
    lines.append(f"api_ok: {bool(resp)}")
    lines.append(f"account: {resp}")
except Exception as exc:
    lines.append(f"error: {exc}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")
