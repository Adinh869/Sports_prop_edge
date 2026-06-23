"""Convert UTF-16 / null-byte corrupted project files back to UTF-8."""
from __future__ import annotations

from pathlib import Path

root = Path(__file__).resolve().parents[1]
fixed = 0
patterns = ("*.py", "*.ps1", "*.csv", "*.bat", "*.mdc", "*.json", "*.toml", "*.md", "*.txt")
skip_dirs = {".venv", "__pycache__", ".git"}

for pattern in patterns:
    for path in root.rglob(pattern):
        if skip_dirs.intersection(path.parts):
            continue
        data = path.read_bytes()
        if b"\x00" not in data and not data.startswith(b"\xff\xfe"):
            continue
        enc = "utf-16" if data.startswith(b"\xff\xfe") else "utf-16-le"
        path.write_text(data.decode(enc), encoding="utf-8", newline="\n")
        print("fixed", path)
        fixed += 1

old = root / "tools" / "daily_sync.py"
if old.exists():
    old.unlink()
    print("removed", old)

print("done", fixed, "files")
