"""Provenance: every result file records exactly what produced it."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]


def run_record(config: dict) -> dict:
    import numpy
    try:
        import torch
        tv = torch.__version__
    except ImportError:  # pragma: no cover
        tv = None
    return {
        "config": config,
        "config_hash": config_hash(config),
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": tv,
        "machine": platform.machine(),
    }


def save_result(experiment: str, record: dict, metrics: dict, extra: dict | None = None) -> Path:
    out_dir = REPO_ROOT / "results" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = record["config"].get("seed", "na")
    path = out_dir / f"{record['config_hash']}_seed{seed}.json"
    payload = {"provenance": record, "metrics": metrics, **(extra or {})}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    return path


def load_results(experiment: str) -> list[dict]:
    d = REPO_ROOT / "results" / experiment
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []
