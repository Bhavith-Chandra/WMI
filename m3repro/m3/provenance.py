"""Provenance stamping: every result file records exactly what produced it.

stamp(args) -> dict with git commit (+dirty flag), argv, library versions, UTC time and SHA-256 of
every checkpoint/data file the run read. Hashes are cached by (path, size, mtime) so re-stamping is cheap.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = Path("/home/user/ext/m3-market-microstructure")
INPUTS = {
    "tokenizer": REL / "tokenizer/base/best.pt",
    "tiny": REL / "tiny/best.pt",
    "small": REL / "small/best.pt",
    "base": REL / "base/best.pt",
    "example_prompt_ids": REL / "examples/prompt_ids.npy",
    "example_conditioning": REL / "examples/conditioning.npz",
    "example_metadata": REL / "examples/example_metadata.json",
}
_CACHE = ROOT / "results/.hash_cache.json"


def sha256(path: Path) -> str:
    path = Path(path)
    st = path.stat()
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    cache = json.loads(_CACHE.read_text()) if _CACHE.exists() else {}
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    cache[key] = h.hexdigest()
    _CACHE.parent.mkdir(exist_ok=True)
    _CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]


def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def stamp(used: list[str]) -> dict:
    import numpy
    import torch
    import transformers

    return dict(
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain", "--", ".")),
        argv=sys.argv,
        utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        versions=dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
                      numpy=numpy.__version__),
        inputs={k: dict(path=str(INPUTS[k]), sha256=sha256(INPUTS[k])) for k in used},
        data_scope="ONLY the released held-out example window (split=test, 2025-12-01) is real market data; "
                   "everything else is model-generated.",
    )


def load_ckpt_meta(path) -> dict:
    """Checkpoint metadata only (weights_only + mmap: no code execution, no full copy in RAM)."""
    import torch

    d = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    return {k: v for k, v in d.items() if "state_dict" not in k}
