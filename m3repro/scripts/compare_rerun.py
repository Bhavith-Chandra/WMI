"""Determinism check: compare regenerated results/*.json against the committed version (git HEAD).

python -I scripts/compare_rerun.py [file ...]
Ignores provenance and wall-clock fields. Prints max |diff| per file and the first differing keys.
Exit code 1 if any numeric field differs by more than 1e-9 (relative) or structure differs.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"_provenance", "wall_s", "gen_seconds", "elapsed_seconds"}


def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in SKIP:
                continue
            yield from flat(v, f"{pre}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{pre}[{i}]")
    else:
        yield pre, x


def main(files):
    bad = False
    for f in files:
        rel = f"m3repro/results/{Path(f).name}"
        old = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True)
        if old.returncode:
            print(f"{Path(f).name}: not in HEAD (new file)")
            continue
        a = dict(flat(json.loads(old.stdout)))
        b = dict(flat(json.loads(Path(f).read_text())))
        if isinstance(a.get(".rows"), type(None)) and ".rows[0].size" in b:  # legacy list layout -> {"rows":...}
            a = {f".rows{k}": v for k, v in a.items()}
        keys = set(a) | set(b)
        diffs = []
        for k in sorted(keys):
            x, y = a.get(k, "<missing>"), b.get(k, "<missing>")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                d = abs(x - y) / max(1.0, abs(x))
                if d > 1e-9:
                    diffs.append((d, k, x, y))
            elif x != y:
                diffs.append((float("inf"), k, x, y))
        diffs.sort(reverse=True)
        status = "IDENTICAL" if not diffs else f"{len(diffs)} DIFFERENT fields"
        print(f"{Path(f).name}: {status}")
        for d in diffs[:5]:
            print(f"   {d[1]}: {d[2]!r} -> {d[3]!r}")
        bad |= bool(diffs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(str(p) for p in (ROOT / "results").glob("*.json") if "bootstrap" not in p.name))
