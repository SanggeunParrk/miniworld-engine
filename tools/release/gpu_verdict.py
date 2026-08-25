"""Turn "the author ran it on their card" into a dated, checkable artifact.

Both CI jobs run on `ubuntu-latest`. The one step that mentions the GPU runs
`pytest --collect-only -m gpu`, which proves GPU tests can be *collected*. 103 kernels, none
executed by anything automatic -- so every correctness claim in this repository rested on a
person remembering to run something.

A self-hosted GPU runner is not available here, so the gate is Slurm-shaped instead: this script
runs the GPU stages on a compute node and writes `verdicts/<version>-<arch>.json` recording what
ran, on which device, at which commit. `tests/test_gpu_verdict_is_current.py` then refuses a
release whose version has no passing verdict. Day-to-day commits stay green; an unverified
release does not.

Stages are read from machine-readable output -- JUnit XML from pytest, `--json` from run_all --
never from the human summary. A check that re-parses prose breaks when the prose is reworded,
which is a check failing for the wrong reason.

    python tools/release/gpu_verdict.py [--out DIR] [--allow-fail]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    """`capture` as an explicit flag rather than `**kw`: forwarding kwargs to `subprocess.run`
    defeats its overloads, and the type checker then cannot tell `CompletedProcess[str]` from
    `CompletedProcess[bytes]`."""
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO, text=True, check=False, capture_output=capture)


def declared_version() -> str:
    match = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.MULTILINE)
    if not match:
        raise SystemExit("no `version = ...` in pyproject.toml")
    return match.group(1)


def _git(*args: str) -> str:
    return _run(["git", *args], capture=True).stdout.strip()


def device_facts() -> dict[str, str]:
    import torch

    from miniworld_engine.autotune.run_all import device_arch
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda"
    triton_version = ""
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        pass
    return {"device": name, "arch": device_arch(), "torch": torch.__version__,
            "cuda": torch.version.cuda or "", "triton": triton_version,
            "python": ".".join(str(p) for p in sys.version_info[:3])}


def pytest_gpu(report: Path) -> dict[str, object]:
    """Every gpu-marked test, counted from JUnit XML rather than from the summary line."""
    proc = _run(["python", "-m", "pytest", "tests/", "-m", "gpu", "-q", "--no-header",
                 "-p", "no:cacheprovider", f"--junitxml={report}"])
    if not report.is_file():
        return {"ran": False, "reason": f"pytest produced no report (exit {proc.returncode})"}
    root = ET.parse(report).getroot()
    node = root if root.tag == "testsuite" else root.find("testsuite")
    if node is None:
        return {"ran": False, "reason": "JUnit XML has no <testsuite>"}
    total = int(node.get("tests", "0"))
    bad = int(node.get("failures", "0")) + int(node.get("errors", "0"))
    skipped = int(node.get("skipped", "0"))
    return {"ran": True, "total": total, "failed": bad, "skipped": skipped,
            "passed": total - bad - skipped, "exit": proc.returncode}


def run_all_stage(out: Path) -> dict[str, object]:
    proc = _run(["python", "-m", "miniworld_engine.autotune.run_all", "--json", str(out)])
    if not out.is_file():
        return {"ran": False, "reason": f"run_all wrote no JSON (exit {proc.returncode})"}
    payload: dict[str, object] = json.loads(out.read_text())
    payload["ran"] = True
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "verdicts"), help="directory for the verdict file")
    ap.add_argument("--allow-fail", action="store_true",
                    help="write the verdict even when a stage failed (it records verdict=fail)")
    args = ap.parse_args(argv)

    scratch = Path(args.out) / ".stage"
    scratch.mkdir(parents=True, exist_ok=True)

    facts = device_facts()
    if facts["device"] == "no-cuda":
        raise SystemExit("no CUDA device visible -- a verdict must be produced on a GPU node")

    stages: dict[str, dict[str, object]] = {
        "pytest_gpu": pytest_gpu(scratch / "gpu-junit.xml"),
        "run_all": run_all_stage(scratch / "run_all.json"),
    }
    ok = (bool(stages["pytest_gpu"].get("ran"))
          and stages["pytest_gpu"].get("failed") == 0
          and bool(stages["run_all"].get("ran"))
          and stages["run_all"].get("failed") == 0
          and stages["run_all"].get("accounting_ok") is True)

    version = declared_version()
    verdict = {
        "version": version,
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "describe": _git("describe", "--tags", "--always"),
        "clean_tree": not _git("status", "--porcelain"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **facts,
        "stages": stages,
        "verdict": "pass" if ok else "fail",
    }
    path = Path(args.out) / f"{version}-{facts['arch']}.json"
    if not ok and not args.allow_fail:
        print(json.dumps(verdict, indent=2, sort_keys=True))
        print(f"\nNOT writing {path}: a stage failed. Re-run with --allow-fail to record the "
              f"failure as the verdict.", file=sys.stderr)
        return 1
    path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {path.relative_to(REPO)}  ->  {verdict['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
