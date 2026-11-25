"""Where a record was made: engine version and commit, torch / CUDA / driver, GPU, node, card key."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from functools import cache
from pathlib import Path

CARDS = {"h100_sm90": "h100", "a100_sm80": "a100", "5090_sm120": "rtx5090", "b200_sm100": "b200"}  # engine card key -> short card name used in file names


def _engine_root() -> Path:
    from foldjet_engine.measure import config  # foldjet_engine is a namespace package: no __file__

    return Path(config.__file__).resolve().parents[3]


@cache
def engine_version() -> str:
    """The engine release the process imports: MWE_VERSION, else ``version`` of the pyproject next to the source tree, else the wheel's metadata."""
    if os.environ.get("MWE_VERSION"):
        return os.environ["MWE_VERSION"]
    try:
        text = (_engine_root() / "pyproject.toml").read_text()
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version

        return version("miniworld-engine")
    except Exception:  # recorded as unknown
        return "unknown"


@cache
def engine_commit() -> str:
    """MWE_COMMIT, else the git HEAD of the source tree the engine is imported from, else ``unknown`` (an installed wheel: set MWE_COMMIT)."""
    if os.environ.get("MWE_COMMIT"):
        return os.environ["MWE_COMMIT"]
    try:
        return subprocess.run(
            ["git", "-C", str(_engine_root()), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20, check=True
        ).stdout.strip()
    except Exception:  # recorded as unknown
        return "unknown"


def card_key() -> str:
    from foldjet_engine.measure.config import result_gpu

    return result_gpu()


def card_name() -> str:
    return CARDS.get(card_key(), card_key())


def describe() -> dict:
    """The provenance block written next to every raw record file."""
    import torch

    from foldjet_engine.measure import config

    env = dict(config.environment())
    return {
        "engine_version": engine_version(),
        "commit": engine_commit(),
        "card": card_name(),
        "card_key": card_key(),
        "gpu": env.get("gpu") or "",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_driver": env.get("cuda_driver"),
        "driver": env.get("driver"),
        "triton": env.get("triton"),
        "python": platform.python_version(),
        "node": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "paper_bench_commit": _own_commit(),
    }


def _own_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20, check=True
        ).stdout.strip()
    except Exception:  # recorded as unknown
        return "unknown"
