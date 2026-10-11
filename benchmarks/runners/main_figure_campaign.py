"""Measure the points of the main figure: every module at L = 384 tokens (atom blocks 4096 atoms), inference and training.

Drives ``bench.py`` once per (module, mode, implementation group) and collects the ``time=`` lines into one CSV. Run it with the
engine environment's python on an otherwise idle card; ``PREC=32`` measures the fp32 runs of the diffusion blocks. The Anthropic
rows need ``MINIWORLD_ANTHROPIC_ROOT`` (and, for TriMul, the build directory variables of docs/kernels/anthropic-trimul-payload.md)
in the environment. Draw the figure with ``main_figure_plot.py``.

usage: python benchmarks/runners/main_figure_campaign.py OUT.csv key [key ...]
keys: trimul1 trimul2 triattn trans apb opm pwa tdit bodit ldit swadit3
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(r"^(\S+) seq_len=(\d+) d_pair=(\d+) implementation=(\S+) time=([0-9.eE+-]+) ms")
# key: (target, common overrides, seq_len, implementations of the main run, Anthropic inference row)
JOBS = {
    "trimul1": ("triangle_multiplication", "d_pair=128", 384, ["pytorch", "cuequivariance", "miniworld"], True),
    "trimul2": ("triangle_multiplication_bidirectional", "d_pair=128", 384, ["pytorch", "cuequivariance", "miniworld"], True),
    "triattn": ("triangle_attention", "d_pair=128 +tri_d_hidden=128 +tri_n_head=4", 384,
                ["pytorch", "cuequivariance", "miniworld"], True),
    "trans": ("transition", "d_pair=128", 384, ["pytorch", "miniworld"], False),
    "apb": ("attention_pair_bias", "sweep_axis=seq_len +apb_n_head=8 d_single=384", 384,
            ["pytorch", "cuequivariance", "miniworld"], False),
    "opm": ("outer_product", "+n_msa=1024", 384, ["pytorch", "miniworld"], True),
    "pwa": ("msa_pair_weighted_averaging", "+n_msa=1024", 384, ["pytorch", "miniworld"], True),
    "tdit": ("dit", "+shared_cond=true", 384, ["pytorch", "miniworld"], True),
    "bodit": ("bias_only_dit", "", 384, ["pytorch", "miniworld"], False),
    "ldit": ("local_dit", "", 512, ["pytorch", "miniworld"], True),
    "swadit3": ("swa_dit", "compile=false +shared_cond=true", 512, ["miniworld"], True),
}


def run(rows: list, key: str, mode: str, label: str, impls: list[str], extra: str = "") -> None:
    target, common, seq, _impls, _anthropic = JOBS[key]
    precision = os.environ.get("PREC", "")
    mask = " mask_prob=0.2" if key == "tdit" and mode == "inference" else ""
    graph = " cudagraph=manual" if mode == "training" and "cudagraph=disabled" not in extra else ""
    if "compile=true" in extra:
        common = common.replace("compile=false", "")
    cmd = (f"{sys.executable} benchmarks/runners/bench.py target={target} level=module mode={mode} {common}{mask}{graph}"
           f"{f' precision={precision}' if precision else ''} min_seq_len={seq} max_seq_len={seq} seq_len_step=128 "
           f"'implementations=[{','.join(impls)}]' {extra}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, check=False)
    found = 0
    for line in (res.stdout + res.stderr).splitlines():
        m = LINE.match(line.strip())
        if m:
            rows.append((key, mode, m.group(4), float(m.group(5)), label))
            found += 1
    print(f"{key} {mode} {label}: {found} rows, rc={res.returncode}, {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    out, keys = sys.argv[1], sys.argv[2:]
    rows: list = []
    for key in keys:
        _target, common, _seq, impls, anthropic = JOBS[key]
        for mode in ("inference", "training"):
            if key == "ldit" and mode == "training":
                # the compiled PyTorch backward is refused under a CUDA graph (repeatability check): time it without one
                run(rows, key, mode, "main", ["miniworld"])
                run(rows, key, mode, "ptc_nograph", ["pytorch"], "cudagraph=disabled")
            else:
                run(rows, key, mode, "main", impls)
            if key == "swadit3":
                run(rows, key, mode, "ptc", ["pytorch"], "compile=true")
            if anthropic and mode == "inference":
                run(rows, key, mode, "anthropic", ["anthropic"], "" if "compile=false" in common else "compile=false")
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["module", "mode", "implementation", "latency_ms", "run"])
            writer.writerows(rows)


if __name__ == "__main__":
    main()
