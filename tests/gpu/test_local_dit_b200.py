"""B200 (sm_100a, card ``b200_sm100``): the local atom DiT blocks' ``"cuda"`` path (``kernels/integrations/local_dit_b200.py``,
``local_dit_esm_b200.py``) is reproducible ACROSS PROCESSES: two fresh interpreters given the same weights and inputs produce the same output
and (training) the same input / weight gradients, bit for bit; and within one process two identical calls agree bit for bit.

Background (2026-11-18): an earlier probe saw two processes disagree on the ``to_out`` / ``squeeze`` weight gradients and on Chai-1's
inference output. The cause was the probe, not a kernel: the modules' default (``legacy``) initialisation draws its truncated normals from
NumPy's global random state (``primitives._trunc_normal_init`` via scipy), which ``torch.manual_seed`` does not seed, so each process built
different weights. Here the weights are set explicitly from a seeded torch generator, so any cross-process difference is the kernels'
(uninitialised workspace reads, atomics order, launch-order or stream races)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only"),
]

#: (class, atoms, samples, dtype, mode, input stream): the paths of the six classes, inference and training, fp32 and master (bf16)
CASES = [
    ("AtomDiTProtenix", 1024, 4, "float32", "train", "atom_single"),
    ("AtomDiTProtenix", 1024, 4, "master", "train", "atom_single"),
    ("AtomDiTProtenix", 1024, 5, "float32", "eval", "atom_single_shared_cond"),
    ("AtomDiTProtenix", 2048, 1, "master", "eval", "atom_single"),
    ("AtomDiTOpenFold3", 1024, 2, "float32", "train", "atom_single"),
    ("AtomDiTBoltz2", 1024, 2, "master", "train", "atom_single"),
    ("AtomDiTRoseTTAFold3", 1024, 2, "float32", "train", "atom_single"),
    ("AtomDiTChai1", 1024, 1, "float32", "eval", "atom_single"),
    ("AtomDiTChai1", 1024, 2, "master", "train", "atom_single"),
    ("AtomDiTESMFold2Pair", 1024, 2, "float32", "train", "atom_single"),
    ("AtomDiTESMFold2Pair", 1024, 1, "master", "eval", "atom_single"),
]

_SCRIPT = r"""
import hashlib, json, sys
import torch
from miniworld_engine.modules import atom_dit as ad

def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

def run(cls, length, samples, dt, mode, stream):
    torch.manual_seed(0)
    m = getattr(ad, cls)().cuda().set_backend("cuda")
    gp = torch.Generator("cuda").manual_seed(7)
    with torch.no_grad():  # explicit weights (the default init draws from NumPy's unseeded global state)
        for p in m.parameters():
            if p.dim() >= 2:
                p.copy_(torch.randn(p.shape, device="cuda", generator=gp) * p.shape[-1] ** -0.5)
            else:
                p.add_(0.1 * torch.randn(p.shape, device="cuda", generator=gp))
    g = torch.Generator("cuda").manual_seed(0)
    dtype = torch.bfloat16 if dt == "master" else torch.float32
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=dtype, device="cuda", generator=g, stream=stream)
    train = mode == "train"
    m.train(train)
    for k in ("single", "pair"):
        inp[k] = inp[k].detach().requires_grad_(train)
    calls = []
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dt == "master"), torch.set_grad_enabled(train):
            out = m(**inp)
        r = {"out": digest(out)}
        if train:
            gd = torch.Generator("cuda").manual_seed(1)
            out.backward(torch.randn(out.shape, device="cuda", dtype=out.dtype, generator=gd))
            for k in ("single", "pair"):
                if inp[k].grad is not None:
                    r["d" + k] = digest(inp[k].grad)
                    inp[k].grad = None
            for n, p in m.named_parameters():
                if p.grad is not None:
                    r["g." + n] = digest(p.grad)
                    p.grad = None
        calls.append(r)
    return calls

res = {" ".join(map(str, c)): run(*c) for c in json.loads(sys.argv[1])}
print("RESULT " + json.dumps(res))
"""


def _process(cases) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)  # each process its own hash seed (a dict / set order must not reach the numbers)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, json.dumps(cases)], env=env, capture_output=True, text=True, timeout=1200, check=False
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT ") :])


def test_local_dit_cuda_is_reproducible_across_processes() -> None:
    first, second = _process(CASES), _process(CASES)
    bad = []
    for case, (a1, a2) in first.items():
        if a1 != a2:  # two calls in one process
            bad.append(f"{case}: repeated call differs in {[k for k in a1 if a1[k] != a2.get(k)]}")
        b1 = second[case][0]
        if a1 != b1:  # first call of two processes
            bad.append(f"{case}: processes differ in {[k for k in a1 if a1[k] != b1.get(k)]}")
    assert not bad, "\n".join(bad)
