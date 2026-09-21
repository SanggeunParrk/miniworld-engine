"""CPU-only integrity checks: the vendored Anthropic headers this kernel builds against, portable paths, recorded measurements.

    python verify_package.py        # prints OK, or raises

Runs anywhere: it never imports torch, never touches a GPU and never shells out to nvcc.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

# 1. the build's only include is the sibling experiment's vendored Anthropic v5 header set, unmodified
vendor = root.parent / "trimul_b7b12" / "vendor" / "anthropic_v5"
up = json.loads((vendor / "UPSTREAM.json").read_text())
assert up["revision"] == "f4f62fa6592ae4938d49b1757bea0cfeff9f468e", up["revision"]
for name in ("csrc/tmn_ptx.cuh", "csrc/tmn_kernels.cuh", "csrc/common/tmn_math.cuh"):
    assert sha(vendor / name) == up["files"][name], name
assert 'csrc="$here/../trimul_b7b12/vendor/anthropic_v5/csrc"' in (root / "build.sh").read_text()

# 2. the kernel pulls in nothing else, and its entry points are the ones the harness loads
src = (root / "src" / "transition_bwd.cu").read_text()
fsrc = (root / "src" / "transition_fwd.cu").read_text()
for t in (src, fsrc):
    assert re.findall(r'^\s*#include\s+"([^"]+)"', t, re.M) == ["tmn_kernels.cuh"], "unexpected include"
assert not re.findall(r"^\s*#include\s+<(?!cu)", src, re.M), "unexpected system include"
for entry in ("transition_bwd_fused", "reduce_partials"):
    assert f'extern "C" __global__ void' in src and entry in src, entry
    assert entry in (root / "bench.py").read_text(), entry
assert "atomicAdd" not in src, "the kernel is meant to be atomic-free, hence bit-reproducible"

# 3. nothing in the package hard-codes a checkout path, and the Python parses
for path in sorted(root.rglob("*.py")):
    text = path.read_text()
    ast.parse(text, filename=str(path))
    if path.name != "verify_package.py":            # this file names the pattern it looks for
        assert "/home/" not in text, path
for path in (root / "build.sh", root / "README.md"):
    assert "/home/" not in path.read_text(), path

# 4. the recorded measurements say what the README says
rec = {L: json.loads((root / "records" / f"bench-L{L}.json").read_text()) for L in (384, 768)}
for L, r in rec.items():
    assert r["ndw"] == 64 and r["ndx"] == 68 and r["dw_repl"] == 8, (L, r["ndw"], r["ndx"])
    assert r["lmem"] == 0 and r["regs"] <= 255, (L, r["regs"], r["lmem"])
    assert all(r["reproducible"].values()), (L, r["reproducible"])
    for name, e in r["fused_vs_fp32"].items():
        assert e["finite"] and e["rel_rms"] < 5e-3, (L, name, e["rel_rms"])
        ref = r["engine_vs_fp32"][name]["rel_rms"]          # no looser than the path it replaces
        assert e["rel_rms"] <= ref * 1.15 + 1e-4, (L, name, e["rel_rms"], ref)
    fused, engine = r["time_us"]["fused"]["median"], r["time_us"]["engine"]["median"]
    assert engine / fused > 1.9, (L, fused, engine)
assert rec[384]["time_us"]["fused"]["median"] < 430 and rec[768]["time_us"]["fused"]["median"] < 1600

# 4b. the forward records
for L in (384, 768):
    fr = json.loads((root / "records" / f"fwd-L{L}.json").read_text())
    assert all(fr["reproducible"].values()), (L, fr["reproducible"])
    fo = fr["cmp"]["fused_vs_fp32"]["out"]["rel_rms"]
    assert fo <= fr["cmp"]["engine_vs_fp32"]["out"]["rel_rms"] + 1e-5, (L, fo)   # no looser than the three-kernel path
    assert fr["time_us"]["engine"]["median"] / fr["time_us"]["fused"]["median"] > 1.6, L

# 5. the module-level records agree with the op-level ones and with the README
for L in (384, 768):
    mr = json.loads((root / "records" / f"module-L{L}.json").read_text())
    assert mr["speedup"]["fused-backward"] > 1.4 and mr["speedup"]["fused-both"] > 1.65, (L, mr["speedup"])
    for name, e in mr["agreement"].items():
        assert e["rel_rms"] < 1e-3, (L, name, e["rel_rms"])
        assert mr["engine_self"][name] < 1e-5, (L, name)          # the engine path's own run-to-run noise
    assert mr["ms"]["engine"]["median"] > mr["ms"]["fused-backward"]["median"]

# 6. the ratio sweep backs the compiled-in default
ratio = {8: rec[384]["time_us"]["fused"]["median"]}
for path in (root / "records").glob("ratio-r*-L384.json"):
    r = json.loads(path.read_text())
    ratio[r["dw_repl"]] = r["time_us"]["fused"]["median"]
assert min(ratio, key=ratio.get) == 8, ratio

print("OK  upstream", up["revision"][:12], "| L384", round(ratio[8]), "us |",
      "ratio sweep", {k: round(v) for k, v in sorted(ratio.items())})
