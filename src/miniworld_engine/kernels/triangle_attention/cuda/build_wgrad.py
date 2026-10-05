"""Rebuild the qualified H100 shared-input weight backward. Requires CUTLASS 4.2."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

import torch
from miniworld_engine.kernels._nvcc import load_extension as load   # the lock-guarded torch load

ROOT = Path(__file__).resolve().parent
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
os.environ.setdefault("MAX_JOBS", "2")
cutlass = os.environ.get("CUTLASS_PATH")
if not cutlass:
    cutlass = next((str(p / "anthropic_adoption_20260919/cutlass-4.2")
                    for p in ROOT.parents if p.name == "runs"), None)
if not cutlass or not (Path(cutlass) / "include/cute/tensor.hpp").is_file():
    raise RuntimeError("Set CUTLASS_PATH to a CUTLASS 4.2 checkout")
build = ROOT / "build_wgrad"
build.mkdir(exist_ok=True)
previous = json.loads((ROOT / "wgrad_manifest.json").read_text())
module_name = "triattn_wgrad_shared_z_master"
binary = module_name + ".so"
module = load(name=module_name, sources=[str(ROOT / "wgrad.cu")],
              build_directory=str(build), extra_include_paths=[str(ROOT), str(Path(cutlass) / "include")],
              extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
              "--expt-extended-lambda", "-lineinfo", "-Xptxas=-v"], verbose=True)
shutil.copy2(module.__file__, ROOT / (binary + ".new"))
os.replace(ROOT / (binary + ".new"), ROOT / binary)
files = ["wgrad.cu", "wgrad_backward.py", "build_wgrad.py", "fa3_utils.h", binary]
data = dict(torch=str(torch.__version__), python_abi=sys.implementation.cache_tag, arch="sm_90a",
            module_name=module_name, binary=binary, lengths=previous["lengths"],
            partial_dtype="fp32", splits=previous["splits"], artifact="package_rebuild",
            files={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files})
(ROOT / "wgrad_manifest.json.new").write_text(json.dumps(data, indent=2) + "\n")
os.replace(ROOT / "wgrad_manifest.json.new", ROOT / "wgrad_manifest.json")
