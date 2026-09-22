"""Rebuild the qualified H100 projection backward. Requires CUTLASS 4.2."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
os.environ.setdefault("MAX_JOBS", "2")
cutlass = os.environ.get("CUTLASS_PATH")
if not cutlass:
    cutlass = next((str(p / "anthropic_adoption_20260919/cutlass-4.2")
                    for p in ROOT.parents if p.name == "runs"), None)
if not cutlass or not (Path(cutlass) / "include/cute/tensor.hpp").is_file():
    raise RuntimeError("Set CUTLASS_PATH to a CUTLASS 4.2 checkout")
build = ROOT / "build"
build.mkdir(exist_ok=True)
module = load(name="triattn_projection_bwd", sources=[str(ROOT / "projection_dgrad.cu")],
              build_directory=str(build), extra_include_paths=[str(ROOT), str(Path(cutlass) / "include")],
              extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
              "--expt-extended-lambda", "-lineinfo", "-Xptxas=-v"], verbose=True)
shutil.copy2(module.__file__, ROOT / "triattn_projection_bwd.so")
files = ["__init__.py", "projection_dgrad.cu", "fa3_utils.h", "build_native.py", "triattn_projection_bwd.so"]
data = dict(torch=str(torch.__version__), python_abi=sys.implementation.cache_tag, arch="sm_90a",
            module_name="triattn_projection_bwd", binary="triattn_projection_bwd.so", tile_m=64, tile_k=64,
            sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files})
(ROOT / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
