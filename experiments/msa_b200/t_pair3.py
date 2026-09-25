"""pair3 forward: the H100 build (fp32 logits, 2 CTAs/SM) vs the sm_100a build (bf16 logits, 3 CTAs/SM), same source."""
import os, pathlib, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import timeit
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/pair3.cu"
root = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"])
exts = {}
for tag, defs in (("a", []), ("b", ["-DPAIR3_SM100"])):
    d = root / f"tp3_{tag}"; d.mkdir(parents=True, exist_ok=True)
    exts[tag] = load(f"tp3_{tag}", [str(src)], build_directory=str(d), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"] + defs)
N = 384; bf = torch.bfloat16
z = torch.randn(N, N, 128, device="cuda", dtype=bf); mask = (torch.rand(N, N, device="cuda") > 0.1).to(bf); mask[5] = 0
lw = 1 + 0.1 * torch.randn(128, device="cuda"); lb = 0.1 * torch.randn(128, device="cuda"); wb = torch.randn(8, 128, device="cuda") * 0.1
ra, rb = (exts[t].pair_fwd3(z, mask, lw, lb, 1e-5, wb) for t in "ab")
print("max |a - b|", (ra.float() - rb.float()).abs().max().item())
for t in "ab": print(t, f"{timeit(lambda: exts[t].pair_fwd3(z, mask, lw, lb, 1e-5, wb)) * 1e3:.1f} us")
