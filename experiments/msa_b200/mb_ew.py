import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; src = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "mb_ew"; d.mkdir(parents=True, exist_ok=True)
ext = load("mb_ew", [str(here / "mb_ew.cu")], extra_include_paths=[str(src)], build_directory=str(d), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
names = ["full (ex2 + 2 packs + dS)", "no packs", "ex2 only", "no ex2", "packs only"]
for v in range(5):
    c = ext.run(v, 2000); torch.cuda.synchronize()
    clk = c.float().median().item() / 2000          # clk per "stage" = 8 warps x 32 lanes x 32 elements = 8192 elements
    print(f"{names[v]:28s} {clk:7.1f} clk / 8192-element stage  ({8192 / clk:5.1f} elem/clk/SM)", flush=True)
