import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; inc = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
build = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "t_rate"; build.mkdir(parents=True, exist_ok=True)
ext = load("t_rate", [str(here / "t_rate.cu")], extra_include_paths=[str(inc)], build_directory=str(build), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"])
names = {0: "B K-major SW128", 1: "B MN-major SW128", 2: "B MN-major SW64", 3: "A in TMEM (ts)", 4: "M=64 ss"}
for mode, N in [(0, 96), (2, 96), (0, 128), (1, 128), (2, 128), (1, 256), (0, 256), (2, 64), (0, 64)]:
    c = ext.run(2000, mode, N, 148, 11)
    print(f"{names[mode]:18s} M=128 N={N:3d} chained: {c:6.1f} cycles/MMA -> {128*N*16*2/c/8192*100:.0f}% of 8192", flush=True)
