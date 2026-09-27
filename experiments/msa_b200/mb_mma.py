import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; src = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
lays = ["K-SW128", "K-SW64 ", "MN-SW64", "K128x4 ", "MN-SW128", "MN-SW32"]
for tm in (0, 1):
    d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / f"mb_mma_tm{tm}"; d.mkdir(parents=True, exist_ok=True)
    ext = load(f"mb_mma_tm{tm}", [str(here / "mb_mma.cu")], extra_include_paths=[str(src)], build_directory=str(d), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", f"-DMB_TM={tm}"])
    for ts in (1, 0):
        for lay in (3, 0, 1, 2):
            for N in (32, 64):
                c = ext.run(256, N, ts, lay, 0, 148, 128); torch.cuda.synchronize()
                print(f"tcgen05.ld/st in kernel={tm} {'ts' if ts else 'ss'} B {lays[lay]} N={N}: {c.float().median().item() / 256:6.1f} clk/MMA", flush=True)
