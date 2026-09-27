import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent; src = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "mb_stage5"; d.mkdir(parents=True, exist_ok=True)
ext = load("mb_stage5", [str(here / "mb_stage.cu")], extra_include_paths=[str(src)], build_directory=str(d), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"])
for comp, name in ((3, "random, S+dP+grad"), (6, "+ fence::after_thread_sync per group")):
    c = ext.run(comp, 0, 64); torch.cuda.synchronize()
    print(f"{name:28s}: {c.float().median().item() / 64:6.0f} clk/stage", flush=True)
