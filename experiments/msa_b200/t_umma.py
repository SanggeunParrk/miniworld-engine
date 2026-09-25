import os, pathlib, torch
from torch.utils.cpp_extension import load
here = pathlib.Path(__file__).parent
inc = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
build = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "t_umma"; build.mkdir(parents=True, exist_ok=True)
ext = load("t_umma", [str(here / "t_umma.cu")], extra_include_paths=[str(inc)], build_directory=str(build),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-lineinfo"], extra_ldflags=["-lcuda"], verbose=False)
torch.manual_seed(0)
ok = True
for N, bmn, atm in [(64, 0, 0), (128, 0, 0), (256, 0, 0), (32, 0, 0), (64, 1, 0), (128, 1, 0), (128, 0, 1), (32, 0, 1), (64, 1, 1)]:
    K = 256
    A = torch.randn(128, K, device="cuda", dtype=torch.bfloat16)
    Bk = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    B = Bk.t().contiguous() if bmn else Bk
    C = ext.run(A, B, bool(bmn), bool(atm))
    ref = A.float() @ Bk.float().t()
    err = ((C - ref).norm() / ref.norm()).item()
    print(f"N={N} b_mn={bmn} a_tmem={atm}: rel err {err:.2e}", flush=True)
    ok &= err < 1e-5
print("ALL OK" if ok else "FAIL")
