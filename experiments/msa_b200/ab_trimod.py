"""ablations of the TriangleAttention module kernels: python ab_trimod.py FLAG[,FLAG] ... (timing of head_bwd only)"""
import os, sys, pathlib, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import sustained
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
L, B, C = 384, 1, 128
for flags in sys.argv[1:]:
    tag = flags.replace(",", "_")
    f = pathlib.Path(os.environ["TMPDIR"]) / f"trimod_{tag}.cu"
    f.write_text("".join(f"#define {x} 1\n" for x in flags.split(",") if x != "base") + (src / "triattn_mod_sm100.cu").read_text())
    d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / f"trimod_{tag}"; d.mkdir(parents=True, exist_ok=True)
    ext = load(f"trimod_{tag}", [str(f)], extra_include_paths=[str(src)], build_directory=str(d), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B * L * L, C, device="cuda", generator=g).to(torch.bfloat16)
    d4 = [torch.randn(B * L * L, C, device="cuda", generator=g).to(torch.bfloat16) for _ in range(4)]
    db = torch.randn(B, 4, L, L, device="cuda", generator=g)
    w4 = torch.randn(512, C, device="cuda", generator=g).to(torch.bfloat16); wb = torch.randn(4, C, device="cuda", generator=g)
    lnw = torch.ones(C, device="cuda"); lnb = torch.zeros(C, device="cuda"); dp = torch.zeros_like(x)
    e0 = torch.empty(0, device="cuda")
    r = sustained(lambda: ext.tri_wgrad(*d4, db, x, 1e-5, B, L, w4, wb, lnw, lnb, False, e0), secs=1.5)
    print(f"{flags:20s} wgrad {r['ms']*1e3:8.1f} us", flush=True)
