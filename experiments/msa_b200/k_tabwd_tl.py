"""clock64 timeline of one CTA of the triattn backward KV kernel (TA_TL=<cta> build)"""
import os, sys, pathlib, torch
from torch.utils.cpp_extension import load
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
cta = int(os.environ.get("TL_CTA", "300"))
out = pathlib.Path(os.environ["TMPDIR"]) / "ta_tl.cu"
out.write_text(f"#define TA_TL {cta}\n" + os.environ.get("TL_DEFS", "").replace(";", "\n") + "\n" + (src / "triattn_sm100.cu").read_text())
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "tabwd_tl"; d.mkdir(parents=True, exist_ok=True)
ext = load("tabwd_tl", [str(out)], extra_include_paths=[str(src)], build_directory=str(d), extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
H, D, L = 4, 32, 384
sc = D ** -0.5
qn, kn, vn, don = (torch.randn(1, L, L, H, D, device="cuda").to(torch.bfloat16) for _ in range(4))
bias = torch.randn(1, H, L, L, device="cuda").to(torch.bfloat16)
out, lse, _ = ext.triattn_fwd(qn, kn, vn, bias, sc, True)
delta = (out.float() * don.float()).sum(-1).permute(0, 1, 3, 2).contiguous()
for _ in range(3): ext.triattn_bwd(qn, kn, vn, bias, don, lse, delta, sc)
torch.cuda.synchronize()
t = ext.triattn_tl()
t0 = t[0, 0].item()
names = os.environ.get("TL_NAMES", "mma:stt,mma:qf,mma:grd,mma:pf,g0:start,g0:bias,g0:sf,g0:cmp,g0:pf,-,g1:start,g1:bias,g1:sf,g1:cmp,g1:pf").split(",")
print("y  " + " ".join(f"{n:>8s}" for n in names))
for y in range(int(os.environ.get("TL_N", "24"))):
    print(f"{y:2d} " + " ".join(f"{(t[e, y].item() - t0) if t[e, y].item() else 0:8d}" for e in range(len(names))))
