"""sm100 triangle attention backward: gradients vs fp32 autograd, sustained fwd+bwd timing vs PyTorch / opt_core.
    TA_L=384 python t_triattn_bwd.py"""
import os, sys, pathlib, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import sustained
sys.path.insert(0, os.environ["OPT_CORE_DIR"])
from opt_core.kernels import triattn as ta
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
tag = os.environ.get("TA_TAG", "cur")
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / f"triattn_{tag}"; d.mkdir(parents=True, exist_ok=True)
ext = load(f"triattn_{tag}", [os.environ.get("TA_SRC", str(src / "triattn_sm100.cu"))], extra_include_paths=[str(src)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
H, D = 4, 32
sc = D ** -0.5
Ls = [int(x) for x in os.environ.get("TA_L", "384").split(",")]
NS = [int(x) for x in os.environ.get("TA_N", "0").split(",")]
TIME = int(os.environ.get("TA_TIME", "1"))
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
for L in Ls:
  for Nq in NS:
    N = Nq or L
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(1, N, H, L, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(3))
    bias = torch.randn(1, 1, H, L, L, device="cuda", generator=g).to(torch.bfloat16)
    do = torch.randn(1, N, H, L, D, device="cuda", generator=g).to(torch.bfloat16)
    # fp32 reference gradients
    qf, kf, vf, bf = (t.float().requires_grad_() for t in (q, k, v, bias))
    ref = ta.sdpa_reference(qf, kf, vf, bf)
    ref.backward(do.float())
    gref = [qf.grad, kf.grad, vf.grad, bf.grad[0, 0]]
    del ref, qf, kf, vf, bf
    # ours
    P = lambda t: t.permute(0, 1, 3, 2, 4).contiguous()          # [B, N, H, S, D] -> projection layout [B, N, S, H, D]
    qn, kn, vn, don = P(q), P(k), P(v), P(do)
    def ours_train():
        out, lse, _ = ext.triattn_fwd(qn, kn, vn, bias[:, 0], sc, True)
        delta = (out.float() * don.float()).sum(-1).permute(0, 1, 3, 2).contiguous()   # [B, N, H, S]
        return ext.triattn_bwd(qn, kn, vn, bias[:, 0], don, lse, delta, sc)
    dq, dk, dv, db = ours_train()
    U = lambda t: t.permute(0, 1, 3, 2, 4)
    got = [U(dq), U(dk), U(dv), db[0]]
    # bf16 PyTorch autograd for the error scale
    qb, kb, vb, bb = (t.clone().requires_grad_() for t in (q, k, v, bias))
    ta.sdpa_reference(qb, kb, vb, bb).backward(do)
    gbf = [qb.grad, kb.grad, vb.grad, bb.grad[0, 0]]
    print(f"L={L} N={N}  rel err vs fp32 (ours | torch bf16):  " + "  ".join(f"{n} {rel(a, r):.2e} | {rel(c, r):.2e}" for n, a, c, r in zip(("dq", "dk", "dv", "db"), got, gbf, gref)), flush=True)
    del qb, kb, vb, bb, gbf, gref
    if not TIME: continue
    def torch_train():
        qb, kb, vb, bb = (t.detach().requires_grad_() for t in (q, k, v, bias))
        ta.sdpa_reference(qb, kb, vb, bb).backward(do)
    def flash_train():
        qb, kb, vb = (t.detach().requires_grad_() for t in (q, k, v)); bb = bias.float().requires_grad_()
        ta.triangle_attention(qb, kb, vb, bb, None, None, word="flash").backward(do)
    rows = [("torch sdpa", torch_train), ("flash", flash_train), ("ours", ours_train),
            ("ours fwd", lambda: ext.triattn_fwd(qn, kn, vn, bias[:, 0], sc, True))]
    out, lse, _ = ext.triattn_fwd(qn, kn, vn, bias[:, 0], sc, True)
    delta = (out.float() * don.float()).sum(-1).permute(0, 1, 3, 2).contiguous()
    rows.append(("ours bwd", lambda: ext.triattn_bwd(qn, kn, vn, bias[:, 0], don, lse, delta, sc)))
    for name, fn in rows:
        try:
            r = sustained(fn, secs=2.0)
            print(f"   {name:10s} {r['ms']*1e3:8.1f} us  {r['J']*1e3:7.2f} mJ  {r['W']:5.0f} W", flush=True)
        except Exception as e:
            print(f"   {name:10s} failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
