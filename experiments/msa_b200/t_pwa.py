"""Correctness + timing of the sm100 PWA kernels against fp32 references of the same math."""
import os, pathlib, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import timeit
here = pathlib.Path(__file__).parent
src = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
build = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "pwa_sm100_dev"; build.mkdir(parents=True, exist_ok=True)
ext = load("pwa_sm100_dev", [str(src / "pwa_sm100.cu")], extra_include_paths=[str(src)], build_directory=str(build),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-lineinfo", "--use_fast_math"], verbose=False)
torch.manual_seed(0)
bf = torch.bfloat16
N, S = int(os.environ.get("N", 384)), int(os.environ.get("S", 1024))
H, C, D, DZ = 8, 32, 64, 128
what = sys.argv[1:] or ["lnvg"]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


if "lnvg" in what:
    m = torch.randn(S, N, D, device="cuda", dtype=bf)
    lnw = 1 + 0.1 * torch.randn(D, device="cuda"); lnb = 0.1 * torch.randn(D, device="cuda")
    wv = (torch.randn(H * C, D, device="cuda") * 0.1).to(bf)
    v, y = ext.ln_vg(m, lnw, lnb, wv, 1e-5)
    yr = torch.nn.functional.layer_norm(m.float(), (D,), lnw, lnb, 1e-5)
    vr = (yr.to(bf).float() @ wv.float().t()).view(S, N, H, C).permute(2, 1, 0, 3).reshape(H, N, S * C)
    print(f"ln_vg: y {rel(y, yr):.2e}  v {rel(v, vr):.2e}", flush=True)
    t = timeit(lambda: ext.ln_vg(m, lnw, lnb, wv, 1e-5))
    byts = (m.numel() * 2 + v.numel()) * 2
    print(f"ln_vg: {t*1e3:.1f} us  {byts / t / 1e9:.2f} TB/s ({byts/1e6:.0f} MB)", flush=True)

if "fwd" in what:
    m = torch.randn(S, N, D, device="cuda", dtype=bf)
    y = torch.randn(S, N, D, device="cuda", dtype=bf)
    w = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf)
    v = torch.randn(H, N, S * C, device="cuda", dtype=bf)
    wg = (torch.randn(H * C, D, device="cuda") * 0.2).to(bf); wo = (torch.randn(D, H * C, device="cuda") * 0.05).to(bf)
    dmask = (torch.rand(N, D, device="cuda") > 0.15).to(bf); dsc = 1 / 0.85
    out, o = ext.pwa_fwd(w, v, y, wg, wo, m, True, dmask, dsc)
    vv = v.float().view(H, N, S, C)
    orf = torch.einsum("hij,hjsc->sihc", w.float(), vv)                          # [S, N, H, C]
    gr = (y.float() @ wg.float().t()).view(S, N, H, C)
    u = (orf / (1 + torch.exp(-gr))).to(bf).float()
    upd = (u.reshape(S, N, H * C) @ wo.float().t()).to(bf).float()
    upd = (upd * dmask.float() * dsc).to(bf).float()
    ref = m.float() + upd
    print(f"pwa_fwd: out {rel(out, ref):.2e}  update {rel(out.float() - m.float(), upd):.2e}  o {rel(o, orf.reshape(S, N, H * C)):.2e}", flush=True)
    for so in (True, False):
        t = timeit(lambda: ext.pwa_fwd(w, v, y, wg, wo, m, so, dmask, dsc))
        byts = (v.numel() + 3 * m.numel() + (o.numel() if so else 0)) * 2
        print(f"pwa_fwd save_o={so}: {t*1e3:.1f} us  {byts / t / 1e9:.2f} TB/s ({byts/1e6:.0f} MB)", flush=True)

if "glue" in what:
    o = torch.randn(S, N, H * C, device="cuda", dtype=bf)
    y = torch.randn(S, N, D, device="cuda", dtype=bf)
    dres = torch.randn(S, N, D, device="cuda", dtype=bf)
    wg = (torch.randn(H * C, D, device="cuda") * 0.2).to(bf); wo = (torch.randn(D, H * C, device="cuda") * 0.05).to(bf)
    dmask = (torch.rand(N, D, device="cuda") > 0.15).to(bf); dsc = 1 / 0.85
    dgv = torch.zeros(S, N, 2 * H * C, device="cuda", dtype=bf)
    dO, dwo = ext.pwa_glue(o, y, dres, wg, wo.t().contiguous(), dgv, dmask, dsc)
    dr = (dres.float() * dmask.float() * dsc).to(bf).float()
    g = torch.sigmoid(y.float() @ wg.float().t())                      # [S, N, HC]
    du = dr @ wo.float()                                               # [S, N, HC]
    do_ref = (du * g).view(S, N, H, C).permute(2, 1, 0, 3).reshape(H, N, S * C)
    dgp_ref = du * o.float() * g * (1 - g)
    dwo_ref = dr.reshape(-1, D).t() @ (g * o.float()).to(bf).float().reshape(-1, H * C)
    print(f"glue: do {rel(dO, do_ref):.2e}  dgp {rel(dgv[..., :H * C], dgp_ref):.2e}  dWo {rel(dwo, dwo_ref):.2e}  (dgv second half untouched: {dgv[..., H * C:].abs().max().item()})", flush=True)
    t = timeit(lambda: ext.pwa_glue(o, y, dres, wg, wo.t().contiguous(), dgv, dmask, dsc))
    byts = (o.numel() * 3 + 2 * y.numel()) * 2
    print(f"glue: {t*1e3:.1f} us  {byts / t / 1e9:.2f} TB/s ({byts/1e6:.0f} MB)", flush=True)

if "plain" in what:
    w = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf)
    dO = torch.randn(H, N, S * C, device="cuda", dtype=bf)
    dgv = torch.zeros(S, N, 2 * H * C, device="cuda", dtype=bf)
    ext.pwa_plain(w, dO, dgv)
    dv = torch.einsum("hij,his->hjs", w.float(), dO.float()).view(H, N, S, C).permute(2, 1, 0, 3).reshape(S, N, H * C)
    print(f"plain: dv {rel(dgv[..., H * C:], dv):.2e}  (dgp half untouched: {dgv[..., :H * C].abs().max().item()})", flush=True)
    t = timeit(lambda: ext.pwa_plain(w, dO, dgv))
    byts = 2 * dO.numel() * 2
    print(f"plain: {t*1e3:.1f} us  {byts / t / 1e9:.2f} TB/s ({byts/1e6:.0f} MB), {2*H*N*N*S*C/t/1e9:.0f} TFLOP/s", flush=True)

if "dgv" in what:
    M = S * N
    dgv = torch.randn(M, 2 * H * C, device="cuda", dtype=bf)
    x = torch.randn(M, D, device="cuda", dtype=bf)
    lnw = 1 + 0.1 * torch.randn(D, device="cuda"); lnb = 0.1 * torch.randn(D, device="cuda")
    y = torch.nn.functional.layer_norm(x.float(), (D,), lnw, lnb, 1e-5).to(bf)
    dout = torch.randn(M, D, device="cuda", dtype=bf)
    wgv = (torch.randn(2 * H * C, D, device="cuda") * 0.05).to(bf)
    dm, dW, dgam, dbet = ext.dgv_bwd(dgv, y, x, dout, wgv.t().contiguous(), lnw, 1e-5)
    dy = dgv.float() @ wgv.float()
    xr = x.float().requires_grad_(True); g_ = lnw.clone().requires_grad_(True); b_ = lnb.clone().requires_grad_(True)
    yy = torch.nn.functional.layer_norm(xr, (D,), g_, b_, 1e-5)
    yy.backward(dy)
    print(f"dgv_bwd: dm {rel(dm, xr.grad + dout.float()):.2e}  dWgv {rel(dW, dgv.float().t() @ y.float()):.2e}  dgamma {rel(dgam, g_.grad):.2e}  dbeta {rel(dbet, b_.grad):.2e}", flush=True)
    t = timeit(lambda: ext.dgv_bwd(dgv, y, x, dout, wgv.t().contiguous(), lnw, 1e-5))
    byts = (dgv.numel() + 4 * x.numel()) * 2
    print(f"dgv_bwd: {t*1e3:.1f} us  {byts / t / 1e9:.2f} TB/s ({byts/1e6:.0f} MB)", flush=True)
