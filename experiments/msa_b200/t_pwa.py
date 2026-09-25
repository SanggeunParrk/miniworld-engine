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
