"""Correctness + timing of the sm100 OPM kernels against fp32 references of the same math."""
import os, pathlib, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import timeit
here = pathlib.Path(__file__).parent
src = here.parent / "src/miniworld_engine/integrations/csrc/sm100"
build = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "opm_sm100_dev"; build.mkdir(parents=True, exist_ok=True)
ext = load("opm_sm100_dev", [str(src / "opm_sm100.cu")], extra_include_paths=[str(src)], build_directory=str(build),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-lineinfo"], verbose=False)
torch.manual_seed(0)
bf = torch.bfloat16
N, CH, CZ = int(os.environ.get("N", 384)), 32, 128
what = sys.argv[1:] or ["epi"]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


if "epi" in what:
    O = torch.randn(N * CH, N * CH, device="cuda", dtype=bf)
    norm = torch.randint(1, 1000, (N, N), device="cuda").float()
    wo = (torch.randn(CZ, CH * CH, device="cuda") * 0.03).to(bf)
    bias = (torch.randn(CZ, device="cuda") * 0.1).to(bf).float()
    res = torch.randn(1, N, N, CZ, device="cuda", dtype=bf)
    P = O.view(N, CH, N, CH).permute(0, 2, 1, 3).reshape(N * N, CH * CH).float()
    ref = ((P @ wo.float().t()) / norm.view(-1, 1) + bias).view(1, N, N, CZ)
    z = ext.opm_epilogue(O, norm, wo, bias, N, N, None)
    zr = ext.opm_epilogue(O, norm, wo, bias, N, N, res)
    print(f"epilogue: rel err {rel(z, ref):.2e}  with residual {rel(zr, ref.to(bf) + res):.2e}", flush=True)
    t = timeit(lambda: ext.opm_epilogue(O, norm, wo, bias, N, N, res))
    byts = O.numel() * 2 + 2 * res.numel() * 2
    print(f"epilogue: {t*1e3:.1f} us  {byts / t / 1e9:.2f} TB/s ({byts/1e6:.0f} MB)", flush=True)

if "pro" in what:
    S = int(os.environ.get("S", 1024)); CM = 64
    m = torch.randn(S, N, CM, device="cuda", dtype=bf)
    mask = torch.rand(S, N, device="cuda") > 0.1
    lnw = (1 + 0.1 * torch.randn(CM, device="cuda")); lnb = 0.1 * torch.randn(CM, device="cuda")
    wa = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf); wb = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf)
    A2, BT, norm, stats, bits = ext.opm_prologue(m, mask, lnw, lnb, 1e-5, wa, wb, True, True)
    y = torch.nn.functional.layer_norm(m.float(), (CM,), lnw, lnb, 1e-5).to(bf)
    a = (y.float() @ wa.float().t()).to(bf).float() * mask[..., None]
    b = (y.float() @ wb.float().t()).to(bf).float() * mask[..., None]
    A2r = a.permute(1, 2, 0).reshape(N * CH, S); BTr = b.permute(1, 2, 0).reshape(N * CH, S)
    mf = mask.float(); normr = (mf.t() @ mf).clamp(min=1)
    mean = m.float().mean(-1); rstd = 1 / (m.float().var(-1, unbiased=False) + 1e-5).sqrt()
    print(f"prologue: A2 {rel(A2, A2r):.2e} BT {rel(BT, BTr):.2e} norm maxdiff {(norm - normr).abs().max().item()} "
          f"mean {rel(stats[..., 0], mean.t()):.2e} rstd {rel(stats[..., 1], rstd.t()):.2e}", flush=True)
    wo = (torch.randn(CZ, CH * CH, device="cuda") * 0.03).to(bf); bias = torch.zeros(CZ, device="cuda")
    O = (A2 @ BT.t())
    zb = ext.opm_epilogue(O, bits, wo, bias, N, N, None); zn = ext.opm_epilogue(O, norm, wo, bias, N, N, None)
    print(f"epilogue(bits) vs epilogue(norm): {rel(zb, zn):.2e}")
    t = timeit(lambda: ext.opm_prologue(m, mask, lnw, lnb, 1e-5, wa, wb, False))
    byts = m.numel() * 2 * 2 + mask.numel()
    print(f"prologue: {t*1e3:.1f} us  {byts / t / 1e9:.2f} TB/s ({byts/1e6:.0f} MB)", flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5): ext.opm_prologue(m, mask, lnw, lnb, 1e-5, wa, wb, False); ext.opm_epilogue(O, bits, wo, bias, N, N, None)
        torch.cuda.synchronize()
    for e in p.key_averages(): print(f"   {e.key[:60]:60s} {e.device_time:.1f} us")

if "dgrad" in what:
    S = int(os.environ.get("S", 1024))
    mask = torch.rand(S, N, device="cuda") > 0.1
    mf = mask.float(); normr = (mf.t() @ mf).clamp(min=1)
    bits = torch.zeros(N, S // 32, dtype=torch.int64, device="cuda")
    for w in range(32): bits |= (mask.t().reshape(N, S // 32, 32)[..., w].long() << w)
    bits = bits.to(torch.int32) if False else (bits - ((bits >> 31) & 1) * (1 << 32)).to(torch.int32)
    dz = torch.randn(1, N, N, CZ, device="cuda", dtype=bf)
    wo = (torch.randn(CZ, CH * CH, device="cuda") * 0.03).to(bf)
    dO, dzp, dbo = ext.opm_dgrad(dz[0].contiguous(), bits, wo, N, N)
    ref = (dz[0].float().reshape(N * N, CZ) @ wo.float()) / normr.reshape(-1, 1)          # [(i,j), (c,e)]
    ref = ref.view(N, N, CH, CH).permute(0, 2, 1, 3).reshape(N * CH, N * CH)
    print(f"dgrad: dO {rel(dO, ref):.2e}  dzp {rel(dzp, dz[0].float() / normr[..., None]):.2e}  dbo {rel(dbo, dz[0].float().sum((0, 1))):.2e}", flush=True)
    t = timeit(lambda: ext.opm_dgrad(dz[0], bits, wo, N, N))
    byts = dO.numel() * 2 + 2 * dz.numel() * 2
    print(f"dgrad: {t*1e3:.1f} us  {byts / t / 1e9:.2f} TB/s ({byts/1e6:.0f} MB)", flush=True)
