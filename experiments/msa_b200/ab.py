"""Interleaved A/B timing of two builds of one extension source (the B200 is shared: absolute times are inflated,
the ratio is what this measures).  python ab.py <old.cu> <new.cu> <case>"""
import os, pathlib, statistics, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import timeit
src_inc = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
root = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"])
def build(path, tag):
    d = root / f"ab_{tag}"; d.mkdir(parents=True, exist_ok=True)
    return load(f"ab_{tag}", [path], extra_include_paths=[str(src_inc)], build_directory=str(d),
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"], verbose=False)
old, new, case = sys.argv[1], sys.argv[2], sys.argv[3]
A, B = build(old, "old_" + case), build(new, "new_" + case)
bf = torch.bfloat16; N, S = 384, 1024; torch.manual_seed(0)
if case == "dgrad":
    CZ, CH = 128, 32
    mask = torch.rand(S, N, device="cuda") > 0.1
    bits = torch.zeros(N, S // 32, dtype=torch.int64, device="cuda")
    for w in range(32): bits |= (mask.t().reshape(N, S // 32, 32)[..., w].long() << w)
    bits = (bits - ((bits >> 31) & 1) * (1 << 32)).to(torch.int32)
    dz = torch.randn(N, N, CZ, device="cuda", dtype=bf); wo = (torch.randn(CZ, CH * CH, device="cuda") * 0.03).to(bf)
    fa = lambda: A.opm_dgrad(dz, bits, wo, N, N, 0); fb = lambda: B.opm_dgrad(dz, bits, wo, N, N, 0)
    ra, rb = fa(), fb()
    print("max |old - new| dO", (ra[0].float() - rb[0].float()).abs().max().item(), "dzp", (ra[1].float() - rb[1].float()).abs().max().item(),
          "dbo", (ra[2] - rb[2]).abs().max().item())
elif case == "epi":
    CZ, CH = 128, 32
    mask = torch.rand(S, N, device="cuda") > 0.1
    bits = torch.zeros(N, S // 32, dtype=torch.int64, device="cuda")
    for w in range(32): bits |= (mask.t().reshape(N, S // 32, 32)[..., w].long() << w)
    bits = (bits - ((bits >> 31) & 1) * (1 << 32)).to(torch.int32)
    O = torch.randn(N * CH, N * CH, device="cuda", dtype=bf); wo = (torch.randn(CZ, CH * CH, device="cuda") * 0.03).to(bf)
    bias = (torch.randn(CZ, device="cuda") * 0.1).to(bf).float(); res = torch.randn(1, N, N, CZ, device="cuda", dtype=bf)
    fa = lambda: A.opm_epilogue(O, bits, wo, bias, N, N, res); fb = lambda: B.opm_epilogue(O, bits, wo, bias, N, N, res)
    ra, rb = fa(), fb()
    print("max |old - new| z", (ra.float() - rb.float()).abs().max().item(),
          "(no res)", (A.opm_epilogue(O, bits, wo, bias, N, N, None).float() - B.opm_epilogue(O, bits, wo, bias, N, N, None).float()).abs().max().item())
elif case == "pro":
    CH, CM = 32, 64
    m = torch.randn(S, N, CM, device="cuda", dtype=bf); mask = torch.rand(S, N, device="cuda") > 0.1
    lnw = 1 + 0.1 * torch.randn(CM, device="cuda"); lnb = 0.1 * torch.randn(CM, device="cuda")
    wa = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf); wb = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf)
    tr = os.environ.get("PRO_TRAIN") == "1"
    fa = lambda: A.opm_prologue(m, mask, lnw, lnb, 1e-5, wa, wb, tr, False); fb = lambda: B.opm_prologue(m, mask, lnw, lnb, 1e-5, wa, wb, tr, False)
    ra, rb = fa(), fb()
    print("max |old - new|", [(x.float() - y.float()).abs().max().item() for x, y in zip(ra, rb) if x.numel()])
elif case == "dwo":
    CZ, CH = 128, 32
    dzp = (torch.randn(N, N, CZ, device="cuda") * 0.01).to(bf); O = torch.randn(N * CH, N * CH, device="cuda", dtype=bf)
    fa = lambda: A.opm_dwo(dzp, O, N, N, 0); fb = lambda: B.opm_dwo(dzp, O, N, N, 0)
    ra, rb = fa(), fb()
    print("max |old - new| dWo", (ra - rb).abs().max().item())
elif case == "pbwd":
    CH, CM = 32, 64
    m = torch.randn(S, N, CM, device="cuda", dtype=bf); mask = torch.rand(S, N, device="cuda") > 0.1
    lnw = 1 + 0.1 * torch.randn(CM, device="cuda"); lnb = 0.1 * torch.randn(CM, device="cuda")
    wa = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf); wb = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf)
    _, _, _, stats, _ = A.opm_prologue(m, mask, lnw, lnb, 1e-5, wa, wb, True, False)
    dA = torch.randn(S, N * CH, device="cuda", dtype=bf); dB = torch.randn(S, N * CH, device="cuda", dtype=bf)
    fa = lambda: A.opm_prologue_bwd(dA, dB, m, stats, mask, lnw, lnb, wa, wb); fb = lambda: B.opm_prologue_bwd(dA, dB, m, stats, mask, lnw, lnb, wa, wb)
    ra, rb = fa(), fb()
    print("max |old - new|", [(x.float() - y.float()).abs().max().item() for x, y in zip(ra, rb)])
elif case == "glue":
    H, C, D = 8, 32, 64
    o = torch.randn(S, N, H * C, device="cuda", dtype=bf); y = torch.randn(S, N, D, device="cuda", dtype=bf); dres = torch.randn(S, N, D, device="cuda", dtype=bf)
    wg = (torch.randn(H * C, D, device="cuda") * 0.2).to(bf); wot = (torch.randn(D, H * C, device="cuda") * 0.05).to(bf).t().contiguous()
    dmask = (torch.rand(N, D, device="cuda") > 0.15).to(bf)
    ga, gb = torch.zeros(S, N, 2 * H * C, device="cuda", dtype=bf), torch.zeros(S, N, 2 * H * C, device="cuda", dtype=bf)
    fa = lambda: A.pwa_glue(o, y, dres, wg, wot, ga, dmask, 1 / 0.85); fb = lambda: B.pwa_glue(o, y, dres, wg, wot, gb, dmask, 1 / 0.85)
    ra, rb = fa(), fb()
    print("max |old - new| do", (ra[0].float() - rb[0].float()).abs().max().item(), "dWo", (ra[1] - rb[1]).abs().max().item(), "dgp", (ga.float() - gb.float()).abs().max().item())
elif case in ("fwd2", "fwd2save"):              # old = the fused pwa_fwd, new = the split pwa_fwd2 (same inputs)
    H, C, D = 8, 32, 64
    m = torch.randn(S, N, D, device="cuda", dtype=bf); y = torch.randn(S, N, D, device="cuda", dtype=bf)
    w = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf); v = torch.randn(H, N, S * C, device="cuda", dtype=bf)
    wg = (torch.randn(H * C, D, device="cuda") * 0.2).to(bf); wo = (torch.randn(D, H * C, device="cuda") * 0.05).to(bf)
    dmk = (torch.rand(N, D, device="cuda") > 0.15).to(bf) if os.environ.get("AB_DMASK") else None
    so = case == "fwd2save"
    fa = lambda: A.pwa_fwd(w, v, y, wg, wo, m, so, dmk, 1 / 0.85); lnw_ = 1 + 0.1 * torch.randn(D, device="cuda"); lnb_ = 0.1 * torch.randn(D, device="cuda")
    y = torch.nn.functional.layer_norm(m.float(), (D,), lnw_, lnb_, 1e-5).to(bf)
    fb = lambda: B.pwa_fwd2(w, v, m, lnw_, lnb_, 1e-5, wg, wo, so, dmk, 1 / 0.85)
    ra, rb = fa(), fb()
    ref_u = (m.float() - ra[0].float())
    print("out: max |old - new|", (ra[0].float() - rb[0].float()).abs().max().item(),
          " rel(update)", ((ra[0].float() - rb[0].float()).norm() / ref_u.norm()).item(),
          " o max", (ra[1].float() - rb[1].view(H, N, S, C).permute(2, 1, 0, 3).reshape(S, N, H * C).float()).abs().max().item() if so else 0)
elif case in ("fwdsave", "fwd"):
    H, C, D = 8, 32, 64
    m = torch.randn(S, N, D, device="cuda", dtype=bf); y = torch.randn(S, N, D, device="cuda", dtype=bf)
    w = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf); v = torch.randn(H, N, S * C, device="cuda", dtype=bf)
    wg = (torch.randn(H * C, D, device="cuda") * 0.2).to(bf); wo = (torch.randn(D, H * C, device="cuda") * 0.05).to(bf)
    so = case == "fwdsave"
    fa = lambda: A.pwa_fwd(w, v, y, wg, wo, m, so, None, 1.0); fb = lambda: B.pwa_fwd(w, v, y, wg, wo, m, so, None, 1.0)
    ra, rb = fa(), fb()
    print("max |old - new| out", (ra[0].float() - rb[0].float()).abs().max().item(), "o", (ra[1].float() - rb[1].float()).abs().max().item() if so else 0)
elif case == "plain":
    H, C = 8, 32
    w = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf); dO = torch.randn(H, N, S * C, device="cuda", dtype=bf)
    ga, gb = torch.zeros(S, N, 2 * H * C, device="cuda", dtype=bf), torch.zeros(S, N, 2 * H * C, device="cuda", dtype=bf)
    fa = lambda: A.pwa_plain(w, dO, ga); fb = lambda: B.pwa_plain(w, dO, gb)
    fa(); fb()
    print("max |old - new| dv", (ga.float() - gb.float()).abs().max().item())
elif case == "dgv":
    H, C, D = 8, 32, 64; M = S * N
    dgv = torch.randn(M, 2 * H * C, device="cuda", dtype=bf); x = torch.randn(M, D, device="cuda", dtype=bf)
    lnw = 1 + 0.1 * torch.randn(D, device="cuda"); y = torch.nn.functional.layer_norm(x.float(), (D,), lnw, None, 1e-5).to(bf)
    dout = torch.randn(M, D, device="cuda", dtype=bf); wgvT = (torch.randn(2 * H * C, D, device="cuda") * 0.05).to(bf).t().contiguous()
    fa = lambda: A.dgv_bwd(dgv, y, x, dout, wgvT, lnw, 1e-5); fb = lambda: B.dgv_bwd(dgv, y, x, dout, wgvT, lnw, 1e-5)
    ra, rb = fa(), fb()
    print("max |old - new|", [(a.float() - b.float()).abs().max().item() for a, b in zip(ra, rb)])
ta, tb = [], []
for _ in range(9):
    ta.append(timeit(fa, rounds=3)); tb.append(timeit(fb, rounds=3))
print(f"A/B {case}: old {statistics.median(ta)*1e3:.1f} us  new {statistics.median(tb)*1e3:.1f} us  ratio {statistics.median(ta)/statistics.median(tb):.2f}x")
if os.environ.get("AB_ENERGY"):                 # sustained (power-capped) time and energy per call, old then new
    from energy_sol import sustained
    for tag, fn in (("old", fa), ("new", fb)):
        r = sustained(fn, secs=2.0)
        print(f"sustained {case} {tag}: {r['ms']*1e3:.1f} us  {r['J']*1e3:.2f} mJ  {r['W']:.0f} W", flush=True)
