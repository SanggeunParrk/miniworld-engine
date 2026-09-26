"""Per-kernel sustained energy of the PWA path vs each kernel's energy floor (R e_r + W e_w + F e_f + idle x T),
with the constants of a previous energy_sol.py --out json.   python kenergy_pwa.py esol_g4c.json"""
import json, os, pathlib, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import sustained
cal = json.load(open(sys.argv[1]))["ceilings"]
er, ew, ef, idle, pmax = cal["e_read"], cal["e_write"], cal["e_flop"], cal["idle_W"], cal["pmax_W"]
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "kenergy"; d.mkdir(parents=True, exist_ok=True)
ext = load("kenergy", [str(src / "pwa_sm100.cu")], extra_include_paths=[str(src)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
bf = torch.bfloat16; N, S, H, C, D = 384, 1024, 8, 32, 64; M = S * N
MB = lambda x: x * 1e6
m = torch.randn(S, N, D, device="cuda", dtype=bf); lnw = 1 + 0.1 * torch.randn(D, device="cuda"); lnb = 0.1 * torch.randn(D, device="cuda")
wv = (torch.randn(H * C, D, device="cuda") * 0.1).to(bf); wg = (torch.randn(H * C, D, device="cuda") * 0.2).to(bf); wo = (torch.randn(D, H * C, device="cuda") * 0.05).to(bf)
w16 = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf)
v, y = ext.ln_vg(m, lnw, lnb, wv, 1e-5, True)
o = ext.pwa_ctr(w16, v)
dres = torch.randn(S, N, D, device="cuda", dtype=bf)
d_o, dgp, _ = ext.pwa_glue(o, y, dres, wg, wo.t().contiguous(), None, 1.0)
dvh = ext.pwa_plain(w16, d_o)
vsz, msz = H * N * S * C * 2, M * D * 2
fl = 2 * H * N * N * S * C
wgvT = torch.cat([wg, wv], 0).t().contiguous()
cases = [
    ("ln_vg (infer, no y)", lambda: ext.ln_vg(m, lnw, lnb, wv, 1e-5, False), msz, vsz, 2 * M * D * H * C),
    ("ln_vg (train, y)",    lambda: ext.ln_vg(m, lnw, lnb, wv, 1e-5, True), msz, vsz + msz, 2 * M * D * H * C),
    ("pwa_ctr",             lambda: ext.pwa_ctr(w16, v), vsz, vsz, fl),
    ("pwa_fwd2 (infer)",    lambda: ext.pwa_fwd2(w16, v, m, lnw, lnb, 1e-5, wg, wo, False, None, 1.0), vsz + vsz + msz, vsz + msz, fl + 4 * M * D * H * C),
    ("pwa_glue",            lambda: ext.pwa_glue(o, y, dres, wg, wo.t().contiguous(), None, 1.0), vsz + 2 * msz, 2 * vsz, 6 * M * D * H * C),
    ("pwa_plain (dv)",      lambda: ext.pwa_plain(w16, d_o), vsz, vsz, fl),
    ("dw bmm (cuBLAS)",     lambda: torch.bmm(d_o, v.transpose(1, 2)), 2 * vsz, 0, fl),
    ("dgv_bwd",             lambda: ext.dgv_bwd(dgp, dvh, y, m, dres, wgvT, lnw, 1e-5), 2 * vsz + 3 * msz, msz, 2 * 2 * M * D * 2 * H * C),
]
print(f"idle {idle:.0f} W, pmax {pmax:.0f} W, e_r {er*1e12:.0f} e_w {ew*1e12:.0f} pJ/B, e_f {ef*1e12:.2f} pJ/FLOP")
print(f"{'kernel':22s} {'us':>7s} {'mJ':>7s} {'W':>6s} | floor: {'dyn mJ':>7s} {'us':>6s}  energy-SoL")
for name, fn, R, W, F in cases:
    r = sustained(fn, secs=2.0)
    edyn = R * er + W * ew + F * ef
    t_floor = edyn / (pmax - idle)
    meas_dyn = r["J"] - idle * r["ms"] * 1e-3
    print(f"{name:22s} {r['ms']*1e3:7.1f} {r['J']*1e3:7.2f} {r['W']:6.0f} | {edyn*1e3:7.2f} {t_floor*1e6:6.1f}  {t_floor/(r['ms']*1e-3)*100:5.1f}%   (measured dyn {meas_dyn*1e3:.1f} mJ)")
