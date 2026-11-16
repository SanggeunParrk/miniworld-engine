"""Kernel A/B of opm_fused_infer variants on the same operands (prologue once), interleaved; cool (idle before a short burst) and sustained
(back-to-back for --sustain s, then timed) per variant per round; SM clock + power (NVML) after the sustained burst; bitwise compare to
cluster 1.  python kbench.py L S OUT.jsonl [--rounds 4] [--variants cl1,cl2,cublas]"""
import argparse, json, statistics, sys, time
import torch
from miniworld_engine.kernels.integrations import opm_train as ot

ap = argparse.ArgumentParser()
ap.add_argument("L", type=int); ap.add_argument("S", type=int); ap.add_argument("out")
ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--variants", default="cl1,cl2,cublas")
ap.add_argument("--sustain", type=float, default=0.6); ap.add_argument("--idle", type=float, default=2.0)
ap.add_argument("--residual", action="store_true")
a = ap.parse_args()
T0 = time.time(); L, S = a.L, a.S; dev = torch.device("cuda"); bf = torch.bfloat16
torch.manual_seed(0)
m = torch.randn(S, L, 64, device=dev, dtype=bf); mk = torch.rand(S, L, device=dev) > 0.1
gam = torch.ones(64, device=dev); bet = torch.zeros(64, device=dev)
wa = (torch.randn(32, 64, device=dev) * 0.1).to(bf); wb = (torch.randn(32, 64, device=dev) * 0.1).to(bf)
wo = (torch.randn(128, 1024, device=dev) * 0.02).to(bf); ob = torch.randn(128, device=dev) * 0.1
ext = ot._ext(dev, native=True, tma=False, cm=64, cz=128, fused=True)
print("ext ready", round(time.time() - T0, 1), flush=True)
a2, bt, stats, mask16, norm = ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False)
del stats
wp = ot._fused_weight(wo)
res = torch.randn(1, L, L, 128, device=dev, dtype=bf) if a.residual else None
V = {f"cl{c}": (lambda c=c: ext.opm_fused_infer(a2, bt, norm, wp, ob, res, 1, cluster=c)) for c in (1, 2, 3, 4)}   # 1 none, 2 2x1, 3 1x2, 4 2x2
V["cublas"] = lambda: torch.mm(a2, bt.t())
if any(v in a.variants for v in ("pro_s", "nt_unfused", "cnt")):   # the s-major prologue, the NT product + epilogue on it, the mask counts
    pro_s = ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False, smajor=True)
    a_s, b_s, mask16s = pro_s[0], pro_s[1], pro_s[3]
    print("s-major prologue bitwise == transposed", bool(torch.equal(a_s, a2.t())), bool(torch.equal(b_s, bt.t())), flush=True)
    V["pro_s"] = lambda: ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False, smajor=True)
    V["nt_mm"] = lambda: torch.mm(a_s.t(), b_s)
    V["nt_unfused"] = lambda: ext.opm_epilogue(torch.mm(a_s.t(), b_s), norm, wo, ob, L, L, res, 1)
    from miniworld_engine.kernels.conditioned_transition.cuda.train import owned_gemm_ext
    og = owned_gemm_ext()
    V["cnt"] = lambda: ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, True)   # (prologue + count kernel; minus pro_s)
    V["cnt_mm"] = lambda: og.owned_mm(mask16s.t(), mask16s, False, True)
    cm_ = og.owned_mm(mask16s.t(), mask16s, False, True).clamp_min(1.0)
    print("count by mm == count kernel table", bool(torch.equal(cm_, ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, True)[4])), flush=True)
if any(v.startswith("chain") or v == "triton" for v in a.variants.split(",")):   # whole inference chains (eager), as the routes run them
    from miniworld_engine.kernels.conditioned_transition.cuda.train import owned_gemm_ext as _og
    from miniworld_engine.kernels.outer_product_mean.triton.main import triton_outer_product_mean
    og2 = _og()
    def _chain(nt):
        p = ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False, smajor=nt)
        nrm = og2.owned_mm(p[3].t(), p[3], False, True).clamp_min_(1.0)
        o = torch.mm(p[0].t(), p[1]) if nt else torch.mm(p[0], p[1].t())
        return ext.opm_epilogue(o, nrm, wo, ob, L, L, res, 1)
    V["chain_tn"] = lambda: _chain(False)
    V["chain_nt"] = lambda: _chain(True)
    msa4 = m[None]; mask4 = mk[None]
    lnw = torch.ones(64, device=dev); lnb = torch.zeros(64, device=dev)
    V["triton"] = lambda: triton_outer_product_mean(msa4, mask4, lnw, lnb, wa, wb, wo, ob, None, eps=1e-5, normalize_before_proj=False)
    _x, _y = V["chain_tn"](), V["chain_nt"]()
    _t = V["triton"]()
    print("chain nt == tn", bool(torch.equal(_x, _y)), "max|tn - triton|", float((_x.float() - _t.float()).abs().max()), "max|ref|", float(_t.float().abs().max()), flush=True)
if "cublas_nt" in a.variants:   # the same product from s-major operands [S, N*32] (the Triton / Anthropic layout): cuBLAS NT
    a2s, bts = a2.t().contiguous(), bt.t().contiguous()
    V["cublas_nt"] = lambda: torch.mm(a2s.t(), bts)
V["unfused"] = lambda: ext.opm_epilogue(torch.mm(a2, bt.t()), norm, wo, ob, L, L, res, 1)   # cuBLAS product + the CUDA epilogue kernel
if any(v in a.variants for v in ("pro", "epi")):   # (also "<build>.epi" / "<build>.pro")
    O0 = torch.mm(a2, bt.t())
    V["pro"] = lambda: ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False)   # the TMA prologue alone
    V["epi"] = lambda: ext.opm_epilogue(O0, norm, wo, ob, L, L, res, 1)   # the epilogue alone on a kept O
    normf = ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, True)[4]   # the fp32 [N, N] count table (count kernel)
    V["epin"] = lambda: ext.opm_epilogue(O0, normf, wo, ob, L, L, res, 1)   # the epilogue on the count table instead of the bit rows
    V["pron"] = lambda: ext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, True)   # the prologue + count kernel
    for nm, kw in (("epi_tma", dict(epi_tma=True)), ("epi_half", dict(epi_tma=True, epi_half=True))):
        if nm in a.variants.split(","):
            e2 = ot._ext(dev, native=True, tma=True, **kw)
            V[nm] = (lambda e2=e2: e2.opm_epilogue(O0, norm, wo, ob, L, L, res, 1))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from builds import build   # noqa: E402
for k in a.variants.split(","):
    if "." in k:   # "<build>.cl<code>": a macro variant of the extension (builds.py)
        bname, cl = k.split(".")
        bext = build(bname)
        if cl == "epi":   # "<build>.epi": that build's epilogue alone on the kept O
            V[k] = (lambda bext=bext: bext.opm_epilogue(O0, norm, wo, ob, L, L, res, 1))
        elif cl == "pro":   # "<build>.pro": that build's wide prologue alone
            V[k] = (lambda bext=bext: bext.opm_prologue_wide(m, mk, gam, bet, wa, wb, 1e-5, False, None, False))
        else:
            V[k] = (lambda bext=bext, c=int(cl[2:]): bext.opm_fused_infer(a2, bt, norm, wp, ob, res, 1, cluster=c))
V = {k: V[k] for k in a.variants.split(",")}
ref = V["cl1"]() if "cl1" in V else None
for k in V:
    if k not in ("cublas", "pro", "pro_s", "nt_mm", "cnt", "cnt_mm", "chain_tn", "chain_nt", "triton") and not k.endswith(".pro") and ref is not None:  # ("epi" variants vs the fused kernel: the bf16 O rounding differs)
        o = V[k](); o2 = V[k]()
        print(k, "bitwise vs cl1", bool(torch.equal(o, ref)), "run-twice", bool(torch.equal(o, o2)), "max|d|", float((o.float() - ref.float()).abs().max()), flush=True)
if "pro" in V:   # the prologue variants: bitwise the same A2 / BT / mask bits as this build's
    rp = [t for t in V["pro"]() if t is not None]
    for k in V:
        if k.endswith(".pro"):
            op = [t for t in V[k]() if t is not None]
            print(k, "prologue bitwise vs pro", all(torch.equal(x, y) for x, y in zip(op, rp)), [tuple(t.shape) for t in op], flush=True)
if "epi" in V:   # the epilogue variants: bitwise the same z as this build's epilogue
    re = V["epi"]()
    for k in V:
        if k.endswith(".epi"):
            print(k, "epilogue bitwise vs epi", bool(torch.equal(V[k](), re)), flush=True)
print("max clusters per code", ext.opm_fused_max_clusters(), flush=True)
flush = torch.empty(256 << 20, dtype=torch.int8, device=dev)
fl = 2 * S * L * L * 1024

def timed(fn, k):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(k)]; en = [torch.cuda.Event(enable_timing=True) for _ in range(k)]
    for s_, e_ in zip(st, en):
        flush.zero_(); s_.record(); fn(); e_.record()
    en[-2 if k > 1 else -1].synchronize()
    try: ghz, w = torch.cuda.clock_rate() / 1000.0, torch.cuda.power_draw() / 1000.0
    except Exception: ghz = w = None
    torch.cuda.synchronize()
    return statistics.median(s_.elapsed_time(e_) for s_, e_ in zip(st, en)), ghz, w

est = {}
for k, fn in V.items():
    fn(); torch.cuda.synchronize(); est[k] = max(timed(fn, 2)[0], 0.05)
with open(a.out, "a") as f:
    for r in range(a.rounds):
        names = list(V); names = names[r % len(names):] + names[: r % len(names)]
        for k in names:
            torch.cuda.synchronize(); time.sleep(a.idle)
            c, cg, cw = timed(V[k], max(2, min(5, int(40 / est[k]))))
            n = max(2, int(a.sustain * 1000 / est[k]))
            for _ in range(n):
                flush.zero_(); V[k]()
            s, g, w = timed(V[k], max(3, min(10, int(100 / est[k]))))
            rec = {"L": L, "S": S, "variant": k, "round": r, "cool_ms": c, "sus_ms": s, "sus_ghz": g, "sus_w": w, "residual": a.residual}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{k:<7} r{r} cool {c:.3f} sus {s:.3f} ms  TF {fl / s / 1e9:.0f}  {g} GHz {w} W", flush=True)
print("done", round(time.time() - T0, 1), flush=True)
