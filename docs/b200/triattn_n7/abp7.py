# abp.py "MAC=v,MAC2=v" L cfgjson... : ONE variant per process (two extensions of one build in one process break the tail launch: invalid argument).
# prints graph median (60 samples, L2 flushed) and the kernel spans. Macros are -D for the mod (front / tail) and core extensions.
import sys, json, torch, hashlib
from miniworld_engine.modules import triangle as T
from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn as K
from miniworld_engine.kernels._nvcc import load_extension
d = [kv for kv in sys.argv[1].split(",") if kv]; L = int(sys.argv[2])
tag = hashlib.md5(",".join(d).encode()).hexdigest()[:6] if d else ""
dc = [x for x in d if x.startswith("TA_")]; dm = [x for x in d if not x.startswith("TA_")]; tagc = hashlib.md5(",".join(dc).encode()).hexdigest()[:6]; tagm = hashlib.md5(",".join(dm).encode()).hexdigest()[:6]
_core0, _mod0 = K._core, K._mod; cache = {}; CS = {True: chr(99), False: chr(120)[:0]}
def core(rows=4, hb=False, d16=False, poly=None, d16c=False, qs=False):
    if not dc: return _core0(rows, hb, d16, poly, d16c, qs)
    key = (rows, hb, d16, poly, d16c, qs)
    if key not in cache:
        p = (K.POLY_INFER(rows) if poly is None else poly) if hb else 0
        base = f"triattn_b200_core_r{rows}_hb_d16{CS[d16c]}_p{p}_x{tagc}" if d16 else (f"triattn_b200_core_r{rows}_hb_p{p}{'_qs' if qs else ''}_x{tagc}" if hb else f"triattn_b200_core_r{rows}_x{tagc}")
        cache[key] = load_extension(name=base, sources=[str(K._SRC / "triattn_sm100.cu")], extra_cuda_cflags=[*K._flags(), f"-DTA_R={rows}", *(("-DTA_HB=1", "-DTA_CK=16", f"-DTA_POLY={p}") if hb else ()), *(("-DTA_D16=1",) if hb and d16 else ()), *(("-DTA_D16C=1",) if hb and d16 and d16c else ()), *(("-DTA_QS=1",) if hb and not d16 and qs else ()), *[f"-D{x}" for x in dc]], verbose=False)
    return cache[key]
def mod():
    if not dm: return _mod0()
    if "m" not in cache: cache["m"] = load_extension(name=f"triattn_b200_mod_x{tagm}", sources=[str(K._SRC / "triattn_mod_sm100.cu")], extra_cuda_cflags=[*K._flags(), *[f"-D{x}" for x in dm]], verbose=False)
    return cache["m"]
K._core = core; K._mod = mod
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
from torch.profiler import profile, ProfilerActivity
for cj in sys.argv[3:]:
    kw = json.loads(cj); D = kw["d_pair"]
    torch.manual_seed(0)
    m = T.TriangleAttention(**kw, backend="cuda").cuda().float().eval()
    x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16)
    def step():
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): return m(x)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): y = step()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): y = step()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(60):
        flush.zero_(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        flush.zero_(); g.replay(); torch.cuda.synchronize()
    ev = sorted([e for e in p.events() if e.device_type.name == "CUDA" and "memset" not in e.name.lower() and e.name != "" and "elementwise" not in e.name], key=lambda e: e.time_range.start)
    t0 = ev[0].time_range.start
    sp = " | ".join(f"{e.name.split(chr(40))[0].split(chr(58))[-1][:14]} {e.time_range.start-t0:.1f}+{e.time_range.end-e.time_range.start:.1f}" for e in ev)
    print(f"L{L} d{D}{'S' if kw.get('starting') else 'E'} [{sys.argv[1] or 'base'}] med {ts[30]:.1f} min {ts[0]:.1f} p25 {ts[15]:.1f} | {sp}", flush=True)
    if any(x.startswith("TG_STAMP") for x in d):
        st = K._mod().tri_tail_stamps()
        G = torch.cuda.get_device_properties(0).multi_processor_count
        s0, s1, rk, nn = st[0, :G].double(), st[1, :G].double(), st[2, :G], st[3, :G]
        t0_ = s0.min()
        a = (s0 - t0_) / 1e3; e = (s1 - t0_) / 1e3
        qs = lambda v: " ".join(f"{float(v.sort().values[int(len(v) * f)]):.1f}" for f in (0, .25, .5, .75, .999))
        print(f"   stamps(us from first tail start) start q0/25/50/75/100: {qs(a)} | end: {qs(e)} | tiles min/max {int(nn.min())}/{int(nn.max())} | last-start CTA n={int(nn[a.argmax()])} end {float(e[a.argmax()]):.1f}; latest end {float(e.max()):.1f}; mean(end-start) {float((e-a).mean()):.1f}", flush=True)
    if any(x.startswith("TG_TL") for x in d):
        tl = K._mod().tri_tail_tl().double()
        nt = int((tl[:, 0] > 0).sum())
        base_ = tl[tl > 0].min()
        names = ["x_iss", "x_land", "LN_end", "g_start", "g_ok", "o_land", "u_done", "OUT_rdy", "slot_free", "o_iss"]
        order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print("   tile " + " ".join(f"{n:>9s}" for n in names) + "   (us from the CTA's first stamp; tile gaps = period)", flush=True)
        for i in range(min(nt, 12)):
            print(f"   {i:4d} " + " ".join(f"{float((tl[i, e] - base_) / 1e3):9.2f}" if tl[i, e] > 0 else f"{'-':>9s}" for e in order), flush=True)
    if any(x.startswith("FR_TL") for x in d):
        tl = K._mod().tri_front_tl().double()
        nt = int((tl[:, 0] > 0).sum())
        base_ = tl[tl > 0].min()
        names = ["x_iss", "x_land", "LN_end", "mma_got_y", "mma_issued", "qk_rdy", "vg_rdy"]
        order = [0, 1, 2, 3, 4, 5, 6]
        print("   tile " + " ".join(f"{n:>10s}" for n in names) + "   (front CTA 10, us from its first stamp)", flush=True)
        for i in range(min(nt, 12)):
            print(f"   {i:4d} " + " ".join(f"{float((tl[i, e] - base_) / 1e3):10.2f}" if tl[i, e] > 0 else f"{'-':>10s}" for e in order), flush=True)
    import os
    if os.environ.get("OUTSAVE"):
        g.replay(); torch.cuda.synchronize()
        torch.save(y.detach().cpu(), os.environ["OUTSAVE"] + f"_L{L}_{'S' if kw.get('starting') else 'E'}.pt")
