# gf8.py S N bf name=FLAGS... : like gf7 but kernel durations from the profiler (ns resolution), interleaved rounds, L2 flushed, median/min per variant
import statistics, sys, torch, time, os
GAP = float(os.environ.get("GAP", "0"))
from torch.profiler import profile, ProfilerActivity
from miniworld_engine.kernels.swa_dit.cuda import sm100
S = int(sys.argv[1]); N = int(sys.argv[2]); bf = int(sys.argv[3])
H, D, C = 4, 32, 128
dev = torch.device("cuda")
io = () if bf else (f"-DF32IO={sm100.F32IO_MODE}",)
sets = {"base": ()}
for spec in sys.argv[4:]:
    nm, fl = spec.split("=", 1); sets[nm] = tuple(fl.split(",")) if fl else ()
gs = {}
for nm, fl in sets.items():
    sm100.CUBINS[f"x_{nm}_w"] = ("gattn_fwd", ("-DSPLITB=0",) + io + fl)
    sm100.CUBINS[f"x_{nm}_s"] = ("gattn_fwd", ("-DSPLITB=1",) + io + fl)
    sm100.CUBINS[f"x_{nm}_c"] = ("gattn_fwd", ("-DSPLITB=0", "-DCSPLIT=1") + io + fl)
    gs[nm] = sm100.GattnFwd(f"x_{nm}_w", split_cubin=f"x_{nm}_s", f32io=0 if bf else sm100.F32IO_MODE, csplit_cubin=f"x_{nm}_c")
torch.manual_seed(0)
dt = torch.bfloat16 if bf else torch.float32
x = torch.randn(N, H, S, D, device=dev); x = x / x.norm(dim=-1, keepdim=True) * (D ** 0.5)
kvdt = torch.bfloat16 if (bf or sm100.F32IO_MODE == 2) else torch.float32
Qh = x.to(dt); Kh = x.roll(1, 2).contiguous().to(kvdt); Vh = torch.randn(N, H, S, D, device=dev).to(kvdt)
seq = torch.full((N,), S, device=dev, dtype=torch.int32); flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
runs, outs = {}, {}
for nm, g in gs.items():
    O = torch.empty(N * S, C, device=dev, dtype=dt if bf else torch.float32); L = torch.empty(N, H, S, device=dev)
    run, _, _ = g.bind(Qh, Kh, Vh, seq, O=O, LSE=L, log2=True)
    run(); torch.cuda.synchronize(); outs[nm] = (O.clone().float(), L.clone()); runs[nm] = run
names = list(runs); ROUNDS, IT = 6, 8
for _ in range(2):
    for nm in names: flush.zero_(); runs[nm]()
torch.cuda.synchronize()
order = []
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for r in range(ROUNDS):
        for nm in names:
            for _ in range(IT):
                
                if GAP: torch.cuda.synchronize(); time.sleep(GAP)
                flush.zero_(); runs[nm](); order.append(nm)
        torch.cuda.synchronize()
ks = sorted([e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA and "gattn_fwd" in e.name], key=lambda e: e.time_range.start)
assert len(ks) == len(order), (len(ks), len(order))
d = {nm: [] for nm in names}
for e, nm in zip(ks, order): d[nm].append(e.time_range.end - e.time_range.start)
o0, l0 = outs["base"]
for nm in names:
    print(f"S {S} N {N} {'bf16' if bf else 'f32'} {nm:10s} median {statistics.median(d[nm]):7.2f} us  min {min(d[nm]):7.2f}  mean {statistics.mean(d[nm]):7.2f}  |dO| {(outs[nm][0]-o0).abs().max().item():.1e} |dLSE| {(outs[nm][1]-l0).abs().max().item():.1e}")
