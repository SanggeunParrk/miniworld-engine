# polprof.py CFG TARGET : walk every case of CFG in measurer order (build, graph, 30 timed replays), profile (CUPTI) the kernels of the case whose label contains TARGET
import collections, copy, statistics, sys, torch
from torch.profiler import profile, ProfilerActivity
from foldjet_engine.measure import config
from foldjet_engine.measure.speed import make_step, unpadded_masks
cfg = config.load(sys.argv[1]); target = sys.argv[2]
dev = torch.device("cuda"); flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
for case in config.cases(cfg):
    lab = config.describe(case, cfg)["label"]
    base = config.build(case, device=dev); module = copy.deepcopy(base)
    with unpadded_masks(): inputs = config.inputs(case, module, device=dev)
    train = case.mode == "train"
    step = make_step(module, inputs, backend="cuda", train=train, autocast=config.autocast(case), seed=0, compile_mode=None)
    for _ in range(2): step()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): step()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(25):
        flush.zero_(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
    print(f"{lab[-70:]:70s} {statistics.median(ts):8.1f} us", flush=True)
    if target in lab:
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            flush.zero_(); g.replay(); torch.cuda.synchronize()
        tot = collections.defaultdict(lambda: [0.0, 0])
        for e in pr.events():
            if e.device_type.name == "CUDA":
                tot[e.name[:60]][0] += e.time_range.end - e.time_range.start; tot[e.name[:60]][1] += 1
        for k, (t, n) in sorted(tot.items(), key=lambda kv: -kv[1][0])[:14]: print(f"      {t:7.1f} us x{n:<3d} {k}", flush=True)
        break
    del step, module, inputs, g
