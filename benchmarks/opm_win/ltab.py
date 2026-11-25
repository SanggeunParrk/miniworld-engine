"""In-graph A/B of the product route's cuBLASLt heuristic index (opm_train._LT_ALGO) on one OuterProductMean row, fair protocol
(idle -> cool burst; idle -> sustained back-to-back -> timed), PyTorch as the drift reference, outputs checked bitwise.
   python ltab.py ROW L S IDX... --out OUT.jsonl   (IDX -1 = torch.mm)"""
import argparse, copy, dataclasses, json, statistics, time
import torch
from foldjet_engine.measure import config, speed
from foldjet_engine.modules.primitives import unpadded_masks
from foldjet_engine.kernels.integrations import opm_train as ot

ap = argparse.ArgumentParser()
ap.add_argument("row", type=int); ap.add_argument("L", type=int); ap.add_argument("S", type=int); ap.add_argument("idx", type=int, nargs="+")
ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--idle", type=float, default=2.0)
ap.add_argument("--sustain", type=float, default=1.0)
a = ap.parse_args()
dev = torch.device("cuda"); timer = speed._Timer(dev, True)
cfg = config.Config(modules=("OuterProductMean",), rows=(a.row,), lengths=(a.L,), depths="min", modes=("eval",), samples="min")
case = dataclasses.replace(next(iter(config.cases(cfg))), depth=a.S)
base = config.build(case, device=dev)
graphs, outs = {}, {}
npad = (a.L + 63) // 64 * 64
for name in ["pytorch"] + [f"lt{i}" for i in a.idx]:
    module = copy.deepcopy(base)
    with unpadded_masks():
        inputs = config.inputs(case, module, device=dev)
    ot._LT_ALGO.clear()
    if name != "pytorch" and int(name[2:]) >= 0:
        ot._LT_ALGO[npad] = int(name[2:])
    step = speed.make_step(module, inputs, backend="pytorch" if name == "pytorch" else "cuda", train=False, autocast=config.autocast(case),
                           seed=config.seed(case), compile_mode=None)
    side = torch.cuda.Stream(dev); side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        for _ in range(3):
            o = step()
    torch.cuda.current_stream(dev).wait_stream(side); torch.cuda.synchronize(dev)
    outs[name] = (o[0] if isinstance(o, (tuple, list)) else o).detach().clone()
    graphs[name] = (speed._capture(step, side), module, inputs, step)
    torch.cuda.synchronize(dev)
ot._LT_ALGO.clear()
ref = [k for k in outs if k != "pytorch"][0]
for k in outs:
    if k != "pytorch":
        print(k, "bitwise vs", ref, bool(torch.equal(outs[k], outs[ref])), flush=True)
est = {k: max(min(timer.run(g[0].replay, 2)), 0.05) for k, g in graphs.items()}
res = {k: {"cool": [], "sus": []} for k in graphs}
for rep in range(a.reps):
    order = list(graphs); order = order[rep % len(order):] + order[: rep % len(order)]
    for k in order:
        g = graphs[k][0]
        torch.cuda.synchronize(dev); time.sleep(a.idle)
        res[k]["cool"].append(statistics.median(timer.run(g.replay, max(1, min(5, int(40 / est[k]))))))
        torch.cuda.synchronize(dev); time.sleep(a.idle)
        for _ in range(max(2, int(a.sustain * 1000 / est[k]))):
            timer.flush.zero_(); g.replay()
        res[k]["sus"].append(statistics.median(timer.run(g.replay, max(3, min(10, int(100 / est[k]))))))
with open(a.out, "a") as f:
    pc, ps = statistics.median(res["pytorch"]["cool"]), statistics.fmean(res["pytorch"]["sus"])
    for k, v in res.items():
        c, s = statistics.median(v["cool"]), statistics.fmean(v["sus"])
        rec = {"row": a.row, "L": a.L, "S": a.S, "variant": k, "cool_ms": c, "sus_ms": s, "cool_x": pc / c, "sus_x": ps / s, **v}
        f.write(json.dumps(rec) + "\n")
        print(f"{k:<8} cool {c:.3f} ({pc / c:.3f}x) sus {s:.3f} ({ps / s:.3f}x)  {['%.3f' % x for x in v['cool']]} {['%.3f' % x for x in v['sus']]}", flush=True)
