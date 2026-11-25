# dethash.py CFG FILTER : sha1 of the outputs + every parameter gradient of one measurer step (eager, then graph replay), for run-to-run (process-to-process) determinism
import copy, hashlib, sys, torch
from foldjet_engine.measure import config
from foldjet_engine.measure.speed import make_step, unpadded_masks
cfg = config.load(sys.argv[1]); flt = sys.argv[2]
dev = torch.device("cuda")
for case in config.cases(cfg):
    lab = config.describe(case, cfg)["label"]
    if flt not in lab: continue
    base = config.build(case, device=dev); module = copy.deepcopy(base)
    with unpadded_masks(): inputs = config.inputs(case, module, device=dev)
    step = make_step(module, inputs, backend="cuda", train=case.mode == "train", autocast=config.autocast(case), seed=0, compile_mode=None)
    def digest():
        out = step(); outs = out if isinstance(out, (list, tuple)) else [out]
        h = hashlib.sha1()
        for t in [o.detach() for o in outs if torch.is_tensor(o)] + [p.grad.detach() for p in module.parameters() if p.grad is not None]:
            h.update(t.float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]
    d1 = digest(); d2 = digest()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o_ = step()
    g.replay(); torch.cuda.synchronize()
    oo = list(o_) if isinstance(o_, (list, tuple)) else [o_]
    h = hashlib.sha1()
    for t in [t.detach() for t in oo if torch.is_tensor(t)] + [p.grad.detach() for p in module.parameters() if p.grad is not None]:
        h.update(t.float().cpu().numpy().tobytes())
    print(f"{lab[:90]} eager {d1} {d2} graph {h.hexdigest()[:16]}", flush=True)
