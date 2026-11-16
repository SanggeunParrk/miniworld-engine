"""Bitwise: AF3-order OPM inference at S>4096 with the native prologue (main default) vs the wide one."""
import copy, dataclasses, sys, torch
from miniworld_engine.measure import config, speed
from miniworld_engine.modules.primitives import unpadded_masks
from miniworld_engine.kernels.integrations import opm_train
dev = torch.device("cuda")
for row, L, S in [(57, 256, 8192), (57, 384, 6144), (57, 768, 16384), (57, 130, 5120)]:
    cfg = config.Config(modules=("OuterProductMean",), rows=(row,), lengths=(L,), depths="min", modes=("eval",), samples="min")
    case = dataclasses.replace(next(iter(config.cases(cfg))), depth=S)
    base = config.build(case, device=dev); mod = copy.deepcopy(base)
    with unpadded_masks():
        inp = config.inputs(case, mod, device=dev)
    step = speed.make_step(mod, inp, backend="cuda", train=False, autocast=config.autocast(case), seed=config.seed(case), compile_mode=None)
    outs = {}
    for name, f in (("new_wide", opm_train._wide_prologue_deep), ("old_native", lambda s, t: False)):
        opm_train._wide_prologue_deep = f
        o = step(); o = o[0] if isinstance(o, (tuple, list)) else o
        outs[name] = o.clone()
    ref = None
    opm_train._wide_prologue_deep = lambda s, t: not t and s > 4096
    a, b = outs["new_wide"].float(), outs["old_native"].float()
    print(row, L, S, "bitwise" if torch.equal(outs["new_wide"], outs["old_native"]) else "DIFF", "maxabs", float((a - b).abs().max()), "ref_max", float(b.abs().max()), flush=True)
