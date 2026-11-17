"""Static-scale table after one trunk run on the shipped route: per site key kind, #sites, log2(scale) min / max per head, sites with a degenerate head (scale <= 2^-20).  usage: scales.py MODEL FORCE TARGET"""
import sys, collections
from pathlib import Path
import torch
SPECS = "/home/psk6950/foldforge-scratch/foldbench/specs_all"
MODEL, FORCE, T = sys.argv[1:4]
def main():
    import foldforge as ff
    import foldforge.models.checkpoints.lock as _LK
    _LK.check_size = lambda _p: None
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as HS
    HS.FP8_PV_WHITELIST[MODEL] = frozenset(FORCE.split(","))
    cfg = {"backend": "miniworld", "precision": "bf16", "execution": {"compile": False, "cuda_graph": False, "bucketing": True, "scope": "trunk"}}
    with ff.load_model(MODEL, config=cfg, device="cuda:0") as model:
        assert HS.enable_fp8_pv_for(MODEL)
        inputs = ff.load_inputs(Path(SPECS) / f"{T}.yaml")
        with torch.no_grad(), model.run_trunk(inputs, seed=0) as trunk: pass
        print("overflow:", HS.take_overflow(), "sites:", len(HS._VS_AUTO), flush=True)
        kinds = collections.defaultdict(list)
        for k, s in HS._VS_AUTO.items():
            kind = ("vq", k[2], k[3]) if isinstance(k, tuple) else ("front-fused", int(s.numel()), None)
            kinds[kind].append(torch.log2(s.float()).cpu())
        for kind, v in sorted(kinds.items(), key=str):
            x = torch.stack(v); print(f"  {kind}: sites {len(v)}, log2(scale) min {x.min():.0f} max {x.max():.0f}, heads at <= -20: {int((x <= -20).sum())} of {x.numel()}", flush=True)
            bad = [i for i, a in enumerate(v) if (a <= -20).any()]
            if bad: print("    degenerate sites (index, per-head log2):", [(i, [int(z) for z in v[i]]) for i in bad[:6]], flush=True)
    print("END", flush=True)
main()
