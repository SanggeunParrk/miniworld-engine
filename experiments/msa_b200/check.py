"""Module-level correctness of ours vs the PyTorch statements (same weights), inference output and training gradients.
Both are compared against an fp32 reference of the module (the PyTorch path run on fp32 copies), so the bf16 paths
are judged on the same scale:  rel = ||x - ref|| / ||ref||.
    python check.py opm pwa [--mode infer train]"""
import argparse, copy, sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench import make_inputs, make_module, module_fn

ap = argparse.ArgumentParser()
ap.add_argument("ops", nargs="*", default=["opm", "pwa"])
ap.add_argument("--mode", nargs="+", default=["infer", "train"])
ap.add_argument("--L", type=int, default=384)
ap.add_argument("--S", type=int, default=1024)
a = ap.parse_args()


def rel(x, r):
    return ((x.float() - r.float()).norm() / r.float().norm().clamp_min(1e-30)).item()


for op in a.ops:
    msa, pair, mask = make_inputs(op, a.L, a.S)
    ours = make_module(op, "ours")
    torch_mod = make_module(op, "pytorch")
    torch_mod.load_state_dict(ours.state_dict())
    ref_mod = copy.deepcopy(torch_mod).float()
    for mode in a.mode:
        res = {}
        for name, mod, cast in (("ours", ours, torch.bfloat16), ("pytorch", torch_mod, torch.bfloat16), ("fp32", ref_mod, torch.float32)):
            x, z = msa.detach().to(cast).clone(), pair.detach().to(cast).clone()
            mod.train(mode == "train")
            if mode == "infer":
                with torch.no_grad():
                    res[name] = [module_fn(op, mod, x, z, mask)()]
            else:
                x.requires_grad_(True); z.requires_grad_(True)
                out = module_fn(op, mod, x, z, mask)()
                g = torch.randn(out.shape, generator=torch.Generator(device="cuda").manual_seed(1), device="cuda").to(out.dtype)
                grads = torch.autograd.grad(out, [x, z, *mod.parameters()], g, allow_unused=True)
                res[name] = [out] + [gg if gg is not None else torch.zeros_like(p) for gg, p in zip(grads, [x, z, *mod.parameters()])]
        names = ["out"] if mode == "infer" else ["out", "d_msa", "d_pair"] + [n for n, _ in ours.named_parameters()]
        print(f"{op} {mode}: rel err vs fp32 (ours | pytorch)")
        worst = 0.0
        for k, n in enumerate(names):
            eo, ep = rel(res["ours"][k], res["fp32"][k]), rel(res["pytorch"][k], res["fp32"][k])
            worst = max(worst, eo / max(ep, 1e-4))
            print(f"   {n:28s} {eo:.2e} | {ep:.2e}")
        print(f"   worst ours/pytorch error ratio: {worst:.2f}")
