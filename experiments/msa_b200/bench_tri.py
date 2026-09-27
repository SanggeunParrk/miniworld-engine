"""TriangleAttention (starting node, d_pair 128, 4 heads x 32) on this B200: PyTorch vs Anthropic opt_core vs ours, inference and
training (forward + backward to the input and every parameter, dropout 0).  CUDA-graph replay medians (bench.timeit), plus the
sustained power-capped time of each.  opt_core's triangle-attention rows are forward-only (their backward is the stock op), so the
Anthropic training cell is n/a.
    python bench_tri.py --L 384 768"""
import argparse, os, sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bench import timeit
from energy_sol import sustained
from tri_b200 import TriangleAttentionB200
from miniworld_engine.modules.triangle_attention.module import TriangleAttention
from miniworld_engine.modules.exceptions import ImplementationType as IT
sys.path.insert(0, os.environ["OPT_CORE_DIR"])
from opt_core.kernels import triattn as ta
from einops import rearrange


def make(impl, seed=0):
    torch.manual_seed(seed)
    cls = TriangleAttentionB200 if impl == "ours" else TriangleAttention
    m = cls(128, 4, starting=True, implementation=IT.PYTORCH, p_drop=0.0)
    torch.nn.init.normal_(m.to_out.weight, std=0.02)                 # the zero init would hide the projections' gradients
    with torch.no_grad():
        m.ln_pair.weight.add_(0.1 * torch.randn_like(m.ln_pair.weight)); m.ln_pair.bias.add_(0.1 * torch.randn_like(m.ln_pair.bias))
    return m.cuda().to(torch.bfloat16)


def anthropic_forward(m, pair, word):
    B, L, _, _ = pair.shape
    H = m.n_head
    x = m.ln_pair(pair)
    q, k, v = (rearrange(f(x), "B I J (H D) -> B I H J D", H=H).contiguous() for f in (m.to_query, m.to_key, m.to_value))
    bias = rearrange(m.to_bias(x), "B J K H -> B H J K")[:, None].float()
    o = ta.triangle_attention(q, k, v, bias, None, None, word=word)
    o = rearrange(o, "B I H J D -> B I J (H D)")
    return pair + m.to_out(torch.sigmoid(m.to_gate(x)) * o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[384])
    ap.add_argument("--check", type=int, default=1)
    a = ap.parse_args()
    for L in a.L:
        g = torch.Generator(device="cuda").manual_seed(0)
        pair = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        gout = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        mods = {"pytorch": make("pytorch"), "ours": make("ours")}
        mods["ours"].load_state_dict(mods["pytorch"].state_dict())
        if a.check:
            ref = make("pytorch").float(); ref.load_state_dict({k: v.float() for k, v in mods["pytorch"].state_dict().items()})
            x32 = pair.float().requires_grad_(True)
            y32 = ref(x32); gr = torch.autograd.grad(y32, [x32, *ref.parameters()], gout.float())
            for name in ("pytorch", "ours"):
                m = mods[name]; xb = pair.clone().requires_grad_(True)
                y = m(xb); gg = torch.autograd.grad(y, [xb, *m.parameters()], gout)
                rel = lambda u, w: ((u.float() - w.float()).norm() / w.float().norm()).item()
                ey = rel(y - pair, y32 - x32)
                eg = max(rel(u, w) for u, w in zip(gg, gr))
                print(f"L={L} {name:8s} rel err vs fp32 module: out(sans residual) {ey:.2e}  worst grad {eg:.2e}", flush=True)
            del ref, x32, y32, gr
        rows = []
        for name in ("pytorch", "anthropic", "ours"):
            m = mods["ours" if name == "ours" else "pytorch"]
            m.eval()
            with torch.no_grad():
                f = (lambda m=m: anthropic_forward(m, pair, "k2b")) if name == "anthropic" else (lambda m=m: m(pair))
                inf = timeit(f); inf_s = sustained(f, secs=2.0)
            if name == "anthropic":
                tr = tr_s = None
            else:
                m.train(); xb = pair.clone().requires_grad_(True); params = [xb, *m.parameters()]
                step = lambda m=m, xb=xb, params=params: torch.autograd.grad(m(xb), params, gout)
                tr = timeit(step); tr_s = sustained(step, secs=2.0)
            rows.append((name, inf, inf_s, tr, tr_s))
            fmt = lambda v: "n/a" if v is None else f"{v*1e3:8.1f} us"
            print(f"L={L} {name:10s} infer {fmt(inf)} (sustained {fmt(inf_s['ms'])})   train {fmt(tr)} (sustained {fmt(None if tr_s is None else tr_s['ms'])})", flush=True)


if __name__ == "__main__":
    main()
