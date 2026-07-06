import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/snu_hwle/psk/mw-bidir/src")
import torch
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication as TMB)
from miniworld_kernels.modules import ImplementationType
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training_b200 import BidirTriMulB200Train

torch.manual_seed(0)
d = 128; dev = "cuda"
Ls = [int(x) for x in sys.argv[1:]] or [384, 768, 1024]


def rnd(m):
    for n, p in m.named_parameters():
        if n.endswith("ln_pair.weight") or n.endswith("ln_out.weight"): p.data.normal_(1.0, 0.05)
        elif n.endswith(".bias"): p.data.normal_(0.0, 0.02)
        else: p.data.normal_(0.0, 0.05)
    return m


def sc(y, r):
    e = (y - r).abs()
    return (torch.nn.functional.cosine_similarity(y.flatten(), r.flatten(), dim=0).item(),
            (e.mean() / r.abs().mean().clamp_min(1e-12)).item(), e.max().item())


# map: my-module param name -> (ref param name, transpose?)
MAP = {
    "WL": ("to_left.weight", True), "WLg": ("to_left_gate.weight", True),
    "WR": ("to_right.weight", True), "WRg": ("to_right_gate.weight", True),
    "Wg": ("to_gate.weight", True), "Wp": ("to_out.weight", True),
    "ln_in_w": ("ln_pair.weight", False), "ln_in_b": ("ln_pair.bias", False),
    "ln_out_w": ("ln_out.weight", False), "ln_out_b": ("ln_out.bias", False),
}

for L in Ls:
    print(f"=== TRAIN BIDIR L={L} ===", flush=True)
    pair = torch.randn(1, L, L, d, device=dev)
    cot = torch.randn(1, L, L, d, device=dev)
    ref = rnd(TMB(d_pair=d, d_hidden=d, implementation=ImplementationType("pytorch")).to(dev))
    refp = {n: p for n, p in ref.named_parameters()}

    # fp32 reference fwd+bwd
    p32 = pair.clone().double().requires_grad_(True) if False else pair.clone().requires_grad_(True)
    ref32 = ref  # fp32 weights
    y_ref = ref32(pair.float())
    (y_ref * cot.float()).sum().backward()
    gref = {n: p.grad.detach().clone() for n, p in ref.named_parameters()}
    yref = y_ref.detach().float()

    # my sm100 training module (bf16)
    mod = BidirTriMulB200Train(ref).to(dev)
    mod = mod.to(torch.bfloat16)
    for p in mod.parameters(): p.grad = None
    pb = pair.to(torch.bfloat16).requires_grad_(True)
    y = mod(pb)
    (y.float() * cot.float()).sum().backward()

    c, r, mx = sc(y.detach().float(), yref)
    print(f"  fwd  cos={c:.6f} relmean={r:.3e} maxabs={mx:.3e}", flush=True)
    worst = 1.0
    for myn, (refn, tr) in MAP.items():
        g = dict(mod.named_parameters())[myn].grad
        if g is None:
            print(f"  grad {myn:9s} NONE"); worst = 0; continue
        g = g.float().t() if tr else g.float()
        gr = gref[refn].float()
        cc, rr, mm = sc(g, gr)
        worst = min(worst, cc)
        print(f"  grad {myn:9s}<-{refn:22s} cos={cc:.6f} relmean={rr:.3e} maxabs={mm:.3e}", flush=True)
    # input grad
    cc, rr, mm = sc(pb.grad.float(), ref.ln_pair.weight.new_tensor(0)*0 + 0) if False else (0,0,0)
    print(f"  WORST grad cos = {worst:.6f}", flush=True)
