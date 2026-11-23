"""Default-route (FP8 off) bytes of the pair modules with a 5090 e4m3 route (TriMul, TriAttn, Transition): sha256 of eager, of a CUDA-graph replay and of
eager after the capture, one line per case. Run once per tree (PYTHONPATH = that tree's src, every MINIWORLD_*FP8* variable unset) and compare the outputs.

    python pair_default_bytes.py [case ...]      # case = trimul:D:kind:L (kind 0 bidir / 1 out / 2 in), triattn:D:H:L, transition:D:n:rows
"""

import hashlib
import os
import sys

import torch

assert not any("FP8" in k for k in os.environ if k.startswith("MINIWORLD")), "unset every MINIWORLD_*FP8* variable"

CASES = [
    "trimul:128:1:256", "trimul:128:2:512", "trimulrf3:128:1:256", "trimulrf3:128:2:384", "trimul:512:1:256", "trimul:64:2:384", "trimul:64:0:256", "trimul:128:0:384", "trimul:256:1:384", "trimul:384:2:256", "trimul:64:1:256",
    "triattn:128:4:256", "triattn:128:4:512", "triattn:256:8:256", "triattn:64:4:256", "triattn:128:4:200",
    "chai1:256:4:256", "chai1:256:4:512", "chai1end:256:4:512", "chai1conf:256:4:512",
    "transition:128:4:16384", "transition:128:2:38477", "transition:256:4:16384", "transition:256:2:8192", "transition:384:4:8192", "transition:512:4:8192",
    "transition:512:2:4096",
]  # fmt: skip


def _init(m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    return m


def build(spec):
    from miniworld_engine.modules import TriangleAttention
    from miniworld_engine.modules.transition import Transition
    from miniworld_engine.modules.triangle import BidirectionalTriangleMultiplication, TriangleMultiplication

    kind, *a = spec.split(":")
    a = [int(v) for v in a]
    g = torch.Generator(device="cpu").manual_seed(sum(a))
    if kind == "transition":
        d, n, rows = a
        m = _init(Transition(d, n, master_weights=True, backend="cuda"), 1).cuda().eval()
        x = torch.randn(rows, d, generator=g).to("cuda", torch.bfloat16)
        return m, (x,)
    if kind in ("trimul", "trimulrf3"):
        d, k, L = a
        if k == 0:
            m = BidirectionalTriangleMultiplication(d, d_hidden=d, master_weights=True, backend="cuda")
        else:
            m = TriangleMultiplication(d, outgoing=k == 1, master_weights=True, backend="cuda", **({"model": "rf3"} if kind == "trimulrf3" else {}))
    elif kind.startswith("chai1"):  # Chai-1's 64-channel heads (sm80_wide.variant_forward_ops; kind d64 of the e4m3 route)
        d, h, L = a
        model = "chai1_confidence" if kind == "chai1conf" else "chai1"
        m = TriangleAttention(d, h, model=model, starting=kind != "chai1end", p_drop=0.0, master_weights=True, backend="cuda")
    else:
        d, h, L = a
        m = TriangleAttention(d, n_head=h, master_weights=True, backend="cuda")
    m = _init(m, 2).cuda().eval()
    x = torch.randn(1, L, L, d, generator=g).to("cuda", torch.bfloat16)
    mask = torch.ones(1, L, device="cuda", dtype=torch.bool)
    mask[:, L - 3 :] = False
    return m, (x, mask)


def sha(t):
    t = t if torch.is_tensor(t) else t[0]
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


for spec in sys.argv[1:] or CASES:
    m, inp = build(spec)

    def step(m=m, inp=inp):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            return m(*inp)

    y0 = step()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        yg = step()
    gr.replay()
    torch.cuda.synchronize()
    y1 = step()
    torch.cuda.synchronize()
    print(spec, sha(y0), sha(yg), sha(y1), flush=True)
    del m, inp, gr, yg, y0, y1
    torch.cuda.empty_cache()
