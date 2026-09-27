"""Packaged core against the sample-batched one, at the step's shapes."""
import statistics
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit.attn import attention_gated_in_place2, bias_descriptor  # noqa: E402
from attn_sb import attention_gated_sb  # noqa: E402

S, H, DS, NB, dev, bf = 5, 16, 768, 24, "cuda", torch.bfloat16
D = DS // H


def t_us(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    out = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / reps)
    return statistics.median(out)


for L in (384, 768):
    torch.manual_seed(0)
    base = torch.randn(S * L, 4 * DS, device=dev, dtype=bf) * DS ** -0.5
    bias = torch.randn(NB * H, L, L, device=dev, dtype=bf) * 0.3
    bdesc = bias_descriptor(bias)
    outs = []
    variants = [("packaged", attention_gated_in_place2)]
    for gs in ((1,) * S, (2, 3), (2, 2, 1), (3, 2), (5,)):
        variants.append((f"batched {gs}", lambda *a, _g=gs, **kw: attention_gated_sb(*a, groups=_g, **kw)))
    for fn_name, fn in variants:
        qkvg = base.clone()
        q4, k4, v4, g4 = (qkvg.view(S, L, 4 * DS)[..., i * DS:(i + 1) * DS].unflatten(-1, (H, D)) for i in range(4))
        fn(q4, k4, v4, g4, bdesc, 0, "tf32")
        torch.cuda.synchronize()
        outs.append(qkvg[:, :DS].float().clone())
        t = t_us(lambda: fn(q4, k4, v4, g4, bdesc, 0, "tf32"))
        flops = 4 * S * H * L * L * D
        print(f"L{L} {fn_name:<16s} {t:6.1f} us  {flops / t / 1e6:4.0f} TFLOP/s", flush=True)
    print(f"L{L} rel diff {float((outs[1] - outs[0]).norm() / outs[0].norm()):.2e}", flush=True)
