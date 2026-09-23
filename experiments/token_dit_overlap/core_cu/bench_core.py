"""CUDA core against the packaged Triton core, at the step's shapes."""
import statistics
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdit.attn import attention_gated_in_place2, bias_descriptor  # noqa: E402
from core_cu import attn_core  # noqa: E402

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
    qkvg = (torch.randn(S * L, 4 * DS, device=dev) * DS ** -0.5).to(bf)
    q4, k4, v4, g4 = (qkvg.view(S, L, 4 * DS)[..., i * DS:(i + 1) * DS].unflatten(-1, (H, D)) for i in range(4))
    bias = (torch.randn(NB * H, L, L, device=dev) * 0.3).to(bf)
    bdesc = bias_descriptor(bias)
    tt = t_us(lambda: attention_gated_in_place2(q4, k4, v4, g4, bdesc, 0, "tf32"))
    tc = t_us(lambda: attn_core(qkvg, bias, 0, S, H))
    flops = 4 * S * H * L * L * D
    print(f"L{L}: triton {tt:5.1f} us ({flops / tt / 1e6:3.0f} TF/s)   cuda {tc:5.1f} us ({flops / tc / 1e6:3.0f} TF/s)   "
          f"{tt / tc:.2f}x", flush=True)
