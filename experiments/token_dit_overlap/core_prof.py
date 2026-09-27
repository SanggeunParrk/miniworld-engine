"""The attention core alone at the step's shapes, for NCU."""
import statistics
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
from tdit.attn import attention_gated_in_place2, bias_descriptor  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 768
S, H, DS, NB, dev, bf = 5, 16, 768, 24, "cuda", torch.bfloat16
D = DS // H
qkvg = torch.randn(S * L, 4 * DS, device=dev, dtype=bf) * DS ** -0.5
q4, k4, v4, g4 = (qkvg.view(S, L, 4 * DS)[..., i * DS:(i + 1) * DS].unflatten(-1, (H, D)) for i in range(4))
bias = torch.randn(NB * H, L, L, device=dev, dtype=bf) * 0.3
bdesc = bias_descriptor(bias)
fn = lambda: attention_gated_in_place2(q4, k4, v4, g4, bdesc, 0, "tf32")
for _ in range(5):
    fn()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    fn()
torch.cuda.synchronize()
with torch.cuda.graph(g, stream=s):
    for _ in range(10):
        fn()
out = []
for _ in range(7):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    out.append(a.elapsed_time(b) * 1e3 / 10)
t = statistics.median(out)
flops = 4 * S * H * L * L * D
print(f"L{L}: core {t:.1f} us   {flops / t / 1e6:.0f} TFLOP/s   bias {NB and H * L * L * 2 / 1e6:.1f} MB/block", flush=True)
