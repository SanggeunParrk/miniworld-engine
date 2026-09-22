"""cuDNN fused attention (torch SDPA, CUDNN_ATTENTION backend) with an additive pair bias, at the token DiT shape."""
import argparse, statistics
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=768); a = p.parse_args()
S, L, H, D, dev, bf = 5, a.length, 16, 48, "cuda", torch.bfloat16
torch.manual_seed(0)
buf = torch.randn(S, L, 4 * H * D, device=dev, dtype=bf)
q, k, v = (buf[..., i * H * D:(i + 1) * H * D].unflatten(-1, (H, D)).transpose(1, 2) for i in range(3))   # [S,H,L,D] views
bias = torch.randn(1, H, L, L, device=dev, dtype=bf)
with torch.no_grad():
    ref = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * D ** -0.5 + bias.float(), -1) @ v.float()
def time_us(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.synchronize(); out = []
    for _ in range(5):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps): g.replay()
        en.record(); torch.cuda.synchronize(); out.append(st.elapsed_time(en) * 1000 / reps)
    return statistics.median(out)
for name, backend, bb in (("cudnn, bias broadcast over samples", SDPBackend.CUDNN_ATTENTION, bias),
                          ("cudnn, bias expanded [S,H,L,L]", SDPBackend.CUDNN_ATTENTION, bias.expand(S, H, L, L).contiguous()),
                          ("efficient (mem-eff), broadcast", SDPBackend.EFFICIENT_ATTENTION, bias)):
    try:
        with sdpa_kernel(backend):
            fn = lambda bb=bb: F.scaled_dot_product_attention(q, k, v, attn_mask=bb, scale=D ** -0.5)
            o = fn(); torch.cuda.synchronize()
            e = float((o.float() - ref).norm() / ref.norm())
            print(f"  L={L} {name:<36s} {time_us(fn):7.1f} us  rel {e:.2e}", flush=True)
    except Exception as ex:
        print(f"  L={L} {name}: FAILED {type(ex).__name__}: {str(ex)[:200]}", flush=True)
