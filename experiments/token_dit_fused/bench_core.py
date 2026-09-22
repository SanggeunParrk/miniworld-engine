"""Attention core alone at the token DiT shape (S samples, 16 heads x 48, pair bias shared by the samples), bf16:
tdit.attn vs torch flex_attention (FLASH = FA4 CuTe on sm90, and its Triton backend) vs Anthropic apb_views."""
import argparse, statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit.attn import attention_gated_in_place, attention_gated_in_place2, bias_descriptor
p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=768); a = p.parse_args()
S, L, H, D, dev, bf = 5, a.length, 16, 48, "cuda", torch.bfloat16
torch.manual_seed(0)
buf = torch.randn(S, L, 4 * H * D, device=dev, dtype=bf)
q, k, v, g = (buf[..., i * H * D:(i + 1) * H * D].unflatten(-1, (H, D)) for i in range(4))
bias = torch.randn(H, L, L, device=dev, dtype=bf)
keep = torch.ones(S, L, device=dev, dtype=torch.bool)
with torch.no_grad():
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    ref = torch.softmax(qf @ kf.transpose(-1, -2) * D ** -0.5 + bias.float()[None], -1) @ vf        # [S,H,L,D]
    ref_gated = (ref.transpose(1, 2) * torch.sigmoid(g.float())).contiguous()

def time_us(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.synchronize()
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph, stream=s): fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(5):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps): gph.replay()
        en.record(); torch.cuda.synchronize(); out.append(st.elapsed_time(en) * 1000 / reps)
    return statistics.median(out)

def rel(x, y): return float((x.float() - y.float()).norm() / y.float().norm())
print(f"core S={S} L={L} H={H} D={D} bf16")
qkvg0 = buf.clone()
def mine():
    buf.copy_(qkvg0)
    return attention_gated_in_place(q, k, v, g, bias, keep)
out = mine().clone()
print(f"  tdit.attn (gate fused, includes a {buf.numel()*2/1e6:.0f} MB reset copy) {time_us(mine):8.1f} us  rel {rel(out, ref_gated):.2e}")
t_copy = time_us(lambda: buf.copy_(qkvg0)); print(f"    (reset copy alone {t_copy:.1f} us)")
# v2 core: pre-scaled logits, as the runner packs them -- q * sm_scale*log2e (copy), bias * log2e
LOG2E = 1.4426950408889634
qs0 = qkvg0.clone(); qs0[..., : H * D] *= D ** -0.5 * LOG2E
bias2 = (bias.float() * LOG2E).to(bf).reshape(1 * H, L, L).contiguous()
bd = bias_descriptor(bias2)
def mine2():
    buf.copy_(qs0)
    return attention_gated_in_place2(q, k, v, g, bd, 0)
out = mine2().clone()
print(f"  tdit.attn v2 (TMA bias, no masks)  {time_us(mine2) - t_copy:8.1f} us (copy subtracted)  rel {rel(out, ref_gated):.2e}")

from torch.nn.attention.flex_attention import flex_attention
def score_mod(score, b, h, qi, ki):
    return score + bias[h, qi, ki]
for backend in ("FLASH", "TRITON"):
    try:
        fa = torch.compile(lambda Q, K, V: flex_attention(Q, K, V, score_mod=score_mod, scale=D ** -0.5,
                                                          kernel_options={"BACKEND": backend}), dynamic=False)
        Qt, Kt, Vt = (t.transpose(1, 2) for t in (q, k, v))
        o = fa(Qt, Kt, Vt)
        print(f"  flex_attention {backend:<7s} (no gate)  {time_us(lambda: fa(Qt, Kt, Vt)):8.1f} us  rel {rel(o, ref):.2e}")
    except Exception as e:
        print(f"  flex_attention {backend}: FAILED {type(e).__name__}: {str(e)[:300]}")
try:
    from opt_core.kernels.apb.fpf_apb.apb_triton import apb_views
    o = apb_views(q, k, v, bias, g, scale=D ** -0.5)
    print(f"  Anthropic apb_views (gate fused)  {time_us(lambda: apb_views(q, k, v, bias, g, scale=D ** -0.5)):8.1f} us  rel {rel(o, ref_gated):.2e}")
except Exception as e:
    print("  apb_views FAILED", e)
