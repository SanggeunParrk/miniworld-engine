"""Upper bound for a hand-written core: FA4's sm90 kernel at the token DiT shape WITHOUT the pair bias (it has no plain
bias input), against tdit.attn v2 WITH the bias. If FA4 without bias is not well below v2, a custom core cannot pay."""
import argparse, statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit.attn import attention_gated_in_place2, bias_descriptor
p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=768); a = p.parse_args()
S, L, H, D, dev, bf = 5, a.length, 16, 48, "cuda", torch.bfloat16
torch.manual_seed(0)
base = torch.randn(S, L, 4 * H * D, device=dev, dtype=bf); buf = base.clone()
q, k, v, g = (buf[..., i * H * D:(i + 1) * H * D].unflatten(-1, (H, D)) for i in range(4))
bias = torch.randn(H, L, L, device=dev, dtype=bf); bd = bias_descriptor(bias)
def time_us(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.synchronize(); gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph, stream=s): fn()
    torch.cuda.synchronize(); out = []
    for _ in range(5):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps): gph.replay()
        en.record(); torch.cuda.synchronize(); out.append(st.elapsed_time(en) * 1000 / reps)
    return statistics.median(out)
t_copy = time_us(lambda: buf.copy_(base))
print(f"L={L}: tdit.attn v2 (bias + gate)    {time_us(lambda: (buf.copy_(base), attention_gated_in_place2(q, k, v, g, bd, 0))) - t_copy:7.1f} us")
print(f"L={L}: tdit.attn v2, bias load removed   {time_us(lambda: (buf.copy_(base), attention_gated_in_place2(q, k, v, g, bd, 0, _has_bias=False))) - t_copy:7.1f} us")
from flash_attn.cute.interface import flash_attn_func
out = torch.empty(S, L, H, D, device=dev, dtype=bf)
for tag, args in (("strided q/k/v views", (q, k, v)), ("contiguous q/k/v", (q.contiguous(), k.contiguous(), v.contiguous()))):
    try:
        fn = lambda args=args: flash_attn_func(*args, softmax_scale=D ** -0.5)
        fn(); torch.cuda.synchronize()
        print(f"L={L}: FA4 sm90, NO bias, {tag:<20s} {time_us(fn):7.1f} us")
    except Exception as e:
        print(f"L={L}: FA4 {tag} FAILED {type(e).__name__}: {str(e)[:240]}")
