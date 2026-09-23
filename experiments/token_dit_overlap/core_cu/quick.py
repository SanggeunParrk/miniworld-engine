"""Fast loop for the CUDA core: correctness against a torch reference in the exp2 domain (the packaged core's
convention: sm_scale*log2(e) is folded into q and log2(e) into the bias at hoist time), then timing. No Triton, so no
autotune wait."""
import math
import statistics
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_cu import attn_core  # noqa: E402

S, H, DS, NB, dev, bf = 5, 16, 768, 24, "cuda", torch.bfloat16
D = DS // H
LN2 = math.log(2.0)


def ref_attn(qkvg, bias, block, S, L):
    x = qkvg.view(S, L, 4 * DS).float()
    q, k, v, g = (x[..., i * DS:(i + 1) * DS].unflatten(-1, (H, D)).permute(0, 2, 1, 3) for i in range(4))
    s = (q @ k.transpose(-1, -2) + bias[block * H:(block + 1) * H].float()[None]) * LN2   # exp2 -> exp
    o = torch.softmax(s, -1) @ v
    return (o * torch.sigmoid(g)).permute(0, 2, 1, 3).reshape(S * L, DS)


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


TRITON = {384: 15.7, 768: 49.4}                                     # packaged core on this machine, for reference
for L in (int(sys.argv[1]),) if len(sys.argv) > 1 else (384, 768):
    torch.manual_seed(0)
    qkvg = (torch.randn(S * L, 4 * DS, device=dev) * DS ** -0.5).to(bf)
    bias = (torch.randn(NB * H, L, L, device=dev) * 0.3).to(bf)
    ref = ref_attn(qkvg, bias, 1, S, L)
    got = qkvg.clone()
    attn_core(got, bias, 1, S, H)
    torch.cuda.synchronize()
    err = float((got[:, :DS].float() - ref).norm() / ref.norm())
    t = t_us(lambda: attn_core(qkvg, bias, 1, S, H))
    flops = 4 * S * H * L * L * D
    print(f"L{L}: rel {err:.1e} {'ok' if err < 8e-3 else 'FAIL'}   cuda {t:5.1f} us ({flops / t / 1e6:3.0f} TF/s)   "
          f"triton {TRITON[L]:.1f}   {TRITON[L] / t:.2f}x", flush=True)
