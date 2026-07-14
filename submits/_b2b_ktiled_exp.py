"""Large-d forward experiment: current split path (module d>=256) vs the fully-fused K-tiled
b2b (transition_b2b_ktiled) on A100. Latency + correctness (cos vs pytorch ref). If ktiled is
faster AND correct, it is the large-d forward optimization (route module d>=256 there)."""
import torch
import triton

from miniworld_kernels.modules import Transition, ImplementationType as IT
from miniworld_kernels.kernels.transition.triton.fused import transition_b2b_ktiled

dev = "cuda"; bf16 = torch.bfloat16
torch.manual_seed(0)
print(f"{'shape':<15}{'split ms':>11}{'ktiled ms':>12}{'speedup':>10}{'cos':>10}   verdict")
print("-" * 74)
for d in (256, 512):
    for L in (384, 768, 1024):
        M = L * L
        mw = Transition(d, implementation=IT.MINIWORLD).to(dev).to(bf16)
        ref = Transition(d, implementation=IT.PYTORCH).to(dev).to(bf16)
        ref.load_state_dict(mw.state_dict())
        x4 = torch.randn(1, L, L, d, device=dev, dtype=bf16) * 0.1
        x2 = x4.reshape(M, d).contiguous()
        wa, wb, ws = mw.expand_a.weight, mw.expand_b.weight, mw.squeeze.weight
        lnw, lnb, eps = mw.ln_in.weight, mw.ln_in.bias, mw.ln_in.eps

        with torch.no_grad():
            ref_out = ref(x4).reshape(M, d).float()
            kt = transition_b2b_ktiled(x2, lnw, lnb, wa, wb, ws, eps).reshape(M, d).float()
        cos = torch.nn.functional.cosine_similarity(kt.flatten(), ref_out.flatten(), dim=0).item()

        run_split = lambda: mw(x4)                                                # noqa: E731
        run_kt = lambda: transition_b2b_ktiled(x2, lnw, lnb, wa, wb, ws, eps)     # noqa: E731
        with torch.no_grad():
            ms_s = triton.testing.do_bench(run_split, warmup=25, rep=100, return_mode="median")
            ms_k = triton.testing.do_bench(run_kt, warmup=25, rep=100, return_mode="median")
        sp = ms_s / ms_k
        verdict = "ktiled FASTER" if sp > 1.03 else ("~same" if sp > 0.97 else "split faster")
        if cos < 0.99: verdict += "  (!! cos low)"
        print(f"d={d} L={L:<8}{ms_s:>11.4f}{ms_k:>12.4f}{sp:>9.2f}x{cos:>10.4f}   {verdict}")
print("\nB2B KTILED EXP DONE")
