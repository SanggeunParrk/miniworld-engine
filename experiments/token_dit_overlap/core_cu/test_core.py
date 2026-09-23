"""The CUDA core against the packaged Triton core."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdit.attn import attention_gated_in_place2, bias_descriptor  # noqa: E402
from core_cu import attn_core  # noqa: E402

S, H, DS, NB, dev, bf = 5, 16, 768, 24, "cuda", torch.bfloat16
D = DS // H
for L in (256, 384, 768):
    torch.manual_seed(0)
    base = (torch.randn(S * L, 4 * DS, device=dev) * DS ** -0.5).to(bf)
    bias = (torch.randn(NB * H, L, L, device=dev) * 0.3).to(bf)
    for block in (0, 3):
        ref = base.clone()
        q4, k4, v4, g4 = (ref.view(S, L, 4 * DS)[..., i * DS:(i + 1) * DS].unflatten(-1, (H, D)) for i in range(4))
        attention_gated_in_place2(q4, k4, v4, g4, bias_descriptor(bias), block, "tf32")
        got = base.clone()
        attn_core(got, bias, block, S, H)
        torch.cuda.synchronize()
        a, b = got[:, :DS].float(), ref[:, :DS].float()
        err = float((a - b).norm() / b.norm())
        print(f"L{L} block {block}: rel {err:.2e}  {'ok' if err < 6e-3 else 'FAIL'}", flush=True)
