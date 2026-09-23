"""Does the max-free softmax help the Triton core too? Timed the engine's way (do_bench)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import us  # noqa: E402
from tdit.attn import attention_gated_in_place2, bias_descriptor  # noqa: E402
from attn_ab import attention_ab  # noqa: E402

S, H, DS, NB, dev, bf = 5, 16, 768, 24, "cuda", torch.bfloat16
D = DS // H
for L in (384, 768):
    torch.manual_seed(0)
    base = (torch.randn(S * L, 4 * DS, device=dev) * DS ** -0.5).to(bf)
    bias = (torch.randn(NB * H, L, L, device=dev) * 0.3).to(bf)
    bdesc = bias_descriptor(bias)
    outs = {}
    for name, kw in (("packaged", None), ("ablation copy", {}), ("maxless", {"maxless": True})):
        qkvg = base.clone()
        q4, k4, v4, g4 = (qkvg.view(S, L, 4 * DS)[..., i * DS:(i + 1) * DS].unflatten(-1, (H, D)) for i in range(4))
        fn = (lambda: attention_gated_in_place2(q4, k4, v4, g4, bdesc, 1, "tf32")) if kw is None else \
             (lambda _kw=kw: attention_ab(q4, k4, v4, g4, bdesc, 1, "tf32", **_kw))
        fn()
        torch.cuda.synchronize()
        outs[name] = qkvg[:, :DS].float().clone()
        t = us(fn)
        d = float((outs[name] - outs["packaged"]).norm() / outs["packaged"].norm())
        print(f"L{L} {name:<14s} {t:6.1f} us   vs packaged {d:.1e}", flush=True)
