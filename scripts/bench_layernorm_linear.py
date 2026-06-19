"""Baseline comparison: NVIDIA Transformer Engine ``LayerNormLinear`` vs a
``torch.compile``-d eager (LayerNorm -> Linear) reference.

Fused LayerNorm+GEMM is the op we are about to develop in
``kernels/layernorm_linear``. This script establishes the reference point:
forward + backward latency and numerical agreement on H100, bf16, over the
AF3-style shapes miniworld-kernels cares about.

Run inside the te-env pixi environment::

    cd te-env && pixi run python ../scripts/bench_layernorm_linear.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import transformer_engine.pytorch as te
    HAVE_TE = True
except Exception as e:  # noqa: BLE001
    print(f"[warn] transformer_engine not importable: {e}")
    te = None
    HAVE_TE = False


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


# AF3-style shapes: (tokens M = batch*seq, in_features, out_features).
SHAPES = [
    (512 * 512, 384, 384),
    (256 * 256, 384, 384),
    (128 * 128, 768, 768),
    (512 * 512, 768, 3072),  # MLP-style fan-out
]


class TorchLNLinear(nn.Module):
    """Eager LayerNorm -> Linear, the thing we hand to torch.compile."""

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(in_features, eps=eps)
        self.fc = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.ln(x))


def copy_weights(te_mod: nn.Module, ref: TorchLNLinear) -> None:
    """Make the TE module and the reference numerically identical."""
    with torch.no_grad():
        # TE LayerNormLinear exposes layer_norm_weight / layer_norm_bias / weight / bias.
        te_mod.layer_norm_weight.copy_(ref.ln.weight)
        te_mod.layer_norm_bias.copy_(ref.ln.bias)
        te_mod.weight.copy_(ref.fc.weight)
        te_mod.bias.copy_(ref.fc.bias)


def do_bench(fn, *, n_warmup: int = 25, n_iter: int = 100) -> float:
    """Median wall time (ms) via CUDA events."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def max_err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a32, b32 = a.float(), b.float()
    abs_err = (a32 - b32).abs().max().item()
    rel_err = ((a32 - b32).abs() / (b32.abs() + 1e-6)).max().item()
    return abs_err, rel_err


def run_shape(M: int, d_in: int, d_out: int) -> None:
    print(f"\n=== M={M}  d_in={d_in}  d_out={d_out}  dtype={DTYPE} ===")

    ref = TorchLNLinear(d_in, d_out).to(DEVICE, DTYPE)
    ref_c = torch.compile(ref)

    x = torch.randn(M, d_in, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_te = x.detach().clone().requires_grad_(True)
    g = torch.randn(M, d_out, device=DEVICE, dtype=DTYPE)

    # --- torch.compile reference (also used as the correctness oracle) ---
    y_ref = ref_c(x)
    y_ref.backward(g)

    # --- Transformer Engine ---
    if HAVE_TE:
        te_mod = te.LayerNormLinear(d_in, d_out, bias=True, params_dtype=DTYPE).to(DEVICE)
        copy_weights(te_mod, ref)
        y_te = te_mod(x_te)
        y_te.backward(g)

        a, r = max_err(y_te, y_ref)
        print(f"  fwd  max|abs|={a:.3e}  max|rel|={r:.3e}")
        a, r = max_err(x_te.grad, x.grad)
        print(f"  dx   max|abs|={a:.3e}  max|rel|={r:.3e}")

    # --- timing: forward, then forward+backward ---
    def fwd_ref():
        ref_c(x)

    def fwdbwd_ref():
        x.grad = None
        ref_c(x).backward(g)

    print(f"  torch.compile  fwd={do_bench(fwd_ref):.4f} ms  fwd+bwd={do_bench(fwdbwd_ref):.4f} ms")

    if HAVE_TE:
        def fwd_te():
            te_mod(x_te)

        def fwdbwd_te():
            x_te.grad = None
            te_mod(x_te).backward(g)

        print(f"  TE             fwd={do_bench(fwd_te):.4f} ms  fwd+bwd={do_bench(fwdbwd_te):.4f} ms")


def main() -> None:
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    if HAVE_TE:
        print(f"transformer_engine {getattr(te, '__version__', '?')}")
    for shape in SHAPES:
        run_shape(*shape)


if __name__ == "__main__":
    main()
