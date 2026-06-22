r"""Baseline + kernel bench for `layernorm_linear`.

Compares, forward and forward+backward, on H100/bf16:
  - `torch.compile` of the eager `LayerNormLinear` reference (correctness oracle),
  - NVIDIA Transformer Engine `te.LayerNormLinear` (if importable),
  - our Triton `layernorm_linear_triton` (once implemented — skipped until then).

Sweeps square LayerNormLinear (d_in == d_out) over hidden dims × token counts M.
Results are rendered to `benchmark/` via `scripts/plot_bench.py` (table + graph).

Run from the repo root (miniworld_kernels is editable-installed), via srun:

    pixi run --frozen bash -c \
      "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; \
       python -m miniworld_kernels.kernels.layernorm_linear.bench"
"""

from __future__ import annotations

import sys
from pathlib import Path

# When run as a path script (`python .../layernorm_linear/bench.py`), Python puts
# this file's dir on sys.path[0] — and it contains a `triton/` subpackage that
# would shadow the real Triton. Drop our own dir so `import triton` resolves to
# the installed package. (miniworld_kernels itself is editable-installed.)
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]

import torch

from miniworld_kernels.kernels.layernorm_linear.cute import fold_for_gemm, layernorm_linear_cute
from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_kernels.kernels.layernorm_linear.interface import layernorm_linear_triton
from miniworld_kernels.kernels.layernorm_linear.reference import LayerNormLinearRef

try:
    import transformer_engine.pytorch as te
    HAVE_TE = True
except Exception as e:  # noqa: BLE001
    print(f"[warn] transformer_engine not importable: {e}")
    te = None
    HAVE_TE = False


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# Square LayerNormLinear (d_in == d_out) over hidden dims, swept over the same
# token counts M as the baseline (= L^2 for L in 128/256/512).
D_LIST = [128, 256, 384, 512, 768]
M_LIST = [128 * 128, 256 * 256, 512 * 512]  # 16384, 65536, 262144
SHAPES = [(M, d, d) for d in D_LIST for M in M_LIST]


def copy_weights(dst, ref: LayerNormLinearRef) -> None:
    """Copy the reference's params into a TE module (matching attribute names)."""
    with torch.no_grad():
        dst.layer_norm_weight.copy_(ref.layer_norm_weight)
        dst.layer_norm_bias.copy_(ref.layer_norm_bias)
        dst.weight.copy_(ref.weight)
        dst.bias.copy_(ref.bias)


import triton


def do_bench(fn, *, grad_to_none: list | None = None) -> float:
    """Median ms via `triton.testing.do_bench` (L2 flush + warmup, repo convention)."""
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8], grad_to_none=grad_to_none or []
    )[0]


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    """Agreement vs the oracle: max abs error, relative Frobenius error, cosine.

    bf16 GEMMs round at ~1e-2 abs, so element-wise relative error blows up on
    near-zero outputs and is meaningless — hence the norm-ratio + cosine.
    """
    a32, b32 = a.float(), b.float()
    abs_err = (a32 - b32).abs().max().item()
    rel_fro = ((a32 - b32).norm() / (b32.norm() + 1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(a32.flatten(), b32.flatten(), dim=0).item()
    return f"  {name:4s} max|abs|={abs_err:.3e}  rel_fro={rel_fro:.3e}  cos={cos:.6f}"


def run_shape(M: int, d_in: int, d_out: int) -> None:
    print(f"\n=== M={M}  d_in={d_in}  d_out={d_out}  dtype={DTYPE} ===")

    ref = LayerNormLinearRef(d_in, d_out).to(DEVICE, DTYPE)
    ref_c = torch.compile(ref)

    x = torch.randn(M, d_in, device=DEVICE, dtype=DTYPE, requires_grad=True)
    g = torch.randn(M, d_out, device=DEVICE, dtype=DTYPE)

    # --- correctness oracle: ONE clean fwd+bwd, capture grads before any timing
    # loop runs (the timing loops below re-run backward many times and would
    # otherwise accumulate the reference's parameter grads). ---
    ref.zero_grad(set_to_none=True)
    x.grad = None
    y_ref = ref_c(x)
    y_ref.backward(g)
    y_ref_val = y_ref.detach().clone()
    dx_ref = x.grad.clone()
    dWl_ref = ref.weight.grad.clone()
    dg_ref = ref.layer_norm_weight.grad.clone()

    te_mod = None
    if HAVE_TE:
        x_te = x.detach().clone().requires_grad_(True)
        te_mod = te.LayerNormLinear(d_in, d_out, bias=True, params_dtype=DTYPE).to(DEVICE)
        copy_weights(te_mod, ref)
        y_te = te_mod(x_te)
        y_te.backward(g)
        print(compare("fwd", y_te, y_ref_val))
        print(compare("dx", x_te.grad, dx_ref))
        print(compare("dWl", te_mod.weight.grad, dWl_ref))
        print(compare("dg", te_mod.layer_norm_weight.grad, dg_ref))

    # --- timing (triton.testing.do_bench zeroes grad_to_none between reps) ---
    ref_grads = [x, *ref.parameters()]

    def fwd_ref():
        ref_c(x)

    def fwdbwd_ref():
        ref_c(x).backward(g)

    print(
        f"  torch.compile  fwd={do_bench(fwd_ref):.4f} ms  "
        f"fwd+bwd={do_bench(fwdbwd_ref, grad_to_none=ref_grads):.4f} ms"
    )

    if HAVE_TE:
        te_grads = [x_te, *te_mod.parameters()]

        def fwd_te():
            te_mod(x_te)

        def fwdbwd_te():
            te_mod(x_te).backward(g)

        print(
            f"  TE             fwd={do_bench(fwd_te):.4f} ms  "
            f"fwd+bwd={do_bench(fwdbwd_te, grad_to_none=te_grads):.4f} ms"
        )

    # Our fused CuTeDSL (quack SM90) kernel. Prologue (fold) is cached — fixed
    # weights — so we time only the stats + fused GEMM (inference-style).
    xd = x.detach()
    gamma, beta, W, bias = ref.layer_norm_weight, ref.layer_norm_bias, ref.weight, ref.bias
    try:
        prefold = fold_for_gemm(W, gamma, beta, bias, w2_dtype=xd.dtype)
        y_cute = layernorm_linear_cute(xd, gamma, beta, W, bias, prefolded=prefold)
    except Exception as e:  # noqa: BLE001
        print(f"  cute           [skipped: {type(e).__name__}: {e}]")
    else:
        print(compare("cute-fwd", y_cute, y_ref))

        def fwd_cute():
            layernorm_linear_cute(xd, gamma, beta, W, bias, prefolded=prefold)

        print(f"  cute           fwd={do_bench(fwd_cute):.4f} ms  (fold cached)")

    # Milestone-2 fused kernel (stats inside the mainloop).
    try:
        y_cf = layernorm_linear_cute_fused(xd, gamma, beta, W, bias, prefolded=prefold)
    except Exception as e:  # noqa: BLE001
        print(f"  cute-fused     [skipped: {type(e).__name__}: {e}]")
    else:
        print(compare("cutefused-fwd", y_cf, y_ref))

        def fwd_cf():
            layernorm_linear_cute_fused(xd, gamma, beta, W, bias, prefolded=prefold)

        print(f"  cute-fused     fwd={do_bench(fwd_cf):.4f} ms  (fold cached)")

    # Our Triton kernel (once implemented).
    try:
        y_tr = layernorm_linear_triton(
            x.detach(), ref.layer_norm_weight, ref.layer_norm_bias, ref.weight, ref.bias,
        )
    except NotImplementedError:
        pass
    else:
        print(compare("triton-fwd", y_tr, y_ref))


def main() -> None:
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    if HAVE_TE:
        import transformer_engine
        print(f"transformer_engine {transformer_engine.__version__}")
    for shape in SHAPES:
        run_shape(*shape)


if __name__ == "__main__":
    main()
