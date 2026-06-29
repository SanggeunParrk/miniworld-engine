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

from miniworld_kernels.kernels.layernorm_linear import (
    layernorm_linear,            # inference dispatch (cute best: M2 N<=256 / M1)
    layernorm_linear_fn,         # trainable cute
    layernorm_linear_triton,     # portable inference forward
    layernorm_linear_triton_fn,  # trainable portable
)
from miniworld_kernels.kernels.layernorm_linear.cute import fold_for_gemm
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
    """Canonical bench: 4 backends x {inference fwd, training fwd+bwd}.

    Backends — all COMPILED (see benchmark/BENCHMARKING.md; eager is never benched):
      pytorch = torch.compile(naive ref) | TE | triton (portable) | cute (best variant).
    The "fwd" column is inference (cute = tuned dispatch w/ cached fold; triton = fused
    kernel); the "fwd+bwd" column is training (cute = layernorm_linear_fn; triton =
    layernorm_linear_triton_fn). pytorch/TE forward is identical in both.
    """
    print(f"\n=== M={M}  d_in={d_in}  d_out={d_out}  dtype={DTYPE} ===")
    eps = 1e-5
    ref = LayerNormLinearRef(d_in, d_out).to(DEVICE, DTYPE)
    ref_c = torch.compile(ref)
    x = torch.randn(M, d_in, device=DEVICE, dtype=DTYPE, requires_grad=True)
    g = torch.randn(M, d_out, device=DEVICE, dtype=DTYPE)
    gamma, beta, W, bias = ref.layer_norm_weight, ref.layer_norm_bias, ref.weight, ref.bias
    xd = x.detach()

    # oracle (compiled pytorch naive) — capture once before timing loops re-run backward
    ref.zero_grad(set_to_none=True); x.grad = None
    y_ref = ref_c(x); y_ref.backward(g)
    y_ref_val = y_ref.detach().clone()

    def leaves():  # fresh requires_grad leaves for a fwd+bwd timing loop
        return [t.detach().clone().requires_grad_(True) for t in (x, gamma, beta, W, bias)]

    # 1) pytorch naive — COMPILED (never eager)
    rg = [x, *ref.parameters()]
    print(f"  pytorch        fwd={do_bench(lambda: ref_c(x)):.4f} ms  "
          f"fwd+bwd={do_bench(lambda: ref_c(x).backward(g), grad_to_none=rg):.4f} ms")

    # 2) TE
    if HAVE_TE:
        te_mod = te.LayerNormLinear(d_in, d_out, bias=True, params_dtype=DTYPE).to(DEVICE)
        copy_weights(te_mod, ref)
        x_te = x.detach().clone().requires_grad_(True)
        print(compare("TE-fwd", te_mod(x_te), y_ref_val))
        teg = [x_te, *te_mod.parameters()]
        print(f"  TE             fwd={do_bench(lambda: te_mod(x_te)):.4f} ms  "
              f"fwd+bwd={do_bench(lambda: te_mod(x_te).backward(g), grad_to_none=teg):.4f} ms")

    # 3) triton (portable): inference forward + trainable fwd+bwd
    try:
        y_tr = layernorm_linear_triton(xd, gamma, beta, W, bias, eps)
    except Exception as e:  # noqa: BLE001
        print(f"  triton         [skipped: {type(e).__name__}: {e}]")
    else:
        print(compare("triton-fwd", y_tr, y_ref_val))
        xt, gt, bt, Wt, bit = leaves()
        trg = [xt, gt, bt, Wt, bit]
        print(f"  triton         fwd={do_bench(lambda: layernorm_linear_triton(xd, gamma, beta, W, bias, eps)):.4f} ms  "
              f"fwd+bwd={do_bench(lambda: layernorm_linear_triton_fn(xt, gt, bt, Wt, bit, eps).backward(g), grad_to_none=trg):.4f} ms")

    # 4) cute (best): inference = tuned dispatch w/ cached fold; training = layernorm_linear_fn
    try:
        prefold = fold_for_gemm(W, gamma, beta, bias, w2_dtype=xd.dtype)
        y_cu = layernorm_linear(xd, gamma, beta, W, bias, eps, prefolded=prefold)
    except Exception as e:  # noqa: BLE001
        print(f"  cute           [skipped: {type(e).__name__}: {e}]")
    else:
        print(compare("cute-fwd", y_cu, y_ref_val))
        xc, gc, bc, Wc, bic = leaves()
        cg = [xc, gc, bc, Wc, bic]
        print(f"  cute           fwd={do_bench(lambda: layernorm_linear(xd, gamma, beta, W, bias, eps, prefolded=prefold)):.4f} ms  "
              f"fwd+bwd={do_bench(lambda: layernorm_linear_fn(xc, gc, bc, Wc, bic, eps).backward(g), grad_to_none=cg):.4f} ms")


def main() -> None:
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    if HAVE_TE:
        import transformer_engine
        print(f"transformer_engine {transformer_engine.__version__}")
    for shape in SHAPES:
        run_shape(*shape)


if __name__ == "__main__":
    main()
