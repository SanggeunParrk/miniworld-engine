r"""Bidirectional TriangleMultiplication bench — team-gm harness format.

Compares, forward (inference) and forward+backward (training), on H100/bf16:
  - pytorch  = torch.compile(BidirectionalTriangleMultiplication ref)   [oracle + baseline]
  - ours     = BidirV6TriMul (fused front + split back, merged-backward, cute/quack + cuBLAS)
  - dtv1     = a fused bidirectional dt-v1 (dt-v1's own kernels, same architecture)
  - cuequiv  = cuequivariance triangle_multiplicative_update run for both directions (×2)

Sweeps M = L^2 (L = seq_len) × d_pair. ALL COMPILED
(benchmarks/CONVENTIONS.md hard rule: the pytorch
baseline is torch.compile, never eager; ours/dtv1/cuequiv are already-compiled kernels).
Emits the parseable `=== M=.. d_in=.. d_out=.. ===` + `<backend> fwd=.. ms fwd+bwd=.. ms`
format that `benchmarks/runners/plot_bench.py` renders into a table + graph.

Lives under benchmarks/suites because it is an executable benchmark, not package
code.

Run (compute node):
    K=benchmarks/artifacts/triangle_multiplication
    pixi run --frozen bash -c \
      "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; \
       python benchmarks/suites/triangle_multiplication_bidirectional.py" \
      | tee $K/bidirectional.out
    python benchmarks/runners/plot_bench.py $K/bidirectional.out $K --name bidirectional \
      --title 'Bidirectional TriangleMultiplication (H100, bf16)'
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn as nn
import triton

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch, dispatch
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training import BidirV6TriMul
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1_bidir import (
    fused_bidirectional_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
L_LIST = [256, 384, 512, 768, 1024]
D_LIST = [128, 256, 512]


def do_bench(fn, *, grad_to_none=None) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median",
                                   grad_to_none=grad_to_none or [])


def cos(a, b) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


class _DtV1Bidir(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.b, self.h = base, base.d_hidden

    def forward(self, pair):
        b = self.b
        return fused_bidirectional_dtv1(
            pair, None, norm_in_weight=b.ln_pair.weight, norm_in_bias=b.ln_pair.bias,
            p_in_weight=torch.cat([b.to_left.weight, b.to_right.weight], dim=0),
            g_in_weight=torch.cat([b.to_left_gate.weight, b.to_right_gate.weight], dim=0),
            norm_out_weight=b.ln_out.weight, norm_out_bias=b.ln_out.bias,
            p_out_weight=b.to_out.weight, g_out_weight=b.to_gate.weight, h=self.h, eps=1e-5)


class _Cuequiv2(nn.Module):
    """Bidirectional = cuequiv outgoing + incoming on the same input (work-matched ×2)."""

    def __init__(self, d):
        super().__init__()
        self.o = TriangleMultiplication(d_pair=d, d_hidden=d, outgoing=True,
                                        implementation=ImplementationType.CUEQUIVARIANCE).to(DEVICE, DTYPE)
        self.i = TriangleMultiplication(d_pair=d, d_hidden=d, outgoing=False,
                                        implementation=ImplementationType.CUEQUIVARIANCE).to(DEVICE, DTYPE)

    def forward(self, pair):
        return self.o(pair) + self.i(pair)


def _timed(label: str, fwd, full, leaves):
    t_fwd = do_bench(fwd)
    t_full = do_bench(full, grad_to_none=leaves)
    print(f"  {label:13s} fwd={t_fwd:.4f} ms  fwd+bwd={t_full:.4f} ms", flush=True)


def run_shape(L: int, d: int) -> None:
    M = L * L
    print(f"\n=== M={M}  d_in={d}  d_out={d}  L={L}  dtype={DTYPE} ===", flush=True)
    torch.manual_seed(0)
    base = BidirectionalTriangleMultiplication(
        d_pair=d, d_hidden=d, implementation=ImplementationType.PYTORCH).to(DEVICE)
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=d**-0.5)
    base = base.to(DTYPE)

    pair = torch.randn(1, L, L, d, device=DEVICE, dtype=DTYPE)
    g = torch.randn_like(pair)

    # oracle: compiled pytorch ref (forward value for correctness)
    ref_c = torch.compile(base)
    with torch.no_grad():
        y_ref = ref_c(pair).detach()

    def mk(model):
        p = pair.detach().clone().requires_grad_(True)
        params = [pr for pr in model.parameters() if pr.requires_grad]

        def fwd():
            return model(p)

        def full():
            p.grad = None
            for pr in params:
                pr.grad = None
            model(p).backward(g)
        return fwd, full, [p, *params]

    # 1) pytorch (compiled) — baseline
    f, fu, lv = mk(ref_c)
    _timed("pytorch", f, fu, lv)

    # 2) ours
    try:
        ours = BidirV6TriMul(base)
        with torch.no_grad():
            print(f"  ours-fwd cos={cos(y_ref, ours(pair)):.6f}", flush=True)
        f, fu, lv = mk(ours)
        _timed("ours", f, fu, lv)
    except Exception as e:  # noqa: BLE001
        print(f"  ours          [skipped: {type(e).__name__}: {str(e)[:70]}]", flush=True)

    # 3) dtv1 (fused bidirectional)
    try:
        dtv1 = _DtV1Bidir(base)
        with torch.no_grad():
            print(f"  dtv1-fwd cos={cos(y_ref, dtv1(pair)):.6f}", flush=True)
        f, fu, lv = mk(dtv1)
        _timed("dtv1", f, fu, lv)
    except Exception as e:  # noqa: BLE001
        print(f"  dtv1          [skipped: {type(e).__name__}: {str(e)[:70]}]", flush=True)

    # 4) cuequivariance (×2)
    try:
        f, fu, lv = mk(_Cuequiv2(d))
        _timed("cuequivariance", f, fu, lv)
    except Exception as e:  # noqa: BLE001
        print(f"  cuequivariance [skipped: {type(e).__name__}: {str(e)[:70]}]", flush=True)

    del base, pair, g, ref_c
    torch.cuda.empty_cache()


def main() -> None:
    assert torch.cuda.is_available()
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    for d in D_LIST:
        for L in L_LIST:
            if d == 512 and L >= 768:   # d512 L>=768 fwd+bwd activations exceed sane bench mem
                continue
            run_shape(L, d)


if __name__ == "__main__":
    main()
