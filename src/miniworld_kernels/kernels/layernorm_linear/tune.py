r"""Autotuner + recorded bench for `layernorm_linear` (Triton-`autotune` style).

For every shape (M, d) it sweeps candidate GEMM configs for BOTH fused paths,
verifies each against the torch reference (discards any config whose cosine drops
below threshold — M2's larger tiles can reintroduce the sA-recycle race), times
the survivors with `triton.testing.do_bench`, and keeps the fastest *correct*
config per cell. It then prints the winners in the standard bench `.out` format
(so `benchmarks/runners/plot_bench.py` renders the table + bar chart unchanged) plus a
`[tune]` line per cell naming the chosen config, and writes the best-config map
to `benchmarks/kernels/layernorm_linear/artifacts/tuned_configs.json`.

  - cute        = M1 (separate stats + folded-GEMM epilogue), config = quack GemmConfig
  - cute-fused  = M2 (stats inside the mainloop), config = tile_m/tile_n/cluster/pingpong

Compiles are shape-generic (keyed on dtype/major/config, not M/N/K), so each
distinct config compiles once and is reused across every shape.

Run on a GPU compute node via srun (NEVER the login node):

    pixi run --frozen bash -c \
      "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH QUACK_CACHE_ENABLED=0; \
       python -m miniworld_kernels.kernels.layernorm_linear.tune"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Drop our own dir from sys.path so `import triton` finds the real package, not
# the local `triton/` subpackage (same guard as bench.py).
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]

import torch
import triton

from quack.gemm_config import GemmConfig, get_all_configs

from miniworld_kernels.kernels.layernorm_linear.cute import fold_for_gemm, layernorm_linear_cute
from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
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
COS_OK = 0.999  # bf16 fused GEMM lands at 0.999997 when correct; the race drops to ~0.996

D_LIST = [128, 256, 384, 512, 768]
M_LIST = [128 * 128, 256 * 256, 512 * 512]  # 16384, 65536, 262144
SHAPES = [(M, d, d) for d in D_LIST for M in M_LIST]


# --- M1 candidate configs: all SM90 quack configs (no swap_ab — our epilogue's
#     col/row-vec loads assume A=X, B=W2), deduped, plus the shipped default. ---
def m1_candidates() -> list[GemmConfig]:
    seen, out = set(), []
    for c in get_all_configs():
        if c.device_capacity != 9 or c.swap_ab:
            continue
        key = (c.tile_m, c.tile_n, c.cluster_m, c.cluster_n, c.pingpong, c.is_dynamic_persistent)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# --- M2 candidate configs: pingpong only (non-pingpong+persistent is a known
#     acc-reuse hazard); tile_m in {64,128}; tile_n swept. Correctness is verified
#     per shape so the fragile large-tile races get filtered out automatically. ---
M2_CANDIDATES = [
    dict(tile_m=128, tile_n=128, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=128, tile_n=160, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=128, tile_n=192, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=128, tile_n=208, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=64, tile_n=128, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=64, tile_n=192, cluster_m=1, cluster_n=1, pingpong=True),
    dict(tile_m=64, tile_n=256, cluster_m=1, cluster_n=1, pingpong=True),
]


def do_bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])[0]


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def m1_label(c: GemmConfig) -> str:
    return f"{c.tile_m}x{c.tile_n} cl({c.cluster_m},{c.cluster_n}) pp={int(c.pingpong)} dyn={int(c.is_dynamic_persistent)}"


def m2_label(c: dict) -> str:
    return f"{c['tile_m']}x{c['tile_n']} cl({c['cluster_m']},{c['cluster_n']}) pp={int(c['pingpong'])}"


def tune_backend(call, candidates, label_fn, y_ref) -> tuple[object, float, str, int]:
    """Return (best_config, best_ms, best_label, n_correct) over candidates."""
    best_cfg, best_ms, best_label, n_ok = None, float("inf"), "—", 0
    for cfg in candidates:
        try:
            y = call(cfg)
            c = cos(y, y_ref)
        except Exception as e:  # noqa: BLE001 — invalid tile/cluster -> compile/runtime error
            print(f"      [skip {label_fn(cfg)}] {type(e).__name__}: {str(e)[:80]}")
            continue
        if c < COS_OK:
            print(f"      [bad  {label_fn(cfg)}] cos={c:.6f}")
            continue
        n_ok += 1
        ms = do_bench(lambda cfg=cfg: call(cfg))
        flag = ""
        if ms < best_ms:
            best_cfg, best_ms, best_label, flag = cfg, ms, label_fn(cfg), "  <= best"
        print(f"      [ok   {label_fn(cfg)}] cos={c:.6f}  fwd={ms:.4f} ms{flag}")
    return best_cfg, best_ms, best_label, n_ok


def run_shape(M: int, d_in: int, d_out: int, tuned: dict) -> None:
    print(f"\n=== M={M}  d_in={d_in}  d_out={d_out}  dtype={DTYPE} ===")
    ref = LayerNormLinearRef(d_in, d_out).to(DEVICE, DTYPE)
    x = torch.randn(M, d_in, device=DEVICE, dtype=DTYPE)
    gamma, beta, W, bias = ref.layer_norm_weight, ref.layer_norm_bias, ref.weight, ref.bias
    with torch.no_grad():
        y_ref = ref(x)

    # baselines (torch.compile + TE), fwd only — recorded for the table/graph.
    ref_c = torch.compile(ref)
    print(f"  torch.compile  fwd={do_bench(lambda: ref_c(x)):.4f} ms")
    if HAVE_TE:
        te_mod = te.LayerNormLinear(d_in, d_out, bias=True, params_dtype=DTYPE).to(DEVICE)
        with torch.no_grad():
            te_mod.layer_norm_weight.copy_(gamma)
            te_mod.layer_norm_bias.copy_(beta)
            te_mod.weight.copy_(W)
            te_mod.bias.copy_(bias)
        print(f"  TE             fwd={do_bench(lambda: te_mod(x)):.4f} ms")

    prefold = fold_for_gemm(W, gamma, beta, bias, w2_dtype=x.dtype)

    print("    --- M1 (cute) sweep ---")
    _, m1_ms, m1_lab, m1_n = tune_backend(
        lambda c: layernorm_linear_cute(x, gamma, beta, W, bias, prefolded=prefold, config=c),
        m1_candidates(), m1_label, y_ref,
    )
    print(f"  [tune] M1 best = {m1_lab}  ({m1_n} correct)")
    print(f"  cute           fwd={m1_ms:.4f} ms")

    print("    --- M2 (cute-fused) sweep ---")
    _, m2_ms, m2_lab, m2_n = tune_backend(
        lambda c: layernorm_linear_cute_fused(x, gamma, beta, W, bias, prefolded=prefold, config=c),
        M2_CANDIDATES, m2_label, y_ref,
    )
    print(f"  [tune] M2 best = {m2_lab}  ({m2_n} correct)")
    print(f"  cute-fused     fwd={m2_ms:.4f} ms")

    tuned[f"{M}x{d_in}"] = {"M1": m1_lab, "M1_ms": m1_ms, "M2": m2_lab, "M2_ms": m2_ms}


def main() -> None:
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    if HAVE_TE:
        import transformer_engine
        print(f"transformer_engine {transformer_engine.__version__}")
    print(f"M1 candidates: {len(m1_candidates())}   M2 candidates: {len(M2_CANDIDATES)}")
    tuned: dict = {}
    for shape in SHAPES:
        run_shape(*shape, tuned=tuned)
    out = (
        _HERE.parents[3]
        / "benchmarks"
        / "kernels"
        / "layernorm_linear"
        / "artifacts"
        / "tuned_configs.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tuned, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
