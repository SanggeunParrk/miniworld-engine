"""EXPERIMENTAL, opt-in (never reached by default): OuterProductMean inference on H100 with the grouped outer product
``O[(i,c), (j,e)] = sum_s A2[(i,c), s] BT[(j,e), s]`` -- the K = S GEMM that grows with the MSA depth and is ~90 % of the call past
depth 4096 -- on e4m3 operands.

  wide prologue, FP8 build (``opm_train._ext(fp8=True)``, ``-DMINIWORLD_OPM_FP8``): LN + both projections + mask as in the default,
  the fp32 accumulator times a static power-of-two ``1 / scale`` per operand, saturating e4m3 cast, A2 / BT written as bytes (half the
  default's bytes), the max |scaled value| of each operand atomically kept in a device buffer (the guard)
  -> cuBLASLt e4m3 GEMM (``opm_lt_f8``, TN, fp32 accumulation, A / B scale pointers = the dequantisation scales, bf16 O as the default
  product route's) per chunk of i rows -> the default epilogue on the fp32 pair-count table (``opm_train._count_table``: exact counts,
  the division stays in the fp32 accumulator).

Scales (``POLICY``):
  ``bound``  from the weights alone: |LN(x)_k| <= sqrt(d_msa - 1) |gamma_k| + |beta_k| for any input, so
             max_c sum_k |W_ck| (sqrt(d_msa - 1) |gamma_k| + |beta_k|) (+ |projection bias|) bounds every A2 / BT value: the cast can
             never saturate, no calibration call, no guard needed (the guard buffer is still written).
  ``calib``  the docs/fp8.md convention: the first eager call measures max |A2|, max |BT| (one extra FP8 prologue with scale 1) and
             takes the power of two above it times ``HEADROOM`` (16); lazy guard ``overflow_peak()`` / ``rescale_all()`` + redo.

Environment (read at import; the module globals can be set before a call / capture): ``MINIWORLD_OPM_FP8`` = ``bound`` | ``calib``,
``MINIWORLD_OPM_FP8_GUARD`` = ``sync`` (eager: redo an overflowing call itself) | ``lazy`` | ``off``, ``MINIWORLD_OPM_FP8_HEADROOM``,
``MINIWORLD_OPM_FP8_FASTACC=1`` (cuBLASLt fast accumulation), ``MINIWORLD_OPM_FP8_MINS`` (one depth gate replacing the shape rule `_gate`;
below it the bf16 default runs unchanged).
"""

from __future__ import annotations

import functools
import math
import os
import weakref
from pathlib import Path

import torch

from miniworld_engine.fp8 import env as _fp8_env

E4M3_MAX = 448.0

POLICY = _fp8_env("MINIWORLD_OPM_FP8") or None
GUARD = os.environ.get("MINIWORLD_OPM_FP8_GUARD", "sync")
HEADROOM = float(os.environ.get("MINIWORLD_OPM_FP8_HEADROOM", "16"))
FAST_ACCUM = os.environ.get("MINIWORLD_OPM_FP8_FASTACC") == "1"
#: the (padded) shapes the FP8 route takes; below them the bf16 default route runs unchanged. Module time, CUDA graph, same-cubin control
#: (docs/fp8.md "OuterProductMean"): FP8 / default 0.74-0.93 at S 1024 from L 256 (1.05 at L 128), 0.62-0.93 at S 2048 (every L), 0.93-1.34
#: at S 512 (no gain). ``MINIWORLD_OPM_FP8_MINS=<s>`` replaces the rule by one depth gate (0 = none).
MIN_S: int | None = int(os.environ["MINIWORLD_OPM_FP8_MINS"]) if "MINIWORLD_OPM_FP8_MINS" in os.environ else None


def _gate(s: int, n: int) -> bool:
    if MIN_S is not None:
        return s >= MIN_S
    return s >= 2048 or (s >= 1024 and n >= 256)


#: the O workspace per row chunk (bytes), as the default product route's
_O_CHUNK_BYTES = 2 << 30

GUARD_STATS = {"calls": 0, "retries": 0, "fallbacks": 0, "capture_fallbacks": 0}
_CACHES: dict = {}  # per module (key: data_ptr of to_left.weight, dropped when that tensor is freed): {"s": [2] dequant scales, "q": [2] 1 / s, "sab": s as two tensors}
_LAZY: dict = {}


def _lazy_buffer(device):
    if device not in _LAZY:
        _LAZY[device] = torch.zeros(2, device=device)
    return _LAZY[device]


def overflow_peak(reset: bool = True) -> float:
    """Largest |scaled A2 / BT value| since the last read (> 448: some call saturated the e4m3 cast). One host sync: once per forward, outside any graph."""
    out = 0.0
    for buf in _LAZY.values():
        out = max(out, float(buf.max().item()))
        if reset:
            buf.zero_()
    return out


def rescale_all(peak: float) -> float:
    """Raise every calibrated module's scales so a forward whose guard read ``peak`` (> 448) fits with room x2 (in place: captured graphs follow); returns the factor."""
    f = 2.0 ** math.ceil(math.log2(peak / E4M3_MAX * 2.0))
    for cache in _CACHES.values():
        _scale_up(cache, f)
    return f


def _scale_up(cache, f: float) -> None:
    cache["s"].mul_(f)
    cache["q"].copy_(1.0 / cache["s"])
    cache["sab"][0].copy_(cache["s"][0:1])
    cache["sab"][1].copy_(cache["s"][1:2])


def pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    """Smallest power of two 2^e with amax / 2^e <= 448 (the TriMul FP8 helper, same floor)."""
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_fp8 import pow2_scale as p2

    return p2(amax)


def bound_scales(lnw, lnb, wa, wb, pbias=None) -> torch.Tensor:
    """[2] dequantisation scales (powers of two) from the weights alone: no input can push a scaled A2 / BT value past 448."""
    cm = lnw.numel()
    g = math.sqrt(cm - 1) * lnw.detach().double().abs() + lnb.detach().double().abs()  # max |LN(x)_k| over every input
    out = []
    for k, w in enumerate((wa, wb)):
        b = w.detach().to(torch.bfloat16).double().abs() @ g  # [32]: the bf16 weights the prologue multiplies with
        if pbias is not None:
            b = b + pbias.detach().double().abs()[k * 32 : (k + 1) * 32]
        out.append(b.max() * (1.0 + 2.0**-6))  # the bf16 rounding of the LayerNorm output (<= 2^-9 relative) and the fp32 sums
    return pow2_scale(torch.stack(out).float()).contiguous()


_LT_F8_CPP = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
// O[(i,c),(j,e)] = sa sb (A2q . BTq^T): e4m3 A2q [M, K], BTq [N, K] (K contiguous: cuBLASLt's TN e4m3 form), fp32 accumulation, bf16 O.
// sa / sb: device fp32 scalars (the dequantisation scales; read at run time, so a captured graph follows an in-place rescale).
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulHeuristicResult_t res[16]; int found, k; };
void opm_lt_f8(torch::Tensor a2, torch::Tensor bt, torch::Tensor out, torch::Tensor sa, torch::Tensor sb, int64_t index, bool fast) {
  TORCH_CHECK(a2.scalar_type() == at::kFloat8_e4m3fn && bt.scalar_type() == at::kFloat8_e4m3fn && out.scalar_type() == torch::kBFloat16
              && a2.stride(1) == 1 && bt.stride(1) == 1 && out.stride(1) == 1 && a2.size(1) == bt.size(1) && a2.size(1) % 16 == 0
              && out.size(0) == a2.size(0) && out.size(1) == bt.size(0) && sa.scalar_type() == torch::kFloat32
              && sb.scalar_type() == torch::kFloat32, "opm_lt_f8: e4m3 A2 [M, K], BT [N, K] (K % 16), bf16 out [M, N], fp32 scales");
  auto ck = [](cublasStatus_t s, const char* w) { TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "opm_lt_f8 ", w, ": status ", int(s)); };
  const int64_t M = a2.size(0), K = a2.size(1), N = bt.size(0);
  static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int>, LtPlan> plans;
  const auto key = std::make_tuple(M, N, K, a2.stride(0), bt.stride(0), out.stride(0), index, int(fast), a2.get_device());
  auto it = plans.find(key);
  if (it == plans.end()) {
    LtPlan p{};
    ck(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    int8_t fa = fast ? 1 : 0;
    ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fa, sizeof(fa)), "fast accum");
    const void* pa0 = sb.data_ptr(); const void* pb0 = sa.data_ptr();   // (the scale pointers are part of the description the heuristic sees)
    ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa0, sizeof(pa0)), "a scale");
    ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb0, sizeof(pb0)), "b scale");
    ck(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8F_E4M3, K, N, bt.stride(0)), "layout BT");    // col-major K x N, op T
    ck(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8F_E4M3, K, M, a2.stride(0)), "layout A2");    // col-major K x M
    ck(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, M, out.stride(0)), "layout O");       // O^T col-major == O row-major
    cublasLtMatmulPreference_t pref;
    ck(cublasLtMatmulPreferenceCreate(&pref), "pref");
    size_t ws = at::cuda::getCUDABlasLtWorkspaceSize();
    ck(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)), "pref ws");

    p.found = 0;
    ck(cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.a, p.b, p.c, p.c, pref, 16, p.res, &p.found), "heuristic");
    cublasLtMatmulPreferenceDestroy(pref);
    TORCH_CHECK(p.found > 0, "opm_lt_f8: no algorithm");
    p.k = index < p.found ? (int)index : 0;
    it = plans.emplace(key, p).first;
  }
  auto& p = it->second;
  const void* pa = sb.data_ptr();   // cuBLAS's A is BT
  const void* pb = sa.data_ptr();
  ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa)), "a scale");
  ck(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb)), "b scale");
  float alpha = 1.f, beta = 0.f;
  cublasStatus_t st = CUBLAS_STATUS_NOT_SUPPORTED;
  for (; p.k < p.found; ++p.k) {   // (an algorithm the heuristic listed but cublasLtMatmul refuses is skipped once, for good)
    const size_t wsz = p.res[p.k].workspaceSize;
    auto ws = torch::empty({(int64_t)std::max<size_t>(wsz, 1)}, a2.options().dtype(torch::kByte));
    st = cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), p.op, &alpha, bt.data_ptr(), p.a, a2.data_ptr(), p.b, &beta, out.data_ptr(), p.c,
                        out.data_ptr(), p.c, &p.res[p.k].algo, ws.data_ptr(), wsz, at::cuda::getCurrentCUDAStream());
    if (st != CUBLAS_STATUS_NOT_SUPPORTED) break;
  }
  ck(st, "matmul");
}
"""

#: heuristic index per padded N (default 0); every index is a valid algorithm, the bits may differ between them
LT_ALGO: dict[int, int] = {}


@functools.lru_cache(maxsize=1)
def _lt_ext():
    from miniworld_engine.kernels._codegraph import CUDA_HOME_STANDIN
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, load_inline_extension

    cuda_home = ensure_cuda_home() or CUDA_HOME_STANDIN
    return load_inline_extension(
        name="miniworld_opm_lt_f8",
        cpp_sources=_LT_F8_CPP,
        functions=["opm_lt_f8"],
        with_cuda=True,
        extra_include_paths=[str(Path(cuda_home) / "targets" / "x86_64-linux" / "include")],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )


def product(a8, b8, s, n, out=None):
    """O = sa sb A2q . BTq^T (bf16). ``s``: (sa, sb) as two separately allocated fp32 [1] device tensors (the scale pointers), or a [2] tensor (copied)."""
    if isinstance(s, torch.Tensor):
        s = (s[0:1].clone(), s[1:2].clone())
    if out is None:
        out = torch.empty((a8.shape[0], b8.shape[0]), device=a8.device, dtype=torch.bfloat16)
    _lt_ext().opm_lt_f8(a8, b8, out, s[0], s[1], LT_ALGO.get(n, 0), FAST_ACCUM)
    return out


def serves(args: list[torch.Tensor]) -> bool:
    """Does the FP8 route take this ``opm_train`` inference call (padded msa [1, S, N, d_msa], bf16, sm_90)?"""
    if POLICY not in ("bound", "calib"):
        return False
    m = args[0]
    if m.dim() != 4 or m.shape[0] != 1 or m.dtype != torch.bfloat16 or not m.is_cuda:
        return False
    if torch.cuda.get_device_capability(m.device) != (9, 0):
        return False
    s, n = m.shape[1], m.shape[2]
    return _gate(s, n) and s % 256 == 0 and n % 64 == 0


def _forms(lnw, lnb, wa, wb, wo, bo, pbias):
    from miniworld_engine.kernels import _capture

    bf = torch.bfloat16

    def build():
        return (
            lnw.detach().float().contiguous(),
            lnb.detach().float().contiguous(),
            wa.detach().to(bf).contiguous(),
            wb.detach().to(bf).contiguous(),
            wo.detach().to(bf).contiguous(),
            bo.detach().to(bf).float().contiguous(),
            pbias.detach().float().contiguous() if pbias is not None else None,
        )

    return _capture.param_forms("opm_fp8_infer", [t for t in (lnw, lnb, wa, wb, wo, bo, pbias) if t is not None], build)


def _cache_for(ext, m, mk, gam, bet, wa16, wb16, eps, pb, lnw, lnb, wa, wb, pbias):
    """The module's static scales (made on its first eager call); None under a capture with no calibration (the default route runs)."""
    key = wa.data_ptr()
    cache = _CACHES.get(key)
    if cache is not None and cache.get("policy") == POLICY:
        return cache
    if torch.cuda.is_current_stream_capturing():
        GUARD_STATS["capture_fallbacks"] += 1
        return None
    if POLICY == "bound":
        s = bound_scales(lnw, lnb, wa, wb, pbias).to(m.device)
    else:  # calib: one FP8 prologue at scale 1 measures max |A2|, max |BT| (saturated values are not used)
        amax = torch.zeros(2, device=m.device)
        ext.opm_prologue_wide(m, mk, gam, bet, wa16, wb16, float(eps), False, pb, False, f8q=torch.ones(2, device=m.device), f8amax=amax)
        s = pow2_scale(amax) * HEADROOM
    cache = _CACHES[key] = {"policy": POLICY, "s": s.contiguous(), "q": (1.0 / s).contiguous(), "sab": (s[0:1].clone(), s[1:2].clone())}
    weakref.finalize(wa, _CACHES.pop, key, None)
    return cache


def inference(args: list[torch.Tensor], eps: float, after: bool, dmin: float, doff: float):
    """z of the FP8 route for ``opm_train._inference_op``'s arguments, or None (no calibration under a capture: the caller runs the default)."""
    from miniworld_engine.kernels.integrations import opm_train as ot

    msa, mask, lnw, lnb, wa, wb, wo, bo = args[:8]
    residual = args[8] if args[8].numel() else None
    pbias = args[9] if args[9].numel() else None
    m = msa[0].contiguous()
    mk = mask[0].contiguous()
    if mk.dtype != torch.bool:
        mk = mk != 0
    n, cm, cz = m.shape[1], m.shape[2], wo.shape[0]
    ext = ot._ext(msa.device, native=True, tma=False, cm=cm, cz=cz, fp8=True)
    gam, bet, wa16, wb16, wo16, ob, pb = _forms(lnw, lnb, wa, wb, wo, bo, pbias)
    cache = _cache_for(ext, m, mk, gam, bet, wa16, wb16, eps, pb, lnw, lnb, wa, wb, pbias)
    if cache is None:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    lazy = GUARD in ("lazy", "off") or capturing
    dkw = {} if (dmin, doff) == (1.0, 0.0) else {"dmin": float(dmin), "doff": float(doff)}
    for _ in range(4):
        amax = _lazy_buffer(m.device) if lazy else torch.zeros(2, device=m.device)
        a8, b8, _stats, mask16, _bits = ext.opm_prologue_wide(
            m, mk, gam, bet, wa16, wb16, float(eps), False, pb, False, f8q=cache["q"], f8amax=amax
        )
        GUARD_STATS["calls"] += 1
        if not lazy:
            peak = float(amax.max().item())
            if peak > E4M3_MAX:
                GUARD_STATS["retries"] += 1
                _scale_up(cache, 2.0 ** math.ceil(math.log2(peak / E4M3_MAX * 2.0)))
                continue
        table = ot._count_table(mask16, dmin, doff)
        rows = max(16, min(n, _O_CHUNK_BYTES // (n * ot.CH * ot.CH * 2) // 16 * 16))
        if rows >= n:
            return ext.opm_epilogue(product(a8, b8, cache["sab"], n), table, wo16, ob, n, n, residual, int(after), **dkw)
        z = torch.empty((1, n, n, cz), device=m.device, dtype=torch.bfloat16)
        o = torch.empty((rows * ot.CH, n * ot.CH), device=m.device, dtype=torch.bfloat16)
        for r0 in range(0, n, rows):
            nr = min(rows, n - r0)
            oc = o[: nr * ot.CH]
            product(a8[r0 * ot.CH : (r0 + nr) * ot.CH], b8, cache["sab"], n, out=oc)
            res = residual[:, r0 : r0 + nr] if residual is not None else None
            ext.opm_epilogue(oc, table[r0 : r0 + nr], wo16, ob, nr, n, res, int(after), out=z[:, r0 : r0 + nr], **dkw)
        return z
    GUARD_STATS["fallbacks"] += 1
    return None
