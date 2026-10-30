"""H100 (sm_90a) TriangleAttention at any (d_pair, heads x 32 [or x 16]) the fused d_pair-128 kernels do not cover, bf16.

The attention core (forward and the two backward kernels, ``h100_sources/triattn_sm90.cu`` / ``triattn_bwd_sm90.cu``) is the d_pair-128
path's: heads of 32 channels at any head count, with L a multiple of 128 (the integration pads other lengths and masks the padded
keys).  What surrounds it is width-generic: one hand-written LayerNorm kernel, cuBLAS GEMMs for the projections and the weight
gradients, and the small hand-written kernels of ``h100_sources/triattn_generic_sm90.cu`` between them (see its header).

  forward   LN -> [q | k | v | g | bias] = y Wcat^T (one GEMM) -> head-major scaled / masked bias -> core -> u = sigmoid(g) o -> t = u Wo^T
            -> x + drop o t
  backward  ddrop = dy o drop -> du = ddrop Wo (GEMM), dWo = ddrop^T u -> do, dg -> core backward (delta, key side, query side) ->
            dall = [dq | dk | dv | dg | dbias] -> dyln = dall Wcat (GEMM), dW = dall^T y (GEMM) -> LN backward (+ residual)

A head of 16 channels is zero-padded to 32 in the packed weights (padded q / k / v / g channels are 0 and meet zero columns of Wo); the
softmax scale stays 1 / sqrt(16).
"""

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

from miniworld_engine.kernels._nvcc import (
    ensure_cuda_home,
    gencodes,
    host_flags,
    load_extension,
)
from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as _core_mod

_SRC = Path(__file__).with_name("h100_sources")
D = 32  # head dim the core takes
HEAD_DIMS = (16, 32)  # head widths served (16 is padded to 32)
MAX_HEADS = 16
MAX_C = 512
TOK = _core_mod.TOK


@functools.lru_cache(maxsize=None)
def _glue():
    ensure_cuda_home()
    return load_extension(
        name="triattn_h100_generic_v5",
        sources=[str(_SRC / "triattn_generic_sm90.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            *gencodes("90a"),
            "--use_fast_math",
            "-DNDEBUG",
        ],
        verbose=True,
    )


# out = u W^T (+ x) through cublasLt with D != C (torch.addmm would first copy x into the output), the algorithm picked by timing the
# heuristic's best candidates once per shape (outside a graph capture).
_LT_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <vector>
#define LT_CHECK(x) do { auto st_ = (x); TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, "cublasLt status ", int(st_), " at " #x); } while (0)
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc; cublasLtMatmulAlgo_t algo; size_t ws; bool tuned; };
torch::Tensor lt_mm(torch::Tensor u, torch::Tensor w, c10::optional<torch::Tensor> x, int64_t L, bool trans, c10::optional<torch::Tensor> out_into,
                    c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "bf16 cuda");
  TORCH_CHECK(u.dim() == 2 && w.dim() == 2 && u.stride(1) == 1 && w.stride(1) == 1 && u.size(1) == w.size(1), "shapes");
  c10::cuda::CUDAGuard guard(u.device());
  const int64_t M = u.size(0), K = u.size(1), C = w.size(0);
  const bool has_c = x.has_value();
  if (has_c) TORCH_CHECK(x->scalar_type() == at::kBFloat16 && x->is_contiguous() && x->size(0) == M && x->size(1) == C, "x");
  // bias: bf16 [C] added in the epilogue (fp32, before the one rounding of x + u W^T + bias)
  const bool has_b = bias.has_value();
  if (has_b) TORCH_CHECK(bias->scalar_type() == at::kBFloat16 && bias->is_contiguous() && bias->numel() == C, "bias");
  // trans (ending node): the output / residual rows are the pair's physical rows (b, j, i) of the logical rows (b, i, j): per (b, i) a
  // [C x L] matrix at column stride L C, batched over i with batch stride C
  const int64_t LL = trans ? L * L : M, NB = trans ? M / LL : 1;
  TORCH_CHECK(!trans || (L > 0 && M % LL == 0), "trans needs L with M = B L L");
  // out_into: a contiguous bf16 [M, C] to write into (a chunk's rows of a batch's output), else a new tensor
  if (out_into.has_value())
    TORCH_CHECK(out_into->is_cuda() && out_into->scalar_type() == at::kBFloat16 && out_into->is_contiguous() && out_into->numel() == M * C, "out");
  auto out = out_into.has_value() ? out_into->view({M, C}) : torch::empty({M, C}, u.options());
  static std::map<std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t, int, int64_t>, LtPlan> plans;
  const auto key = std::make_tuple((int)u.get_device(), M, K, C, u.stride(0), w.stride(0), (int)has_c + 2 * (int)trans + 4 * (int)has_b,
                                   trans ? L : 0);
  auto lt = reinterpret_cast<cublasLtHandle_t>(at::cuda::getCurrentCUDABlasLtHandle());
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
  auto it = plans.find(key);
  std::vector<cublasLtMatmulHeuristicResult_t> cands;
  if (it == plans.end() || (!it->second.tuned && cap == cudaStreamCaptureStatusNone)) {
    LtPlan p;
    if (it == plans.end()) {
      LT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
      LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
      LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
      if (has_b) {
        cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
        cudaDataType_t bt = CUDA_R_16BF;
        LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)));
        LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
      }
      LT_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, K, C, w.stride(0)));
      if (trans) {
        LT_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, K, L, u.stride(0)));
        LT_CHECK(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, C, L, L * C));
        int32_t nb = (int32_t)L;
        int64_t sa = 0, sb = L * u.stride(0), sc = C;
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(p.la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &nb, sizeof(nb)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(p.la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(p.lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &nb, sizeof(nb)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(p.lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(p.lc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &nb, sizeof(nb)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(p.lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sc, sizeof(sc)));
      } else {
        LT_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, K, M, u.stride(0)));
        LT_CHECK(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, C, M, C));
      }
    } else p = it->second;
    if (has_b) {   // this call's bias, for the heuristic's alignment checks and the timing runs
      const void* bp = bias->data_ptr();
      LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
    }
    cublasLtMatmulPreference_t pref;
    LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t wsmax = 32u << 20;
    LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsmax, sizeof(wsmax)));
    cands.resize(16);
    int n = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.la, p.lb, p.lc, p.lc, pref, 16, cands.data(), &n));
    cublasLtMatmulPreferenceDestroy(pref);
    TORCH_CHECK(n > 0, "no cublasLt algorithm");
    cands.resize(n);
    p.algo = cands[0].algo; p.ws = cands[0].workspaceSize; p.tuned = false;
    if (cap == cudaStreamCaptureStatusNone && n > 1) {          // time the candidates once: not inside a capture
      auto ws = torch::empty({(int64_t)wsmax + 16}, u.options().dtype(at::kByte));
      auto tmp = torch::empty({M, C}, u.options());
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      const float alpha = 1.f, beta = has_c ? 1.f : 0.f;
      float best = 1e30f; int bi = 0;
      for (int c = 0; c < n; ++c) {
        auto run = [&]() {
          cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
          for (int64_t bb = 0; bb < NB; ++bb) {
            const char* ub = (const char*)u.data_ptr() + bb * LL * u.stride(0) * 2;
            char* tb = (char*)tmp.data_ptr() + bb * LL * C * 2;
            const char* xb = has_c ? (const char*)x->data_ptr() + bb * LL * C * 2 : tb;
            st = cublasLtMatmul(lt, p.op, &alpha, w.data_ptr(), p.la, ub, p.lb, &beta, xb, p.lc, tb, p.lc, &cands[c].algo, ws.data_ptr(),
                                cands[c].workspaceSize, stream);
            if (st != CUBLAS_STATUS_SUCCESS) break;
          }
          return st;
        };
        if (run() != CUBLAS_STATUS_SUCCESS) continue;
        cudaEventRecord(e0, stream);
        for (int r = 0; r < 8; ++r) run();
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) { best = ms; bi = c; }
      }
      cudaEventDestroy(e0); cudaEventDestroy(e1);
      p.algo = cands[bi].algo; p.ws = cands[bi].workspaceSize; p.tuned = true;
    }
    if (it == plans.end()) it = plans.emplace(key, p).first; else it->second = p;
  }
  const LtPlan& p = it->second;
  if (has_b) {     // the descriptor is shared by every call of this shape: set this call's bias
    const void* bp = bias->data_ptr();
    LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
  }
  auto ws = torch::empty({(int64_t)p.ws + 16}, u.options().dtype(at::kByte));
  const float alpha = 1.f, beta = has_c ? 1.f : 0.f;
  for (int64_t bb = 0; bb < NB; ++bb) {
    const char* ub = (const char*)u.data_ptr() + bb * LL * u.stride(0) * 2;
    char* ob = (char*)out.data_ptr() + bb * LL * C * 2;
    const char* xb = has_c ? (const char*)x->data_ptr() + bb * LL * C * 2 : ob;
    LT_CHECK(cublasLtMatmul(lt, p.op, &alpha, w.data_ptr(), p.la, ub, p.lb, &beta, xb, p.lc, ob, p.lc, &p.algo, ws.data_ptr(), p.ws, stream));
  }
  return out;
}
"""


@functools.lru_cache(maxsize=1)
def _lt():
    from miniworld_engine.kernels._codegraph import CUDA_HOME_STANDIN
    from miniworld_engine.kernels._nvcc import load_inline_extension as load_inline

    cuda_home = ensure_cuda_home() or CUDA_HOME_STANDIN  # the toolkit's headers are needed only to build
    return load_inline(
        name="triattn_h100_lt5",
        cpp_sources=_LT_CPP,
        functions=["lt_mm"],
        with_cuda=True,
        extra_include_paths=[
            str(Path(cuda_home) / "targets" / "x86_64-linux" / "include")
        ],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _front():
    """LN + [Wq; Wk; Wv; Wg; Wb] + head-major scaled / masked bias in one kernel (d_pair 256 inference at short L); None without CUTLASS."""
    from miniworld_engine.kernels._nvcc import CutlassMissing, cutlass_include

    try:  # prebuilt or cached: no CUTLASS headers needed; a build without them -> None, as before
        return _front_ext(cutlass_include())
    except CutlassMissing:
        return None


def _front_ext(cutlass_inc):
    return load_extension(
        name="triattn_h100_front3_sm90_v1",
        sources=[str(_SRC / "front_fused3_sm90.cu")],
        extra_include_paths=[str(_SRC), str(_SRC.parent), str(cutlass_inc)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            *host_flags(),
            *gencodes("90a"),
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
        ],
        verbose=False,
    )


#: (d_pair, 32 x heads) -> the longest L served by the one-kernel front (LN + projection + bias); to be tuned per entry
_FRONT_L = {
    (64, 64): 768,
    (64, 128): 768,
    (128, 128): 128,
    (128, 256): 128,
    (256, 256): 128,
    (384, 384): 128,
    (512, 512): 128,
}


def _wall(wcat, hd):
    """The fused front's weights: [4 HD + 64, C] (the bias rows zero-padded to 64), cached on the packed tensor."""
    w = getattr(wcat, "_triattn_wall", None)
    if w is None:
        w = torch.cat(
            [
                wcat[: 4 * hd],
                torch.nn.functional.pad(
                    wcat[4 * hd :], (0, 0, 0, 64 - (wcat.shape[0] - 4 * hd))
                ),
            ]
        ).contiguous()
        wcat._triattn_wall = w
    return w


#: fp16 inference beyond L 256, d_pair <= 256, <= 8 heads: ``ln_bias_fwd`` (LayerNorm + bias projection + head-major bias in one kernel)
LN_BIAS = True


def _half(wcat):
    """The packed weights in fp16 (the fp16 inference GEMMs), cached on the packed tensor (a cast kernel per call otherwise)."""
    w = getattr(wcat, "_triattn_w16", None)
    if w is None:
        w = wcat.half()
        wcat._triattn_w16 = w
    return w


#: fp16 inference beyond L 256 at d_pair 256 with 8 heads x 32: ``front_c256`` (LayerNorm + q | k | v | g + head-major bias
#: in one kernel; replaces ln_bias_fwd and the projection GEMM)
FRONT_C256 = True


def _front_w16(wcat, hd):
    """front_c256's weights: q | k | v | g rows [4 HD, C] fp16 permuted within each 64-row chunk, cached on the packed tensor."""
    w = getattr(wcat, "_triattn_front_w16", None)
    if w is None:
        # within each 64-row chunk, accumulator column a = 8 i + 2 t + e of a lane (t = lane % 4, i = 4 s + j) is output column
        # 32 s + 8 t + 2 j + e: a lane's 16 columns of a row are two runs of 8 (16-byte stores).  Output rows (s, t, j, e) -> (s, j, t, e):
        # views only (no host-to-device copy: the pack may be rebuilt inside a CUDA-graph capture)
        c = wcat.shape[1]
        w = _half(wcat)[: 4 * hd].view(-1, 2, 4, 4, 2, c).permute(0, 1, 3, 2, 4, 5).reshape(4 * hd, c).contiguous()
        wcat._triattn_front_w16 = w
    return w


def supports(d_pair: int, d_hidden: int, n_head: int, length: int) -> bool:
    """Shapes the generic path takes: d_pair a multiple of 8 up to 512, heads of 16 or 32 channels (at most 16 heads), L a multiple
    of 16 (padded to a multiple of 128 by the caller)."""
    if (
        d_pair % 8
        or not 8 <= d_pair <= MAX_C
        or n_head < 1
        or n_head > MAX_HEADS
        or d_hidden % n_head
    ):
        return False
    return d_hidden // n_head in HEAD_DIMS and length >= 16 and length % 16 == 0


def _hp(n_head: int) -> int:
    return (n_head + 7) // 8 * 8


def pack_weights(wq, wk, wv, wg, wb, wo):
    """-> (wcat [4 HD + Hp, C] bf16 = [Wq; Wk; Wv; Wg; Wb (zero rows below H)], wo_p [C, HD] bf16), HD = 32 H (heads padded to 32)."""
    n_head = wb.shape[0]
    hd = wq.shape[0] // n_head
    C = wq.shape[1]
    bf = torch.bfloat16
    ws = (wq, wk, wv, wg, wb, wo)
    if (
        hd <= D
        and wq.dtype in (torch.float32, bf)
        and all(w.dtype == wq.dtype and w.is_cuda and w.is_contiguous() for w in ws)
    ):  # one launch (the packing kernels are replayed with every CUDA-graph call, so their count is latency)
        wall, wo_p = _glue().pack(*ws)
        wcat = wall[: 4 * n_head * D + _hp(n_head)]
        wcat._triattn_wall = wall
        return wcat, wo_p

    def pad_rows(w):
        w = w.to(bf)
        if hd == D:
            return w
        out = w.new_zeros(n_head, D, C)
        out[:, :hd] = w.view(n_head, hd, C)
        return out.view(n_head * D, C)

    wbp = wb.new_zeros(_hp(n_head), C, dtype=bf)
    wbp[:n_head] = wb.to(bf)
    wcat = torch.cat(
        [pad_rows(wq), pad_rows(wk), pad_rows(wv), pad_rows(wg), wbp]
    ).contiguous()
    wo = wo.to(bf)
    if hd != D:
        wop = wo.new_zeros(C, n_head, D)
        wop[:, :, :hd] = wo.view(C, n_head, hd)
        wo = wop.view(C, n_head * D)
    return wcat, wo.contiguous()


def _views(qkvg, B, L, HD, H):
    """q, k, v, g of the GEMM output [M, 4 HD] as [B, N, S, H, 32] views at the common row stride."""
    ld = qkvg.shape[1]
    s = (L * L * ld, L * ld, ld, D, 1)
    base = qkvg.view(-1)
    shp = (B, L, L, H, D)
    return [base.as_strided(shp, s, i * HD) for i in range(4)]


def _forward(
    pair,
    lnw,
    lnb,
    wcat,
    wo_p,
    mask,
    eps,
    drop,
    n_head,
    head_dim,
    need,
    trans=False,
    out=None,
    flip=False,
    gate_bias=None,
    out_bias=None,
):
    B, L, _, C = pair.shape
    H = n_head
    HD = H * D
    M = B * L * L
    g = _glue()
    x = pair.contiguous().view(M, C)
    scale = head_dim**-0.5
    empty = torch.empty(0, device=pair.device)
    # ``flip`` (the ending node on AF3's bias orientation, inference): the bias is made unmasked and masked after the transpose
    front_mask = mask if mask is not None and not flip else empty.bool()
    fused = (
        (not need)
        and (C, HD) in _FRONT_L
        and L <= _FRONT_L[(C, HD)]
        and _front() is not None
        and tuple(wcat.shape) == (4 * HD + _hp(H), C)
    )
    # inference on the fp16 core (h100_sm90.INFERENCE_FP16): q / k / v and the bias in fp16, from the fused front or (L > 256) fp16 GEMMs
    h16 = (not need) and _core_mod.INFERENCE_FP16 and (fused or L > 256)
    # fp16 inference beyond L 256 at d_pair <= 256 and <= 8 heads: LayerNorm, the bias projection and the head-major bias in one kernel
    # (the bias GEMM re-read all of y for its 8 columns)
    # d_pair 256: LayerNorm, q | k | v | g and the bias in one kernel (the y round trip and the projection GEMM gone)
    front256 = (not fused) and h16 and L > 256 and C == 256 and HD == 256 and FRONT_C256
    lnbias = (not fused) and (not front256) and h16 and L > 256 and C <= 256 and H <= 8 and LN_BIAS
    if fused:  # LN + projection + head-major scaled bias: one kernel
        y = mean = rstd = None
        proj, sbias, _ = _front().front(
            x,
            lnw.float().contiguous(),
            lnb.float().contiguous(),
            _wall(wcat, HD),
            front_mask,
            L,
            H,
            HD,
            eps,
            1.0 / scale,
            trans,
            h16,
        )
        qkvg = None
        q, k, v, gt = (
            (proj[i].view(torch.float16) if h16 and i < 3 else proj[i]).view(B, L, L, H, D) for i in range(4)
        )
    elif front256:
        y = mean = rstd = None
        qkvg, sbias = g.front_c256(
            x, lnw.float().contiguous(), lnb.float().contiguous(), _front_w16(wcat, HD), _half(wcat)[4 * HD : 4 * HD + 8], front_mask, eps,
            B, L, H, trans, scale, False,  # (two-CTA multicast of the weight chunks: measured level, 2026-10-30)
        )
    elif lnbias:
        mean = rstd = None
        y, sbias = g.ln_bias_fwd(
            x, lnw.float().contiguous(), lnb.float().contiguous(), wcat[4 * HD : 4 * HD + 8], front_mask, eps, B, L, H, trans, scale
        )
    else:
        y, mean, rstd = g.ln_fwd(
            x, lnw.float().contiguous(), lnb.float().contiguous(), eps, L, trans, h16  # h16: y in fp16 for fp16 GEMMs
        )
    if fused or front256:
        pass
    elif need or L > 256:
        ww = _half(wcat) if h16 else wcat
        qkvg = torch.matmul(y, ww[: 4 * HD].t())  # [M, 4 HD]: q | k | v | g (fp16 with h16, the gate too)
        if not lnbias:
            bias = torch.matmul(y, ww[4 * HD :].t())  # [M, Hp]
            sbias = g.front_bias(
                bias, front_mask, B, L, H, 0, scale
            )
    else:
        # short inference calls: one GEMM for q | k | v | g | bias (a launch less); beyond L 256 the extra tile column costs more
        qkvg = _lt().lt_mm(y, wcat, None, 0, False, None, None)  # [M, 4 HD + Hp]
        sbias = g.front_bias(
            qkvg, front_mask, B, L, H, 4 * HD, scale
        )
    if not fused:
        q, k, v, gt = _views(qkvg, B, L, HD, H)
    if flip:
        sbias = _core_mod.flip_bias(sbias, mask)
    if gate_bias is not None:  # RoseTTAFold3's gate bias, laid out like the gate's (heads padded to 32 channels)
        hd = gate_bias.numel() // H
        gt.add_(F.pad(gate_bias.view(H, hd), (0, D - hd)).to(gt.dtype))
    # the core's epilogue applies the gate: u = sigmoid(g) o (and writes o only when the backward needs it)
    core = _core_mod._core_h16().triattn_fwd_h16 if h16 else _core_mod._core().triattn_fwd
    o, lse, _, u = core(q, k, v, sbias, scale, need, gt, need)
    # (the ending node's out-projection writes its transposed rows as a strided batch over i: one cublasLt GEMM per sample; one GEMM
    # + the residual kernel's transposed store was slower except at d_pair 64 L128, measured 2026-10-20)
    # RoseTTAFold3's out bias in the same epilogue, before its one rounding: added to the bf16 sum afterwards it was a second
    # rounding at the residual's scale
    bo = None if out_bias is None else out_bias.to(torch.bfloat16).contiguous()
    if drop is None:
        out = _lt().lt_mm(
            u.view(M, HD), wo_p, x, L, trans, out, bo
        )  # x + u Wo^T [+ bo]: the residual in the GEMM epilogue (D != C, no staging copy)
    else:
        t = torch.matmul(u.view(M, HD), wo_p.t())
        if bo is not None:
            t = t + bo
        out = g.residual(x, t, drop, L, trans)
    out = out.view(B, L, L, C)
    return out, (y, mean, rstd, qkvg, sbias, o, lse, u)


@torch.autocast("cuda", enabled=False)  # the fp16 core takes the fp16 projections as made: an enclosing bf16 autocast would turn the GEMMs bf16
def inference(
    pair,
    lnw,
    lnb,
    wcat,
    wo_p,
    n_head,
    head_dim,
    mask=None,
    eps=1e-5,
    trans=False,
    out=None,
    flip=False,
    gate_bias=None,
    out_bias=None,
):
    """pair + TriangleAttention(pair), starting node, no grad; wcat, wo_p = pack_weights(...); L a multiple of 128; `out`: a contiguous
    tensor like pair to write into."""
    return _forward(
        pair, lnw, lnb, wcat, wo_p, mask, eps, None, n_head, head_dim, False, trans, out, flip, gate_bias, out_bias
    )[0]


def _mm_f32(a, b):
    """a @ b with bf16 operands and an fp32 result (fp32 accumulation, no bf16 rounding of the result)."""
    return torch.mm(a, b, out_dtype=torch.float32)


def train_forward(
    pair,
    lnw,
    lnb,
    wq,
    wk,
    wv,
    wg,
    wb,
    wo,
    mask,
    eps,
    drop,
    n_head,
    head_dim,
    trans=False,
):
    """-> (out [B, L, L, C], (y, mean, rstd, qkvg, sbias, o, lse, u)): the tensors the backward reads (o, u as [B, N, S, H, 32]).  Every
    returned tensor is a fresh allocation (none aliases an argument), so it can be an output of a custom op."""
    wcat, wo_p = pack_weights(wq, wk, wv, wg, wb, wo)
    return _forward(
        pair, lnw, lnb, wcat, wo_p, mask, eps, drop, n_head, head_dim, True, trans
    )


def train_backward(
    dy,
    pair,
    lnw,
    lnb,
    wq,
    wk,
    wv,
    wg,
    wb,
    wo,
    eps,
    drop,
    n_head,
    head_dim,
    saved,
    trans=False,
):
    """Gradients (pair, lnw, lnb, wq, wk, wv, wg, wb, wo) of `train_forward`, each in its input's dtype."""
    y, mean, rstd, qkvg, sbias, o, lse, u = saved
    B, L, _, C = pair.shape
    H, hd = n_head, head_dim
    HD = H * D
    M = B * L * L
    wcat, wo_p = pack_weights(wq, wk, wv, wg, wb, wo)
    lw = lnw.float().contiguous()
    x = pair.contiguous().view(M, C)
    o, u = o.view(M, HD), u.view(M, HD)
    g = _glue()
    e = _core_mod._bwd()
    dy = dy.contiguous().view(M, C)
    ddrop = g.gate_bwd_in(dy, drop if drop is not None else dy.new_empty(0), L, trans)
    du = torch.matmul(ddrop, wo_p)
    dwo = _mm_f32(ddrop.t(), u)
    dall = torch.empty(M, wcat.shape[0], device=x.device, dtype=torch.bfloat16)
    do, delta = g.gate_bwd_out(du, qkvg, o, dall, 3 * HD, H, L)
    shp = (B, L, L, H, D)
    q, k, v, _ = _views(qkvg, B, L, HD, H)
    do5 = do.view(shp)
    scale = hd**-0.5
    dall5 = dall.view(-1).as_strided(
        shp, (L * L * dall.shape[1], L * dall.shape[1], dall.shape[1], D, 1), 0
    )

    def slot(i):  # columns [i HD, (i + 1) HD) of dall
        return dall5.as_strided(shp, dall5.stride(), i * HD)

    e.triattn_bwd_kv(q, k, v, sbias, do5, lse, delta, scale, slot(1), slot(2))
    dbias = torch.zeros(B, H, L, L, device=x.device, dtype=torch.float32)
    _core_mod.bwd_q(e, q, k, v, sbias, do5, lse, delta, scale, dbias, slot(0))
    g.assemble(dbias, dall, H, L, HD)
    dyln = torch.matmul(dall, wcat)  # bf16 (the reference module's own precision here)
    dx, part = g.ln_bwd(dyln, x, dy, lw, mean, rstd, L, trans)
    dg_db = part.sum(0)
    dw = _mm_f32(dall.t(), y)  # [4 HD + Hp, C]

    def unpad(w):  # [H * 32, C] -> [H * hd, C]
        return w if hd == D else w.view(H, D, -1)[:, :hd].reshape(H * hd, -1)

    # (clones: the outputs of a custom op may not alias each other, and these are slices of one buffer)
    dwq, dwk, dwv, dwg = (
        unpad(dw[i * HD : (i + 1) * HD]).to(wq.dtype).clone() for i in range(4)
    )
    dwb = dw[4 * HD : 4 * HD + H].to(wb.dtype).clone()
    dwo = dwo if hd == D else dwo.view(C, H, D)[:, :, :hd].reshape(C, H * hd)
    return (
        dx.view(B, L, L, C),
        dg_db[0].to(lnw.dtype).clone(),
        dg_db[1].to(lnb.dtype).clone(),
        dwq,
        dwk,
        dwv,
        dwg,
        dwb,
        dwo.to(wo.dtype).clone(),
    )


class TriAttnGeneric(torch.autograd.Function):
    """pair + drop o TriangleAttention(pair) (starting node), any served (d_pair, heads); parameters bf16 or fp32 (gradients come back in
    their dtype); drop = the module's broadcast dropout scale [B, L, C] bf16 or None.
    (The module path goes through the custom ops of ``integrations.triattn_h100_sm90``, which call `train_forward` / `train_backward`.)"""

    @staticmethod
    def forward(
        ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, n_head, head_dim
    ):
        need = any(ctx.needs_input_grad)
        wcat, wo_p = pack_weights(wq, wk, wv, wg, wb, wo)
        out, saved = _forward(
            pair, lnw, lnb, wcat, wo_p, mask, eps, drop, n_head, head_dim, need
        )
        if need:
            ctx.save_for_backward(
                pair,
                lnw,
                lnb,
                wq,
                wk,
                wv,
                wg,
                wb,
                wo,
                drop if drop is not None else pair.new_empty(0),
                *saved,
            )
            ctx.dims = (eps, n_head, head_dim)
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop, *saved = ctx.saved_tensors
        eps, n_head, head_dim = ctx.dims
        grads = train_backward(
            dy,
            pair,
            lnw,
            lnb,
            wq,
            wk,
            wv,
            wg,
            wb,
            wo,
            eps,
            drop if drop.numel() else None,
            n_head,
            head_dim,
            saved,
        )
        return (*grads, None, None, None, None, None)


def triangle_attention(
    pair,
    lnw,
    lnb,
    wq,
    wk,
    wv,
    wg,
    wb,
    wo,
    n_head,
    head_dim,
    mask=None,
    eps=1e-5,
    drop=None,
):
    """pair + drop o TriangleAttention(pair) for the starting node; the caller transposes for the ending node."""
    return TriAttnGeneric.apply(
        pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, n_head, head_dim
    )
