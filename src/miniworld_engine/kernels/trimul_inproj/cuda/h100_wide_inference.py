"""Bidirectional wide (D256/384/512) TriMul inference: K1 + two contractions + streamed K3.

K1 (input LN + four gated projections into packed left|right planes) is the native K1
extended to H = 2D; the two contractions are cuBLAS bmm; K3 (``output_stream.cu``) is a
streamed-operand GEMM that reads the contraction output in its channel-major layout and
has both LayerNorms folded into its weights, so it needs only per-token input-LN
statistics (fp32 mean, rstd) instead of the normalised rows. K1 writes those statistics
(D256/D384, in-kernel LN); at D512 the separate input LN writes them next to x_n.
B1, BF16 pair, FP32 LN affine, L384/768, H100 (sm_90a). Selected schedules are in
``h100_sources/wide_forward/selection.json``; ``MW_WIDE_INFER_CFG`` (JSON) overrides an
entry for development measurements.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

R = T.SOURCES / "wide_forward"
WIDTHS = (256, 384, 512)
LENGTHS = (384, 768)


def supports(width: int, length: int) -> bool:
    return width in WIDTHS and length in LENGTHS


def _k1_smem(D, cfg):
    bi, bj, slots, sk, mb = cfg[:5]
    stream = len(cfg) > 5 and cfg[5] == 2
    if D // 64 % sk or (not stream and slots < D // 64 // sk):
        raise ValueError("K1 slots")
    size = (bi * bj * D * 2 + slots * sk * 8192 + bi * bj // 64 * 8192 + 8 * D
            + ((2 + 2 * slots) * 8 + 127) // 128 * 128)
    if size * mb > 232448:
        raise ValueError("K1 resources")
    return size


def _k3_smem(D, bn, stages):
    size = stages * (128 * 128 + bn * 128) + 2 * 64 * bn * 2 + 16 * D + 4096 + 3 * 512 + 16 * stages + 16
    if size > 232448:
        raise ValueError("K3 resources")
    return size


@lru_cache(None)
def _headers():
    """The inference headers with the wide K1 residency/streaming switches exposed."""
    source = T.SOURCES / "inference"
    dest = T.cache_dir() / ("wide_forward_headers_" + T._source_digest().hex()[:16])
    for name in ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh"):
        content = (source / name).read_text()
        if name == "tmn_kernels.cuh":
            content = content.replace(
                "static constexpr int MINB = NCWG == 1 ? 2 : 1;",
                "static constexpr int MINB = MW_MINB;")
            content = content.replace(
                "W_RESIDENT || NSLOT >= 2 * SPB",
                "W_RESIDENT || (SCHED == 1 && (MW_K1_STREAM || NSLOT >= SPB)) || NSLOT >= 2 * SPB")
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        T.publish_header(path, content)
    return dest


@T.device_cache
def _compile(kind, D, cfg, normalize=True, stats=False, has_mask=True):
    if kind == "k1":
        bi, bj, slots, sk, mb = cfg[:5]
        shared = len(cfg) > 5 and cfg[5]
        stream = int(shared == 2)
        include = "front_shared.cuh" if shared else "tmn_kernels.cuh"
        function = "mw_k1_shared" if shared else "k1_body"
        # <HAS_MASK, LNM (1: in-kernel input LN, 0: pre-normalised), SAVE (LN statistics), EMITX (x_n), bf16 mask, K1ParamsQ
        # (the four projection weights read in place when pad0 = 1; the token mask when pad1 = 1)>
        body = (
            f'#include "{include}"\n'
            f"using C=tmn::K1Cfg<{D},{2 * D},false,{bi},{bj},{slots},{sk},{1 if stream else -1}>;\n"
            'extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB) '
            "void mw_wide_front(__grid_constant__ const tmn::K1ParamsQ p){"
            f"tmn::sm90::{function}<C,{str(has_mask).lower()},{int(normalize)},{str(stats).lower()},"
            f"{str(normalize and not stats).lower()},1,tmn::K1ParamsQ>(p);}}")
        smem = _k1_smem(D, cfg)
        name = "mw_wide_front"
        flags = [f"-DMW_MINB={mb}", f"-DMW_K1_STREAM={stream}"]
    else:
        bn, stages, gfold = cfg
        body = (R / "output_stream.cu").read_text()
        smem = _k3_smem(D, bn, stages)
        name = "mw_wide_output_stream"
        flags = ["-DMW_MINB=1", "-DMW_K1_STREAM=0", f"-DWIDTH={D}", f"-DBLOCKN={bn}",
                 f"-DSTAGES={stages}", f"-DGFOLD={int(gfold)}"]
    flags += ["-DTMN_SIGMOID_TANH=1", "-DTMN_WSKIP=1", "-DTMN_MASK_TEMPLATE=1", "-std=c++17",
              "-O3", "-arch=sm_90a", "--cubin", "-lineinfo", "-I" + str(_headers()), "-I" + str(R)]
    unit = T.load_unit(str(T.compile_text(body, flags)), name)
    k = unit.kernel(name)
    k.set_max_dynamic_smem(smem)
    return k, smem


@triton.jit
def _fold(WP, GO, BO, WPF, WG, GI, BI, WGF, UV, D: tl.constexpr, BLOCK: tl.constexpr, GATE: tl.constexpr):
    """Fold the LayerNorm affines into K3's weights, one output row per program:
    Wpf = bf16(Wp * go), up = sum(Wpf), vp = Wp . bo (axis 1 == 0; K = 2D) and, GATE,
    Wgf = bf16(Wg * gi), ug = sum(Wgf), vg = Wg . bi (axis 1 == 1; K = D)."""
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    if tl.program_id(1) == 0:
        m = k < 2 * D
        w = tl.load(WP + c * 2 * D + k, m, other=0).to(tl.float32)
        wf = (w * tl.load(GO + k, m, other=0)).to(tl.bfloat16)
        tl.store(WPF + c * 2 * D + k, wf, m)
        tl.store(UV + c, tl.sum(wf.to(tl.float32), 0))
        tl.store(UV + D + c, tl.sum(w * tl.load(BO + k, m, other=0), 0))
    else:
        if GATE:
            m = k < D
            w = tl.load(WG + c * D + k, m, other=0).to(tl.float32)
            wf = (w * tl.load(GI + k, m, other=0)).to(tl.bfloat16)
            tl.store(WGF + c * D + k, wf, m)
            tl.store(UV + 2 * D + c, tl.sum(wf.to(tl.float32), 0))
            tl.store(UV + 3 * D + c, tl.sum(w * tl.load(BI + k, m, other=0), 0))


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _normalize_stats_into(dst, stats, x, g, b):
    """x_n = LN(x) (bf16) for K1 and the per-token (mean, rstd) for K3's folded gate."""
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    rstd = torch.rsqrt((xf - mean).square().mean(-1, keepdim=True) + 1e-5)
    dst.copy_(F.layer_norm(xf, (x.shape[-1],), g, b, 1e-5).to(x.dtype))
    stats.copy_(torch.cat([mean, rstd], -1).reshape(stats.shape))


def _selection(D, N):
    sel = dict(T.read_config("wide_forward/selection.json")[f"{D}-{N}"])
    if os.environ.get("MW_WIDE_INFER_CFG"):
        sel.update(json.loads(os.environ["MW_WIDE_INFER_CFG"]))
    return sel


def _map(t, box, dims, strides):
    return T._launch_module().tensor_map(t, box, dims=dims, strides_bytes=strides,
                                         swizzle="128B", l2="128B")


def _sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _run(x, weights, mask):
    wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = weights
    _, N, _, D = x.shape
    H, M = 2 * D, N * N
    sel = _selection(D, N)
    cfg, normalize, gfold = tuple(sel["k1"]), sel["normalize"], sel.get("gfold", False)
    bn, stages = sel["k3s"]
    launch = T._launch_module()

    # K1: input LN (in-kernel, or separately at D512) + gated projections -> ab. The four projection
    # weights are read in place when row-major (no packed copy); the pair mask is formed from the
    # token mask inside K1. The gate operand handed to K3 is x plus the LN statistics (gfold) or x_n.
    quad = (wl, wlg, wr, wrg)
    in_place = all(t.stride() == (D, 1) for t in quad)
    if in_place:
        w_map = launch.TensorMap(b"\0" * 128)
        wq_maps = [_map(t, [64, 32], [D, 2 * D], [D * 2]) for t in quad]
    else:
        w = x.new_empty((8 * D, D))
        T.pack_into(w, wl, wlg, wr, wrg)
        w_map = _map(w, [64, 64], [D, 8 * D], [D * 2])
        wq_maps = [launch.TensorMap(b"\0" * 128)] * 4
    ab = x.new_empty((4 * D, N, N))
    xs = torch.empty((M, 2), device=x.device, dtype=torch.float32) if gfold else None
    xn = torch.empty_like(x) if not (gfold and normalize) else None
    if not normalize:
        if gfold:
            _normalize_stats_into(xn, xs, x, gi, bi)
        else:
            T.normalize_into(xn, x, gi, bi)
    k1, smem1 = _compile("k1", D, cfg, normalize, gfold and normalize, mask is not None)
    a, b, _, _, mb = cfg[:5]
    tj = (N + b - 1) // b
    tiles = ((N + a - 1) // a) * tj
    p1 = launch.Struct([
        _map(x if normalize else xn, [64, b, a], [D, N, N], [D * 2, N * D * 2]),
        w_map,
        _map(ab, [64, 1, 32], [N, N, 4 * D], [N * 2, N * N * 2]),
        mask, gi, bi, ab, xs if normalize else None, xn if normalize else None,
        N, N, tj, tiles, 1, N, 1, 1e-5, N * D, D, int(in_place), int(mask is not None),
        *wq_maps])
    k1.launch((min(tiles, _sms(x.device) * mb), 1, 1), (128 * (a * b // 64 + 1), 1, 1), [p1], smem1)

    # Outgoing and incoming contractions into tri = [out | in].
    tri = x.new_empty((H, N, N))
    torch.bmm(ab[:D], ab[H:H + D].transpose(-1, -2), out=tri[:D])
    torch.bmm(ab[D:H].transpose(-1, -2), ab[H + D:], out=tri[D:])

    # K3: folded output LN + projection, gate, residual.
    y = torch.empty_like(x)
    uv = torch.empty((4, D), device=x.device, dtype=torch.float32)
    wpf = x.new_empty((D, H))
    wgf = x.new_empty((D, D)) if gfold else wg
    _fold[(D, 2 if gfold else 1)](wp, go, bo, wpf, wg, gi, bi, wgf, uv, D,
                                  BLOCK=triton.next_power_of_2(H), GATE=gfold, num_warps=4)
    k3, smem3 = _compile("k3", D, (bn, stages, gfold))
    p3 = launch.Struct([
        _map(tri, [64, 64], [M, H], [M * 2]),
        _map(x if gfold else xn, [64, 128], [D, M], [D * 2]),
        _map(wpf, [64, bn], [H, D], [H * 2]),
        _map(wgf, [64, bn], [D, D], [D * 2]),
        _map(x, [64, 64], [D, M], [D * 2]),
        _map(y, [64, 64], [D, M], [D * 2]),
        uv, xs, M, M // 128])
    k3.launch((min(M // 128, _sms(x.device)), 1, 1), (384, 1, 1), [p3], smem3)
    return y


def _wide_inference_fake(x, weights, mask):
    """Return the input shape and dtype (the residual output)."""
    return torch.empty_like(x)


@opaque(fake=_wide_inference_fake, name="trimul_h100_infer_wide")
def wide_inference(x: torch.Tensor, weights: list[torch.Tensor], mask: torch.Tensor | None) -> torch.Tensor:
    """y = x + bidirectional TriMul(x); weights as in ``h100_inference.inference``; mask None or a bool token mask [N]."""
    with T.native_context(x.device):
        return _run(x.contiguous(), [w.contiguous() for w in weights],
                    None if mask is None else mask.to(torch.bool).contiguous())
