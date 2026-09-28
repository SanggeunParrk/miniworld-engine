"""Single-direction wide (D512) TriMul inference: LN, K1, one contraction, gate GEMM, K3.

Hidden width = pair width (H = D = 512), outgoing or incoming.

* LN_in: one compiled pass writes ``xn`` (K1's A operand and the gate GEMM's input).
* K1: the four gated projections into packed left|right planes ``ab`` [2H, N, N] (the wide
  bidirectional shared-A / streamed-weight K1 with the hidden width decoupled from the input width).
* Contraction: outgoing ``left @ right^T``, incoming ``left^T @ right`` per channel (cuBLAS).
* Gate: ``g = bf16(xn @ Wg^T)`` on cuBLAS (exactly the bf16 rounding the update applies).
* K3 (``h100_sources/uni_wide/output.cu``): output projection with LN_out folded into the weight
  (``Wp' = bf16(gamma o Wp)``, per-token mean/rstd from the resident channel-major triangle tile,
  ``P = rs (tri . Wp'^T - mu u) + v``) and the fused ``x + bf16(P) * sigmoid(g)`` epilogue.

B1, BF16 pair, FP32 LN affine, eps 1e-5, L384/768, H100 (sm_90a). Selected schedules are in
``h100_sources/uni_wide/selection.json``.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

R = T.SOURCES / "uni_wide"
WIDTHS = (512,)
LENGTHS = (384, 768)


def supports(width: int, length: int) -> bool:
    return width in WIDTHS and length in LENGTHS


def serves(module, pair: torch.Tensor) -> bool:
    """Shape/parameter contract on top of ``trimul_h100.serves_inference``'s generic checks."""
    return (
        supports(pair.shape[-1], pair.shape[1])
        and module.d_hidden == pair.shape[-1]
        and module.ln_pair.eps == 1e-5
        and module.ln_out.eps == 1e-5
    )


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


@lru_cache(None)
def _headers():
    """The inference headers with the K1 residency/streaming switches exposed."""
    source = T.SOURCES / "inference"
    dest = T.cache_dir() / ("uni_wide_headers_" + T._source_digest().hex()[:16])
    for name in ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh"):
        content = (source / name).read_text()
        if name == "tmn_kernels.cuh":
            content = content.replace(
                "static constexpr int MINB = NCWG == 1 ? 2 : 1;",
                "static constexpr int MINB = MW_MINB;")
            content = content.replace(
                "W_RESIDENT || NSLOT >= 2 * SPB",
                "W_RESIDENT || (SCHED == 1 && (MW_K1_STREAM || NSLOT >= SPB)) || NSLOT >= 2 * SPB")
        T.publish_header(dest / name, content)
    return dest


@T.device_cache
def _compile(kind, D, H, cfg, normalize=False):
    if kind == "k1":
        bi, bj, slots, sk, mb = cfg[:5]
        shared = len(cfg) > 5 and cfg[5]
        stream = int(shared == 2)
        include = "front_shared.cuh" if shared else "tmn_kernels.cuh"
        function = "mw_k1_shared" if shared else "k1_body"
        body = (
            f'#include "{include}"\n'
            f"using C=tmn::K1Cfg<{D},{H},false,{bi},{bj},{slots},{sk},{1 if stream else -1}>;\n"
            'extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB) '
            "void mw_uni_wide_front(__grid_constant__ const tmn::K1Params p){"
            f"tmn::sm90::{function}<C,true,{int(normalize)},false,{str(normalize).lower()},1>(p);}}")
        smem = _k1_smem(D, cfg)
        name = "mw_uni_wide_front"
        flags = [f"-DMW_MINB={mb}", f"-DMW_K1_STREAM={stream}"]
    else:
        slots, waitn, pregs = cfg
        body = (R / "output.cu").read_text()
        smem = 2 * H * 128 + slots * 16384 + 1024 + (2 * slots + 6) * 8
        name = "mw_uni_wide_output"
        flags = ["-DMW_MINB=1", "-DMW_K1_STREAM=0", f"-DWIDTH={D}", f"-DHIDDEN={H}",
                 f"-DMW_NSLOT={slots}", f"-DWAITN={waitn}", f"-DPREGS={pregs}"]
    flags += ["-DTMN_SIGMOID_TANH=1", "-DTMN_WSKIP=1", "-DTMN_MASK_TEMPLATE=1", "-std=c++17",
              "-O3", "-arch=sm_90a", "--cubin", "-lineinfo", "-I" + str(_headers()), "-I" + str(R)]
    unit = T.load_unit(str(T.compile_text(body, flags)), name)
    k = unit.kernel(name)
    k.set_max_dynamic_smem(smem)
    return k, smem


def _map(t, box, dims, strides):
    return T._launch_module().tensor_map(t, box, dims=dims, strides_bytes=strides,
                                         swizzle="128B", l2="128B")


@lru_cache(None)
def _sms(index):
    return torch.cuda.get_device_properties(index).multi_processor_count


def _run(x, weights, mask, outgoing, sel=None):
    wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = weights
    _, N, _, D = x.shape
    H = wl.shape[0]
    sel = sel or T.read_config("uni_wide/selection.json")[f"{D}-{N}"]
    cfg, normalize = tuple(sel["k1"]), sel["normalize"]
    launch = T._launch_module()
    sms = _sms(x.device.index)

    # LN_in (in K1, or one separate pass) + K1: packed gated projections -> ab = [left | right].
    w = x.new_empty((4 * H, D))
    T.pack_into(w, wl, wlg, wr, wrg)
    xn = torch.empty_like(x)
    ab = x.new_empty((2 * H, N, N))
    if not normalize:
        T.normalize_into(xn, x, gi, bi)
    k1, smem1 = _compile("k1", D, H, cfg, normalize)
    a, b, _, _, mb = cfg[:5]
    tj = (N + b - 1) // b
    tiles = ((N + a - 1) // a) * tj
    p1 = launch.Struct([
        _map(x if normalize else xn, [64, b, a], [D, N, N], [D * 2, N * D * 2]),
        _map(w, [64, 64], [D, 4 * H], [D * 2]),
        _map(ab, [64, 1, 32], [N, N, 2 * H], [N * 2, N * N * 2]),
        mask, gi, bi, ab, None, xn if normalize else None,
        N, N, tj, tiles, 1, N, 1, 1e-5, N * D, D, 0, 0])
    k1.launch((min(tiles, sms * mb), 1, 1), (128 * (a * b // 64 + 1), 1, 1), [p1], smem1)

    # One contraction: outgoing tri[c] = left[c] @ right[c]^T, incoming left[c]^T @ right[c].
    tri = x.new_empty((H, N, N))
    left, right = ab[:H], ab[H:]
    if outgoing:
        torch.bmm(left, right.transpose(-1, -2), out=tri)
    else:
        torch.bmm(left.transpose(-1, -2), right, out=tri)

    # Gate GEMM (bf16 out) and the LN_out fold of the output projection.
    g = torch.mm(xn.view(N * N, D), wg.t())
    wpf = wp.float()
    wpp = (wpf * go).to(x.dtype)
    u = wpp.float().sum(1)
    v = wpf @ bo

    # K3: folded output projection + gate + residual.
    y = torch.empty_like(x)
    k3, smem3 = _compile("k3", D, H, tuple(sel["k3"]))
    tiles3 = N * N // 128
    p3 = launch.Struct([
        _map(tri, [64, 64], [N * N, H], [N * N * 2]),
        _map(wpp, [64, 128], [H, D], [H * 2]),
        x, g, y, u, v, tiles3, 0])
    k3.launch((min(tiles3, sms), 1, 1), (384, 1, 1), [p3], smem3)
    return y


def _uni_wide_inference_fake(x, weights, mask, outgoing):
    """Return the input shape and dtype (the residual output)."""
    return torch.empty_like(x)


@opaque(fake=_uni_wide_inference_fake, name="trimul_h100_infer_uni_wide")
def uni_wide_inference(x: torch.Tensor, weights: list[torch.Tensor], mask: torch.Tensor,
                       outgoing: bool) -> torch.Tensor:
    """y = x + single-direction TriMul(x); weights as in ``h100_inference.inference``, mask [N, N]."""
    with T.native_context(x.device):
        return _run(x.contiguous(), [w.contiguous() for w in weights],
                    mask.to(torch.bfloat16).contiguous(), outgoing)
