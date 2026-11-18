"""Shared-resident front retaining half of the input-LN values in registers."""

from miniworld_engine.kernels._device import num_sms
from . import _h100_runtime as T
from . import h100_uni_wide_inference as I


def _config(d, backward):
    return (128, 4, 128, 0, 1, 7)


@T.device_cache
def _kernel(d, h, backward, cfg, f8=False):
    tokens, stages, bn = cfg[:3]
    private = cfg[3] if len(cfg) > 3 else 0
    wait = cfg[4] if len(cfg) > 4 else 1
    cache = cfg[5] if len(cfg) > 5 else d // 128
    smem = tokens * d * 2 + stages * bn * 128 + tokens // 64 * (24576 if backward and private else 16384) + 512
    name = f"mw_b200style_k1ssregtune{'wb' if backward else ''}_n{bn}t{tokens}s{stages}p{private}w{wait}c{cache}"
    name += "f8" if f8 else ""
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        f"-I{I._headers()}",
        f"-I{T.SOURCES / 'wide_forward'}",
        f"-DWIDTH={d}",
        f"-DHIDDEN={h}",
        f"-DBACKWARD={int(backward)}",
        f"-DTOKENS={tokens}",
        f"-DSTAGES={stages}",
        f"-DPRIVATESTORE={private}",
        f"-DWAITN={wait}",
        f"-DCACHEK={cache}",
        "-DMW_MINB=1",
        "-DMW_K1_STREAM=1",
        "-DTMN_SIGMOID_TANH=1",
        f"-DBLOCKN={bn}",
        f"-DKERNEL_NAME={name}",
        *(["-DF8PLANES=1"] if f8 else []),
    ]
    kernel = T.load_unit(
        str(T.compile(T.SOURCES / "b200style/k1ssregtune.cu", flags)),
        "b200style_k1ssregtune",
    ).kernel(name)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem


def front(x, w1, mask, gi, bi, ab, xs, dpl=None, dcat=None, xn=None):
    if dcat is not None:
        from .h100_b200style_ss import front as backward_front

        return backward_front(x, w1, mask, gi, bi, ab, xs, dpl, dcat, xn)
    n, d, h = x.shape[1], x.shape[-1], ab.shape[0] // 2
    m = x.numel() // d  # token-flat: B n n tokens for batched samples (b-major; the pair mask is [B n, n])
    backward = dcat is not None
    cfg = _config(d, backward)
    tokens, stages, bn = cfg[:3]
    private = backward and len(cfg) > 3 and cfg[3]
    kernel, smem = _kernel(d, h, backward, cfg)
    am = I._map(ab, [64, 32], [m, 2 * h], [m * 2])
    fields = [
        I._map(x, [64, tokens], [d, m], [d * 2]),
        I._map(w1, [64, bn], [d, 4 * h], [d * 2]),
        am,
        T._launch_module().tensor_map(
            dpl,
            [16, 64],
            dims=[m, 2 * h],
            strides_bytes=[m * 2],
            swizzle="32B",
            l2="128B",
        )
        if private
        else I._map(dpl, [64, 64], [m, 2 * h], [m * 2])
        if backward
        else am,
        I._map(dcat, [64, 16 if private else 64], [dcat.shape[1], m], [dcat.shape[1] * 2]) if backward else am,
        I._map(xn, [64, tokens], [d, m], [d * 2]) if xn is not None else am,
        mask,
        gi,
        bi,
        xs if backward else None,
        None if backward else xs,
        m,
        n,
        m // tokens,
        int(xn is not None),
    ]
    kernel.launch(
        (min(m // tokens, num_sms(x.device.index)), 1, 1),
        (128 * (tokens // 64 + 1), 1, 1),
        [T._launch_module().Struct(fields)],
        smem,
    )
    return xn is not None


def prescale_pack(weights, qinv, out=None):
    """The packed weights of the FP8-plane front (``front_f8``): the projection rows of plane channel c scaled by qinv[c] (a power of two: exact),
    the gate rows as they are, packed in the front's block order. out: an existing [4 h, d] bf16 buffer to write in place (a captured graph keeps
    reading it)."""
    from ._h100_pack import pack_into_cuda

    wl, wlg, wr, wrg = weights
    h = wl.shape[0]
    dt = wl.dtype
    wl_ = (wl.float() * qinv[:h, None]).to(dt)
    wr_ = (wr.float() * qinv[h:, None]).to(dt)
    if out is None:
        out = wl.new_empty((4 * h, wl.shape[1]))
    pack_into_cuda(out, wl_, wlg, wr_, wrg)
    return out


def front_f8(x, w1, pair_mask, gi, bi, ab, xs, amax=None, transposed=False):
    """FP8 planes of one sample (L % 128 == 0): ab [2 h, L, L] e4m3 (bytes), the planes of the pair (transposed: of its transpose, K-major for
    the incoming contraction) scaled by 1 / scale through the pre-scaled weights ``w1`` (``prescale_pack``); xs [L L, 2] LN_in (mean, rstd) per
    pair token (transposed: at the swapped token); pair_mask bf16 [L, L]; amax: fp32 [1] receives the uint32-ordered max |scaled value|."""
    n, d, h = x.shape[1], x.shape[-1], ab.shape[0] // 2
    if x.shape[0] != 1 or n % 128 or d != 512:
        raise ValueError("the FP8-plane packed front takes one sample, L % 128 == 0, D512")
    m = n * n
    cfg = _config(d, False)
    tokens, stages, bn = cfg[:3]
    kernel, smem = _kernel(d, h, False, cfg, True)
    am = T._launch_module().tensor_map(ab, [64, 32], dims=[m, 2 * h], strides_bytes=[m], swizzle="64B", l2="128B", dtype="uint8")
    if transposed:  # token (r, c) reads pair[c][r]: dims [d][c][r], strides (n d, d) elements
        xm = T._launch_module().tensor_map(x, [64, tokens, 1], dims=[d, n, n], strides_bytes=[n * d * 2, d * 2], swizzle="128B", l2="128B")
    else:
        xm = I._map(x, [64, tokens], [d, m], [d * 2])
    fields = [
        xm,
        I._map(w1, [64, bn], [d, 4 * h], [d * 2]),
        am,
        am,
        am,
        am,
        pair_mask,
        gi,
        bi,
        None,
        xs,
        amax,
        m,
        n,
        m // tokens,
        0,
        int(transposed),
    ]
    kernel.launch(
        (min(m // tokens, num_sms(x.device.index)), 1, 1),
        (128 * (tokens // 64 + 1), 1, 1),
        [T._launch_module().Struct(fields)],
        smem,
    )
