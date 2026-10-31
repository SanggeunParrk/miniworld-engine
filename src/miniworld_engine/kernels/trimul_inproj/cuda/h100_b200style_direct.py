"""K1 consumes row-major projection weights and the original token mask."""

from miniworld_engine.kernels._device import num_sms
from . import h100_b200style as K
from . import h100_uni_wide_inference as I
from . import _h100_runtime as T


@T.device_cache
def _kernel(d, h, cfg):
    a, b, slots, sk, mb = cfg[:5]
    schedule = cfg[5] if len(cfg) > 5 else 1
    name = "mw_b200style_k1_direct"
    body = (
        '#include "tmn_kernels.cuh"\n'
        f"using C=tmn::K1Cfg<{d},{h},false,{a},{b},{slots},{sk},{schedule}>;\n"
        'extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB) '
        f"void {name}(__grid_constant__ const tmn::K1ParamsQ p){{"
        "tmn::sm90::k1_body<C,true,1,true,false,1,tmn::K1ParamsQ>(p);}\n"
    )
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        f"-I{K._headers(False)}",
        f"-DMW_MINB={mb}",
        "-DMW_K1_STREAM=1",
        "-DTMN_MASK_TEMPLATE=1",
        "-DTMN_SIGMOID_TANH=1",
        "-DTMN_WSKIP=1",
        "-DTMN_K1_BULK_STORE=1",
    ]
    smem = I._k1_smem(d, (*cfg[:5], 2))
    kernel = T.load_unit(
        str(T.compile_text(body, flags)), "b200style_k1_direct"
    ).kernel(name)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem


def front(x, weights, token_mask, gi, bi, ab, xs):
    """x [B, n, n, d]; B > 1: the samples stacked along K1's slow token axis (K1Params.rows = B n, planes [2 h, B n, n]), the token
    mask [B n] read per sample. One sample may take planes padded past n (ab [2 h, np, np], np = ab.shape[-1] >= n, a multiple of 16):
    K1 writes zeros into the pad rows and columns, as the packaged K1 does."""
    rows, n, d, h = x.shape[0] * x.shape[1], x.shape[1], x.shape[-1], ab.shape[0] // 2
    np_ = ab.shape[-1]
    if np_ != n and rows != n:
        raise ValueError("padded planes take one sample")
    cfg = K._config(d)
    a, b, _, _, mb = cfg[:5]
    kernel, smem = _kernel(d, h, cfg)
    tj = -(-np_ // b)
    tiles = (-(-(rows if rows != n else np_) // a)) * tj
    fields = [
        I._map(x, [64, b, a], [d, n, rows], [d * 2, n * d * 2]),
        T._launch_module().TensorMap(b"\0" * 128),
        I._map(ab, [64, 1, 32], [np_, rows if rows != n else np_, 2 * h], [np_ * 2, (rows if rows != n else np_) * np_ * 2]),
        token_mask,
        gi,
        bi,
        ab,
        xs,
        None,
        n,
        np_,
        tj,
        tiles,
        int(np_ % 8 == 0),
        n,
        1,
        1e-5,
        n * d,
        d,
        1,
        1,
        0 if rows == n else rows,   # K1Params.rows (batched samples; 0 = one sample), before K1ParamsQ.tm_wq
        *(I._map(w, [64, 32], [d, h], [d * 2]) for w in weights),
    ]
    param = T._launch_module().Struct(fields)
    assert len(param.pack()) == 1024
    kernel.launch(
        (min(tiles, mb * num_sms(x.device.index)), 1, 1),
        (128 * (a * b // 64 + 1), 1, 1),
        [param],
        smem,
    )
