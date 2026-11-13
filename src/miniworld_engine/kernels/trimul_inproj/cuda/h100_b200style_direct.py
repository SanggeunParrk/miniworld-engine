"""K1 consumes row-major projection weights and the original token mask."""

from miniworld_engine.kernels._device import num_sms
from . import h100_b200style as K
from . import h100_uni_wide_inference as I
from . import _h100_runtime as T


def _block(cfg):
    """Weight block rows = the n of one wgmma (cfg[6]): 64 ([32 gate | 32 proj], 32 plane channels) or 128 (64 channels)."""
    return cfg[6] if len(cfg) > 6 else 64


def _fold(cfg):
    """n128 blocks: the kernel takes W_g / 2, W_p / 2 and zeroes masked A rows (tmn_kernels.cuh TMN_K1_FOLD)."""
    return _block(cfg) == 128


def _smem(d, cfg):
    """K1Cfg::SMEM of the direct K1 (tmn_kernels.cuh): z tile, weight ring, staging buffers, LN affine, barriers."""
    a, b, slots, sk, mb = cfg[:5]
    if _block(cfg) == 64:
        return I._k1_smem(d, (*cfg[:5], 2))
    bw = _block(cfg)
    if d // 64 % sk:
        raise ValueError("K1 slots")
    tile = 0 if _ring(cfg) else a * b * d * 2  # schedule 5: the z chunks travel through the weight ring
    bars = 2 + 2 * slots
    size = tile + slots * sk * bw * 128 + a * b // 64 * 2 * bw // 2 * 128 + 8 * d + (bars * 8 + 127) // 128 * 128
    if size * mb > 232448:
        raise ValueError("K1 resources")
    return size


@T.device_cache
def _kernel(d, h, cfg, lnm=1, extra=()):
    """lnm / extra: measurement builds only (lnm 0 = no input LayerNorm, extra e.g. -DTMN_K1_DIAG=2)."""
    a, b, slots, sk, mb = cfg[:5]
    schedule = cfg[5] if len(cfg) > 5 else 1
    name = "mw_b200style_k1_direct"
    body = (
        '#include "tmn_kernels.cuh"\n'
        f"using C=tmn::K1Cfg<{d},{h},false,{a},{b},{slots},{sk},{schedule}>;\n"
        'extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB) '
        f"void {name}(__grid_constant__ const tmn::K1ParamsQ p){{"
        f"tmn::sm90::k1_body<C,true,{lnm},true,false,1,tmn::K1ParamsQ>(p);}}\n"
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
        *([f"-DTMN_K1_BN={_block(cfg)}"] if _block(cfg) != 64 else []),
        *(["-DTMN_K1_FOLD=1"] if _fold(cfg) else []),
        *extra,
    ]
    smem = _smem(d, cfg)
    kernel = T.load_unit(str(T.compile_text(body, flags)), "b200style_k1_direct").kernel(name)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem


def _ring(cfg):
    """Schedule 6: the z chunks and the packed weight blocks stream through one ring (no z tile buffer)."""
    return len(cfg) > 5 and cfg[5] == 6


def _packed(weights, cpb):
    """[block][k chunk][gate rows | proj rows][64 k]: every weight block of the ring is one contiguous image (one 3-D TMA load)."""
    import torch

    wl, wlg, wr, wrg = weights
    h, d = wl.shape
    g = torch.cat((wlg, wrg)).view(2 * h // cpb, cpb, d // 64, 64).permute(0, 2, 1, 3)
    p = torch.cat((wl, wr)).view(2 * h // cpb, cpb, d // 64, 64).permute(0, 2, 1, 3)
    return torch.cat((g, p), dim=2).contiguous().view(-1, 2 * cpb, 64)


def front(x, weights, token_mask, gi, bi, ab, xs, diag=()):
    """x [B, n, n, d]; B > 1: the samples stacked along K1's slow token axis (K1Params.rows = B n, planes [2 h, B n, n]), the token
    mask [B n] read per sample. One sample may take planes padded past n (ab [2 h, np, np], np = ab.shape[-1] >= n, a multiple of 16):
    K1 writes zeros into the pad rows and columns, as the packaged K1 does."""
    rows, n, d, h = x.shape[0] * x.shape[1], x.shape[1], x.shape[-1], ab.shape[0] // 2
    np_ = ab.shape[-1]
    if np_ != n and rows != n:
        raise ValueError("padded planes take one sample")
    cfg = K._config(d, n=n, direct=True)
    a, b, _, _, mb = cfg[:5]
    cpb = _block(cfg) // 2  # plane channels per weight block: the weight maps' row box and the plane store's channel box
    if cpb != 32 and (np_ % 8 or b % 64):
        raise ValueError("the n128 K1 stores planes by TMA: 64-token plane row segments, 16-byte plane rows")
    kernel, smem = _kernel(d, h, cfg, *diag)
    if _fold(cfg):  # halved once per parameter version (exact in bf16)
        import torch
        from miniworld_engine.kernels import _capture

        weights = _capture.param_forms(
            ("trimul_h100_k1_half", d, h, _ring(cfg)),
            weights,
            lambda: [_packed([w * 0.5 for w in weights], cpb)] if _ring(cfg) else [w * 0.5 for w in weights],
            capture_reads_eager=not torch.is_grad_enabled(),
        )
    tj = -(-np_ // b)
    tiles = (-(-(rows if rows != n else np_) // a)) * tj
    fields = [
        I._map(x, [64, b, a], [d, n, rows], [d * 2, n * d * 2]),
        T._launch_module().TensorMap(b"\0" * 128),
        I._map(ab, [64, 1, cpb], [np_, rows if rows != n else np_, 2 * h], [np_ * 2, (rows if rows != n else np_) * np_ * 2]),
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
        0 if rows == n else rows,  # K1Params.rows (batched samples; 0 = one sample), before K1ParamsQ.tm_wq
        *(
            [I._map(weights[0], [64, 4 * cpb], [64, weights[0].shape[0] * 2 * cpb], [128])] * 4
            if _ring(cfg)
            else [I._map(w, [64, cpb], [d, h], [d * 2]) for w in weights]
        ),
    ]
    param = T._launch_module().Struct(fields)
    assert len(param.pack()) == 1024
    kernel.launch(
        (min(tiles, mb * num_sms(x.device.index)), 1, 1),
        (128 * (a * b // 64 + 1), 1, 1),
        [param],
        smem,
    )
