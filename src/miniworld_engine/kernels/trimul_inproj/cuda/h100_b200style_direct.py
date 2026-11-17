"""K1 consumes row-major projection weights and the original token mask."""

from miniworld_engine.kernels._device import num_sms
from . import h100_b200style as K
from . import h100_uni_wide_inference as I
from . import _h100_runtime as T


#: FP8-plane K1 tiles (the n128 blocks the e4m3 store needs; D256 is the shipped direct tile)
F8_CFG: dict = {
    128: (3, 64, 6, 1, 1, 6, 128),
    64: (3, 64, 4, 1, 1),  # n64 blocks (tmn_kernels.cuh TMN_K1_F8 n64 store): the long D64 direct tile (BJ 64: TMA plane stores)
    384: (2, 64, 2, 6, 1),  # n64 blocks: the D384 direct tile of the bf16 route (an n128 D384 ring does not fit shared memory)
}  # FP8 planes with the bf16-compute K1: three warpgroups, 24 KB ring slots (K1 0.84-0.90x of the earlier 2-warpgroup (2, 64, 4) tile, planes bit-identical); (2, 64, 4, 1, 1, 6, 128) is the two-warpgroup form
#: FP8-compute K1 tiles: D128 runs three consumer warpgroups (192-token tiles, 24 KB z chunks in the ring slots)
F8C_CFG: dict = {128: (3, 64, 6, 1, 1, 6, 128), 256: (3, 64, 4, 1, 1, 6, 128)}


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
    size = (
        tile
        + slots * max(sk * bw * 128, a * b * 128 if _ring(cfg) else 0)
        + a * b // 64 * 2 * bw // 2 * 128
        + 8 * d
        + (bars * 8 + 127) // 128 * 128
    )
    if size * mb > 232448:
        raise ValueError("K1 resources")
    return size


@T.device_cache
def _kernel(d, h, cfg, lnm=1, extra=(), f8=False, f8c=False):
    """lnm / extra: measurement builds only (lnm 0 = no input LayerNorm, extra e.g. -DTMN_K1_DIAG=2)."""
    a, b, slots, sk, mb = cfg[:5]
    schedule = cfg[5] if len(cfg) > 5 else 1
    name = "mw_b200style_k1_direct"
    body = (
        '#include "tmn_kernels.cuh"\n'
        f"using C=tmn::K1Cfg<{d},{h},false,{a},{b},{slots},{sk},{schedule}>;\n"
        'extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB) '
        f"void {name}(__grid_constant__ const tmn::{'K1ParamsF8' if f8 else 'K1ParamsQ'} p){{"
        f"tmn::sm90::k1_body<C,true,{lnm},true,false,1,tmn::{'K1ParamsF8' if f8 else 'K1ParamsQ'}>(p);}}\n"
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
        *(["-DTMN_K1_F8=1"] if f8 else []),
        *(["-DTMN_K1_F8C=1"] if f8c else []),
        *(
            ["-DTMN_K1_START_OFFSET=0"] if a == 3 and _ring(cfg) else []
        ),  # three warpgroups: no start offset (a 4-slot ring would deadlock on it)
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


def prescale(weights, qinv, h, cpb, ring, half=True):
    """FP8 planes: the projection rows of plane channel c scaled by qinv[c] (a power of two: exact), all four halved as the fold does (half; the
    n64 blocks take the unhalved weights)."""
    wl_, wlg_, wr_, wrg_ = (w.float() for w in weights)
    dt = weights[0].dtype
    f = 0.5 if half else 1.0
    mats = [(wl_ * (f * qinv[:h, None])).to(dt), (wlg_ * f).to(dt), (wr_ * (f * qinv[h:, None])).to(dt), (wrg_ * f).to(dt)]
    return [_packed(mats, cpb)] if ring else mats


class F8Compute:
    """EXPERIMENTAL FP8-compute K1 operands (``TMN_K1_F8C``): e4m3 weights packed for the n128 ring, the scalar scales, the input scale folded into the LN affine."""

    __slots__ = ("wq", "kscale", "gs", "bs", "sx", "scg", "scp", "h", "d", "src")


def _p2_ceil(x):
    import torch

    return torch.exp2(torch.ceil(torch.log2(x.clamp_min(1e-30))))


def f8c_input_scale(gi, bi, d):
    """Power-of-two scale sx for xn = LN(z) so that sx |xn| <= 448 for EVERY input row: |xn| <= sqrt(d - 1) max|gamma| + max|beta| (no guard needed)."""
    import math

    bound = math.sqrt(d - 1) * float(gi.abs().max()) + float(bi.abs().max())
    return 2.0 ** math.floor(math.log2(448.0 / max(bound, 1e-6)))


def f8c_perm(d):
    """K permutation (virtual -> natural column) of the e4m3 A fragment this kernel builds from the bf16 LN fragments, per 32-column group."""
    import torch

    v = torch.arange(32)
    q, i, hh = (v % 16) // 4, v % 4, v // 16
    nat = 16 * hh + torch.where(i < 2, 2 * q + i, 8 + 2 * q + (i - 2))
    return (torch.arange(d).view(-1, 32) * 0 + 32 * torch.arange(d // 32).view(-1, 1) + nat.view(1, -1)).reshape(-1)


def f8c_prepare(weights, gi, bi, qinv, h, d, sx=None):
    """Quantise the four bf16 projection weights [h, d] to e4m3 (one pow2 scale per matrix), pack the ring images [block][slot][64 gate | 64 proj rows][128 B]."""
    import torch

    dev = weights[0].device
    sx = f8c_input_scale(gi, bi, d) if sx is None else sx
    perm = f8c_perm(d).to(dev)
    scs, qs = [], []
    for w in weights:
        wf = w.float()
        sc = _p2_ceil(wf.abs().amax() / 448.0)
        scs.append(sc)
        qs.append((wf / sc)[:, perm].clamp(-448, 448).to(torch.float8_e4m3fn))
    wl, wlg, wr, wrg = qs
    spb = d // 128
    nblk = 4 * h // 128
    g = torch.cat((wlg, wrg)).view(nblk, 64, spb, 128)
    p = torch.cat((wl, wr)).view(nblk, 64, spb, 128)
    img = torch.cat((g, p), dim=1).permute(0, 2, 1, 3).contiguous()  # [block][slot][128 rows][128 B]
    o = F8Compute()
    o.wq = img.view(torch.uint8).view(nblk * spb * 128, 128)
    o.sx, o.h, o.d = sx, h, d
    o.scg, o.scp = torch.stack([scs[1], scs[3]]), torch.stack([scs[0], scs[2]])
    o.gs, o.bs = (gi.float() * sx).contiguous(), (bi.float() * sx).contiguous()
    o.kscale = torch.zeros(4, device=dev)
    o.src = None
    f8c_kscale(o, qinv)
    return o


def f8c_kscale(o, qinv):
    """[Kg_left, Kp_left, Kg_right, Kp_right] in place (static device tensor): Kg = sc_g / (2 sx), Kp = qinv sc_p / (2 sx)."""
    import torch

    o.kscale.copy_(
        torch.stack([0.5 * o.scg[0] / o.sx, 0.5 * qinv[0] * o.scp[0] / o.sx, 0.5 * o.scg[1] / o.sx, 0.5 * qinv[o.h] * o.scp[1] / o.sx])
    )


def front(x, weights, token_mask, gi, bi, ab, xs, diag=(), qinv=None, transposed=False, wcache=None, amax=None, f8c=None):
    """x [B, n, n, d]; B > 1: the samples stacked along K1's slow token axis (K1Params.rows = B n, planes [2 h, B n, n]), the token
    mask [B n] read per sample. One sample may take planes padded past n (ab [2 h, np, np], np = ab.shape[-1] >= n, a multiple of 16):
    K1 writes zeros into the pad rows and columns, as the packaged K1 does."""
    rows, n, d, h = x.shape[0] * x.shape[1], x.shape[1], x.shape[-1], ab.shape[0] // 2
    np_ = ab.shape[-1]
    if np_ != n and rows != n:
        raise ValueError("padded planes take one sample")
    cfg = (f8c is not None and F8C_CFG.get(d)) or (qinv is not None and F8_CFG.get(d)) or K._config(d, n=n, direct=True)
    a, b, _, _, mb = cfg[:5]
    cpb = _block(cfg) // 2  # plane channels per weight block: the weight maps' row box and the plane store's channel box
    if cpb != 32 and (np_ % 8 or b % 64):
        raise ValueError("the n128 K1 stores planes by TMA: 64-token plane row segments, 16-byte plane rows")
    f8 = qinv is not None  # EXPERIMENTAL FP8 planes: ab is a float8_e4m3fn / uint8 tensor, qinv [2 h] fp32 = 1 / (plane channel scale)
    if f8 and (np_ % 16 or b % 64):
        raise ValueError("FP8 planes need 64-token plane row segments (TMA stores) and planes padded to 16")
    if f8c is not None and (not f8 or _ring(cfg) is False or d % 128):
        raise ValueError("FP8 compute K1 needs the FP8 planes, the ring schedule and a width that is a multiple of 128")
    kernel, smem = _kernel(d, h, cfg, *diag, f8=f8, f8c=f8c is not None)
    if f8c is not None:  # e4m3 weights packed for the ring (the host built them once: static scales), the input scale in the LN affine
        weights = [f8c.wq]
        gi, bi = f8c.gs, f8c.bs
    elif f8 and wcache is not None and "w" in wcache:
        weights = wcache["w"]
    elif f8:  # per call unless wcache (static scales)
        src = weights
        weights = prescale(src, qinv, h, cpb, _ring(cfg), _fold(cfg))
        if wcache is not None:
            wcache["w"] = weights
            wcache["w_args"] = (src, h, cpb, _ring(cfg), _fold(cfg))
    elif _fold(cfg):  # halved once per parameter version (exact in bf16)
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
        I._map(
            x, [64, b, a], [d, n, rows], [n * d * 2, d * 2] if transposed else [d * 2, n * d * 2]
        ),  # transposed (EXPERIMENTAL, one sample): token (r, c) reads pair[c][r]
        T._launch_module().TensorMap(b"\0" * 128),
        (
            T._launch_module().tensor_map(
                ab,
                [64, 1, cpb],
                dims=[np_, rows if rows != n else np_, 2 * h],
                strides_bytes=[np_, (rows if rows != n else np_) * np_],
                swizzle="64B",
                l2="128B",
                dtype="uint8",
            )
            if f8
            else I._map(ab, [64, 1, cpb], [np_, rows if rows != n else np_, 2 * h], [np_ * 2, (rows if rows != n else np_) * np_ * 2])
        ),
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
        int(f8 and transposed),  # K1Params.stats_t (TMN_K1_F8): LN_in stats stored at the swapped pair token
        *(
            (
                [
                    T._launch_module().tensor_map(
                        weights[0],
                        [128, 128],
                        dims=[128, weights[0].shape[0]],
                        strides_bytes=[128],
                        swizzle="128B",
                        l2="128B",
                        dtype="uint8",
                    )
                ]
                * 4
                if f8c is not None
                else [I._map(weights[0], [64, (2 if a == 3 else 4) * cpb], [64, weights[0].shape[0] * 2 * cpb], [128])] * 4
            )
            if _ring(cfg)
            else [I._map(w, [64, cpb], [d, h], [d * 2]) for w in weights]
        ),
        *(
            [qinv, amax, None if f8c is None else f8c.kscale, *([None] * 5)] if f8 else []
        ),  # K1ParamsF8: qinv (unused: the weights are prescaled), amax (uint32-ordered fp32 max of the scaled values; None = off), pad
    ]
    param = T._launch_module().Struct(fields)
    assert len(param.pack()) == (1088 if f8 else 1024)
    kernel.launch(
        (min(tiles, mb * num_sms(x.device.index)), 1, 1),
        (128 * (a * b // 64 + 1), 1, 1),
        [param],
        smem,
    )
