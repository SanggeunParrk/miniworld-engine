"""Shared-resident normalized input and streamed weights for the B200-style front."""

from . import _h100_runtime as T
from . import h100_uni_wide_inference as I


def _config(d, backward):
    return (128, 4, 128, 0, 1)


@T.device_cache
def _kernel(d, h, backward, cfg):
    tokens, stages, bn = cfg[:3]
    private = cfg[3] if len(cfg) > 3 else 0
    wait = cfg[4] if len(cfg) > 4 else 1
    smem = (
        tokens * d * 2
        + stages * bn * 128
        + tokens // 64 * (24576 if backward and private else 16384)
        + 512
    )
    name = f"mw_b200style_k1ss{'wb' if backward else ''}_n{bn}t{tokens}s{stages}p{private}w{wait}"
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
        "-DMW_MINB=1",
        "-DMW_K1_STREAM=1",
        "-DTMN_SIGMOID_TANH=1",
        f"-DBLOCKN={bn}",
        f"-DKERNEL_NAME={name}",
    ]
    kernel = T.load_unit(
        str(T.compile(T.SOURCES / "b200style/k1ss.cu", flags)), "b200style_k1ss"
    ).kernel(name)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem


def front(x, w1, mask, gi, bi, ab, xs, dpl=None, dcat=None, xn=None):
    n, d, h = x.shape[1], x.shape[-1], ab.shape[0] // 2
    m = x.numel() // d   # token-flat: B n n tokens for batched samples (b-major; the pair mask is [B n, n])
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
        I._map(
            dcat, [64, 16 if private else 64], [dcat.shape[1], m], [dcat.shape[1] * 2]
        )
        if backward
        else am,
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
        (min(m // tokens, I._sms(x.device.index)), 1, 1),
        (128 * (tokens // 64 + 1), 1, 1),
        [T._launch_module().Struct(fields)],
        smem,
    )
    return xn is not None
