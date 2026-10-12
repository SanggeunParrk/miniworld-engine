"""Hopper port of the B200 wide TriMul fusion algorithm (explicit candidate).

K1 writes gated planes and LN_in statistics; folded K3 writes y, projection,
gate and LN_out statistics. Neither normalized output rows nor front
preactivations survive forward. Backward uses B200's folding identities and
recomputes front preactivations in a WGMMA K1 epilogue. No architecture-100
instructions are imported: the shared B200 auxiliary sources are plain CUDA.
"""

from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import (
    ensure_cuda_home,
    gencodes,
    host_flags,
    load_extension,
)
from . import _h100_runtime as T
from . import h100_uni_wide_inference as I
from . import h100_wide_inference as BI
from ._h100_pack import owned_bf16_mask, pack_into_cuda, cast_training_weights

R = T.SOURCES / "b200style"
WIDTHS = (256, 384, 512)
LENGTHS = (128, 256, 384, 512, 640, 768, 896, 1024)


@lru_cache(None)
def _runtime_aux():
    """Retain the initial runtime binding for reproducing diagnostic logs."""
    ensure_cuda_home()
    shared = Path(__file__).with_name("b200_sources")
    return load_extension(
        name="trimul_h100_scalar_tmasave_aux_v1",
        sources=[
            str(R / "bind.cpp"),
            str(shared / "wide_aux.cu"),
            str(shared / "wide_bwd.cu"),
        ],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            *gencodes("90a"),
            "-lineinfo",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


@lru_cache(None)
def _aux():
    from miniworld_engine.kernels.trimul_inproj.cuda.h100_b200style_aux import DriverAux

    return DriverAux()


@lru_cache(None)
def _headers(backward=False):
    base = I._headers()
    dest = T.cache_dir() / (
        "b200style_headers_"
        + T._source_digest().hex()[:16]
        + ("_bwd" if backward else "_fwd")
    )
    for name in ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh"):
        source = (base / name).read_text()
        if name == "tmn_kernels.cuh" and backward:
            # Keep the existing K1 producer/ring policy; replace only its gated
            # plane epilogue with token-major gate/projection derivatives.
            anchor = "int xs_i, xs_j, pad0, pad1;"
            assert source.count(anchor) == 1
            source = source.replace(
                anchor,
                anchor + " const __nv_bfloat16* dpl; __nv_bfloat16* dcat; int ldcat;",
            )
            anchor = "  const float* mask;"
            pos = source.index(anchor, source.index("struct K1Params {"))
            source = source[:pos] + "  CUtensorMap tm_dcat, tm_dpl;\n" + source[pos:]
            anchor = (
                '  static_assert((G::NBAR + 1) * 8 <= G::SMEM_BAR, "barrier area");'
            )
            assert source.count(anchor) == 1
            source = source.replace(
                anchor,
                "  uint64_t* barD_full = barGo + 1;\n"
                '  static_assert((G::NBAR + 1 + G::NCWG) * 8 <= G::SMEM_BAR, "dpl barrier area");',
            )
            anchor = "    if (K1OFF > 0) mbar_init(barGo, 1);"
            assert source.count(anchor) == 1
            source = source.replace(
                anchor,
                anchor
                + "\n    for (int i = 0; i < G::NCWG; ++i) mbar_init(barD_full + i, 1);\n"
                "    tma_prefetch_desc(&p.tm_dcat); tma_prefetch_desc(&p.tm_dpl);",
            )
            anchor = "if (TMAST && st_elect) tma_store_wait_all();"
            assert source.count(anchor) == 1
            source = source.replace(anchor, "if (st_elect) tma_store_wait_all();")
            start = source.index(
                "      uint32_t pk[4][2];", source.index("auto epilogue =")
            )
            end = source.index("    };", start)
            source = (
                source[:start]
                + (R / "k1wb_epilogue.inc").read_text()
                + "\n"
                + source[end:]
            )
        T.publish_header(dest / name, source)
    return dest


def _config(d, backward=False):
    if d == 256 and not backward:
        return (2, 64, 4, 4, 1, 0)
    return {256: (2, 64, 4, 4, 1), 384: (2, 64, 2, 6, 1), 512: (2, 64, 4, 2, 1)}[d]


def _output_config(d):
    return (192, 4) if d == 384 else (256, 3)


@T.device_cache
def _front_kernel(d, h, backward, cfg):
    a, b, slots, sk, mb = cfg[:5]
    schedule = cfg[5] if len(cfg) > 5 else 1
    name = "mw_b200style_k1wb" if backward else "mw_b200style_k1"
    body = (
        f'#include "tmn_kernels.cuh"\n'
        f"using C=tmn::K1Cfg<{d},{h},false,{a},{b},{slots},{sk},{schedule}>;\n"
        'extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB) '
        f"void {name}(__grid_constant__ const tmn::K1Params p){{"
        f"tmn::sm90::k1_body<C,true,1,{str(not backward).lower()},false,1>(p);}}"
    )
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        f"-I{_headers(backward)}",
        f"-DMW_MINB={mb}",
        "-DMW_K1_STREAM=1",
        "-DTMN_MASK_TEMPLATE=1",
        "-DTMN_SIGMOID_TANH=1",
        "-DTMN_WSKIP=1",
        f"-DTMN_K1_BULK_STORE={int(not backward)}",
    ]
    smem = I._k1_smem(d, (*cfg[:5], 2))
    kernel = T.load_unit(str(T.compile_text(body, flags)), "b200style_k1").kernel(name)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem


def _front(x, w1, mask, gi, bi, ab, xs, dpl=None, dcat=None, xn=None, normalized=False):
    if x.shape[-1] == 512:
        if dcat is not None and not normalized:
            from .h100_b200style_ss import front
        elif dcat is not None or xn is not None:
            from .h100_b200style_ss_saved import front
        else:
            from .h100_b200style_ss_regtune import front

        return front(x, w1, mask, gi, bi, ab, xs, dpl, dcat, xn)
    return _front_rs(x, w1, mask, gi, bi, ab, xs, dpl, dcat, xn)


def _front_rs(x, w1, mask, gi, bi, ab, xs, dpl=None, dcat=None, xn=None):
    n, d, h = x.shape[1], x.shape[-1], ab.shape[0] // 2
    cfg = _config(d, dcat is not None)
    a, b, _, _, mb = cfg[:5]
    kernel, smem = _front_kernel(d, h, dcat is not None, cfg)
    tj = n // b
    tiles = (n // a) * tj
    fields = [
        I._map(x, [64, b, a], [d, n, n], [d * 2, n * d * 2]),
        I._map(w1, [64, 64], [d, 4 * h], [d * 2]),
        I._map(ab, [64, 1, 32], [n, n, 2 * h], [n * 2, n * n * 2]),
        mask,
        gi,
        bi,
        ab,
        xs,
        None,
        n,
        n,
        tj,
        tiles,
        1,
        n,
        1,
        1e-5,
        n * d,
        d,
        0,
        0,
    ]
    if dcat is not None:
        fields.insert(
            3, I._map(dcat, [64, 64], [dcat.shape[1], n * n], [dcat.shape[1] * 2])
        )
        fields.insert(4, I._map(dpl, [64, 32], [n * n, 2 * h], [n * n * 2]))
        fields += [dpl, dcat, dcat.shape[1]]
    kernel.launch(
        (min(tiles, mb * I._sms(x.device.index)), 1, 1),
        (128 * (a * b // 64 + 1), 1, 1),
        [T._launch_module().Struct(fields)],
        smem,
    )


@T.device_cache
def _output_kernel_standard(d, h, cfg):
    bn, stages = cfg
    flags = [
        "-std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--cubin",
        "-lineinfo",
        f"-I{_headers()}",
        f"-I{T.SOURCES / 'wide_forward'}",
        f"-DWIDTH={d}",
        f"-DHIDDEN={h}",
        f"-DBLOCKN={bn}",
        f"-DSTAGES={stages}",
        "-DGFOLD=1",
        "-DTMN_SIGMOID_TANH=1",
        "-DMW_MINB=1",
        "-DMW_K1_STREAM=1",
    ]
    kernel = T.load_unit(str(T.compile(R / "k3.cu", flags)), "b200style_k3").kernel(
        "mw_b200style_k3"
    )
    smem = BI._k3_smem(d, bn, stages)
    kernel.set_max_dynamic_smem(smem)
    return kernel, smem


def _output_kernel(d, h, cfg, save=False):
    if d == 512:
        if save:
            from .h100_b200style_tmasave import output_kernel
        else:
            from .h100_b200style_reorder import output_kernel
        return output_kernel(d, h, cfg)
    if d == 256 and save:
        from .h100_b200style_scalar_tmasave import output_kernel
        return output_kernel(d, h, cfg)
    return _output_kernel_standard(d, h, cfg)


def _contract(ab, d, direction):
    from .b200_infer import _contract as contract

    return contract(ab, d, direction)


def _saved(x, direction):
    n, d = x.shape[1], x.shape[-1]
    m, h = n * n, d * (2 if direction == 0 else 1)
    return [
        x.new_empty((2 * h, n, n)),
        x.new_empty((h, n, n)),
        x.new_empty((m, 2), dtype=torch.float32),
        x.new_empty((m,), dtype=torch.float32),
        x.new_empty((m,), dtype=torch.float32),
        x.new_empty((4, d), dtype=torch.float32),
        x.new_empty((m, d)),
        x.new_empty((m, d)),
    ] + ([x.new_empty((m, d))] if d == 512 else [])


def _forward_fake(leaves, mask, ds, direction):
    """[y, *saved activations]: y like x, the saves as ``_saved`` lays them out."""
    return [torch.empty_like(leaves[0]), *_saved(leaves[0], direction)]


def _run_forward(leaves, mask, ds, direction, save):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n, d, m = x.shape[1], x.shape[-1], x.shape[1] ** 2
    h = wp.shape[1]
    with T.native_context(x.device):
        ab = x.new_empty((2 * h, n, n))
        xs = x.new_empty((m, 2), dtype=torch.float32)
        vec = x.new_empty((4, d), dtype=torch.float32)
        mu = x.new_empty((m,), dtype=torch.float32) if save else None
        rs = torch.empty_like(mu) if save else None
        pv, gv = (x.new_empty((m, d)), x.new_empty((m, d))) if save else (None, None)
        # Emit the D512 normalized input here; backward reuses it directly.
        saved_xn = x.new_empty((m, d)) if save and d == 512 else None
        weights = (wl, wlg, wr, wrg)
        if (
            d in (256, 384)
            and mask.dtype == torch.bool
            and mask.numel() == n
            and mask.is_contiguous()
            and all(w.dtype == torch.bfloat16 and w.is_contiguous() for w in weights)
        ):
            from .h100_b200style_direct import front

            front(x, weights, mask, gi, bi, ab, xs)
        else:
            w1 = x.new_empty((4 * h, d))
            pack_into_cuda(w1, wl, wlg, wr, wrg)
            pm = owned_bf16_mask(mask, n, x.device)
            _front(x, w1, pm, gi, bi, ab, xs, xn=saved_xn)
        tri = _contract(ab, d, direction)
        wpf, wgf = torch.empty_like(wp), torch.empty_like(wg)
        _aux().fold(wp, go, bo, wg, gi, bi, wpf, wgf, vec)
        y = torch.empty_like(x)
        cfg = _output_config(d)
        if not save and ds is None:
            from .h100_b200style_static import output_kernel

            kernel, smem = output_kernel(d, h, cfg)
        else:
            kernel, smem = _output_kernel(d, h, cfg, save)
        bn, _ = cfg
        fields = [
            I._map(tri, [64, 64], [m, h], [m * 2]),
            I._map(x, [64, 128], [d, m], [d * 2]),
            I._map(wpf, [64, bn], [h, d], [h * 2]),
            I._map(wgf, [64, bn], [d, d], [d * 2]),
            I._map(x, [64, 64], [d, m], [d * 2]),
            I._map(y, [64, 64], [d, m], [d * 2]),
            *(
                [
                    I._map(pv, [64, 64], [d, m], [d * 2]),
                    I._map(gv, [64, 64], [d, m], [d * 2]),
                ]
                if save and d in (256, 512)
                else []
            ),
            vec,
            xs,
            m,
            m // 128,
            ds,
            pv,
            gv,
            mu,
            rs,
            n,
        ]
        kernel.launch(
            (min(m // 128, I._sms(x.device.index)), 1, 1),
            (384, 1, 1),
            [T._launch_module().Struct(fields)],
            smem,
        )
        return (
            ([y, ab, tri, xs, mu, rs, vec, pv, gv] + ([saved_xn] if d == 512 else []))
            if save
            else [y]
        )


@opaque(fake=_forward_fake, name="trimul_h100_scalar_tmasave_forward")
def forward(
    leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, direction: int
) -> list[torch.Tensor]:
    """B200-style TriMul training forward on sm_90a: [y, *saved activations for the backward]."""
    return _run_forward(leaves, mask, ds, direction, True)


def _inference_fake(leaves, mask, ds, direction):
    """y like x."""
    return torch.empty_like(leaves[0])


@opaque(fake=_inference_fake, name="trimul_h100_scalar_tmasave_inference")
def inference(
    leaves: list[torch.Tensor],
    mask: torch.Tensor,
    ds: torch.Tensor | None,
    direction: int,
) -> torch.Tensor:
    """B200-style TriMul forward on sm_90a without saving anything: y like x."""
    return _run_forward(leaves, mask, ds, direction, False)[0]


def _backward_fake(leaves, mask, ds, direction, saved, dy, master_mask):
    """[dx, one gradient per leaf after x]: fp32 where the master_mask bit is set, bf16 otherwise."""
    return [
        torch.empty_like(leaves[0]),
        *(
            torch.empty_like(
                w, dtype=torch.float32 if master_mask & (1 << i) else torch.bfloat16
            )
            for i, w in enumerate(leaves[1:])
        ),
    ]


@opaque(fake=_backward_fake, name="trimul_h100_scalar_tmasave_backward")
def backward(
    leaves: list[torch.Tensor],
    mask: torch.Tensor,
    ds: torch.Tensor,
    direction: int,
    saved: list[torch.Tensor],
    dy: torch.Tensor,
    master_mask: int,
) -> list[torch.Tensor]:
    """B200-style TriMul training backward on sm_90a: [dx, *leaf gradients]; see ``_backward_fake``."""
    from .b200_train import _contract_bwd
    from . import h100_b200style_gemm as G

    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    ab, tri, xs, mu, rs, vec, pv, gv = saved[:8]
    n, d, h = x.shape[1], x.shape[-1], wp.shape[1]
    m, p = n * n, 2 * h
    with T.native_context(x.device):
        E = _aux()
        workspace = (
            x.new_empty(G.WORKSPACE, dtype=torch.uint8)
            if any(
                G._selection(d, h, n, role) is not None
                for role in ("dt", "dwp", "dw", "dxn")
            )
            else None
        )
        dy2 = dy.contiguous().view(m, d)
        dpr = x.new_empty((m, d))
        dcat = x.new_empty((m, 2 * p + d))
        s1, s2 = torch.empty_like(mu), torch.empty_like(mu)
        acc = x.new_empty((4 * d + 2 * h,), dtype=torch.float32)
        E.gate_bwd(
            dy2, pv, gv, ds, vec, mu, rs, dpr, dcat[:, 2 * p :], s1, s2, acc[: 2 * d], n
        )
        dt = torch.empty_like(tri)
        E.lnout_bwd(
            G.mm(wp.t(), dpr.t(), workspace, d, h, n, "dt"),
            tri,
            mu,
            rs,
            s1,
            s2,
            go,
            dt,
            acc[4 * d :],
        )
        dwp = (
            go[None, :]
            * (
                G.mm(dpr.t(), tri.view(h, m).t(), workspace, d, h, n, "dwp", fp32=True)
                - acc[d : 2 * d, None]
            )
            + acc[:d, None] * bo[None, :]
        )
        dpl = _contract_bwd(ab, dt, d, direction)
        w1 = x.new_empty((2 * p, d))
        pack_into_cuda(w1, wl, wlg, wr, wrg)
        pm = owned_bf16_mask(mask, n, x.device)
        if d == 512:
            xn = saved[8]
            _front(xn.view_as(x), w1, pm, gi, bi, ab, xs, dpl, dcat, normalized=True)
            emitted = True
        else:
            xn = x.new_empty((m, d))
            emitted = _front(x, w1, pm, gi, bi, ab, xs, dpl, dcat, xn=xn)
        mean, rstd = xs[:, 0].contiguous(), xs[:, 1].contiguous()
        if not emitted:
            E.ln_apply(x.view(m, d), mean, rstd, gi, bi, xn)
        dw = G.mm(dcat.t(), xn, workspace, d, h, n, "dw", fp32=True)
        dxn = G.mm(dcat, torch.cat((w1, wg)), workspace, d, h, n, "dxn")
        dx = torch.empty_like(x)
        E.lnin_bwd(
            dxn, x.view(m, d), dy2, mean, rstd, gi, dx.view(m, d), acc[2 * d : 4 * d]
        )
        packed = dw[: 2 * p].reshape(p // 32, 2, 32, d)
        dg = packed[:, 0].reshape(p, d)
        dv = packed[:, 1].reshape(p, d)
        grads = [
            dv[:h],
            dg[:h],
            dv[h:],
            dg[h:],
            dw[2 * p :],
            dwp,
            acc[2 * d : 3 * d],
            acc[3 * d : 4 * d],
            acc[4 * d : 4 * d + h],
            acc[4 * d + h :],
        ]
        return [
            dx,
            *(
                g.to(
                    torch.float32 if master_mask & (1 << i) else torch.bfloat16,
                    copy=True,
                )
                for i, g in enumerate(grads)
            ),
        ]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, direction, mask, ds, *leaves):
        ctx.direction = direction
        ctx.master_mask = sum(
            1 << i for i, w in enumerate(leaves[1:]) if w.dtype == torch.float32
        )
        kernel_leaves = [
            leaves[0],
            *cast_training_weights(leaves[1:7]),
            *leaves[7:],
        ]
        y, *saved = forward(kernel_leaves, mask, ds, direction)
        ctx.save_for_backward(mask, ds, *kernel_leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        grads = backward(
            list(vals[2:13]),
            vals[0],
            vals[1],
            ctx.direction,
            list(vals[13:]),
            dy,
            ctx.master_mask,
        )
        return (
            None,
            None,
            None,
            grads[0],
            *grads[1:],
        )


def trimul(leaves, mask, ds, direction=0):
    """Explicit B1/H=D Hopper candidate; direction 0=bidi, 1=outgoing, 2=incoming."""
    x = leaves[0]
    if (
        x.dtype != torch.bfloat16
        or x.ndim != 4
        or x.shape[0] != 1
        or x.shape[1] != x.shape[2]
        or x.shape[-1] not in WIDTHS
        or x.shape[1] not in LENGTHS
        or not x.is_contiguous()
        or direction not in (0, 1, 2)
        or torch.cuda.get_device_capability(x.device) != (9, 0)
    ):
        raise ValueError(
            "B200-style Hopper TriMul requires B1, BF16, D256/384/512, L128..1024"
        )
    h = x.shape[-1] * (2 if direction == 0 else 1)
    if leaves[1].shape != (h, x.shape[-1]) or leaves[6].shape != (x.shape[-1], h):
        raise ValueError("B200-style Hopper TriMul requires hidden=D per direction")
    expected = [(h, x.shape[-1])] * 4 + [(x.shape[-1], x.shape[-1]), (x.shape[-1], h)]
    if (
        len(leaves) != 11
        or any(
            w.device != x.device
            or w.dtype not in (torch.bfloat16, torch.float32)
            or tuple(w.shape) != shape
            for w, shape in zip(leaves[1:7], expected, strict=True)
        )
        or any(
            w.device != x.device
            or w.dtype != torch.float32
            or not w.is_contiguous()
            or w.shape != (width,)
            for w, width in zip(
                leaves[7:], (x.shape[-1], x.shape[-1], h, h), strict=True
            )
        )
        or (
            ds is not None
            and (
                ds.device != x.device
                or ds.dtype != torch.bfloat16
                or not ds.is_contiguous()
                or ds.shape != (x.shape[1], x.shape[-1])
            )
        )
        or mask.device != x.device
        or not mask.is_contiguous()
        or mask.dtype not in (torch.bool, torch.bfloat16, torch.float32)
        or mask.numel() not in (0, x.shape[1], x.shape[1] ** 2)
    ):
        raise ValueError(
            "invalid B200-style Hopper weights, norm affines, mask or dropout scale"
        )
    if not torch.is_grad_enabled():
        kernel_leaves = [
            x,
            *cast_training_weights(leaves[1:7]),
            *leaves[7:],
        ]
        return inference(kernel_leaves, mask, ds, direction)
    if ds is None:
        from ._h100_pack import training_unit_scale

        ds = training_unit_scale(x)
    return _Training.apply(direction, mask, ds, *leaves)
