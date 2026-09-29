"""B200 (sm_100a) bidirectional TriMul, D128 (d_pair = d_hidden = 128), bf16, B=1.

Inference: K1 (input LN + four gated projections + mask -> channel-major planes) -> two cuBLAS
contractions -> K3 (output LN + projection + gate + dropout + residual).
Training adds, in the backward: B1r (output side) -> four cuBLAS contractions -> B7r (input side).
Kernel bodies are tcgen05 / TMEM / TMA hand-CUDA in ``b200_sources/``, ported unchanged from the
B200 research capsule ``experiments/trimul_b200`` (v15). Every call owns its buffers; nothing is
cached across calls except the compiled extension.
"""

import functools
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_SRC = Path(__file__).with_name("b200_sources")

C = 128          # d_pair
H = 256          # packed hidden: [outgoing 128 | incoming 128]
TOK = 128        # token tile of every kernel; L must be a multiple of it
EPS = 1e-5
B1_RSF = 4       # B1r ring slots per front CTA


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="trimul_b200_bidir_d128_v1",
        sources=[str(_SRC / f) for f in ("k1.cu", "k3.cu", "b1r.cu", "b7r.cu", "bind.cpp")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a"), "--expt-relaxed-constexpr",
                           "-DNDEBUG", "-DK1_SIG=2", f"-I{_SRC}"],
        extra_cflags=["-std=c++17", f"-I{_SRC}"], verbose=False)


def supports(length: int) -> bool:
    """Lengths the kernels take: whole 128-token tiles."""
    return length > 0 and length % TOK == 0


def _b1_front_ctas(length: int) -> int:
    """B1r front CTA count: the largest multiple of L/128 that is <= 96 (measured best at 96)."""
    rows = length // TOK
    return (96 // rows) * rows


def _b7_source_groups(length: int) -> int:
    return 11 if length <= 384 else 12


def pack_w1(wl, wlg, wr, wrg):
    """K1/B7 packed weight [1024, 128]: chunk c (64 plane channels) = [its 64 gate rows ; its 64
    projection rows], chunks 0-3 from the left weights and 4-7 from the right. Three launches; the
    inputs may be strided (the module stores these four column-major)."""
    left = torch.stack((wlg.reshape(4, 64, C), wl.reshape(4, 64, C)), 1)
    right = torch.stack((wrg.reshape(4, 64, C), wr.reshape(4, 64, C)), 1)
    return torch.cat((left, right)).view(1024, C)


def pack_wp(wp):
    """K3 output-projection operand: columns k = 2c + u of each 16-column block reordered from
    (c % 2, c // 2 within 8, u) to (c // 2, c % 2, u) -- the extension's ``k3_pack_wp`` without
    its host-built index (a host copy cannot be captured in a CUDA graph)."""
    return wp.reshape(C, H // 16, 2, 4, 2).transpose(2, 3).reshape(C, H)


def _unpack_w1(dw1, cw=64):
    """Inverse of :func:`pack_w1` for the [1024, 128] weight gradient -> (dWl, dWlg, dWr, dWrg)."""
    g = torch.cat([dw1[2 * cw * c:2 * cw * c + cw] for c in range(512 // cw)])
    p = torch.cat([dw1[2 * cw * c + cw:2 * cw * (c + 1)] for c in range(512 // cw)])
    return p[:H], g[:H], p[H:], g[H:]


def _front(leaves, mask):
    """K1 + the two forward contractions. Returns (x2, w1, ab, tri)."""
    x, wl, wlg, wr, wrg, _wg, _wp, gi, bi, _go, _bo = leaves
    n = x.shape[1]
    x2 = x.reshape(n * n, C)
    w1 = pack_w1(wl, wlg, wr, wrg)
    ab = x.new_empty((512, n, n))
    _ext().k1_forward(x2, w1, mask.reshape(-1), gi, bi, ab, EPS, 0)
    tri = x.new_empty((H, n, n))
    h = H // 2
    torch.bmm(ab[:h], ab[2 * h:3 * h].transpose(1, 2), out=tri[:h])
    torch.bmm(ab[h:2 * h].transpose(1, 2), ab[3 * h:], out=tri[h:])
    return x2, w1, ab, tri


def _forward_fake(leaves, mask, ds):
    """y and the saved set: planes, tri, x_n, output-LN mean / rstd, packed W1."""
    x = leaves[0]
    n = x.shape[1]
    f32 = torch.float32
    return [torch.empty_like(x), x.new_empty((512, n, n)), x.new_empty((H, n, n)), x.new_empty((n * n, C)),
            x.new_empty((n * n,), dtype=f32), x.new_empty((n * n,), dtype=f32), x.new_empty((1024, C))]


@opaque(fake=_forward_fake, name="trimul_b200_bidir_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor) -> list[torch.Tensor]:
    """Training forward: y plus the saved set (planes, tri, x_n, output-LN mean / rstd, packed W1)."""
    x, _wl, _wlg, _wr, _wrg, wg, wp, gi, bi, go, bo = leaves
    n = x.shape[1]
    with torch.cuda.device(x.device):
        x2, w1, ab, tri = _front(leaves, mask)
        y = torch.empty_like(x)
        xn = x.new_empty((n * n, C))
        mo = x.new_empty((n * n,), dtype=torch.float32)
        ro = torch.empty_like(mo)
        _ext().k3_forward(x2, tri, pack_wp(wp), wg, gi, bi, go, bo, y.view(n * n, C), n, EPS, 1,
                          ds, xn, mo, ro, 0)
    return [y, ab, tri, xn, mo, ro, w1]


def _forward_nograd_fake(leaves, mask, ds):
    """y only."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_b200_bidir_nograd")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor | None) -> torch.Tensor:
    """No-grad forward. ``ds`` None: the inference K3; else the saving K3 into throwaway buffers
    (the inference K3 has no dropout)."""
    x, _wl, _wlg, _wr, _wrg, wg, wp, gi, bi, go, bo = leaves
    n = x.shape[1]
    with torch.cuda.device(x.device):
        x2, _w1, _ab, tri = _front(leaves, mask)
        y = torch.empty_like(x)
        if ds is None:
            _ext().k3_forward(x2, tri, pack_wp(wp), wg, gi, bi, go, bo, y.view(n * n, C), n, EPS, 0,
                              None, None, None, None, 0)
        else:
            mo = x.new_empty((n * n,), dtype=torch.float32)
            _ext().k3_forward(x2, tri, pack_wp(wp), wg, gi, bi, go, bo, y.view(n * n, C), n, EPS, 1,
                              ds, x.new_empty((n * n, C)), mo, torch.empty_like(mo), 0)
    return y


def _backward_fake(leaves, mask, ds, saved, dy):
    """One gradient per leaf, contiguous, in the leaf's dtype."""
    # Contiguous, like the real gradients -- not empty_like: the module passes the four front
    # weights column-major, and a fake that copied those strides would disagree with the body.
    return [torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in leaves]


@opaque(fake=_backward_fake, name="trimul_b200_bidir_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, saved: list[torch.Tensor],
             dy: torch.Tensor) -> list[torch.Tensor]:
    """B1r -> four contractions -> B7r. Weight / LN gradients accumulate atomically in one fp32
    buffer (one memset); the ring flags live in the same buffer."""
    x, _wl, _wlg, _wr, _wrg, wg, wp, gi, _bi, go, bo = leaves
    ab, tri, xn, mo, ro, w1 = saved
    n = x.shape[1]
    M = n * n
    nf, sg = _b1_front_ctas(n), _b7_source_groups(n)
    shapes = [(C, C), (C, H), (H,), (H,), (1024, C), (C,), (C,)]
    sizes = [int(torch.Size(s).numel()) for s in shapes]
    nflag1, nflag7 = nf * B1_RSF, sg * 12 * 8
    with torch.cuda.device(x.device):
        acc = torch.zeros(sum(sizes) + nflag1 + nflag7, device=x.device, dtype=torch.float32)
        views, o = [], 0
        for s, k in zip(shapes, sizes, strict=True):
            views.append(acc[o:o + k].view(s))
            o += k
        dwg, dwp, dgo, dbo, dw1, dgi, dbi = views
        flags1 = acc[o:o + nflag1].view(torch.int32)
        flags7 = acc[o + nflag1:].view(torch.int32)
        dy2 = dy.contiguous().view(M, C)
        dg = x.new_empty((M, C))
        dtri = x.new_empty((H, n, n))
        ring1 = x.new_empty((nf * B1_RSF * TOK, C))
        _ext().b1r_backward(dy2, xn, tri, ds, mo, ro, wg, wp, go, bo, dg, dtri, dwg, dwp, dgo, dbo, ring1, flags1, n, nf)
        h = H // 2
        dl = torch.empty_like(tri)
        dr = torch.empty_like(tri)
        L_, R_ = ab[:H], ab[H:]
        torch.bmm(dtri[:h], R_[:h], out=dl[:h])
        torch.bmm(dtri[:h].transpose(1, 2), L_[:h], out=dr[:h])
        torch.bmm(R_[h:], dtri[h:].transpose(1, 2), out=dl[h:])
        torch.bmm(L_[h:], dtri[h:], out=dr[h:])
        dx = torch.empty_like(x)
        ring7 = x.new_empty((sg * 12 * 8 * TOK, C))
        _ext().b7r_backward(x.view(M, C), xn, dy2, dl, dr, dg, mask.reshape(-1), w1, wg, gi, dx.view(M, C), dw1,
                            dgi, dbi, ring7, flags7, EPS, sg)
        dwl, dwlg, dwr, dwrg = _unpack_w1(dw1)
    # The small gradients are views of one accumulation buffer; a custom op may not return
    # aliasing outputs, so each leaves as its own tensor (a cast, or a copy of <= 128 KB).
    grads = [dwl, dwlg, dwr, dwrg, dwg, dwp, dgi, dbi, dgo, dbo]
    return [dx, *(g.to(t.dtype, copy=True) for g, t in zip(grads, leaves[1:], strict=True))]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        leaves = list(args[:11])
        mask, ds = args[11:]
        y, *saved = forward(leaves, mask, ds)
        ctx.save_for_backward(*args, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        grads = backward(list(vals[:11]), vals[11], vals[12], list(vals[13:]), dy)
        return (*grads, None, None)


def bidirectional_trimul(x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, mask, ds):
    """x [1, L, L, 128] bf16; wl/wlg/wr/wrg [256, 128], wg [128, 128], wp [128, 256] bf16 ([out, in]);
    gi/bi [128], go/bo [256] fp32; mask [L, L] fp32 pair mask; ds [L, 128] bf16 row dropout scale or None.
    Returns x + ds * trimul(x) (residual included, as the module's other paths)."""
    leaves = [x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo]
    if not torch.is_grad_enabled():
        return forward_nograd(leaves, mask, ds)
    if ds is None:
        ds = x.new_ones((x.shape[1], C))
    return _Training.apply(*leaves, mask, ds)
