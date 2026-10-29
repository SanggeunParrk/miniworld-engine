"""Every bias-only DiT block's pair bias from ONE LayerNorm and ONE GEMM on B200 (sm_100), forward and backward: the fused path
of ``modules.bias_only_dit.pair_bias_all``.

Block b's logits are bias_b = to_bias_b(ln_pair_b(pair)). ``ln_pair`` has no offset, so ln_pair_b(pair) = LN0(pair) diag(g_b), LN0 the
LayerNorm without affine (its statistics are the same for every block), and with the fold W'_b = W_b diag(g_b)

    bias_all^T = W'_all LN0(pair)^T        [N, R] = [nb][H][L][L];  W'_all = [W'_1; ...; W'_nb] [N, 128], N = nb H, R = L^2

-- one LN0 pass over the pair rows (the token DiT rows' ``layernorm128_rows``) and ONE cuBLAS GEMM for every block. Block b's bias is
the contiguous head-major slice [b H, (b + 1) H) of that buffer, the layout the training softmax (``softmax_t``) reads. Backward, from
dbias_all [N, R] -- the blocks' bias gradients, each written by its block's backward straight into its slice of one buffer (``_Sink``),
so nothing is concatenated:

    d LN0   = dbias_all^T W'_all              one cuBLAS GEMM (K = N), fp32 out
    d pair  = LN0 backward, no affine         ``ln0_bwd_rows`` (kernels/bias_only_dit/cuda/bias_only_dit_hoist_rows.cu), which
                                              also writes LN0 (recomputed, fp32) as the next GEMM's operand Y
    dW'_all = dbias_all Y                     one cuBLAS GEMM over the R rows, as a strided batch of S row chunks (fp32 partials,
                                              summed: the [N, 128] output alone would leave most SMs idle)
    dW_b = dW'_b diag(g_b),  dg_b = sum_h dW'_b o W_b                (the unfold: small torch ops on [nb, H, 128], fp32, W_b unfolded)

dW' is not taken against LN0 rounded to bf16: that rounding (2^-9 per product) entered d gamma_b and to_bias's gradient, where the
PyTorch block reads LN0 in fp32 (hoist1: d gamma 1.11-1.22x the PyTorch bf16 block's error in a 24-block stack). On the bf16 path Y
is the two-term split [bf16(LN0) | bf16(LN0 - bf16(LN0))] (K = 256; hi + lo = LN0 to ~2^-17), so dW' is the bf16 dbias against LN0
to fp32 accumulation; on the fp32 path Y = LN0 (fp32, the TF32 GEMM's operand). LN0 is not kept from the forward.

Per training step that replaces, per block, the pair LayerNorm + projection (``pair_bias``), its backward (``pair_bias_bwd``) and
the accumulation of a [L, L, 128] pair gradient, by one pass of each for all blocks. Hopper: the forward is ONE kernel (``bo_hoist_pass.cu`` hoist_fwd, N <= 64: LN0 and the projection on the tensor cores, the normalised pair never
written), the backward of one block of 16 heads is ONE kernel (hoist_bwd1: d LN0, the LN0 backward and dW' in a single pass over the pair, bf16 and
fp32/TF32); more blocks keep the bf16 ``bo_hoist_bwd.cu`` kernel (N <= 192), fp32 the two cuBLAS GEMMs (TF32) around the LN0 rows.

bf16 (pair bf16; the weights in any dtype, folded in fp32 and rounded to bf16 as the per-block pack rounds Wf): bf16 operands, fp32
accumulation, the bias in bf16, d LN0 and dW' in fp32, d pair in the pair's dtype. fp32 (pair and weights fp32, no autocast): the
same structure, the GEMMs on TF32 tensor cores (``tf32_gemms``), every tensor fp32.

``serves()`` is the gate (B200 or H100, an engine implementation, pair [1, L, L, C] bf16 or fp32, C = 128 or 256 on H100, 128 on B200);
anything else takes the PyTorch fold of ``modules.bias_only_dit.hoist``. At C = 256 (the widths written 128 above) every kernel is the same
template at the other width, except the multi-block bf16 backward (``bo_hoist_bwd.cu``, 128 only): more than one block of 16 heads takes the
two GEMMs around ``ln0_bwd_rows`` there.
"""

from __future__ import annotations

import contextlib

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

DP = 128
#: the pair widths of the fused hoist: 128 everywhere, 256 on H100 (the sm_90 kernels are templates on the width)
DPS = (128, 256)
BF = torch.bfloat16
F32 = torch.float32
#: the attribute of each hoisted bias that names its slot of the shared gradient buffer: (``_Sink``, block index)
SINK_ATTR = "_bias_only_dit_hoist_sink"


def serves(implementation, attns, pair, per_sample=False) -> bool:
    """The fused hoist serves ``pair`` and the attention modules ``attns`` (eps / head count / no offset checked by the caller).
    ``per_sample``: ``pair`` is [A, L, L, C] with A > 1 (a pair of its own per sample, the samples in the batch axis) instead of
    [1, L, L, C]; C = 128, or 256 on H100."""
    from miniworld_engine.modules.exceptions import ImplementationType

    if implementation != ImplementationType.MINIWORLD:
        return False
    if settings.current().force_portable_kernels:
        return False
    if not pair.is_cuda or torch.cuda.get_device_capability(pair.device) not in ((10, 0), (9, 0)):
        return False
    dp = pair.shape[-1]
    if (
        pair.ndim != 4
        or (pair.shape[0] < 2 if per_sample else pair.shape[0] != 1)
        or pair.shape[1] != pair.shape[2]
        or dp not in DPS
        or (dp != DP and torch.cuda.get_device_capability(pair.device) != (9, 0))
        or pair.shape[1] == 0
    ):
        return False
    ws = [p for a in attns for p in (a.ln_pair.weight, a.to_bias.weight)]
    if any(
        p is None or not p.is_cuda or p.device != pair.device or p.shape[-1] != dp
        for p in ws
    ):
        return False
    if pair.dtype is F32:
        return not torch.is_autocast_enabled("cuda") and all(p.dtype is F32 for p in ws)
    return pair.dtype is BF


def _split(R: int) -> int:
    """Row chunks of the dW' = dbias_all Y GEMM (its [nb H, K] output alone is a few tiles): 48 where it divides R, else 64, else one
    GEMM."""
    return next((s for s in (48, 64) if R % s == 0), 1)


@contextlib.contextmanager
def _gemms(dt):
    """The GEMMs of the op: autocast off (the operands carry the dtypes); fp32 on TF32 tensor cores (the fp32 path's recipe)."""
    with contextlib.ExitStack() as st:
        st.enter_context(torch.autocast("cuda", enabled=False))
        if dt is F32:
            from miniworld_engine.kernels.bias_only_dit.cuda import tf32

            st.enter_context(tf32.tf32_gemms())
        yield


def _fwd_fake(pair2d, wf, eps):
    return pair2d.new_empty((wf.shape[0], pair2d.shape[0]))


@opaque(fake=_fwd_fake, name="bias_only_dit_hoist_fwd")
def _hoist_fwd(pair2d: torch.Tensor, wf: torch.Tensor, eps: float) -> torch.Tensor:
    """bias_all^T [N, R] (head-major, every block's) in the pair's dtype, freshly allocated (LN0 is a temporary)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    if HK.pass_fwd_served(pair2d, wf):
        return HK.pass_fwd(pair2d, wf, eps)  # LayerNorm + projection in one pass over the pair (bo_hoist_pass.cu)
    ln0 = torch.empty_like(pair2d)
    HK.ln0_rows(pair2d, ln0, eps)
    with _gemms(pair2d.dtype):
        return torch.mm(wf, ln0.t())  # [N, 128] x [128, R]


def _bwd_fake(pair2d, wf, dball, eps, split):
    return [torch.empty_like(pair2d), wf.new_empty(wf.shape, dtype=F32)]


@opaque(fake=_bwd_fake, name="bias_only_dit_hoist_bwd")
def _hoist_bwd(
    pair2d: torch.Tensor, wf: torch.Tensor, dball: torch.Tensor, eps: float, split: int
) -> list[torch.Tensor]:
    """[d pair [R, 128] in the pair's dtype, dW'_all [N, 128] fp32] from dbias_all [N, R]: d LN0 = dbias_all^T W'_all (fp32); the LN0
    backward rows, which also write Y = LN0 as the dW' operand (bf16: [hi | lo], 256 columns); dW'_all = dbias_all Y over ``split``
    row chunks (strided views of dbias_all and Y: no copies), the two halves of the bf16 split added in fp32."""
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    N, R = dball.shape
    f32 = pair2d.dtype is F32
    if HK.pass_bwd_served(pair2d, wf):
        return list(HK.pass_bwd(pair2d, wf, dball, eps))  # one block of 16 heads, bf16 / fp32: one pass over the pair (bo_hoist_pass.cu)
    if HK.fused_bwd_served(pair2d, wf):
        dpair, dwf = HK.hoist_bwd_fused(pair2d, wf, dball, eps)  # one pass over the pair (bo_hoist_bwd.cu)
        return [dpair, dwf]
    dp = pair2d.shape[-1]
    K = HK.operand_cols(pair2d.dtype, dp)
    with _gemms(pair2d.dtype):
        dln = (
            torch.mm(dball.t(), wf) if f32 else torch.mm(dball.t(), wf, out_dtype=F32)
        )  # [R, dp] fp32
        dpair = torch.empty_like(pair2d)
        y = torch.empty(R, K, device=pair2d.device, dtype=pair2d.dtype)
        HK.ln0_bwd_rows(pair2d, dln, dpair, y, eps)
        del dln
        if split > 1:
            c = R // split
            a, b = (
                dball.view(N, split, c).transpose(0, 1),
                y.view(split, c, K),
            )  # [S, N, c], [S, c, K]
            p = (torch.bmm(a, b) if f32 else torch.bmm(a, b, out_dtype=F32)).sum(
                0
            )  # [N, K] fp32
        else:
            p = torch.mm(dball, y) if f32 else torch.mm(dball, y, out_dtype=F32)
        dwf = p if K == dp else p[:, :dp] + p[:, dp:]  # hi + lo
    return [dpair, dwf]


class _Sink:
    """The one [N, R] buffer the hoisted blocks' backwards write their bias gradients into, a [H, R] slice each (``slot``), so the
    hoist's backward GEMMs read them where they were written (``take``). Made by the first block backward that asks; handed to the
    hoist's backward and released there. A slot is handed out once (a bias read by two blocks gets a fresh gradient the second
    time); a gradient that does not sit in its slot (autograd summed it with another, a reentrant checkpoint copied the input) is
    copied in, a missing one is zero."""

    __slots__ = ("buf", "device", "dtype", "given", "h", "nb", "psb", "r")

    def __init__(self, nb, h, r, dtype, device, psb=False):
        self.nb, self.h, self.r, self.dtype, self.device = nb, h, r, dtype, device
        self.buf, self.given, self.psb = None, set(), psb

    def slot(self, i, dtype, device):
        """Block i's [H, R] slice of the buffer, or None (another dtype / device, or the slot already handed out)."""
        if (
            dtype is not self.dtype
            or torch.device(device) != torch.device(self.device)
            or i in self.given
        ):
            return None
        if self.buf is None:
            self.buf = torch.empty(
                self.nb * self.h, self.r, device=self.device, dtype=self.dtype
            )
        self.given.add(i)
        return self.buf[i * self.h : (i + 1) * self.h]

    def take(self, grads):
        """dbias_all [N, R] holding ``grads`` (one per block, None = zero); the buffer leaves the sink."""
        buf, self.buf, self.given = self.buf, None, set()
        if buf is None:
            buf = torch.empty(
                self.nb * self.h, self.r, device=self.device, dtype=self.dtype
            )
        for i, g in enumerate(grads):
            s = buf[i * self.h : (i + 1) * self.h]
            if g is None:
                s.zero_()
                continue
            if not (
                g.data_ptr() == s.data_ptr()
                and g.dtype == s.dtype
                and g.is_contiguous()
                and g.numel() == s.numel()
            ):
                s.copy_(g.reshape(self.h, self.r))
        return buf


def _fold(params, dt):
    """(g [nb, 128] fp32, W [nb, H, 128] fp32, W'_all = W diag(g) [nb H, 128] in ``dt``): fp32 products rounded once to ``dt``, as
    the per-block weight pack rounds Wf."""
    gam = torch.stack([p.detach() for p in params[0::2]]).float()
    wts = torch.stack([p.detach() for p in params[1::2]]).float()
    wf = (wts * gam[:, None, :]).reshape(-1, gam.shape[-1]).to(dt).contiguous()
    return gam, wts, wf


class _Hoist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair, eps, nb, sink, *params):  # params: ln_pair.weight, to_bias.weight of each block
        L, dp = pair.shape[1], pair.shape[-1]
        H = params[1].shape[0]
        A = pair.shape[0] if sink.psb else 1
        gam, wts, wf = _fold(params, pair.dtype)
        ctx.set_materialize_grads(
            False
        )  # an unused block's bias: None, not a zero tensor
        ball = _hoist_fwd(pair.reshape(A * L * L, dp), wf, eps)
        ctx.save_for_backward(pair, wf, gam, wts)
        ctx.meta = (eps, nb, H, sink, [p.dtype for p in params])
        if sink.psb:  # per-sample: block b's slice is the head-major [H, A, L, L]
            return tuple(ball[i * H : (i + 1) * H].view(H, A, L, L) for i in range(nb))
        return tuple(ball[i * H : (i + 1) * H].view(1, H, L, L) for i in range(nb))

    @staticmethod
    def backward(ctx, *grads):
        pair, wf, gam, wts = ctx.saved_tensors
        eps, nb, H, sink, pdts = ctx.meta
        L, dp = pair.shape[1], pair.shape[-1]
        R = (pair.shape[0] if sink.psb else 1) * L * L
        dball = sink.take(grads)
        dpair, dwf = _hoist_bwd(pair.reshape(R, dp), wf, dball, eps, _split(R))
        del dball
        dwf = dwf.view(nb, H, dp)
        dW = dwf * gam[:, None, :]  # dW_b = dW'_b diag(g_b)
        dg = (dwf * wts).sum(1)  # dg_b = sum_h dW'_b o W_b
        out = []
        for i in range(nb):
            out.append(
                dg[i].to(pdts[2 * i]) if ctx.needs_input_grad[4 + 2 * i] else None
            )
            out.append(
                dW[i].to(pdts[2 * i + 1]) if ctx.needs_input_grad[5 + 2 * i] else None
            )
        return (
            dpair.view_as(pair) if ctx.needs_input_grad[0] else None,
            None,
            None,
            None,
            *out,
        )


def pair_bias_all(attns, pair) -> tuple[torch.Tensor, ...]:
    """Each attention module's bias [1, H, L, L] (head-major, contiguous: slices of one [nb H, L L] buffer) from one LN0 pass and one
    GEMM; differentiable in ``pair`` and in every ``ln_pair.weight`` / ``to_bias.weight``. Call ``serves`` first."""
    nb, H, L = len(attns), attns[0].to_bias.weight.shape[0], pair.shape[1]
    sink = _Sink(nb, H, L * L, pair.dtype, pair.device)
    params = [p for a in attns for p in (a.ln_pair.weight, a.to_bias.weight)]
    outs = _Hoist.apply(
        pair.contiguous(), float(attns[0].ln_pair.eps), nb, sink, *params
    )
    for i, o in enumerate(outs):
        setattr(o, SINK_ATTR, (sink, i))
    return tuple(outs)


def _infer_fake(pair2d, wf, eps):
    return pair2d.new_empty((wf.shape[0], pair2d.shape[0]))


@opaque(fake=_infer_fake, name="bias_only_dit_hoist_fwd_infer")
def _hoist_fwd_infer(pair2d: torch.Tensor, wf: torch.Tensor, eps: float) -> torch.Tensor:
    """``_hoist_fwd`` for a call with no gradient (inference): bias_all^T [N, R] head-major in the pair's dtype. For 12, 16 or 24 weight
    rows (one block's heads) ONE pass over the pair: the LayerNorm stays on chip and the projection runs on the tensor cores
    (``pair_bias_k`` bf16, ``pair_bias_tc_k`` fp32 on TF32), so the pair (268 MB at A = 4, L = 512, bf16) is read once and never written
    back as LN0 -- ``_hoist_fwd`` writes LN0 and reads it again for a GEMM (3 passes). More rows: ``_hoist_fwd``."""
    n, r = wf.shape[0], pair2d.shape[0]
    f32 = pair2d.dtype is F32
    if n not in (12, 16, 24) or r % 128:
        return _hoist_fwd(pair2d, wf, eps)
    bias = torch.empty(n, r, device=pair2d.device, dtype=pair2d.dtype)
    pst = torch.empty(r, 2, device=pair2d.device, dtype=F32)  # the row statistics: only the backward reads them
    if f32:
        from miniworld_engine.kernels.bias_only_dit.cuda import tf32

        tf32.rows32().pair_bias_tc_cuda(pair2d, wf, bias, pst, eps, 3 if settings.current().bias_only_dit_pair_exact else 1)
    else:
        from miniworld_engine.kernels.bias_only_dit.cuda.train import ext

        ext().pair_bias_cuda(pair2d, wf, bias, pst, eps)
    return bias


def pair_bias_all_infer(attns, pair) -> tuple[torch.Tensor, ...]:
    """``pair_bias_all_psb`` without a gradient: each block's bias [A, H, L, L] (the permuted view of one head-major [H, A, L, L] slice of
    ONE buffer), from one pass over the pair for a single block (``_hoist_fwd_infer``). Call ``serves(..., per_sample=True)`` first."""
    nb, H, A, L = len(attns), attns[0].to_bias.weight.shape[0], pair.shape[0], pair.shape[1]
    _, _, wf = _fold([p for a in attns for p in (a.ln_pair.weight, a.to_bias.weight)], pair.dtype)
    ball = _hoist_fwd_infer(pair.contiguous().view(A * L * L, pair.shape[-1]), wf, float(attns[0].ln_pair.eps))
    return tuple(ball[i * H : (i + 1) * H].view(H, A, L, L).permute(1, 0, 2, 3) for i in range(nb))


def pair_bias_all_psb(attns, pair) -> tuple[torch.Tensor, ...]:
    """Each attention module's bias [A, H, L, L] for a pair of its own per sample ``pair`` [A, L, L, 128] (A > 1): the ``[A, H, L, L]``
    view of block b's head-major [H, A, L, L] slice of ONE buffer (one LayerNorm pass and one GEMM for every block); differentiable in
    the pair and in every ``ln_pair.weight`` / ``to_bias.weight``. The fused per-sample block reads the head-major buffer without a
    copy and writes its bias gradient into the same slot of the shared gradient buffer. Call ``serves(..., per_sample=True)`` first."""
    nb, H, A, L = len(attns), attns[0].to_bias.weight.shape[0], pair.shape[0], pair.shape[1]
    sink = _Sink(nb, H, A * L * L, pair.dtype, pair.device, psb=True)
    params = [p for a in attns for p in (a.ln_pair.weight, a.to_bias.weight)]
    outs = _Hoist.apply(pair.contiguous(), float(attns[0].ln_pair.eps), nb, sink, *params)
    res = []
    for i, o in enumerate(outs):
        v = o.permute(1, 0, 2, 3)
        setattr(v, SINK_ATTR, (sink, i))
        res.append(v)
    return tuple(res)


__all__ = ["SINK_ATTR", "pair_bias_all", "pair_bias_all_infer", "pair_bias_all_psb", "serves"]
