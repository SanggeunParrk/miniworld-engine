"""Every bias-only DiT block's pair bias from ONE LayerNorm and ONE GEMM on B200 (sm_100), forward and backward: the fused path
of ``modules.token_dit.pair_bias_all``.

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
the accumulation of a [L, L, 128] pair gradient, by one pass of each for all blocks.

THE BIAS IS fp32 AND EXACT, whatever the pair dtype (user rule 2026-11-19: the logits are 1e3..2e5 with a key spread of ~500; bf16 resolves
32..1024 there and broke the peptide bonds): the forward is LN0 in fp32 (``hoist.ln0_fp32``), the unrounded fp32 W' (``_fold``'s last value)
and one cuBLAS fp32 GEMM on the FMA pipe, TF32 off (``hoist.exact_logits``); the consumers (``bias_only_dit.py`` / ``bias_only_dit_train.py``)
take the fp32 bias into an fp32 softmax and round only P to the block's dtype. Backward (the bias gradient is the activations' dtype, the
autograd edge casts it to the bias's fp32 and ``_Sink.take`` copies it into its slot): bf16 (pair bf16; the weights in any dtype, folded in
fp32 and rounded to bf16 for the backward GEMMs): bf16 operands, fp32 accumulation, d LN0 and dW' in fp32, d pair in the pair's dtype.
fp32 (pair and weights fp32, no autocast): the same structure, the backward GEMMs on TF32 tensor cores (``tf32_gemms``), every tensor fp32.

``serves()`` is the gate (B200 or H100, an engine implementation, pair [1, L, L, C] bf16 or fp32, C = 128, or 256 on H100); anything
else (A100 included) takes the PyTorch fold of ``modules.token_dit.pair_bias_all``. H100 (2026-11-12, as engine 2.4.4 served it): the
PyTorch fold hands out head-major views of a [1, L, L, N] product, which the H100 block copies to [H, L, L] on every call (5.5 ms of
9.8 ms in a 24-block L 768 stack); this op's buffer is head-major already.
"""

from __future__ import annotations

import contextlib

import torch

from miniworld_engine.kernels._compile import opaque

DP = 128
#: the pair widths served: 128 everywhere, 256 on H100 (ln0_rows / ln0_bwd_rows are templates on the width; the GEMMs take any K)
DPS = (128, 256)
#: the cards: B200 (sm_100) and H100 (sm_90)
CARDS = ((10, 0), (9, 0))
BF = torch.bfloat16
F32 = torch.float32
#: the attribute of each hoisted bias that names its slot of the shared gradient buffer: (``_Sink``, block index)
SINK_ATTR = "_bias_only_dit_hoist_sink"


def serves(implementation, attns, pair) -> bool:
    """The fused hoist serves ``pair`` [1, L, L, C] (C 128, or 256 on H100) on a B200 or an H100 and the attention modules ``attns``
    (eps / head count / no offset checked by the caller)."""

    if implementation != "miniworld":
        return False
    if not pair.is_cuda or torch.cuda.get_device_capability(pair.device) not in CARDS:
        return False
    dp = pair.shape[-1]
    wide_ok = torch.cuda.get_device_capability(pair.device) == (9, 0)
    if pair.ndim != 4 or pair.shape[0] != 1 or pair.shape[1] != pair.shape[2] or pair.shape[1] == 0:
        return False
    if dp not in DPS or (dp != DP and not wide_ok):
        return False
    ws = [p for a in attns for p in (a.ln_pair.weight, a.to_bias.weight)]
    if any(p is None or not p.is_cuda or p.device != pair.device or p.shape[-1] != dp for p in ws):
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
    return pair2d.new_empty((wf.shape[0], pair2d.shape[0]), dtype=F32)


@opaque(fake=_fwd_fake, name="bias_only_dit_hoist_fwd")
def _hoist_fwd(pair2d: torch.Tensor, wf: torch.Tensor, eps: float) -> torch.Tensor:
    """bias_all^T [N, R] (head-major, every block's), ALWAYS fp32 and exact (user rule 2026-11-19: the bias-only DiT's logits and softmax
    are fp32): LN0 in fp32, ``wf`` the unrounded fp32 W' = W diag(gamma), the product on the FMA pipe (TF32 off); freshly allocated."""
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    ln32, _ = HK.ln0_fp32(pair2d, eps)
    return HK.exact_logits(wf, ln32)  # [N, dp] x [dp, R] fp32


def _tc_ok(p2: torch.Tensor, wf16: torch.Tensor) -> bool:
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    return HK.logits_tc_ok(p2, wf16)


def is_h100(ref: torch.Tensor) -> bool:
    return ref.is_cuda and torch.cuda.get_device_capability(ref.device) == (9, 0)


def dbias_dtype(ref: torch.Tensor) -> torch.dtype:
    """The dtype of the hoisted blocks' bias gradient: fp32 on H100 (the bf16 block's dbias = P o (dP - D) is formed in fp32 there and
    stays fp32 into the pair-bias backward: rounding it to bf16 left the constant-channel / ln_pair gradients 1e-1..1 off), else the
    activations' dtype."""
    return F32 if is_h100(ref) else ref.dtype


def _bwd_exact_fake(pair2d, wf32, dball, eps, split):
    return [torch.empty_like(pair2d), wf32.new_empty(wf32.shape, dtype=F32)]


@opaque(fake=_bwd_exact_fake, name="bias_only_dit_hoist_bwd_exact")
def hoist_bwd_exact(
    pair2d: torch.Tensor, wf32: torch.Tensor, dball: torch.Tensor, eps: float, split: int, pst: torch.Tensor | None = None
) -> list[torch.Tensor]:
    """``_hoist_bwd`` with an fp32 dbias_all [N, R] and the unrounded fp32 W': every product exact fp32 (FMA pipe, TF32 off) --
    [d pair [R, dp] in the pair's dtype, dW'_all [N, dp] fp32]. d LN0 = dbias^T W'; the LN0 backward rows write Y = LN0 in fp32;
    dW' = dbias Y as a strided batch of ``split`` row chunks."""
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    N, R = dball.shape
    dp = pair2d.shape[-1]
    if N == 16 and eps == 1e-5 and HK.hoist_bwd_tc_ok(pair2d, dball, wf32):
        # one block's 16 heads, width 128 / 256: d LN0, the LN0 backward and dW' in ONE pass on error-compensated TF32 mma (no row statistics needed)
        return list(HK.hoist_bwd_tc(pair2d, dball, wf32))
    if (
        HK.OPT["bwd"]
        and pst is not None
        and N == 16
        and dp == 128
        and R % 32 == 0
        and pair2d.is_contiguous()
        and dball.stride(1) == 1
        and wf32.is_contiguous()
    ):
        # one block's 16 heads: d LN0 = dbias^T W', the LN0 backward and dW' = dbias Y in ONE pass over the rows (no d LN0 / Y in memory)
        return list(HK.hoist_bwd_fused(pair2d, dball, wf32, pst))
    with HK.exact_fp32_gemms(), torch.autocast("cuda", enabled=False):
        dln = torch.mm(dball.t(), wf32)  # [R, dp] fp32
        dpair = torch.empty_like(pair2d)
        y = torch.empty(R, dp, device=pair2d.device, dtype=F32)
        HK.ln0_bwd_rows(pair2d, dln, dpair, y, eps)
        del dln
        if split > 1:
            c = R // split
            dwf = torch.bmm(dball.view(N, split, c).transpose(0, 1), y.view(split, c, dp)).sum(0)
        else:
            dwf = torch.mm(dball, y)
    return [dpair, dwf]


def _bwd_fake(pair2d, wf, dball, eps, split):
    return [torch.empty_like(pair2d), wf.new_empty(wf.shape, dtype=F32)]


@opaque(fake=_bwd_fake, name="bias_only_dit_hoist_bwd")
def _hoist_bwd(pair2d: torch.Tensor, wf: torch.Tensor, dball: torch.Tensor, eps: float, split: int) -> list[torch.Tensor]:
    """[d pair [R, 128] in the pair's dtype, dW'_all [N, 128] fp32] from dbias_all [N, R]: d LN0 = dbias_all^T W'_all (fp32); the LN0
    backward rows, which also write Y = LN0 as the dW' operand (bf16: [hi | lo], 256 columns); dW'_all = dbias_all Y over ``split``
    row chunks (strided views of dbias_all and Y: no copies), the two halves of the bf16 split added in fp32."""
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    N, R = dball.shape
    f32 = pair2d.dtype is F32
    dp = pair2d.shape[-1]
    K = HK.operand_cols(pair2d.dtype, dp)
    with _gemms(pair2d.dtype):
        dln = torch.mm(dball.t(), wf) if f32 else torch.mm(dball.t(), wf, out_dtype=F32)  # [R, dp] fp32
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
            p = (torch.bmm(a, b) if f32 else torch.bmm(a, b, out_dtype=F32)).sum(0)  # [N, K] fp32
        else:
            p = torch.mm(dball, y) if f32 else torch.mm(dball, y, out_dtype=F32)
        dwf = p if dp == K else p[:, :dp] + p[:, dp:]  # hi + lo
    return [dpair, dwf]


class _Sink:
    """The one [N, R] buffer the hoisted blocks' backwards write their bias gradients into, a [H, R] slice each (``slot``), so the
    hoist's backward GEMMs read them where they were written (``take``). Made by the first block backward that asks; handed to the
    hoist's backward and released there. A slot is handed out once (a bias read by two blocks gets a fresh gradient the second
    time); a gradient that does not sit in its slot (autograd summed it with another, a reentrant checkpoint copied the input) is
    copied in, a missing one is zero."""

    __slots__ = ("buf", "device", "dtype", "given", "h", "nb", "r")

    def __init__(self, nb, h, r, dtype, device):
        self.nb, self.h, self.r, self.dtype, self.device = nb, h, r, dtype, device
        self.buf, self.given = None, set()

    def slot(self, i, dtype, device):
        """Block i's [H, R] slice of the buffer, or None (another dtype / device, or the slot already handed out)."""
        if dtype is not self.dtype or torch.device(device) != torch.device(self.device) or i in self.given:
            return None
        if self.buf is None:
            self.buf = torch.empty(self.nb * self.h, self.r, device=self.device, dtype=self.dtype)
        self.given.add(i)
        return self.buf[i * self.h : (i + 1) * self.h]

    def take(self, grads):
        """dbias_all [N, R] holding ``grads`` (one per block, None = zero); the buffer leaves the sink."""
        buf, self.buf, self.given = self.buf, None, set()
        if buf is None:
            buf = torch.empty(self.nb * self.h, self.r, device=self.device, dtype=self.dtype)
        for i, g in enumerate(grads):
            s = buf[i * self.h : (i + 1) * self.h]
            if g is None:
                s.zero_()
                continue
            if not (g.data_ptr() == s.data_ptr() and g.dtype == s.dtype and g.is_contiguous() and g.numel() == s.numel()):
                s.copy_(g.reshape(self.h, self.r))
        return buf


def _fold(params, dt):
    """(g [nb, 128] fp32, W [nb, H, 128] fp32, W'_all = W diag(g) [nb H, 128] in ``dt`` (the backward GEMMs' operand, rounded once as the
    per-block weight pack rounds Wf), the same unrounded fp32 (the exact logits' operand))."""
    gam = torch.stack([p.detach() for p in params[0::2]]).float()
    wts = torch.stack([p.detach() for p in params[1::2]]).float()
    wf32 = (wts * gam[:, None, :]).reshape(-1, gam.shape[-1]).contiguous()
    return gam, wts, wf32.to(dt).contiguous(), wf32


class _Hoist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair, eps, nb, sink, *params):  # params: ln_pair.weight, to_bias.weight of each block
        L, dp = pair.shape[1], pair.shape[-1]
        H = params[1].shape[0]
        gam, wts, wf, wf32 = _fold(
            params, pair.dtype
        )  # wf: the backward GEMMs' operand (rounded to the pair dtype); wf32: the exact logits'
        ctx.set_materialize_grads(False)  # an unused block's bias: None, not a zero tensor
        pst = None
        if (
            is_h100(pair) and wf32.shape[0] == 16 and dp == 128 and eps == 1e-5
        ):  # one block of 16 heads: LN0 and the dot products in one pass, + the row statistics
            from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

            ball, pst = HK.pair_logits(pair.reshape(L * L, dp), wf32, stats=True)
        elif is_h100(pair) and wf32.shape[0] % 16 == 0 and eps == 1e-5 and _tc_ok(pair.reshape(L * L, dp), wf32[:16]):
            # the 256-wide pair (or several blocks): per block one pass (``hoist.logits_tc``: LN0 on chip, split-fp32 TF32 products, error
            # below the FMA-pipe GEMM's); the backward needs no row statistics there (it recomputes them)
            from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

            p2 = pair.reshape(L * L, dp)
            ball = torch.empty(wf32.shape[0], L * L, device=pair.device, dtype=F32)
            for b in range(wf32.shape[0] // 16):
                HK.logits_tc(p2, wf32[16 * b : 16 * b + 16], ball[16 * b : 16 * b + 16])
        else:
            ball = _hoist_fwd(pair.reshape(L * L, dp), wf32, eps)  # fp32 logits whatever the pair dtype
        ctx.save_for_backward(pair, wf, gam, wts, *(() if pst is None else (pst,)))
        ctx.meta = (eps, nb, H, sink, [p.dtype for p in params])
        return tuple(ball[i * H : (i + 1) * H].view(1, H, L, L) for i in range(nb))

    @staticmethod
    def backward(ctx, *grads):
        pair, wf, gam, wts, *rest = ctx.saved_tensors
        eps, nb, H, sink, pdts = ctx.meta
        L, dp = pair.shape[1], pair.shape[-1]
        R = L * L
        dball = sink.take(grads)
        if is_h100(pair):  # fp32 dbias (``dbias_dtype``), exact backward products
            dpair, dwf = hoist_bwd_exact(
                pair.reshape(R, dp), (wts * gam[:, None, :]).reshape(-1, dp).contiguous(), dball, eps, _split(R), rest[0] if rest else None
            )
        else:
            dpair, dwf = _hoist_bwd(pair.reshape(R, dp), wf, dball, eps, _split(R))
        del dball
        dwf = dwf.view(nb, H, dp)
        dW = dwf * gam[:, None, :]  # dW_b = dW'_b diag(g_b)
        dg = (dwf * wts).sum(1)  # dg_b = sum_h dW'_b o W_b
        out = []
        for i in range(nb):
            out.append(dg[i].to(pdts[2 * i]) if ctx.needs_input_grad[4 + 2 * i] else None)
            out.append(dW[i].to(pdts[2 * i + 1]) if ctx.needs_input_grad[5 + 2 * i] else None)
        return (
            dpair.view_as(pair) if ctx.needs_input_grad[0] else None,
            None,
            None,
            None,
            *out,
        )


def pair_bias_all(attns, pair) -> tuple[torch.Tensor, ...]:
    """Each attention module's bias [1, H, L, L] in FP32 (head-major, contiguous: slices of one [nb H, L L] buffer) from one LN0 pass and one
    GEMM; differentiable in ``pair`` and in every ``ln_pair.weight`` / ``to_bias.weight``. Call ``serves`` first."""
    nb, H, L = len(attns), attns[0].to_bias.weight.shape[0], pair.shape[1]
    sink = _Sink(nb, H, L * L, dbias_dtype(pair), pair.device)
    params = [p for a in attns for p in (a.ln_pair.weight, a.to_bias.weight)]
    outs = _Hoist.apply(pair.contiguous(), float(attns[0].ln_pair.eps), nb, sink, *params)
    for i, o in enumerate(outs):
        setattr(o, SINK_ATTR, (sink, i))
    return tuple(outs)


__all__ = ["SINK_ATTR", "pair_bias_all", "serves"]
