"""A100 (sm_80) hand-CUDA TriMul: D128, one direction (hidden 128) or bidirectional (hidden 2 x 128), inference and training.

The path developed in ``experiments/a100_trimul_fwd`` (tag ``archive/a100-sm80-branch-20260928``), measured on an A100 80GB PCIe
(300 W, CUDA-graph replay, the baseline's fixture: B1, bf16, 10 % masked tokens, dropout 0.25 in training):

    op                         bidir L384   bidir L768    single L384   single L768
    forward (inference)        600 us       2780 us       365 us        1616 us     (Anthropic best 790 / 3417 / 407 / 1776 us)
    training step (fwd + bwd)  2.55 ms      11.21 ms      1.53 ms       6.52 ms     (engine Triton 3.20 / 14.15 / 1.98 / 8.17 ms)

* forward: K1 (LN_in + gated projections + pair mask -> channel-major a | b planes, also the LN_in statistics) -> contraction (one
  custom launch for both bidirectional halves at L <= 512, cuBLAS elsewhere) -> K3 (LN_out + output projection, gate, residual, dropout
  row scale; in training also the four LayerNorm statistics and x_n).
* backward: B1 (output side: dX planes, d_g, the W_o gradient on chip at hidden 128 or as a split-K GEMM at 256) -> the weight-gradient
  GEMMs on a side stream under the contraction backward (cuBLAS) -> B7 joint (cooperative: 16 source CTAs per group recompute g / p and
  accumulate the input-projection gradients in registers, consumers take dg / dp through an L2 ring for the dx_n GEMM, the LN_in backward
  and the residual).  A token count that is not a whole number of 128-token tiles, or a card too small for one cooperative group,
  runs B7src + B8 instead (the same math through DRAM).

**Ampere-only and shape-specific by construction**: sm_80, bf16, d_pair = d_hidden = 128, one square plane per launch (the integration runs a batch plane by
plane), L a multiple of 16.
``supports()`` is the whole gate; everything it rejects keeps the existing path.
"""

import functools
import hashlib
import os
import warnings
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"

D = 128
#: direction codes, as the H100 inference path: 0 bidirectional, 1 outgoing, 2 incoming
BIDIR, OUTGOING, INCOMING = 0, 1, 2

# The measured defaults (experiments README, "Code map and defaults"): B7 joint consumers per group by plane channels, ring slots, the
# on-chip W_o gradient only at 128 plane channels (its accumulators spill at 256), the custom contraction for bidirectional L <= 512.
_B7J_CONSUMERS = {256: 8, 128: 6}
_B7J_RINGS = 8
_CUSTOM_CONTRACT_MAX_L = 512


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_TRIMUL_SM80_FLAGS: extra -D flags for A/B experiments (their own build)
    extra = os.environ.get("MINIWORLD_TRIMUL_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"trimul_sm80{tag}",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


@functools.lru_cache(maxsize=8)
def _is_ampere(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (8, 0)


def supports(pair: torch.Tensor, d_hidden: int, mask: torch.Tensor | None = None) -> bool:
    """The kernels' own requirements: sm_80, bf16, d_pair = d_hidden = 128 (tile shapes are literals), one square contiguous plane with
    L % 16 == 0 (whole 128-token tiles for the joint backward), and a [B, L] token mask if any.  A batch is run plane by plane by the integration."""
    if os.environ.get("MINIWORLD_TRIMUL_SM80", "1") == "0":
        return False
    if not pair.is_cuda or pair.dtype is not torch.bfloat16 or not pair.is_contiguous():
        return False
    if pair.ndim != 4 or pair.shape[0] < 1 or pair.shape[1] != pair.shape[2] or pair.shape[-1] != D or d_hidden != D:
        return False
    n = pair.shape[1]
    if n <= 0 or n % 16 != 0:
        return False
    if mask is not None and (mask.shape != (pair.shape[0], n) or mask.dtype is not torch.bool):
        return False
    return _is_ampere(pair.device.index if pair.device.index is not None else torch.cuda.current_device())


_BUILD_FAILED = False


def available(pair: torch.Tensor, d_hidden: int, mask: torch.Tensor | None = None) -> bool:
    """``supports()`` plus a successful (cached) build; a build failure warns once and keeps the existing path."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supports(pair, d_hidden, mask):
        return False
    if torch.compiler.is_compiling() or _is_fake(pair):
        return True
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"sm80 TriMul unavailable, keeping the existing path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor
    return any(isinstance(t, FakeTensor) for t in tensors)


def token_mask(mask: torch.Tensor | None, n: int, device) -> torch.Tensor:
    """The [1, L] bool token mask as the kernels' [L] uint8 (reinterpreted in place, no cast kernel); empty = no mask."""
    if mask is None:
        return torch.empty(0, dtype=torch.uint8, device=device)
    m = mask.reshape(n)
    return m.view(torch.uint8) if m.is_contiguous() else m.contiguous().view(torch.uint8)


# ------------------------------------------------------------------------------------------------------------ weight layouts
@functools.lru_cache(maxsize=8)
def _k1_rows(ch: int, dev):
    """K1 packed row -> (plane channel oc, gate row?): blocks of 64 rows = 4 warps x (8 gate rows | 8 projection rows) of output channels
    oc = 32 step + 8 nw + c8; and the packed rows of each input projection (left gate, left, right gate, right) by its output channel."""
    nstep = ch // 16
    step = torch.arange(nstep).view(-1, 1, 1)
    nw = torch.arange(4).view(1, -1, 1)
    n = torch.arange(16).view(1, 1, -1)
    oc = (32 * step + 8 * nw + (n % 8)).reshape(-1)
    is_gate = (n < 8).expand(nstep, 4, 16).reshape(-1)
    rows = torch.empty(4, ch, dtype=torch.long)
    for k, (gate, lo) in enumerate(((True, 0), (False, 0), (True, ch), (False, ch))):
        sel = torch.nonzero((is_gate == gate) & (oc >= lo) & (oc < lo + ch)).flatten()
        rows[k, oc[sel] - lo] = sel
    return oc.to(dev), is_gate.to(dev), rows.to(dev)


def _pack(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, *, train: bool) -> dict:
    """Weights in the kernels' layouts, one launch per call (``pack_sm80.cuh``).  Never cached: a captured CUDA graph must repack after
    an optimizer step.  0.5 is folded into K1's and K3's rows: sigmoid(g) p = 0.5 p (1 + tanh(g / 2)), exact in bf16."""
    ch = wl.shape[0]
    dev = wl.device
    # the four front matrices are read through their strides (the bidirectional module stores them [in, out]: no copy per call)
    # fp32 master parameters are cast here, inside the op and outside autograd (the gradients are written in the parameters' own dtype)
    bf = torch.bfloat16
    ws = [*(w.detach().to(bf) for w in (wl, wlg, wr, wrg)), *(w.detach().to(bf).contiguous() for w in (wg, wo))]
    ln = [t.detach().float().contiguous() for t in (gi, bi, go, bo)]
    w1 = torch.empty((ch // 16, 16, 64, 8), dtype=torch.bfloat16, device=dev)
    wg3 = torch.empty((D, D), dtype=torch.bfloat16, device=dev)
    wo3 = torch.empty((D, ch), dtype=torch.bfloat16, device=dev)
    vec = torch.empty((8, D), dtype=torch.float32, device=dev)
    wdx = torch.empty((4 * ch + D, D) if train else (0,), dtype=torch.bfloat16, device=dev)
    wob1 = torch.empty((D, ch) if train else (0,), dtype=torch.bfloat16, device=dev)
    _ext().pack(*ws, *ln, w1, wg3, wo3, vec, wdx, wob1)
    so, bo_, sg, bg, so1, bo1, sg1, bg1 = vec.unbind(0)
    pk = dict(ch=ch, w1=w1, g_in=ln[0], b_in=ln[1], wo=wo3, wg=wg3, so=so, bo=bo_, sg=sg, bg=bg)
    if train:
        pk.update(wdx=wdx, wo_b1=wob1, wg_b1=ws[4], so_b1=so1, bo_b1=bo1, sg_b1=sg1, bg_b1=bg1)
    return pk


def _pack_reference(wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, *, train: bool) -> dict:
    """``_pack`` in torch operations (the layouts' definition; the tests compare the two)."""
    f = lambda t: t.detach().float()  # noqa: E731
    ch = wl.shape[0]
    nstep = ch // 16
    oc, is_gate, _ = _k1_rows(ch, wl.device)
    wg_all = torch.cat([f(wlg), f(wrg)], 0)                                                      # [2 CH, 128]
    wp_all = torch.cat([f(wl), f(wr)], 0)
    w_rows = torch.where(is_gate[:, None], wg_all[oc], wp_all[oc])                              # [4 CH, 128], K1 row order
    w1 = (0.5 * w_rows).to(torch.bfloat16).view(nstep, 64, 16, 8).transpose(1, 2).contiguous()   # [block][16 B k-granule][64 rows][8]
    # K3: LayerNorm affine folded into the weights (W' = bf16(W diag gamma), s = sum_k W', b = W beta)
    wo_k3 = (0.5 * f(wo) * f(go)[None, :]).to(torch.bfloat16).contiguous()
    wg_k3 = (0.5 * f(wg) * f(gi)[None, :]).to(torch.bfloat16).contiguous()
    pk = dict(ch=ch, w1=w1, g_in=f(gi).contiguous(), b_in=f(bi).contiguous(), wo=wo_k3, wg=wg_k3,
              so=wo_k3.float().sum(1).contiguous(), bo=(0.5 * f(wo) @ f(bo)).contiguous(),
              sg=wg_k3.float().sum(1).contiguous(), bg=(0.5 * f(wg) @ f(bi)).contiguous())
    if train:
        wo_b1 = (f(wo) * f(go)[None, :]).to(torch.bfloat16).contiguous()
        wg_b1 = (f(wg) * f(gi)[None, :]).to(torch.bfloat16)
        pk.update(wdx=torch.cat([w_rows.to(torch.bfloat16), wg.detach().to(torch.bfloat16)], 0).contiguous(),   # [K1 rows ; W_og]
                  wo_b1=wo_b1, wg_b1=wg.detach().to(torch.bfloat16).contiguous(),   # B1's gate runs on the saved x_n: raw W_og
                  so_b1=wo_b1.float().sum(1).contiguous(), bo_b1=(f(wo) @ f(bo)).contiguous(),
                  sg_b1=wg_b1.float().sum(1).contiguous(), bg_b1=(f(wg) @ f(bi)).contiguous())
    return pk


# ------------------------------------------------------------------------------------------------------------------ forward
def _contract(ext, ab, x, ch, n, direction):
    a, b = ab[:ch].view(ch, n, n), ab[ch:].view(ch, n, n)
    h = ch // 2 if direction == BIDIR else (ch if direction == OUTGOING else 0)     # channels [0, h) outgoing (NT), the rest incoming (TN)
    if direction == BIDIR and n <= _CUSTOM_CONTRACT_MAX_L and n % 128 == 0:
        ext.contract(a, b, x, h, 0)                                                   # both halves in one launch
        return
    if h:
        torch.bmm(a[:h], b[:h].transpose(1, 2), out=x[:h])                          # outgoing: sum_k a[i,k] b[j,k]
    if h < ch:
        torch.bmm(a[h:].transpose(1, 2), b[h:], out=x[h:])                          # incoming: sum_k a[k,i] b[k,j]


def _front(ext, zf, mk, pk, n):
    """K1 + contraction -> (a | b planes, X, LN_in statistics)."""
    ch, T = pk["ch"], zf.shape[0]
    ab = zf.new_empty((2 * ch, T))
    x = zf.new_empty((ch, n, n))
    zst = torch.empty(T, 2, device=zf.device, dtype=torch.float32)
    ext.k1z(zf, mk, pk["w1"], pk["g_in"], pk["b_in"], ab, n, pk["eps_in"], zst)
    return ab, x, zst


def _run_forward(leaves, mask, ds, direction, eps_in, eps_out, *, save):
    ext = _ext()
    z = leaves[0]
    n = z.shape[1]
    T = n * n
    zf = z.reshape(T, D)
    pk = _pack(*leaves[1:], train=False)
    pk["eps_in"] = eps_in
    ab, x, zst = _front(ext, zf, mask, pk, n)
    _contract(ext, ab, x, pk["ch"], n, direction)
    out = torch.empty_like(zf)
    if not save and ds.numel() == 0:
        ext.k3z(x.view(pk["ch"], T), zf, pk["wo"], pk["wg"], pk["so"], pk["bo"], pk["sg"], pk["bg"], out, eps_out, zst)
        return out.view_as(z), None
    # training K3 applies the dropout row scale and saves the LayerNorm statistics (mu_o, r_o, mu_i, r_i) and x_n for the backward
    stats = torch.empty(T, 4, device=z.device, dtype=torch.float32)
    xn = torch.empty_like(zf)
    ext.k3_train(x.view(pk["ch"], T), zf, pk["wo"], pk["wg"], pk["so"], pk["bo"], pk["sg"], pk["bg"], out, eps_out, ds, n, stats, xn,
                 pk["g_in"], pk["b_in"], zst)
    return out.view_as(z), (ab, x, stats, xn)


def _forward_nograd_fake(leaves, mask, ds, direction, eps_in, eps_out):
    """The output has the pair's shape and dtype; nothing is saved."""
    return torch.empty_like(leaves[0])


@opaque(fake=_forward_nograd_fake, name="trimul_sm80_forward")
def forward_nograd(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, direction: int, eps_in: float, eps_out: float,
                   ) -> torch.Tensor:
    """z + drop_row(trimul(z)) without the backward's saves.  ``mask`` [L] uint8 or empty, ``ds`` [L, 128] bf16 or empty."""
    return _run_forward(leaves, mask, ds, direction, eps_in, eps_out, save=False)[0]


def _forward_train_fake(leaves, mask, ds, direction, eps_in, eps_out):
    """[out like the pair, a | b planes [2 CH, T], X [CH, L, L], LayerNorm statistics [T, 4] fp32, x_n [T, 128]], CH the first
    weight's leading size and T = L^2."""
    z = leaves[0]
    n, ch = z.shape[1], leaves[1].shape[0]
    T = n * n
    return [torch.empty_like(z), z.new_empty((2 * ch, T)), z.new_empty((ch, n, n)),
            torch.empty((T, 4), dtype=torch.float32, device=z.device), z.new_empty((T, D))]


@opaque(fake=_forward_train_fake, name="trimul_sm80_train_fwd")
def forward_train(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, direction: int, eps_in: float, eps_out: float,
                  ) -> list[torch.Tensor]:
    """[out, a | b planes, X, LayerNorm statistics, x_n]."""
    out, (planes, x, stats, xn) = _run_forward(leaves, mask, ds, direction, eps_in, eps_out, save=True)
    return [out, planes, x, stats, xn]


# ----------------------------------------------------------------------------------------------------------------- backward
_SIDE: dict = {}


def _side_stream(dev):
    if dev not in _SIDE:
        _SIDE[dev] = torch.cuda.Stream(device=dev)
    return _SIDE[dev]


def _mm32(a, b):
    """bf16 x bf16 -> fp32 matmul (fp32 accumulate, fp32 result)."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return torch.mm(a.float(), b.float())


def _gemm_tk(a, b):
    """a [T, M], b [T, N] (or a transposed view) -> a^T b in fp32 with K = T split into chunks (a batched GEMM + a sum): cuBLAS picks a
    non-split-K kernel for these 128 x 256 x L^2 shapes (0.99 ms instead of ~0.3 at L = 768)."""
    T = a.shape[0]
    S = next((s for s in (32, 16, 8, 4, 2) if T % s == 0 and T // s >= 256), 1)
    if S == 1:
        return _mm32(a.t(), b)
    ac = a.unflatten(0, (S, T // S)).transpose(1, 2)
    bc = b.unflatten(0, (S, T // S))
    try:
        part = torch.bmm(ac, bc, out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        part = torch.bmm(ac.float(), bc.float())
    return part.sum(0)


def _run_backward(leaves, mask, ds, saved, dy, direction, eps_in, eps_out):
    ext = _ext()
    z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo = leaves
    ab, x, stats, xn = saved
    n = z.shape[1]
    T, C = n * n, D
    dev = z.device
    zf = z.reshape(T, C)
    pk = _pack(*leaves[1:], train=True)
    ch = pk["ch"]
    dyf = dy.reshape(T, C).to(torch.bfloat16).contiguous()
    # ---- B1
    dX = z.new_empty((ch, n, n))
    joint = T % 128 == 0
    if joint:
        S, Cc = ch // 16, _B7J_CONSUMERS[ch]
        groups = ext.b7j_capacity() // (S + Cc)
        joint = groups >= 1
    if joint:                                   # d_g alone; dg / dp of the planes live only in the joint kernel's L2 ring
        ldd = C
        dgp = z.new_empty((T, C))
        dgv = dgp
    else:
        ldd = 4 * ch + C
        dgp = z.new_empty((T, ldd))            # [dg | dp of the planes (K1 order) | d_g of the output gate]
        dgv = dgp[:, 4 * ch:]
    ao = None
    if ch == 128:                               # G = A_o^T X^T on chip
        nb = ext.b1g_grid(T)
        red = torch.empty(nb, 2, C, device=dev, dtype=torch.float32)
        gpart = torch.empty(nb, C, ch, device=dev, dtype=torch.float32)
        ext.b1g(x.view(ch, T), xn, dyf, ds, pk["wo_b1"], pk["wg_b1"], pk["so_b1"], pk["bo_b1"], stats, dX.view(ch, T), dgv, ldd, red, gpart, n)
    else:
        red = torch.empty(ext.b1_red_rows(T, ch), 2, C, device=dev, dtype=torch.float32)
        gpart = None
        ao = z.new_empty((T, C))
        ext.b1(x.view(ch, T), zf, dyf, ds, pk["wo_b1"], pk["wg_b1"], pk["so_b1"], pk["bo_b1"], pk["sg_b1"], pk["bg_b1"],
               dX.view(ch, T), ao, dgv, ldd, stats, red, xn, pk["g_in"], pk["b_in"], n, eps_out)

    def weight_grads():
        v_o, S_o = red.sum(0).unbind(0)
        # W_o, LN_out affine:  H = sum_t d_o r (X - mu) = A_o^T X^T - (A_o^T mu) 1^T ;  S_o = sum_t d_o
        G = gpart.sum(0) if gpart is not None else _gemm_tk(ao, x.view(ch, T).t())
        H = G - v_o[:, None]
        Wo = wo.detach().float()
        d_wo = H * go.detach().float()[None, :] + S_o[:, None] * bo.detach().float()[None, :]
        d_go = (Wo * H).sum(0)
        d_bo = (Wo * S_o[:, None]).sum(0)
        d_wg = _mm32(dgv.t(), xn)              # W_og: the gate ran on the saved x_n (affine included)
        return d_wo, d_go, d_bo, d_wg

    # the weight-gradient GEMMs are DRAM-bound, the contraction backward is tensor-bound: run them side by side
    main = torch.cuda.current_stream(dev)
    side = _side_stream(dev)
    side.wait_stream(main)
    for t in (red, gpart, ao, x, dgp, xn, wo, go, bo):
        if t is not None:
            t.record_stream(side)
    with torch.cuda.stream(side):
        d_wo, d_go, d_bo, d_wg = weight_grads()
    # ---- contraction backward -> dA | dB planes
    dab = z.new_empty((2 * ch, n, n))
    A, Bp, dA, dB = ab[:ch].view(ch, n, n), ab[ch:].view(ch, n, n), dab[:ch], dab[ch:]
    h = ch // 2 if direction == BIDIR else (ch if direction == OUTGOING else 0)
    if h:                                      # X = A B^T: dA = dX B, dB = dX^T A
        torch.bmm(dX[:h], Bp[:h], out=dA[:h])
        torch.bmm(dX[:h].transpose(1, 2), A[:h], out=dB[:h])
    if h < ch:                                 # X = A^T B: dA = B dX^T, dB = A dX
        torch.bmm(Bp[h:], dX[h:].transpose(1, 2), out=dA[h:])
        torch.bmm(A[h:], dX[h:], out=dB[h:])
    dz = torch.empty_like(zf)
    if joint:
        # ---- B7 joint: sources (dg / dp + dW per weight block) -> L2 ring -> consumers (dx_n GEMM + LN_in backward + residual)
        ring = torch.empty(groups * _B7J_RINGS * 128 * 64 * S, device=dev, dtype=torch.bfloat16)
        flags = torch.zeros(groups * _B7J_RINGS * (S + 1), device=dev, dtype=torch.int32)
        dwp = torch.empty(groups * ext.b7j_dwseg(), 4 * ch, C, device=dev, dtype=torch.float32)
        part = torch.empty(groups * Cc, 2, C, device=dev, dtype=torch.float32)
        ext.b7j(xn, pk["w1"], dab.view(2 * ch, T), mask, pk["wdx"], dgv, zf, dyf, stats, pk["g_in"], ring,
                flags[:groups * _B7J_RINGS * S], flags[groups * _B7J_RINGS * S:], dz, dwp, part, n, Cc, groups, _B7J_RINGS)
    else:
        # ---- B7src: dg / dp + input-weight gradient partials; B8: dx_n = [dgp | d_g] . [W_in ; W_og] + LN_in backward + residual
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        dwp = torch.empty(max(1, sms // (ch // 32)), 4 * ch, C, device=dev, dtype=torch.float32)
        ext.b7src(xn, pk["w1"], dab.view(2 * ch, T), mask, dgp, ldd, dwp, n)
        part = torch.empty(min((T + 127) // 128, 2 * sms), 2, C, device=dev, dtype=torch.float32)
        ext.b8(dgp, ldd, pk["wdx"], zf, dyf, stats, pk["g_in"], dz, part)
    # one launch: B7's fp32 partials -> the four input-projection gradients (K1 row order undone, in the parameters' strides) and the
    # LayerNorm_in gradients (in the parameters' dtype)
    g_w = [_grad_like(t) for t in (wl, wlg, wr, wrg)]
    g_gi, g_bi = torch.empty_like(gi), torch.empty_like(bi)
    ext.finalize(dwp, part, *g_w, g_gi, g_bi)
    main.wait_stream(side)
    for t in (d_wo, d_go, d_bo, d_wg):
        t.record_stream(main)
    out = [dz.view_as(z), *g_w, d_wg.to(wg.dtype), d_wo.to(wo.dtype), g_gi, g_bi, d_go.to(go.dtype), d_bo.to(bo.dtype)]
    # custom ops may not return aliasing outputs (the first, the residual's dz, is the one the caller reads as a view)
    return [g.clone() if i and g._base is not None else g for i, g in enumerate(out)]


def _grad_like(t: torch.Tensor) -> torch.Tensor:
    """An empty gradient for the parameter ``t``: contiguous, or with ``t``'s strides when ``t`` is stored [in, out] (strides (1, rows): the
    bidirectional module's four front matrices), so that autograd takes the gradient over instead of copying it into the parameter's layout."""
    if t.ndim == 2 and t.shape[0] > 1 and t.stride() == (1, t.shape[0]):
        return torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    return torch.empty(t.shape, dtype=t.dtype, device=t.device)


def _backward_fake(leaves, mask, ds, saved, dy, direction, eps_in, eps_out):
    """One gradient per leaf in the leaf's dtype: contiguous, except the four input-projection matrices, whose gradient takes the leaf's
    strides when it is stored [in, out] (``_grad_like``); a compiled caller checks the strides."""
    grads = [torch.empty_like(t, memory_format=torch.contiguous_format) for t in leaves]
    grads[1:5] = [_grad_like(t) for t in leaves[1:5]]
    return grads


@opaque(fake=_backward_fake, name="trimul_sm80_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor, ds: torch.Tensor, saved: list[torch.Tensor], dy: torch.Tensor,
             direction: int, eps_in: float, eps_out: float) -> list[torch.Tensor]:
    """Gradients of every leaf (z, W_l, W_lg, W_r, W_rg, W_g, W_o, ln_in weight / bias, ln_out weight / bias), in the leaves' dtypes."""
    return _run_backward(leaves, mask, ds, list(saved), dy, direction, eps_in, eps_out)


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        leaves = list(args[:11])
        mask, ds, direction, eps_in, eps_out = args[11:]
        out, *saved = forward_train(leaves, mask, ds, direction, eps_in, eps_out)
        ctx.save_for_backward(*leaves, mask, ds, *saved)
        ctx.cfg = (direction, eps_in, eps_out)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        grads = backward(list(vals[:11]), vals[11], vals[12], list(vals[13:]), dy.contiguous(), *ctx.cfg)
        return (*grads, None, None, None, None, None)


def trimul(z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo, mask, ds, direction, eps_in, eps_out):
    """z + drop_row(trimul(z)).  ``mask`` [L] uint8 or empty (``token_mask``), ``ds`` [L, 128] bf16 or empty.  Call ``available()`` first."""
    leaves = (z, wl, wlg, wr, wrg, wg, wo, gi, bi, go, bo)
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in leaves)):
        return forward_nograd(list(leaves), mask, ds, direction, float(eps_in), float(eps_out))
    return _Training.apply(*leaves, mask, ds, direction, float(eps_in), float(eps_out))
