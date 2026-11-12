"""La-Proteina's pair embeddings: ``PairEmbedAdaLN`` (the denoiser's initial pair builder), ``PairEmbedGeom`` (the autoencoder encoder's backbone pair
embedding) and ``PairEmbedCA`` (the autoencoder decoder's pair embedding).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule` that owns its bias-free Linear (``weight``): the Linear acts on a concatenation of
binned one-hots, so a kernel never builds the one-hots (127 + 30 + 30 or 316 fp32 channels per pair): a one-hot times a matrix is a gather of table
rows. ``_forward_pytorch`` (the dense references below) is the source of truth. The kernels live in ``kernels/pair_embed_adaln`` (one CUDA extension) and
``kernels/pair_embed_geom`` (one CUDA extension for the encoder and the decoder pair); the portable Triton kernels are a second backend of every class.

The output is bf16 (what the attention-bias projection consumes) for fp32 / bf16 inputs, in every backend, so every backend is rounded at the same place
(fp64 inputs, the accuracy gold, stay fp64). The binned inputs (coordinates, residue indices) are data: no gradient flows into them.
"""

import math
from typing import ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniworld_engine.modules.primitives import BackendModule, key_mask, random_input, wide

D_PAIR = 256
N_ROWS_GEOM = 317  # 127 sequence separations | 4 x 21 distances | 5 x 21 angles | 1 chain product
N_ROWS_CA = 157  # 127 sequence separations | 30 CA distances


def _cuda_arch(device: torch.device) -> str | None:
    """Why the hand-written CUDA kernels cannot run on ``device`` (their builds are H100 sm_90a, A100 sm_80 and that source for the RTX 5090, sm_120a), or ``None``."""
    if device.type != "cuda":
        return "the kernels run on CUDA tensors only"
    cc = torch.cuda.get_device_capability(device)
    if cc not in ((9, 0), (8, 0), (12, 0)):
        return f"the CUDA kernels are built for H100 (sm_90a), A100 (sm_80) and RTX 5090 (sm_120a), got sm_{cc[0]}{cc[1]}"
    return None


def _first(*reasons: str | None) -> str | None:
    return next((r for r in reasons if r is not None), None)


def _out(y: torch.Tensor) -> torch.Tensor:
    """The op's output precision: bf16 for fp32 inputs, as the kernels return it (an fp64 gold run keeps fp64)."""
    return y.to(torch.bfloat16) if y.dtype == torch.float32 else y


def _one_hot(v: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.bucketize(v, limits), len(limits) + 1).to(v.dtype)


def _pair_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (mask[:, :, None] & mask[:, None, :]).to(dtype)


def _backbone(samples: int, length: int, *, device, generator: torch.Generator) -> torch.Tensor:
    """``[B, n, 4, 3]`` (N, CA, C, CB) in nm: a chain of CA atoms 0.38 nm apart (a random walk), the other three atoms 0.12 nm around each."""
    steps = torch.randn(samples, length, 3, device=device, generator=generator)
    ca = (steps / steps.norm(dim=-1, keepdim=True) * 0.38).cumsum(1)
    off = torch.randn(samples, length, 3, 3, device=device, generator=generator) * 0.12
    return torch.stack([ca + off[:, :, 0], ca, ca + off[:, :, 1], ca + off[:, :, 2]], 2)


def _table(weight: torch.Tensor, track: bool) -> torch.Tensor:
    """``W^T`` [rows, 256] in bf16, contiguous: differentiable in ``weight`` when ``track``, else one transpose-and-cast kernel."""
    if track:
        return weight.t().to(torch.bfloat16).contiguous()
    tbl = torch.empty((weight.shape[1], weight.shape[0]), dtype=torch.bfloat16, device=weight.device)
    tbl.copy_(weight.detach().t())
    return tbl


# --------------------------------------------------------------------------------------------------------------------- PairEmbedAdaLN
class PairEmbedAdaLN(BackendModule):
    """La-Proteina's initial pair representation (the ``PairReprBuilder`` of its denoiser)::

        pair[b, i, j] = pm_ij * ( LN(h_ij) * gamma[b] + beta[b] ),    h_ij = LN_affine( W [onehot_sep(i, j) | onehot_xt(i, j) | onehot_sc(i, j)] ) * pm_ij

    ``pm_ij`` = both residues are real; the one-hots bin the residue-index separation (``n_sep`` bins), the CA distances of the noisy coordinates ``xt``
    (``n_xt`` bins over ``xt_range`` nm) and of the self-conditioning coordinates ``sc`` (``n_sc`` bins; all zeros when there is none). ``gamma`` / ``beta``
    are per sample because the conditioning (the diffusion times) is constant over (i, j): the caller computes them (``gamma`` after the sigmoid).

    ``forward(xt, sc, residue_idx, mask, gamma, beta)``:

    * ``xt`` / ``sc`` ``[B, n, 3]`` -- CA coordinates in nm (``sc`` may be ``None``); data: no gradient;
    * ``residue_idx`` ``[B, n]`` integers; ``mask`` ``[B, n]`` bool;
    * ``gamma`` / ``beta`` ``[B, 256]`` -- the adaptive scale and shift.

    Returns the pair ``[B, n, n, 256]`` bf16 (fp64 for fp64 inputs), differentiable in ``weight`` ``[256, n_sep + n_xt + n_sc]``, ``ln_weight`` / ``ln_bias``
    ``[256]``, ``gamma`` and ``beta``. Parameters of other dtypes than fp32 are used as they are (the kernels compute in fp32).

    Parameters
    ----------
    backend : str
        See :class:`BackendModule`. ``"cuda"`` (H100 sm_90a; A100 sm_80: the same source's sm_80 build): the engine's hand-written CUDA kernels (``kernels/pair_embed_adaln/cuda``: row gather, both
        LayerNorms and the adaptive scale in one kernel; a bitwise deterministic backward), at most 220 table rows (``n_sep + n_xt + n_sc``).
        ``"triton"``: the portable Triton kernels (``kernels/pair_embed_adaln/triton``), any CUDA card. Both pack each bin into 8 bits (every ``n_*`` <= 255).
    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results
    backend_versions: ClassVar[dict[str, int]] = {"cuda:a100_sm80": 2, "cuda:5090_sm120": 1}  # 2026-11-10: the sm_80 build of the H100 kernels; 2026-11-12: its sm_120a build

    backends = ("pytorch", "cuda", "triton")

    def __init__(
        self,
        *,
        n_sep: int = 127,
        n_xt: int = 30,
        n_sc: int = 30,
        xt_range: tuple[float, float] = (0.1, 3.0),
        sc_range: tuple[float, float] = (0.1, 3.0),
        eps1: float = 1e-5,
        eps2: float = 1e-5,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.n_sep, self.n_xt, self.n_sc = n_sep, n_xt, n_sc
        self.xt_range, self.sc_range = cast("tuple[float, float]", tuple(xt_range)), cast("tuple[float, float]", tuple(sc_range))
        self.eps1, self.eps2 = eps1, eps2
        rows = n_sep + n_xt + n_sc
        self.weight = nn.Parameter(torch.randn(D_PAIR, rows) / math.sqrt(rows))
        self.ln_weight = nn.Parameter(torch.ones(D_PAIR))
        self.ln_bias = nn.Parameter(torch.zeros(D_PAIR))
        self.set_backend(backend)

    def _forward_pytorch(self, xt, sc, residue_idx, mask, gamma, beta):
        dt = wide(self.weight, xt, gamma)
        dev = xt.device
        pm = _pair_mask(mask, dt)[..., None]
        sep = residue_idx[:, :, None].to(dt) - residue_idx[:, None, :].to(dt)
        half = self.n_sep / 2.0 - 1
        feats = [_one_hot(sep, torch.linspace(-half, half, self.n_sep - 1, device=dev, dtype=dt))]

        def dist(x):
            x = x.to(dt)
            return torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)

        feats.append(_one_hot(dist(xt), torch.linspace(*self.xt_range, self.n_xt - 1, device=dev, dtype=dt)))
        feats.append(
            _one_hot(dist(sc), torch.linspace(*self.sc_range, self.n_sc - 1, device=dev, dtype=dt))
            if sc is not None
            else torch.zeros_like(feats[1][..., : self.n_sc])
        )
        h = F.linear(torch.cat(feats, -1) * pm, self.weight.to(dt))
        h = F.layer_norm(h, (D_PAIR,), self.ln_weight.to(dt), self.ln_bias.to(dt), self.eps1) * pm
        y = F.layer_norm(h, (D_PAIR,), None, None, self.eps2)
        return _out((y * gamma.to(dt)[:, None, None, :] + beta.to(dt)[:, None, None, :]) * pm)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``samples`` proteins of ``length`` residues: noisy / self-conditioning CA coordinates (Gaussian, 0.8 nm, fp32: coordinates are data), residue
        indices 1 .. ``length``, the last eighth padding, ``gamma`` = sigmoid of a Gaussian and ``beta`` = 0.2 N(0, 1) in ``dtype``."""
        xt = random_input((samples, length, 3), dtype=torch.float32, device=device, generator=generator, scale=0.8)
        sc = random_input((samples, length, 3), dtype=torch.float32, device=device, generator=generator, scale=0.8)
        idx = torch.arange(1, length + 1, device=device).expand(samples, length).contiguous()
        gamma = torch.sigmoid(random_input((samples, D_PAIR), dtype=torch.float32, device=device, generator=generator)).to(dtype)
        beta = random_input((samples, D_PAIR), dtype=dtype, device=device, generator=generator, scale=0.2)
        return {"xt": xt, "sc": sc, "residue_idx": idx, "mask": key_mask(samples, length, device=device), "gamma": gamma, "beta": beta}

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_kernels(self, xt, sc, residue_idx, mask, gamma, beta) -> str | None:
        if xt.device.type != "cuda":
            return "the kernels run on CUDA tensors only"
        if max(self.n_sep, self.n_xt, self.n_sc) > 255:
            return "the kernels pack each bin into 8 bits"
        if any(t.dtype not in (torch.float32, torch.bfloat16) for t in (self.weight, self.ln_weight, self.ln_bias, gamma, beta, xt)):
            return "weight, ln_weight, ln_bias, gamma, beta and xt must be fp32 or bf16 (the kernels compute in fp32)"
        if xt.dim() != 3 or xt.shape[-1] != 3 or residue_idx.shape != xt.shape[:2] or mask.shape != xt.shape[:2]:
            return "xt must be [B, n, 3] and residue_idx / mask [B, n]"
        if sc is not None and sc.shape != xt.shape:
            return "sc must be shaped like xt"
        if tuple(gamma.shape) != (xt.shape[0], D_PAIR) or gamma.shape != beta.shape:
            return f"gamma / beta must be [B, {D_PAIR}]"
        if xt.numel() == 0:
            return "empty input"
        return None

    def _supports_cuda(self, xt, sc, residue_idx, mask, gamma, beta) -> str | None:
        why = _first(
            self._supports_kernels(xt, sc, residue_idx, mask, gamma, beta), _cuda_arch(xt.device) if xt.device.type == "cuda" else None
        )
        if why is not None:
            return why
        if self.n_sep + self.n_xt + self.n_sc > 220:
            return "the CUDA kernel keeps 4 copies of the table in shared memory (at most 220 rows)"
        return None

    def _supports_triton(self, xt, sc, residue_idx, mask, gamma, beta) -> str | None:
        return self._supports_kernels(xt, sc, residue_idx, mask, gamma, beta)

    def _prepared(self, xt, sc, residue_idx, mask, gamma, beta) -> tuple:
        """The kernels' inputs: the table W^T fp32, the affine, gamma / beta, fp32 coordinates, int32 indices, uint8 mask, the bin limits and the sizes."""
        dev = xt.device
        xt32 = xt.float().contiguous()
        return (
            self.weight.t().contiguous(),
            self.ln_weight,
            self.ln_bias,
            gamma,
            beta,
            xt32,
            (sc if sc is not None else xt).float().contiguous(),
            residue_idx.to(torch.int32).contiguous(),
            mask.to(torch.uint8).contiguous(),
            torch.linspace(*self.xt_range, self.n_xt - 1, device=dev),
            torch.linspace(*self.sc_range, self.n_sc - 1, device=dev),
            int(sc is not None),
            self.n_sep,
            self.n_xt,
            self.eps1,
            self.eps2,
        )

    def _forward_cuda(self, xt, sc, residue_idx, mask, gamma, beta):
        from miniworld_engine.kernels.pair_embed_adaln.cuda.function import pair_embed_adaln_cuda

        return pair_embed_adaln_cuda(*self._prepared(xt, sc, residue_idx, mask, gamma, beta))

    def _forward_triton(self, xt, sc, residue_idx, mask, gamma, beta):
        from miniworld_engine.kernels.pair_embed_adaln.triton.function import pair_embed_adaln_triton

        return pair_embed_adaln_triton(*self._prepared(xt, sc, residue_idx, mask, gamma, beta))


# ---------------------------------------------------------------------------------------------------------------------- PairEmbedGeom
def _unit(v):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)


def _dihedral(a, b, c, d):
    b0, b1, b2 = b - a, c - b, d - c
    n1, n2 = _unit(torch.linalg.cross(b0, b1)), _unit(torch.linalg.cross(b1, b2))
    x = torch.linalg.cross(n1, n2, dim=-1)
    return torch.atan2(torch.sign(torch.linalg.vecdot(x, b1)) * torch.linalg.norm(x, dim=-1), torch.linalg.vecdot(n1, n2, dim=-1))


def _bond(a, b, c):
    b0, b1 = _unit(b - a), _unit(c - a)
    return torch.atan2(torch.linalg.norm(torch.linalg.cross(b0, b1, dim=-1), dim=-1), torch.linalg.vecdot(b0, b1, dim=-1))


def _geom_features(bb, ca_mask, has_cb, mask, residue_idx, chains) -> torch.Tensor:
    """The 317 binned channels per pair, ``[B, n, n, 317]`` in ``bb``'s dtype, masked by the batch mask: sequence separation 127 | four distances
    4 x 21 | five inter-residue angles 5 x 21 | chain product 1."""
    dev, dt = bb.device, bb.dtype
    n_, ca, c_, cb = (bb[:, :, k] for k in range(4))
    pm = _pair_mask(ca_mask, dt)
    sep = (residue_idx[:, :, None] - residue_idx[:, None, :]).to(dt)
    feats = [_one_hot(sep, torch.linspace(-62.5, 62.5, 126, device=dev, dtype=dt))]
    dl = torch.linspace(0.1, 2, 20, device=dev, dtype=dt)
    dists = [torch.linalg.norm(ca[:, :, None] - x[:, None, :], dim=-1) for x in (n_, ca, c_, cb)]
    dists[3] = dists[3] * has_cb[:, None, :].to(dt)
    feats += [_one_hot(d * pm, dl) for d in dists]

    def p1(v):
        return v[:, :, None]

    def p2(v):
        return v[:, None, :]

    ang = torch.stack(
        [
            _dihedral(p1(n_), p1(ca), p1(cb), p2(cb)),
            _dihedral(p2(n_), p2(ca), p2(cb), p1(cb)),
            _bond(p1(ca), p1(cb), p2(cb)),
            _bond(p2(ca), p2(cb), p1(cb)),
            _dihedral(p1(ca), p1(cb), p2(cb), p2(ca)),
        ],
        -1,
    )
    feats_ang = _one_hot(ang, torch.linspace(-math.pi, math.pi, 20, device=dev, dtype=dt)).flatten(-2)
    feats_dist = torch.cat(feats[1:], -1) * pm[..., None]
    feats_ang = feats_ang * (pm * has_cb[:, :, None].to(dt))[..., None]
    chain = (chains[:, :, None] * chains[:, None, :]).to(dt)[..., None]
    return torch.cat([feats[0], feats_dist, feats_ang, chain], -1) * _pair_mask(mask, dt)[..., None]


class PairEmbedGeom(BackendModule):
    """La-Proteina's autoencoder-encoder backbone pair embedding::

        pair[b, i, j] = M_ij * W [onehot_sep | onehot(|CA_i - N_j|, |CA_i - CA_j|, |CA_i - C_j|, |CA_i - CB_j|) | onehot(5 inter-residue angles) | chain_i chain_j]

    (binned one-hots, 317 channels), optionally followed by the attention-bias LayerNorm over the 256 channels (no affine, ``ln_eps``): what the engine's
    attention-bias projection reads. The weight is rounded to bf16 first (what a bf16-autocast Linear multiplies by).

    ``forward(bb, ca_mask, has_cb, mask, residue_idx, chains)``:

    * ``bb`` ``[B, n, 4, 3]`` -- the N, CA, C, CB coordinates in nm (data: no gradient);
    * ``ca_mask`` / ``has_cb`` / ``mask`` ``[B, n]`` bool -- residues with a CA / a CB / real residues;
    * ``residue_idx`` / ``chains`` ``[B, n]`` integers.

    Returns the pair ``[B, n, n, 256]`` bf16 (fp64 for fp64 inputs). Without a gradient (the frozen encoder of the flow stage) the kernels write no backward
    state; with one the pair is differentiable in ``weight`` ``[256, 317]`` (the gradient of a bf16-autocast Linear: accumulated in fp32 in a fixed order and
    rounded to bf16 once; the LayerNorm is recomputed in the backward, the pair is never stored).

    Parameters
    ----------
    ln_eps : float | None
        ``None``: the pair itself; else its LayerNorm with this eps.
    backend : str
        See :class:`BackendModule`. ``"cuda"`` (H100 sm_90a; A100 sm_80: the same source's sm_80 build): the engine's hand-written CUDA kernels (``kernels/pair_embed_geom/cuda``: a gather of up to ten
        table rows, the LayerNorm folded in, a tensor-core table gradient). ``"triton"``: the portable Triton kernels, any CUDA card. Both read ``bb`` in
        fp32 or bf16 (computed in fp32).
    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results
    backend_versions: ClassVar[dict[str, int]] = {"cuda:a100_sm80": 2, "cuda:5090_sm120": 1}  # 2026-11-10: the sm_80 build of the H100 kernels; 2026-11-12: its sm_120a build

    backends = ("pytorch", "cuda", "triton")
    rows = N_ROWS_GEOM

    def __init__(self, *, ln_eps: float | None = None, backend: str = "pytorch") -> None:
        super().__init__()
        self.ln_eps = ln_eps
        self.weight = nn.Parameter(torch.randn(D_PAIR, self.rows) / math.sqrt(self.rows))
        self.set_backend(backend)

    def _forward_pytorch(self, bb, ca_mask, has_cb, mask, residue_idx, chains):
        dt = wide(self.weight, bb)
        with torch.autocast(
            "cuda", enabled=False
        ):  # the geometry (cross products, angles) is computed in fp32 whatever the caller's autocast
            f = _geom_features(bb.to(dt), ca_mask, has_cb, mask, residue_idx, chains)
            pair = F.linear(f, self.weight.to(torch.bfloat16).to(dt)) * _pair_mask(mask, dt)[..., None]
        return _layer_norm(_out(pair), self.ln_eps)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``samples`` backbones of ``length`` residues (a CA random walk, 0.38 nm steps; fp32 coordinates), the last eighth padding, 3 % of the real
        residues without a CA and 10 % of those without a CB (their CB coordinates zero), residue indices 1 .. ``length``, chain 0 (padding -1)."""
        bb = _backbone(samples, length, device=device, generator=generator)
        mask = key_mask(samples, length, device=device)
        ca_mask = mask & (torch.rand(samples, length, device=device, generator=generator) > 0.03)
        has_cb = ca_mask & (torch.rand(samples, length, device=device, generator=generator) > 0.1)
        bb[..., 3, :] *= has_cb[..., None]
        idx = torch.arange(1, length + 1, device=device).expand(samples, length).contiguous()
        chains = torch.where(mask, 0, -1).to(torch.int64)
        return {"bb": bb, "ca_mask": ca_mask, "has_cb": has_cb, "mask": mask, "residue_idx": idx, "chains": chains}

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_kernels(self, bb, ca_mask, has_cb, mask, residue_idx, chains) -> str | None:
        if bb.device.type != "cuda":
            return "the kernels run on CUDA tensors only"
        if self.weight.dtype not in (torch.float32, torch.bfloat16) or bb.dtype not in (torch.float32, torch.bfloat16):
            return "weight and bb must be fp32 or bf16 (the kernels compute in fp32)"
        if bb.dim() != 4 or tuple(bb.shape[2:]) != (4, 3):
            return "bb must be [B, n, 4, 3] (N, CA, C, CB)"
        if any(t.shape != bb.shape[:2] for t in (ca_mask, has_cb, mask, residue_idx, chains)):
            return "ca_mask / has_cb / mask / residue_idx / chains must be [B, n]"
        if bb.numel() == 0:
            return "empty input"
        return None

    def _supports_cuda(self, bb, ca_mask, has_cb, mask, residue_idx, chains) -> str | None:
        return _first(
            self._supports_kernels(bb, ca_mask, has_cb, mask, residue_idx, chains),
            _cuda_arch(bb.device) if bb.device.type == "cuda" else None,
        )

    def _supports_triton(self, bb, ca_mask, has_cb, mask, residue_idx, chains) -> str | None:
        return self._supports_kernels(bb, ca_mask, has_cb, mask, residue_idx, chains)

    def _prepared(self, bb, ca_mask, has_cb, mask, residue_idx, chains) -> tuple:
        """The kernels' inputs: the bf16 table W^T, fp32 coordinates, uint8 masks, int32 indices, the bin limits, the LayerNorm eps (-1: none), train."""
        dev = bb.device
        train = torch.is_grad_enabled() and self.weight.requires_grad
        return (
            _table(self.weight, train),
            bb.float().contiguous(),
            ca_mask.to(torch.uint8).contiguous(),
            has_cb.to(torch.uint8).contiguous(),
            mask.to(torch.uint8).contiguous(),
            residue_idx.to(torch.int32).contiguous(),
            chains.to(torch.int32).contiguous(),
            torch.linspace(0.1, 2, 20, device=dev),
            torch.linspace(-torch.pi, torch.pi, 20, device=dev),
            -1.0 if self.ln_eps is None else float(self.ln_eps),
            train,
        )

    def _forward_cuda(self, bb, ca_mask, has_cb, mask, residue_idx, chains):
        from miniworld_engine.kernels.pair_embed_geom.cuda.function import pair_embed_geom_cuda

        return pair_embed_geom_cuda(*self._prepared(bb, ca_mask, has_cb, mask, residue_idx, chains))

    def _forward_triton(self, bb, ca_mask, has_cb, mask, residue_idx, chains):
        from miniworld_engine.kernels.pair_embed_geom.triton.function import pair_embed_geom_triton

        return pair_embed_geom_triton(*self._prepared(bb, ca_mask, has_cb, mask, residue_idx, chains))


def _layer_norm(pair: torch.Tensor, eps: float | None) -> torch.Tensor:
    return pair if eps is None else F.layer_norm(pair, (D_PAIR,), None, None, eps)


# ------------------------------------------------------------------------------------------------------------------------- PairEmbedCA
class PairEmbedCA(BackendModule):
    """La-Proteina's autoencoder-decoder pair embedding::

        pair[b, i, j] = M_ij * W [onehot_sep(i, j) | onehot_dist30(|CA_i - CA_j|)]     (127 + 30 binned channels, distances over 0.1 .. 3.0 nm)

    optionally followed by the attention-bias LayerNorm over the 256 channels (no affine, ``ln_eps``). The weight is rounded to bf16 first.

    ``forward(ca, mask, residue_idx=None)``: ``ca`` ``[B, n, 3]`` -- CA coordinates in nm, padded residues zeroed by the caller as the decoder does (data: no
    gradient); ``mask`` ``[B, n]`` bool; ``residue_idx`` ``[B, n]`` integers (default 1 .. n). Returns the pair ``[B, n, n, 256]`` bf16 (fp64 for fp64
    inputs), differentiable in ``weight`` ``[256, 157]`` (bf16-accumulate rules of :class:`PairEmbedGeom`).

    Parameters
    ----------
    ln_eps : float | None
        ``None``: the pair itself; else its LayerNorm with this eps.
    backend : str
        See :class:`BackendModule`. ``"cuda"`` (H100 sm_90a; A100 sm_80: the same source's sm_80 build): the decoder-pair entry points of the same CUDA extension as :class:`PairEmbedGeom`.
        ``"triton"``: the portable Triton kernels, any CUDA card.
    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results
    backend_versions: ClassVar[dict[str, int]] = {"cuda:a100_sm80": 2, "cuda:5090_sm120": 1}  # 2026-11-10: the sm_80 build of the H100 kernels; 2026-11-12: its sm_120a build

    backends = ("pytorch", "cuda", "triton")
    rows = N_ROWS_CA

    def __init__(self, *, ln_eps: float | None = None, backend: str = "pytorch") -> None:
        super().__init__()
        self.ln_eps = ln_eps
        self.weight = nn.Parameter(torch.randn(D_PAIR, self.rows) / math.sqrt(self.rows))
        self.set_backend(backend)

    def _forward_pytorch(self, ca, mask, residue_idx=None):
        dt = wide(self.weight, ca)
        b, n = mask.shape
        dev = ca.device
        idx = residue_idx if residue_idx is not None else torch.arange(1, n + 1, device=dev).expand(b, n)
        sep = (idx[:, :, None] - idx[:, None, :]).to(dt)
        ca = ca.to(dt)
        d = torch.norm(ca[:, :, None, :] - ca[:, None, :, :], dim=-1)
        f = (
            torch.cat(
                [
                    _one_hot(sep, torch.linspace(-62.5, 62.5, 126, device=dev, dtype=dt)),
                    _one_hot(d, torch.linspace(0.1, 3.0, 29, device=dev, dtype=dt)),
                ],
                -1,
            )
            * _pair_mask(mask, dt)[..., None]
        )
        pair = F.linear(f, self.weight.to(torch.bfloat16).to(dt)) * _pair_mask(mask, dt)[..., None]
        return _layer_norm(_out(pair), self.ln_eps)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``samples`` backbones of ``length`` residues: their CA atoms (a random walk, 0.38 nm steps, fp32; padded residues zeroed), the last eighth
        padding, residue indices 1 .. ``length``."""
        mask = key_mask(samples, length, device=device)
        ca = _backbone(samples, length, device=device, generator=generator)[:, :, 1] * mask[..., None]
        idx = torch.arange(1, length + 1, device=device).expand(samples, length).contiguous()
        return {"ca": ca, "mask": mask, "residue_idx": idx}

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_kernels(self, ca, mask, residue_idx=None) -> str | None:
        if ca.device.type != "cuda":
            return "the kernels run on CUDA tensors only"
        if self.weight.dtype not in (torch.float32, torch.bfloat16) or ca.dtype not in (torch.float32, torch.bfloat16):
            return "weight and ca must be fp32 or bf16 (the kernels compute in fp32)"
        if (
            ca.dim() != 3
            or ca.shape[-1] != 3
            or mask.shape != ca.shape[:2]
            or (residue_idx is not None and residue_idx.shape != mask.shape)
        ):
            return "ca must be [B, n, 3] and mask / residue_idx [B, n]"
        if ca.numel() == 0:
            return "empty input"
        return None

    def _supports_cuda(self, ca, mask, residue_idx=None) -> str | None:
        return _first(self._supports_kernels(ca, mask, residue_idx), _cuda_arch(ca.device) if ca.device.type == "cuda" else None)

    def _supports_triton(self, ca, mask, residue_idx=None) -> str | None:
        return self._supports_kernels(ca, mask, residue_idx)

    def _prepared(self, ca, mask, residue_idx) -> tuple:
        """The kernels' inputs: the bf16 table W^T, fp32 CA coordinates, uint8 mask, int32 indices, the bin limits, the LayerNorm eps (-1: none), train."""
        dev = ca.device
        b, n = mask.shape
        train = torch.is_grad_enabled() and self.weight.requires_grad
        idx = (
            residue_idx.to(torch.int32).contiguous()
            if residue_idx is not None
            else torch.arange(1, n + 1, device=dev, dtype=torch.int32).expand(b, n).contiguous()
        )
        m8 = mask.contiguous().view(torch.uint8) if mask.dtype is torch.bool else mask.to(torch.uint8).contiguous()
        return (
            _table(self.weight, train),
            ca.float().contiguous(),
            m8,
            idx,
            torch.linspace(0.1, 3.0, 29, device=dev),
            -1.0 if self.ln_eps is None else float(self.ln_eps),
            train,
        )

    def _forward_cuda(self, ca, mask, residue_idx=None):
        from miniworld_engine.kernels.pair_embed_geom.cuda.function import pair_embed_ca_cuda

        return pair_embed_ca_cuda(*self._prepared(ca, mask, residue_idx))

    def _forward_triton(self, ca, mask, residue_idx=None):
        from miniworld_engine.kernels.pair_embed_geom.triton.function import pair_embed_ca_triton

        return pair_embed_ca_triton(*self._prepared(ca, mask, residue_idx))
