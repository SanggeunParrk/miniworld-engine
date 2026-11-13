"""H100 TriangleAttention inference for what the sm90 kernels (``h100_sm90`` / ``h100_sm90_generic``: heads of 16 / 32 channels, the
standard output) do not take: heads of 64 channels (Chai-1's trunk, RoseTTAFold3's template stack) and Chai-1's output variants.

    y = LN(pair) (the generic glue kernel; the ending node reads the pair transposed)  ->  q | k | v | g = y W^T and the pair bias
    (cuBLAS; the bias from the transposed pair or, ``transposed_bias=False``, from the pair as it is; masked keys -inf)  ->  the core
    (u = sigmoid(g [+ gate bias]) softmax(q k^T / sqrt(d) + bias) v; the 64-key tiles past each sample's last attended key skipped):
    64-channel heads at L a multiple of 128 on the wgmma core ``h100_sources/triattn_d64_sm90.cu`` (``ext_d64``: two pair rows per CTA
    share the bias tile, TMA ring, fp32 softmax), else the mma.sync core ``h100_sources/triattn_dh_sm90.cu``  ->  u Wo^T
    [+ out bias] (cuBLAS), back into the pair's layout:

    standard           the ending node's update transposed back
    untransposed       Chai-1's ending node: the update as the transposed pair has it (``untransposed_output``)
    dual               Chai-1's confidence head: kept = u Wo^T, other = u Wo_t^T; starting: kept + other^T, ending: kept^T + other
                       (``dual_output``)

``forward`` is inference only (no grad, no dropout). ``train_core`` / ``backward``: the same core in training, for 64-channel heads of the
width-generic training path (``h100_sm90_generic``).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_SRC = Path(__file__).parent / "h100_sources"
HEAD_DIMS = (32, 64)
_PACKS: dict = _capture.weight_cache("triangle_attention_h100_dh.packs")


@functools.lru_cache(maxsize=None)
def ext():
    ensure_cuda_home()
    return load_extension(
        name="triattn_h100_dh",
        sources=[str(_SRC / "triattn_dh_sm90.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), "--use_fast_math", "-DNDEBUG"],
        verbose=False,
    )


#: query tile of the inference core: 128 (4 warps x 32 queries) for 64-channel heads once the length is a multiple of 128 and at least
#: this, else 64 (4 x 16). A 16-query warp tile is bound by its ldmatrix traffic (every K | V fragment feeds one MMA); 32 queries per warp
#: halve it (H100, core alone: 0.98 / 0.93 / 0.87 / 0.94 / 0.83 / 0.86 of the 64 tile at S 384 / 512 / 768 / 1024 / 1536 / 2048; 1.04 at
#: S 256). 32-channel heads keep 64 (the 128 tile is 1.17-1.30x slower there: 4 x 16 queries already fit its registers twice over).
QUERY_TILE_128_MIN = 384


#: inference of 64-channel heads on the wgmma core (h100_sources/triattn_d64_sm90.cu: two pair rows per CTA share the bias tile, TMA, fp32
#: softmax) for L a multiple of 128 from this length on, else the mma.sync core (H100, core alone, padded mask: 0.94 / 0.84 / 0.78 / 0.78 /
#: 0.74 / 0.69 / 0.68 / 0.65 / 0.60 of the mma.sync time at L 128 / 256 / 384 / 512 / 640 / 768 / 1024 / 1536 / 2048)
D64_WGMMA_MIN = 128


@functools.lru_cache(maxsize=None)
def ext_d64():
    """The wgmma inference core of 64-channel heads."""
    from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90 import _flags

    ensure_cuda_home()
    return load_extension(name="triattn_h100_d64", sources=[str(_SRC / "triattn_d64_sm90.cu")], extra_cuda_cflags=_flags(), verbose=False)


def wgmma_d64(length: int, d: int) -> bool:
    """64-channel heads take the wgmma core at ``length`` (the padded S): a pure function of the shape."""
    return d == 64 and length % 128 == 0 and length >= D64_WGMMA_MIN


def query_tile(length: int, d: int) -> int:
    """The inference core's query tile at ``length`` (the padded S) and head width ``d``: a pure function of the shape."""
    return 128 if d == 64 and length % 128 == 0 and length >= QUERY_TILE_128_MIN else 64


def train_core(proj, bias, n_head, d, scale):
    """The training forward of the core (``h100_sm90_generic`` at 64-channel heads): proj [M, 4 H d] (q | k | v | g, the gate as it
    is), bias [B, H, S, S] bf16 natural units (masked keys -inf) -> (o [M, H d], lse fp32 [B, S, H, S] base 2, u = sigmoid(g) o)."""
    return ext().triattn_dh_forward_train(proj, bias, n_head, d, scale)


#: query-side backward: True (default) one bf16 dS plane per pair row, summed here in fp32 (the Triton backward's data flow; RoseTTAFold3 L 640 /
#: 768 training 10.4 / 17.0 ms, the fp32 slab read-modify-write 12.5 / 19.8 ms); False the fp32 slab partials of `slabs`
ROWS = True

#: dbias slabs of the query-side backward: enough (b, h, query tile, slab) CTAs for ~3 waves of 132 SMs, at most this many slabs
MAX_SLABS = 8


def slabs(batch: int, length: int, n_head: int) -> int:
    """Slabs of pair rows the query-side backward splits into (each sums its rows' dbias into its own slab): a pure function of shapes."""
    ctas = batch * n_head * (length // 64)
    return max(1, min(MAX_SLABS, length, -(-3 * 132 // ctas)))


def backward(proj, bias, do, lse, delta, dall, n_head, d, scale):
    """The core backward: dq / dk / dv into columns 0 / H d / 2 H d of ``dall``; -> dbias fp32 [B, H, S, S] (natural units), the slabs (``ROWS``:
    the per-row bf16 planes) summed in fp32 in order."""
    B, _, S, _ = bias.shape
    slabs_ = ext().triattn_dh_backward(
        proj, bias, do, lse, delta, dall, 0, n_head * d, 2 * n_head * d, n_head, d, scale, S if ROWS else slabs(B, S, n_head), ROWS
    )
    return slabs_.sum(0, dtype=torch.float32) if slabs_.shape[0] > 1 else slabs_[0].float()


def head_dim(module) -> int:
    return module.to_query.weight.shape[0] // module.n_head


def serves(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    if torch.is_grad_enabled() or (module.training and module.p_drop > 0.0):
        return False

    if getattr(module, "_backend", None) != "portable":
        return False  # the engine implementation only (implementation="pytorch" keeps the torch statements)
    if not module.use_self_attention or module.use_qk_norm:
        return False
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or pair.ndim != 4 or pair.shape[1] != pair.shape[2]:
        return False
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return False
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
        return False
    c = pair.shape[-1]
    return head_dim(module) in HEAD_DIMS and c % 8 == 0 and c <= 512 and module.to_query.weight.shape[1] == c


def _pack(module):
    bf = torch.bfloat16
    w = [module.to_query.weight, module.to_key.weight, module.to_value.weight, module.to_gate.weight]
    out = {
        "wcat": torch.cat([t.to(bf) for t in w]).contiguous(),
        "wb": module.to_bias.weight.to(bf).contiguous(),
        "wo": module.to_out.weight.to(bf).contiguous(),
        "gb": module.to_gate.bias.float().contiguous() if module.to_gate.bias is not None else None,
        "bo": module.to_out.bias.to(bf) if module.to_out.bias is not None else None,
    }
    if getattr(module, "dual_output", False):
        out["wo_t"] = module.to_out_t.weight.to(bf).contiguous()
    return out


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90 import key_tiles
    from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90_generic import _glue

    B, L0, _, C = pair.shape
    L = (L0 + 63) // 64 * 64
    if L != L0:
        pair_p = F.pad(pair, (0, 0, 0, L - L0, 0, L - L0))
        keep = torch.zeros(B, L, dtype=torch.bool, device=pair.device)
        keep[:, :L0] = True if mask is None else mask
        mask = keep
    else:
        pair_p = pair
    H, D = module.n_head, head_dim(module)
    params = list(module.parameters())
    pack = _capture.lookup(_PACKS, (id(module), *((p.data_ptr(), p._version) for p in params)), lambda: _pack(module), limit=64)
    trans = not module.starting
    x = pair_p.contiguous().view(-1, C)
    lnw, lnb = module.ln_pair.weight.float().contiguous(), module.ln_pair.bias.float().contiguous()
    y = _glue().ln_fwd(x, lnw, lnb, module.ln_pair.eps, L, trans)[0]  # the attention's row layout
    proj = torch.mm(y, pack["wcat"].t())
    # the pair bias [B, H, L, L]: from the rows the attention reads, or (an ending node on ``transposed_bias=False``) the pair as it is
    yb = _glue().ln_fwd(x, lnw, lnb, module.ln_pair.eps, L, False)[0] if trans and not module.transposed_bias else y
    bias = torch.mm(yb, pack["wb"].t()).view(B, L, L, H).permute(0, 3, 1, 2)
    if mask is not None:
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
    gb = pack["gb"] if pack["gb"] is not None else proj.new_empty(0, dtype=torch.float32)
    # the 64-key tiles past each sample's last attended key (masked in every row, the padding included) skipped: the same bits
    kt = key_tiles(mask, L, 64)
    if wgmma_d64(L, D):
        u = ext_d64().triattn_d64_forward(proj, bias.contiguous(), gb, H, D**-0.5, kt)
    else:
        u = ext().triattn_dh_forward(proj, bias.contiguous(), gb, H, D, D**-0.5, query_tile(L, D), kt)

    def project(w):
        t = torch.mm(u, w.t())
        if pack["bo"] is not None:
            t += pack["bo"]
        return t.view(B, L, L, C)

    kept = project(pack["wo"])
    if getattr(module, "dual_output", False):
        other = torch.mm(u, pack["wo_t"].t()).view(B, L, L, C)
        delta = kept.transpose(1, 2) + other if trans else kept + other.transpose(1, 2)
    elif trans and not getattr(module, "untransposed_output", False):
        delta = kept.transpose(1, 2)
    else:
        delta = kept
    out = pair_p + delta
    return out[:, :L0, :L0].contiguous() if L != L0 else out
