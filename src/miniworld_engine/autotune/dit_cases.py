"""The DiT blocks the engine serves as whole modules; dimensions live in registry_module.csv.

``dit`` (``modules.dit.DiTBlock``: the token DiT), ``bias_only_dit`` (``modules.bias_only_dit.BiasOnlyDiTBlock``), ``local_dit``
(``modules.local_dit.LocalDiTBlock``: the AlphaFold 3 atom block, ``esmfold2=1`` the ESMFold2 one) and ``swa_dit`` (``modules.swa_dit.SWADiTBlock``:
the ESMFold2 atom block of MiniWorld). Their kernels used to reach a prebuilt bundle only through the hand-written coverage extras
(``build.bundle_coverage.extra_cases``); a registry row makes each one part of the contract every card answers for, built and verified from
the same expansion as every other row.

Registry dims that are not constructor arguments: ``per_sample_pair`` (a pair of its own per sample, the bias-only block's training layout:
``single`` [1, A, L, d] with ``pair`` [A, L, L, d_pair]), ``masked`` (a key mask with holes instead of all ones) -- both
``builder.INPUT_DIMS`` -- and ``esmfold2`` (the ESMFold2 preset of ``LocalDiTBlock``: cross attention, 192 keys, RMS/SiLU conditioning,
rotary q / k). A block hoisted out of its stack (one pair bias for every block) is a stack behaviour, not a block's, and has no row here.
"""

from __future__ import annotations

import torch

NAMES = ("dit", "bias_only_dit", "local_dit", "swa_dit")


def _mask(rows: int, length: int, dims: dict) -> torch.Tensor:
    if dims.get("masked"):
        mask = torch.rand(rows, length, device="cuda") > 0.15
        mask[:, 0] = True                                                  # never a row with no key at all
        return mask
    return torch.ones(rows, length, dtype=torch.bool, device="cuda")


def _token(b: int, length: int, dims: dict, dt: torch.dtype) -> tuple:
    """``single`` / ``cond`` carry the augmentation axis [A, 1, L, d] and the pair [1, L, L, d_pair] is shared by the samples; with
    ``per_sample_pair`` the samples are the batch axis instead: [1, A, L, d] reading a pair [A, L, L, d_pair] of their own."""
    lead, nb = ((1, b), b) if dims.get("per_sample_pair") else ((b, 1), 1)
    single = torch.randn(*lead, length, dims.get("d_single", 768), device="cuda", dtype=dt)
    cond = torch.randn(*lead, length, dims["d_cond"], device="cuda", dtype=dt)
    pair = torch.randn(nb, length, length, dims["d_pair"], device="cuda", dtype=dt)
    return single, cond, pair, _mask(nb, length, dims)


def _local(b: int, n: int, dims: dict, dt: torch.dtype) -> tuple:
    """The atom block: [A, 1, N, d] streams and the trunked pair [1, N / 32, 32, keys, d_pair] (never a dense N x N pair); the ESMFold2
    preset reads 192 keys and rotates q / k by (cos, sin) [1, N, 16]."""
    from miniworld_engine.modules.local_dit import ESMFOLD2, KEYS

    keys = ESMFOLD2["keys"] if dims.get("esmfold2") else KEYS
    single = torch.randn(b, 1, n, dims["d_single"], device="cuda", dtype=dt)
    cond = torch.randn(b, 1, n, dims["d_cond"], device="cuda", dtype=dt)
    pair = torch.randn(1, (n + 31) // 32, 32, keys, dims["d_pair"], device="cuda", dtype=dt)
    args = (single, cond, pair, _mask(1, n, dims))
    if dims.get("esmfold2"):
        angles = torch.randn(1, n, 16, device="cuda")
        args = (*args, (angles.cos(), angles.sin()))
    return args


def _swa(b: int, n: int, dims: dict, dt: torch.dtype) -> tuple:
    """``SWADiTBlock.forward(x [A, N, d], cond [A, N, d], attention_params)``, one sequence per sample. The rotary tables and the varlen
    packing are built the way the atom transformer builds them (``build_rope`` on one batch, ``build_attention_params`` over the A
    samples), as the bundle coverage of the SWA atom stack always did."""
    from miniworld_engine.modules.swa_atom_attention import build_attention_params
    from miniworld_engine.modules.swa_atom_transformer import SWAAtomTransformer

    rope = SWAAtomTransformer(SWAAtomTransformer.Config(n_block=1, d_atom=dims["d_atom"], d_cond=dims["d_cond"])).cuda()
    cos, sin = rope.build_rope(torch.randn(1, n, 3, device="cuda"), torch.arange(n, device="cuda")[None])
    params = build_attention_params(cos, sin, torch.ones(b, n, dtype=torch.bool, device="cuda"), b)
    x = torch.randn(b, n, dims["d_atom"], device="cuda", dtype=dt)
    cond = torch.randn(b, n, dims["d_cond"], device="cuda", dtype=dt)
    return x, cond, params


def cases():
    from miniworld_engine.autotune.builder import Case, _ctor, _shapes
    from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
    from miniworld_engine.modules.dit.module import DiTBlock
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.local_dit import ESMFOLD2, LocalDiTBlock
    from miniworld_engine.modules.swa_dit import SWADiTBlock

    def build(cls):
        return lambda dims, p, impl, dt: cls(**_ctor(dims), implementation=IT(impl)).cuda().to(dt)

    def build_local(dims, p, impl, dt):
        kw = {k: v for k, v in _ctor(dims).items() if k not in ("esmfold2", "parallel_kv", "slide_window")}
        if dims.get("parallel_kv"):     # AF3: the K/V AdaLN reads the raw atoms (FoldForge's `kv_norm="parallel"`); default chained
            kw["kv_norm"] = "parallel"
        if dims.get("slide_window"):    # AF3's slid 128-key window; default the clipped, OR-masked one ("pad")
            kw["key_window"] = "slide"
        if dims.get("esmfold2"):
            kw.update(ESMFOLD2)
        return LocalDiTBlock(**kw, implementation=IT(impl)).cuda().to(dt)

    return [
        Case("dit", build(DiTBlock), lambda b, l, dims, dt, s: _token(b, l, dims, dt), **_shapes("dit")),
        Case("bias_only_dit", build(BiasOnlyDiTBlock), lambda b, l, dims, dt, s: _token(b, l, dims, dt), **_shapes("bias_only_dit")),
        Case("local_dit", build_local, lambda b, l, dims, dt, s: _local(b, l, dims, dt), **_shapes("local_dit")),
        Case("swa_dit", build(SWADiTBlock), lambda b, l, dims, dt, s: _swa(b, l, dims, dt), **_shapes("swa_dit")),
    ]
