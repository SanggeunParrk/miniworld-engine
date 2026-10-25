"""The ops the performance ladder reaches that have no module of their own: the La-Proteina pair builders, the token-pair initialisation,
smooth lDDT, the sampler glue and the standalone norms (`bench.py level=module target=ladder_ops +ladder_op=<op>`).

These ops are functions (weights-as-arguments), not `nn.Module`s, so `bench.py` has no module bench for them. This file is the shared
harness code for them -- it builds the tensors, names the three rungs of the ladder and hands the timed callable to the same
`measured_result` every other target uses (CUDA graph replay; the PyTorch rung through `torch.compile` when `compile=true`). It holds
no timing loop of its own.

The three rungs (`implementation=`):

* ``pytorch``   -- the op's dense PyTorch reference (`kernels/<op>/reference.py`), compiled when ``compile=true``.
* ``portable``  -- the engine's portable (Triton) kernels: the op called under ``settings.force_portable_kernels`` (set by
  `bench.implementation_environment`), or, for the standalone norms, the Triton entry point itself.
* ``miniworld`` -- the op as the engine routes it on this card (the hand CUDA kernel on an H100), or the CUDA entry point itself for the norms.

``ladder_op``: geom, ca, adaln, tpi, smooth_lddt, sampler_augment, sampler_align, sampler_pool, layernorm, rmsnorm.
``seq_len`` = tokens (atoms for smooth_lddt / sampler_*), ``n_augment`` = batch / samples.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: ops with no gradient path (the sampler glue runs between denoiser calls, under no_grad)
FORWARD_ONLY = {"sampler_augment", "sampler_align", "sampler_pool"}
#: ops that exist only as a CUDA kernel + PyTorch reference (no portable kernel, by design)
NO_PORTABLE: set[str] = set()   # since the portable sampler kernels (kernels/sampler_ops/triton) every op of this file has a portable rung

OPS = (
    "geom", "ca", "adaln", "tpi", "smooth_lddt",
    "sampler_augment", "sampler_align", "sampler_pool", "layernorm", "rmsnorm",
)


def _backbone(B, n, g, dev):
    steps = torch.randn(B, n, 3, device=dev, generator=g)
    ca = (steps / steps.norm(dim=-1, keepdim=True) * 0.38).cumsum(1)
    off = torch.randn(B, n, 3, 3, device=dev, generator=g) * 0.12
    return torch.stack([ca + off[:, :, 0], ca, ca + off[:, :, 1], ca + off[:, :, 2]], 2)


def _ragged_mask(B, n, g, dev):
    lengths = torch.randint(int(n * 0.6), n + 1, (B,), device=dev, generator=g)
    lengths[0] = n
    return torch.arange(n, device=dev)[None] < lengths[:, None]


def _geom(op, impl, B, n, train, dev):
    from miniworld_engine.kernels.pair_embed_geom import (
        pair_embed_ca, pair_embed_ca_reference, pair_embed_geom, pair_embed_geom_reference, pair_embed_geom_train)

    g = torch.Generator(device=dev).manual_seed(0)
    bb = _backbone(B, n, g, dev)
    mask = _ragged_mask(B, n, g, dev)
    ca = mask & (torch.rand(B, n, device=dev, generator=g) > 0.03)
    cb = ca & (torch.rand(B, n, device=dev, generator=g) > 0.1)
    bb[..., 3, :] *= cb[..., None]
    idx = torch.arange(1, n + 1, device=dev).expand(B, n).contiguous()
    chains = torch.where(mask, torch.zeros(B, n, device=dev, dtype=torch.long), torch.full((B, n), -1, device=dev))
    ln = 1e-5
    if op == "geom":
        w = (torch.randn(256, 317, device=dev, generator=g) * 0.3).requires_grad_(train)
        args = (w, bb, ca, cb, mask, idx, chains)
        if impl == "pytorch":
            def fwd(w, bb, ca, cb, mask, idx, chains):
                out = pair_embed_geom_reference(w, bb, ca, cb, mask, idx, chains).bfloat16()
                return F.layer_norm(out, (256,), None, None, ln) if train else out
        elif train:
            def fwd(w, bb, ca, cb, mask, idx, chains):
                return pair_embed_geom_train(w, bb, ca, cb, mask, idx, chains, ln_eps=ln)
        else:
            fwd = pair_embed_geom
        path = "kernels.pair_embed_geom" + (" (+ln_eps, train)" if train else "")
    else:  # ca
        w = (torch.randn(256, 157, device=dev, generator=g) * 0.3).requires_grad_(train)
        cax = bb[:, :, 1] * mask[..., None]
        args = (w, cax, mask)
        if impl == "pytorch":
            def fwd(w, cax, mask):
                out = pair_embed_ca_reference(w, cax, mask).bfloat16()
                return F.layer_norm(out, (256,), None, None, ln) if train else out
        elif train:
            def fwd(w, cax, mask):
                return pair_embed_ca(w, cax, mask, ln_eps=ln)
        else:
            fwd = pair_embed_ca
        path = "kernels.pair_embed_geom.pair_embed_ca" + (" (+ln_eps, train)" if train else "")
    dout = torch.randn(B, n, n, 256, device=dev).bfloat16()
    return fwd, args, (w,), dout, path, "bfloat16", "float32"


def _adaln(impl, B, n, train, dev):
    from miniworld_engine.kernels.pair_embed_adaln import pair_embed_adaln, pair_embed_adaln_reference

    g = torch.Generator(device=dev).manual_seed(0)

    def r(*s):
        return torch.randn(*s, device=dev, generator=g)

    w = (r(256, 187) * 0.3).requires_grad_(train)
    lnw, lnb = (1 + 0.1 * r(256)).requires_grad_(train), (0.1 * r(256)).requires_grad_(train)
    gamma, beta = torch.sigmoid(r(B, 256)).requires_grad_(train), (0.2 * r(B, 256)).requires_grad_(train)
    xt, sc = r(B, n, 3) * 0.8, r(B, n, 3) * 0.8
    idx = torch.arange(1, n + 1, device=dev).expand(B, n).contiguous()
    mask = _ragged_mask(B, n, g, dev)
    if impl == "pytorch":
        def fwd(*a):
            return pair_embed_adaln_reference(*a).bfloat16()
    else:
        fwd = pair_embed_adaln
    dout = r(B, n, n, 256).bfloat16()
    return fwd, (w, lnw, lnb, gamma, beta, xt, sc, idx, mask), (w, lnw, lnb, gamma, beta), dout, \
        "kernels.pair_embed_adaln", "bfloat16", "float32"


def _tpi(impl, B, n, train, dev):
    from miniworld_engine.kernels.token_pair_init import token_pair_init, token_pair_init_reference

    g = torch.Generator().manual_seed(0)

    def ids(hi, dt=torch.int64):
        return torch.randint(0, hi, (B, n), generator=g).to(dev, dt)

    asym, resi, tok, ent, sym = ids(3), ids(90), ids(60), ids(2), ids(6, torch.int32)
    bond = (torch.rand(B, n, n, generator=g) < 0.05).to(dev)
    left = torch.randn(B, n, 128, generator=g).to(dev).requires_grad_(train)
    right = torch.randn(B, n, 128, generator=g).to(dev).requires_grad_(train)
    w_rel = (torch.randn(128, 139, generator=g) * 0.3).to(dev).requires_grad_(train)
    w_bond = (torch.randn(128, 2, generator=g) * 0.3).to(dev).requires_grad_(train)
    dz = torch.randn(B, n, n, 128, device=dev)
    fwd = token_pair_init_reference if impl == "pytorch" else token_pair_init
    return fwd, (left, right, w_rel, w_bond, asym, resi, tok, ent, sym, bond), (left, right, w_rel, w_bond), dz, \
        "kernels.token_pair_init", "float32", "float32"


def _smooth_lddt(impl, A, n, train, dev):
    from miniworld_engine.kernels.smooth_lddt import smooth_lddt
    from miniworld_engine.kernels.smooth_lddt.reference import smooth_lddt_reference

    g = torch.Generator(device=dev).manual_seed(0)
    B = 1
    gt = _backbone(1, n, g, dev)[:, :, 1] * 1.0
    gt = gt - gt.mean(1, keepdim=True)
    x = (gt[None].expand(A, B, n, 3) + 0.5 * torch.randn(A, B, n, 3, device=dev, generator=g)).contiguous().requires_grad_(train)
    mask = torch.rand(B, n, device=dev, generator=g) > 0.03
    is_nuc = torch.rand(B, n, device=dev, generator=g) > 0.9
    if impl == "pytorch":
        def fwd(x, gt, is_nuc, mask):
            return smooth_lddt_reference(x.float(), gt.float(), is_nuc, mask)
    else:
        fwd = smooth_lddt
    return fwd, (x, gt, is_nuc, mask), (x,), torch.ones((), device=dev), "kernels.smooth_lddt", "float32", "float32"


def _sampler(op, impl, B, n, dev):
    from miniworld_engine.kernels.sampler_ops import interface as I, reference as R

    mod = R if impl == "pytorch" else I
    g = torch.Generator(device=dev).manual_seed(0)
    real = int(n * 0.9)
    coords = 10 * torch.randn(B, n, 3, device=dev, generator=g) + 3
    mask = (torch.arange(n, device=dev) < real).float()
    s, sn = torch.tensor(2.0, device=dev), torch.tensor(1.5, device=dev)
    cfg = dict(window=0, gamma=0.8, gamma_min=1.0, sigma_min=0.0, sigma_max=float("inf"), noise_scale=1.003,
               variance_floor=0.0, clamp=False)
    if op == "sampler_augment":
        rv = torch.randn(B, 2, 3, device=dev, generator=g)
        tr = torch.randn(B, 3, device=dev, generator=g)
        noise = torch.randn_like(coords)

        def fwd(coords, mask, rv, tr, noise, s, sn):
            return mod.augment_noise(coords, mask, rv, tr, noise, s, sn, **cfg)
        return fwd, (coords, mask, rv, tr, noise, s, sn), "kernels.sampler_ops.augment_noise"
    if op == "sampler_align":
        noisy = coords + 0.5 * torch.randn_like(coords)
        denoised = coords + 0.1 * torch.randn_like(coords)
        sc = torch.tensor([4.0, 0.3, 0.0, 0.0], device=dev)

        def fwd(noisy, denoised, mask, sc, sn):
            return mod.align_euler(noisy, denoised, mask, sc, sn, 1.5)
        return fwd, (noisy, denoised, mask, sc, sn), "kernels.sampler_ops.align_euler"
    tokens, atoms, ch = max(n // 24, 1), 24, 768
    rows = tokens * atoms + 64
    x = torch.randn(B, rows, ch, device=dev, dtype=torch.bfloat16, generator=g)
    idx = torch.randint(0, rows, (tokens, atoms), device=dev, generator=g)
    atom_mask = torch.rand(tokens, atoms, device=dev, generator=g) > 0.3
    gather_mask = atom_mask | (torch.rand(tokens, atoms, device=dev, generator=g) > 0.9)
    am = atom_mask if impl == "pytorch" else atom_mask.float()
    gm = gather_mask if impl == "pytorch" else gather_mask.float()

    def fwd(x, idx, gm, am):
        return mod.pool_atoms(x, idx, gm, am)
    return fwd, (x, idx, gm, am), "kernels.sampler_ops.pool_atoms"


def _norm(op, impl, B, n, train, conf, dev, dtype):
    g = torch.Generator(device=dev).manual_seed(0)
    if op == "layernorm":
        from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
        from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

        D = conf.d_pair
        x = torch.randn(1, n, n, D, device=dev, dtype=dtype, generator=g).requires_grad_(train)
        w = torch.randn(D, device=dev, dtype=dtype, generator=g).requires_grad_(train)
        b = torch.randn(D, device=dev, dtype=dtype, generator=g).requires_grad_(train)
        if impl == "pytorch":
            def fwd(x, w, b):
                return F.layer_norm(x, (D,), w, b, 1e-5)
            path = "torch.nn.functional.layer_norm"
        elif impl == "portable":
            def fwd(x, w, b):
                return triton_layernorm(x, w, b, 1e-5)
            path = "kernels.layernorm.triton.main.triton_layernorm"
        else:
            def fwd(x, w, b):
                return layernorm_kernel(x, w, b, 1e-5)
            path = "kernels.layernorm.interface.layernorm_kernel"
        return fwd, (x, w, b), (x, w, b), torch.randn_like(x), path
    from miniworld_engine.kernels.rmsnorm.cuda import rmsnorm_cuda
    from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm

    D = conf.d_single_token
    x = torch.randn(B, n, D, device=dev, dtype=dtype, generator=g).requires_grad_(train)
    w = torch.randn(D, device=dev, dtype=dtype, generator=g).requires_grad_(train)
    if impl == "pytorch":
        def fwd(x, w):
            return F.rms_norm(x, (D,), w, 1e-5)
        path = "torch.nn.functional.rms_norm"
    elif impl == "portable":
        def fwd(x, w):
            return triton_rmsnorm(x, w, 1e-5)
        path = "kernels.rmsnorm.triton.main.triton_rmsnorm"
    else:
        def fwd(x, w):
            return rmsnorm_cuda(x, w, 1e-5)
        path = "kernels.rmsnorm.cuda.rmsnorm_cuda"
    return fwd, (x, w), (x, w), torch.randn_like(x), path


def run(bench, conf, seq_len, implementation):
    """The body of ``bench.bench_module_ladder_ops``; ``bench`` is the ``bench.py`` module (its result type, timer and error classes)."""
    from benchmarks.runners.measurement import compile_for_benchmark

    op = (getattr(conf, "ladder_op", "") or "").strip()
    if op not in OPS:
        raise bench.UnsupportedBenchmark(f"ladder_op must be one of {OPS}, got {op!r} (pass +ladder_op=...)")
    key = implementation.strip().lower()
    if key not in {"pytorch", "portable", bench.MINIWORLD_IMPL}:
        raise bench.UnsupportedBenchmark(f"ladder_ops implements pytorch, portable and miniworld, not {implementation!r}")
    if key == "portable" and op in NO_PORTABLE:
        raise bench.UnsupportedBenchmark(f"{op} has no portable kernel (CUDA kernel + PyTorch reference only, by design)")
    is_train = not bench.is_inference_mode(conf.mode)
    if is_train and op in FORWARD_ONLY:
        raise bench.UnsupportedBenchmark(f"{op} runs between denoiser calls under no_grad: no training mode")
    dev = bench.DEVICE
    impl = "pytorch" if key == "pytorch" else key
    B = conf.n_augment
    n = seq_len
    dtype = torch.float32 if conf.precision == bench.FP32_PRECISION else torch.bfloat16
    wrt = dout = None
    in_dt, par_dt = "float32", "float32"
    if op in ("geom", "ca"):
        fwd, args, wrt, dout, path, in_dt, par_dt = _geom(op, impl, B, n, is_train, dev)
    elif op == "adaln":
        fwd, args, wrt, dout, path, in_dt, par_dt = _adaln(impl, B, n, is_train, dev)
    elif op == "tpi":
        fwd, args, wrt, dout, path, in_dt, par_dt = _tpi(impl, B, n, is_train, dev)
    elif op == "smooth_lddt":
        fwd, args, wrt, dout, path, in_dt, par_dt = _smooth_lddt(impl, B, n, is_train, dev)
    elif op in FORWARD_ONLY:
        fwd, args, path = _sampler(op, impl, B, n, dev)
        in_dt = "bfloat16" if op == "sampler_pool" else "float32"
    else:
        fwd, args, wrt, dout, path = _norm(op, impl, B, n, is_train, conf, dev, dtype)
        in_dt = par_dt = str(dtype).replace("torch.", "")

    row_conf = conf
    if key == "pytorch" and conf.compile:
        fwd = compile_for_benchmark(fwd)
    elif conf.compile:
        # the hand / Triton kernels are opaque launches, not Inductor graphs: they are timed as CUDA-graph replays (or eager), never wrapped
        # in torch.compile; the PyTorch rung is the one that is compiled
        row_conf = conf.model_copy(update={"compile": False})

    if is_train:
        def step():
            out = fwd(*args)
            return torch.autograd.grad(out, wrt, dout)
    else:
        def step():
            return fwd(*args)

    return bench.measured_result(
        conf=row_conf,
        func=step,
        grad_to_none=[],
        params=[],
        is_train=is_train,
        input_dtype=in_dt,
        parameter_dtype=par_dt,
        execution_path=path + (" [reference]" if key == "pytorch" else ""),
        reference="kernels/<op>/reference.py (PyTorch)" if op not in ("layernorm", "rmsnorm") else path,
    )
