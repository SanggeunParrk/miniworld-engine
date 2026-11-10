"""The diffusion sampler's glue ops: ``AugmentNoise`` (the pre-denoiser part of a step), ``AlignEuler`` (the post-denoiser part) and ``PoolAtoms`` (the atom encoder's pooling).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule` without parameters, eval only (no autograd); ``_forward_pytorch`` (the dense references of
``modules.functional``: ``augment_noise`` / ``align_euler`` / ``pool_atoms``) is the source of truth. The kernels (``kernels/sampler_ops``) replace ~100 small torch ops of
a step by a few launches; the churn arithmetic stays on device scalars (no host read), the random numbers are drawn by the caller.
"""

import torch

from miniworld_engine.modules import _anthropic
from miniworld_engine.modules.functional import align_euler, augment_noise, pool_atoms
from miniworld_engine.modules.primitives import BackendModule, random_input


def _cuda_h100(device: torch.device) -> str | None:
    """Why the hand-written CUDA kernels cannot run on ``device`` (they are built for the H100, sm_90a, and the A100, sm_80), or ``None``."""
    if device.type != "cuda":
        return "the kernels run on CUDA tensors only"
    cc = torch.cuda.get_device_capability(device)
    if cc not in ((9, 0), (8, 0)):
        return f"the CUDA kernels are the H100 (sm_90a) and A100 (sm_80) builds, got sm_{cc[0]}{cc[1]}"
    return None


def _fp32_cuda(name: str, t: torch.Tensor, ndim: int, shape: tuple | None = None) -> str | None:
    if t.device.type != "cuda":
        return "the kernels run on CUDA tensors only"
    if t.dtype != torch.float32:
        return f"{name} must be fp32, got {t.dtype}"
    if t.dim() != ndim or (shape is not None and tuple(t.shape) != shape):
        return f"{name} must be {list(shape) if shape is not None else f'{ndim}-d'}, got {tuple(t.shape)}"
    return None


def _first(*reasons: str | None) -> str | None:
    return next((r for r in reasons if r is not None), None)


def _coords_scene(length: int, samples: int, device, generator, dtype) -> tuple:
    """``(coords [B, N, 3], mask [N])``: independent Gaussian clouds (sigma 10 A) around the point (3, 3, 3), the last quarter of the atoms padding (mask 0)."""
    coords = random_input((samples, length, 3), dtype=dtype, device=device, generator=generator, scale=10.0) + 3
    mask = (torch.arange(length, device=device) < (length * 3) // 4).to(dtype)
    return coords, mask


class AugmentNoise(BackendModule):
    """The pre-denoiser part of one EDM / AlphaFold 3 sampler step (Algorithm 18 lines 2-5, with the random draws made by the caller)::

        sigma_hat = sigma * (1 + gamma) inside the churn window else sigma;   magnitude = noise_scale * sqrt(sigma_hat^2 - sigma^2)
        noisy     = (rotate(coords - centre, Gram-Schmidt frame of rv) + tr) * mask + magnitude * noise

    ``forward(coords, mask, rv, tr, noise, sigma, sigma_next)``:

    * ``coords`` ``[B, N, 3]`` -- ``B`` samples of one structure of ``N`` atoms; ``mask`` ``[N]`` -- resolved atoms (the weights of the centre);
    * ``rv`` ``[B, 2, 3]`` -- two random vectors per sample (the rigid motion's frame); ``tr`` ``[B, 3]`` -- the translation; ``noise`` ``[B, N, 3]``;
    * ``sigma`` / ``sigma_next`` ``[]`` -- device scalars: the noise level now and of the next step.

    Returns ``(noisy [B, N, 3], sc [2])``, ``sc = (sigma_hat, noise magnitude)`` (what :class:`AlignEuler` reads). fp32 (fp64 for the gold run).

    Parameters
    ----------
    window : bool
        The churn applies when ``sigma_min <= sigma <= sigma_max`` (EDM) instead of ``sigma_next > gamma_min`` (AF3).
    gamma, gamma_min, sigma_min, sigma_max, noise_scale, variance_floor, clamp
        The churn of the sampler (defaults: AlphaFold 3). ``clamp`` clamps the variance at ``variance_floor``.
    backend : str
        See :class:`BackendModule`. ``"cuda"``: the sm_90a kernels (``kernels/sampler_ops/cuda``, H100), as opaque ops. ``"triton"``: the portable Triton
        kernels (``kernels/sampler_ops/triton``), any CUDA card, bitwise reproducible. Both need fp32 CUDA tensors.
    """
    version = 1   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "cuda", "triton")

    def __init__(self, *, window: bool = False, gamma: float = 0.8, gamma_min: float = 1.0, sigma_min: float = 0.0, sigma_max: float = float("inf"),
                 noise_scale: float = 1.003, variance_floor: float = 0.0, clamp: bool = False, backend: str = "pytorch") -> None:
        super().__init__()
        self.churn = {"window": bool(window), "gamma": float(gamma), "gamma_min": float(gamma_min), "sigma_min": float(sigma_min),
                      "sigma_max": float(sigma_max), "noise_scale": float(noise_scale), "variance_floor": float(variance_floor), "clamp": bool(clamp)}
        self.set_backend(backend)

    def _forward_pytorch(self, coords, mask, rv, tr, noise, sigma, sigma_next):
        return augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, **self.churn)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``samples`` samples of one structure of ``length`` atoms (a Gaussian cloud around an offset origin, the last quarter padding), random frame vectors,
        translations and noise, ``sigma = 2``, ``sigma_next = 1.5`` (inside the EDM window and above ``gamma_min``)."""
        coords, mask = _coords_scene(length, samples, device, generator, dtype)
        return {"coords": coords, "mask": mask,
                "rv": random_input((samples, 2, 3), dtype=dtype, device=device, generator=generator),
                "tr": random_input((samples, 3), dtype=dtype, device=device, generator=generator),
                "noise": random_input((samples, length, 3), dtype=dtype, device=device, generator=generator),
                "sigma": torch.tensor(2.0, dtype=dtype, device=device), "sigma_next": torch.tensor(1.5, dtype=dtype, device=device)}

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_kernels(self, coords, mask, rv, tr, noise, sigma, sigma_next) -> str | None:
        if coords.dim() != 3 or coords.shape[-1] != 3:
            return f"coords must be [B, N, 3], got {tuple(coords.shape)}"
        b, n, _ = coords.shape
        return _first(
            _fp32_cuda("coords", coords, 3), _fp32_cuda("mask", mask, 1, (n,)), _fp32_cuda("rv", rv, 3, (b, 2, 3)), _fp32_cuda("tr", tr, 2, (b, 3)),
            _fp32_cuda("noise", noise, 3, (b, n, 3)),
            None if sigma.numel() == 1 and sigma_next.numel() == 1 else "sigma / sigma_next must be scalars",
            _fp32_cuda("sigma", sigma, sigma.dim()), _fp32_cuda("sigma_next", sigma_next, sigma_next.dim()),
            "empty input" if coords.numel() == 0 else None)

    def _supports_cuda(self, coords, mask, rv, tr, noise, sigma, sigma_next) -> str | None:
        return _first(self._supports_kernels(coords, mask, rv, tr, noise, sigma, sigma_next), _cuda_h100(coords.device))

    def _supports_triton(self, coords, mask, rv, tr, noise, sigma, sigma_next) -> str | None:
        return self._supports_kernels(coords, mask, rv, tr, noise, sigma, sigma_next)

    def _forward_cuda(self, coords, mask, rv, tr, noise, sigma, sigma_next):
        from miniworld_engine.kernels.sampler_ops.cuda.function import augment_noise_cuda

        return augment_noise_cuda(coords, mask, rv, tr, noise, sigma, sigma_next, **self.churn)

    def _forward_triton(self, coords, mask, rv, tr, noise, sigma, sigma_next):
        from miniworld_engine.kernels.sampler_ops.triton.function import augment_noise_triton

        return augment_noise_triton(coords, mask, rv, tr, noise, sigma, sigma_next, **self.churn)


class AlignEuler(BackendModule):
    """The post-denoiser part of one sampler step: the Euler update from ``noisy`` Kabsch-aligned onto the denoiser's ``denoised`` (weights ``mask``)::

        aligned   = Kabsch(noisy -> denoised) (centred on each set's mask-weighted centre, rotation by Horn's quaternion method)
        direction = (aligned - denoised) / sigma_hat;      out = aligned + step_scale * (sigma_next - sigma_hat) * direction

    ``forward(noisy, denoised, mask, sc, sigma_next)``: ``noisy`` / ``denoised`` ``[B, N, 3]``, ``mask`` ``[N]``, ``sc`` ``[>= 2]`` with ``sigma_hat = sc[0]`` (the
    second output of :class:`AugmentNoise`), ``sigma_next`` ``[]``. Returns the updated coordinates ``[B, N, 3]``. fp32 (fp64 for the gold run).

    Parameters
    ----------
    step_scale : float
        The sampler's step scale (AlphaFold 3: 1.5).
    backend : str
        See :class:`BackendModule`. ``"cuda"``: the sm_90a kernels (H100); ``"triton"``: the portable kernels (any CUDA card). Both need fp32 CUDA tensors.
    """
    version = 1   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")

    def __init__(self, *, step_scale: float = 1.5, backend: str = "pytorch") -> None:
        super().__init__()
        self.step_scale = float(step_scale)
        self.set_backend(backend)

    def _forward_pytorch(self, noisy, denoised, mask, sc, sigma_next):
        return align_euler(noisy, denoised, mask, sc, sigma_next, self.step_scale)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``samples`` rotated, shifted and noised copies of one structure of ``length`` atoms (``noisy``), the denoiser's answer (the structure plus 0.1 A),
        the last quarter padding, ``sc = (4.0, 0.3)``, ``sigma_next = 2.5``."""
        coords, mask = _coords_scene(length, samples, device, generator, dtype)
        angle = 6.0 * torch.rand((samples,), device=device, generator=generator)
        rot = torch.zeros((samples, 3, 3), device=device)
        rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = angle.cos(), -angle.sin(), angle.sin(), angle.cos(), 1.0
        noise = random_input((samples, length, 3), dtype=dtype, device=device, generator=generator)
        noisy = (coords.float() @ rot + 5 + 0.5 * noise.float()).to(dtype)
        denoised = (coords.float() + 0.1 * random_input((samples, length, 3), dtype=torch.float32, device=device, generator=generator)).to(dtype)
        return {"noisy": noisy, "denoised": denoised, "mask": mask, "sc": torch.tensor([4.0, 0.3], dtype=dtype, device=device),
                "sigma_next": torch.tensor(2.5, dtype=dtype, device=device)}

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_kernels(self, noisy, denoised, mask, sc, sigma_next) -> str | None:
        if noisy.dim() != 3 or noisy.shape[-1] != 3:
            return f"noisy must be [B, N, 3], got {tuple(noisy.shape)}"
        n = noisy.shape[1]
        return _first(
            _fp32_cuda("noisy", noisy, 3), _fp32_cuda("denoised", denoised, 3, tuple(noisy.shape)), _fp32_cuda("mask", mask, 1, (n,)),
            _fp32_cuda("sc", sc, 1), None if sc.numel() >= 2 else "sc must hold (sigma_hat, magnitude)",
            None if sigma_next.numel() == 1 else "sigma_next must be a scalar", _fp32_cuda("sigma_next", sigma_next, sigma_next.dim()),
            "empty input" if noisy.numel() == 0 else None)

    def _supports_cuda(self, noisy, denoised, mask, sc, sigma_next) -> str | None:
        return _first(self._supports_kernels(noisy, denoised, mask, sc, sigma_next), _cuda_h100(noisy.device))

    def _supports_triton(self, noisy, denoised, mask, sc, sigma_next) -> str | None:
        return self._supports_kernels(noisy, denoised, mask, sc, sigma_next)

    def _forward_cuda(self, noisy, denoised, mask, sc, sigma_next):
        from miniworld_engine.kernels.sampler_ops.cuda.function import align_euler_cuda

        return align_euler_cuda(noisy, denoised, mask, sc, sigma_next, self.step_scale)

    def _forward_triton(self, noisy, denoised, mask, sc, sigma_next):
        from miniworld_engine.kernels.sampler_ops.triton.function import align_euler_triton

        return align_euler_triton(noisy, denoised, mask, sc, sigma_next, self.step_scale)

    # ------------------------------------------------------------------------------------------------------------- "anthropic"
    # The rotation on Boltz-2's sync-free Kabsch kernel (boltz2/opt/forward/sampler/src/bz_sampler.py ``kabsch_rotation_device``, its
    # ``align_jacobi64`` fast-tier lever: one program per batch element, a one-sided Jacobi SVD of the 3 x 3 covariance in fp64 registers,
    # R = U diag(1, 1, det(U Vh)) Vh -- the rotation of :func:`~miniworld_engine.modules.functional.kabsch_rotation` for ``a @ R ~ b``). The
    # only Kabsch kernels in the release are Boltz-2's and ESMFold2's copy of it (ef2_dit ``_kabsch3_kernel``); the centring, covariance and
    # Euler update stay the module's statements, as in Boltz-2's step (the kernel replaces the eager gesvd section only).
    _release = ("bz_sampler", "boltz2/opt/forward/sampler/src")

    def _supports_anthropic(self, noisy, denoised, mask, sc, sigma_next) -> str | None:
        return _first(_anthropic.unavailable(self._release[0], noisy.device, path=self._release[1]),
                      self._supports_kernels(noisy, denoised, mask, sc, sigma_next))

    def _forward_anthropic(self, noisy, denoised, mask, sc, sigma_next):
        bz = _anthropic.load(self._release[0], path=self._release[1])
        w = mask[None, :, None]
        total = w.sum(1, keepdim=True)
        cm = (noisy * w).sum(1, keepdim=True) / total
        cr = (denoised * w).sum(1, keepdim=True) / total
        a, b = noisy - cm, denoised - cr
        cov = torch.einsum("bnc,bnd->bcd", a * w, b).contiguous()
        rot = torch.empty_like(cov)
        det = torch.empty(cov.shape[0], dtype=torch.float32, device=cov.device)
        smin = torch.empty_like(det)
        bz.kabsch_rotation_device(cov, rot, det, smin)
        aligned = a @ rot + cr
        sigma_hat = sc[0]
        return aligned + self.step_scale * (sigma_next - sigma_hat) * ((aligned - denoised) / sigma_hat)


class PoolAtoms(BackendModule):
    """The atom encoder's pooling of atom features to tokens, in one pass::

        out[s, t] = sum_a m[t, a] * relu(g[t, a] * x[s, idx[t, a]]) / max(sum_a m[t, a], 1e-10)

    ``forward(x, idx, gather_mask, atom_mask)``: ``x`` ``[S, Q, C]`` -- atom features of ``S`` samples (the queries layout, flattened); ``idx`` ``[T, A]`` int --
    the atoms of each of ``T`` tokens (the token-atoms layout), ``gather_mask`` / ``atom_mask`` ``[T, A]`` -- the gather's and the atoms' masks (bool). Returns
    ``[S, T, C]`` in ``x``'s dtype (bf16; the sums run in fp32, fp64 for the gold run).

    Parameters
    ----------
    backend : str
        See :class:`BackendModule`. ``"cuda"``: the sm_90a kernel (H100), ``C % 4 == 0`` and ``C <= 4096``; ``"triton"``: the portable kernel (any CUDA card).
        Both need bf16 ``x`` on CUDA and int32 / int64 ``idx``.
    """
    version = 1   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")

    #: Channels of ``make_inputs``' atom features (the atom encoder's ``d_single``) and atoms per token (the token-atoms layout's width).
    _C, _A = 128, 24

    def __init__(self, *, backend: str = "pytorch") -> None:
        super().__init__()
        self.set_backend(backend)

    def _forward_pytorch(self, x, idx, gather_mask, atom_mask):
        return pool_atoms(x, idx, gather_mask, atom_mask)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """Atom features ``[samples, length + 64, 128]`` in ``dtype``, ``length // 8`` tokens of 24 atom slots with random atom indices, 70 % of the slots
        real, the gather mask a superset of them, token 5 without atoms (its output is zero)."""
        tokens, rows = max(length // 8, 6), length + 64
        x = random_input((samples, rows, self._C), dtype=dtype, device=device, generator=generator)
        idx = torch.randint(0, rows, (tokens, self._A), device=device, generator=generator)
        atom_mask = torch.rand((tokens, self._A), device=device, generator=generator) > 0.3
        gather_mask = atom_mask | (torch.rand((tokens, self._A), device=device, generator=generator) > 0.9)
        atom_mask[5] = False
        return {"x": x, "idx": idx, "gather_mask": gather_mask, "atom_mask": atom_mask}

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_kernels(self, x, idx, gather_mask, atom_mask) -> str | None:
        if x.device.type != "cuda":
            return "the kernels run on CUDA tensors only"
        if x.dtype != torch.bfloat16 or x.dim() != 3:
            return f"x must be bf16 [S, Q, C], got {x.dtype} {tuple(x.shape)}"
        if idx.dim() != 2 or idx.dtype not in (torch.int32, torch.int64):
            return f"idx must be int32 / int64 [T, A], got {idx.dtype} {tuple(idx.shape)}"
        for name, m in (("gather_mask", gather_mask), ("atom_mask", atom_mask)):
            if tuple(m.shape) != tuple(idx.shape):
                return f"{name} must be {tuple(idx.shape)}, got {tuple(m.shape)}"
            if m.dtype not in (torch.bool, torch.float32, torch.int32, torch.int64, torch.bfloat16):
                return f"{name} must be bool / fp32 / bf16 / int32 / int64, got {m.dtype}"
        if x.numel() == 0 or idx.numel() == 0:
            return "empty input"
        return None

    def _supports_cuda(self, x, idx, gather_mask, atom_mask) -> str | None:
        why = self._supports_kernels(x, idx, gather_mask, atom_mask)
        if why is None:
            c = x.shape[-1]
            why = None if c % 4 == 0 and c // 4 <= 1024 else f"the CUDA kernel needs C % 4 == 0 and C <= 4096, got C={c}"
        return _first(why, _cuda_h100(x.device))

    def _supports_triton(self, x, idx, gather_mask, atom_mask) -> str | None:
        return self._supports_kernels(x, idx, gather_mask, atom_mask)

    def _forward_cuda(self, x, idx, gather_mask, atom_mask):
        from miniworld_engine.kernels.sampler_ops.cuda.function import pool_atoms_cuda

        return pool_atoms_cuda(x, idx, gather_mask, atom_mask)

    def _forward_triton(self, x, idx, gather_mask, atom_mask):
        from miniworld_engine.kernels.sampler_ops.triton.function import pool_atoms_triton

        return pool_atoms_triton(x, idx, gather_mask, atom_mask)

    # ------------------------------------------------------------------------------------------------------------- "anthropic"
    # AlphaFold 3's kit kernel for exactly this statement (af3_torch/opt/forward/dtk/af3t_token_agg.py ``token_mean_relu``, the ``token_agg``
    # lever of the atom-attention encoder): per (sample, token, channel block) one program reads the token's gathered rows once (masked slots
    # not read), relu, the weights w = atom_mask * gather_mask and the hoisted 1 / clamp(sum atom_mask, 1e-10) (``operands``); fp32 out, cast
    # to x's dtype like the reference. relu(g x) = g relu(x) needs g >= 0: a bool / non-negative gather mask. The other pooling kernels of the
    # release (dtk_kernels.seg_reduce, of3_sampler.token_agg, Boltz-2 _k_segmean, ESMFold2 _e6_segmean) pool a sorted atom -> token map, not
    # this token-atoms gather.
    _release = ("af3t_token_agg", "af3_torch/opt/forward/dtk")

    def _supports_anthropic(self, x, idx, gather_mask, atom_mask) -> str | None:
        why = _first(_anthropic.unavailable(self._release[0], x.device, path=self._release[1]), self._supports_kernels(x, idx, gather_mask, atom_mask))
        if why is None and gather_mask.dtype != torch.bool and bool((gather_mask < 0).any()):
            return "a negative gather mask weight (the kernel applies the weight after the relu)"
        return why

    def _forward_anthropic(self, x, idx, gather_mask, atom_mask):
        agg = _anthropic.load(self._release[0], path=self._release[1])
        ops = agg.operands(idx, gather_mask, atom_mask)
        return agg.token_mean_relu(x, ops["agg_idx"], ops["agg_w"], ops["agg_inv"]).to(x.dtype)
