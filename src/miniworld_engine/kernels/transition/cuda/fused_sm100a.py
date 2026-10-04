"""Fused sm_100a Transition: one kernel for the forward, one (plus a partial reduction) for the backward.

The B200 hand-CUDA path developed in ``experiments/transition_fused_sm100`` (branch ``perf/transition-sm100-b200``, rounds
v1-v22). The fusion and the numerics contract are the sm_90a kernels' (``fused_sm90a``); the mapping to Blackwell is new:
tcgen05 MMA with the ``[a|b]`` / h / ``d_xn`` accumulators in tensor memory, 2-CTA clusters (the forward's expand and squeeze
are ``cta_group::2`` products whose weight operand is split across the pair; the backward multicasts inputs and weight chunks
over the pair). Measured on a B200 (1000 W cap) at the AF3 pair width, CUDA-graph replay:

    op                          L=384     L=768
    inference (forward)         58.7 us   215.6 us    (Anthropic v2 108.4 / 496.8 us, torch.compile 203 / 741 us)
    training step (fwd + bwd)   275.8 us  1127.5 us   (torch.compile 614 / 2203 us)

* forward: persistent, 128-row tiles, eight 64-unit hidden chunks; SwiGLU on two warpgroups from tensor memory, h written
  back to tensor memory and consumed there by the squeeze; LayerNorm of the next tile on its own warpgroup. Inference skips the
  ``xn`` / statistics stores at run time.
* backward: 8 x ``DW_REPL`` weight-role CTAs (one hidden slice each: dh / a / b recompute, gate, dWa | dWb | dWs in tensor
  memory) and the remaining input-role CTAs (d_xn from ``[dA | dB]`` over the streamed weight chunks, LayerNorm backward,
  residual); a small kernel sums the partials.

**Blackwell-only and shape-specific by construction**: sm_100 (B200), bf16, ``d_hidden == 128`` and hidden 512, rows a
multiple of 128. ``supported()`` is the whole gate; everything it rejects keeps the existing path. ``dgamma`` / ``dbeta`` and
the weight gradients are summed in a fixed order (no atomics), so a replay is bit-identical.
"""

import functools
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import torch

from ..._compile import opaque
from ..._nvcc import _HOST_CANDIDATES, ensure_cuda_home, gencodes, host_flags, load_extension, nvcc_path

_dir = Path(__file__).parent / "sm100"

#: Rows must be a whole number of these (the persistent grid's tile).
ROWS = 128
D, H = 128, 512
#: Hidden-slice replicas of the backward's weight role (8 * DW_REPL CTAs, the rest run the input role). 9 measured fastest at
#: L384 and L768 on a 148-SM B200 (rounds v9, v13, v22); MINIWORLD_TRANSITION_SM100_REPL overrides it for A/B runs.
DW_REPL = 9


def _repl() -> int:
    return int(os.environ.get("MINIWORLD_TRANSITION_SM100_REPL", DW_REPL))


# ------------------------------------------------------------------------------------------------------------------ build
# The kernels are built into cubins by the newest nvcc on the machine that knows sm_100a, not by the toolkit torch pins, and the
# extension loads them through the driver API. Measured on the B200: the same backward source through CUDA 12.9's ptxas runs ~8 %
# slower than through 13.1's (L384 250 vs 230 us, L768 990 vs 885 us); a cubin does not depend on the runtime torch links.
_KERNEL_SOURCES = ("tr_fwd_sm100.cu", "tr_bwd_sm100.cu")
_PROBE = "__global__ void k(float* p) { p[0] = 1.0f; }\n#include <cuda_bf16.h>\n#include <type_traits>\n"


def _release(nvcc: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _knows_sm100(nvcc: str) -> bool:
    try:
        out = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "compute_100" in out.split()


def _host_for(nvcc: str) -> list[str] | None:
    """``-ccbin`` flags that let ``nvcc`` build an sm_100a cubin here ([] = its default), or None if no host compiler works."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cu"
        src.write_text(_PROBE)
        for cc in [None, *(c.format(prefix=sys.prefix) for c in _HOST_CANDIDATES)]:
            if cc is not None and not Path(cc).is_file():
                continue
            flags = [] if cc is None else ["-ccbin", cc]
            cmd = [nvcc, *flags, "-std=c++17", "-arch=sm_100a", "-cubin", str(src), "-o", str(Path(tmp) / "probe.cubin")]
            try:
                if subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False).returncode == 0:
                    return flags
            except (OSError, subprocess.SubprocessError):
                continue
    return None


def _driver_major() -> int | None:
    """CUDA major version the installed driver supports (13 for the B200 box's 580 driver), or None if unknown."""
    try:
        return int(torch._C._cuda_getDriverVersion()) // 1000
    except Exception:  # noqa: BLE001 -- unknown means "do not filter"
        return None


@functools.lru_cache(maxsize=1)
def kernel_toolchain() -> tuple[str, tuple[int, int], tuple[str, ...]]:
    """(nvcc, release, host flags) that build the kernel cubins: MINIWORLD_TRANSITION_SM100_NVCC if set, else the newest of
    ``/usr/local/cuda*/bin/nvcc`` and the torch-matched nvcc that knows sm_100, is no newer (major) than the driver, and can
    drive a host compiler."""
    override = os.environ.get("MINIWORLD_TRANSITION_SM100_NVCC")
    cands = [override] if override else [*map(str, Path("/usr/local").glob("cuda*/bin/nvcc")), nvcc_path()]
    driver_major = _driver_major()
    found = {}
    for c in cands:
        if c and Path(c).is_file() and os.path.realpath(c) not in found and _knows_sm100(c):
            rel = _release(c)
            # a cubin from a newer major toolkit than the driver supports does not load (cssb's 575 driver stops at CUDA 12.9)
            if rel and (driver_major is None or rel[0] <= driver_major or override):
                found[os.path.realpath(c)] = rel
    for nvcc, rel in sorted(found.items(), key=lambda kv: kv[1], reverse=True):
        flags = _host_for(nvcc)
        if flags is not None:
            return nvcc, rel, tuple(flags)
    raise RuntimeError(f"no nvcc here can build sm_100a cubins (tried {sorted(found) or cands})")


def build_cubins(group: str, specs: tuple[tuple[str, str, tuple[str, ...]], ...]) -> dict[str, str]:
    """Build (or reuse) cubins: ``specs`` = (name, source relative to ``sm100/``, extra nvcc flags). The output directory is keyed
    by the group, the toolchain, the flags and every source under ``sm100/`` (the kernels include ``sm100.cuh``). Concurrent
    builders write private temporaries and rename them into place, so no lock is needed: the results are identical."""
    nvcc, rel, flags = kernel_toolchain()
    digest = hashlib.sha256(f"{group} {nvcc} {rel} {flags} {specs}".encode())
    for src in sorted({s for _, s, _ in specs} | {"sm100.cuh"}):
        digest.update((_dir / src).read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "transition_sm100a" / f"{group}-{digest.hexdigest()[:16]}"
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, src, extra in specs:
        dst = out / f"{name}.cubin"
        if not dst.exists():
            tmp = out / f"{name}.{os.getpid()}.tmp"
            cmd = [nvcc, *flags, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", f"-I{_dir}", *extra, str(_dir / src), "-o", str(tmp)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
            if r.returncode != 0:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"nvcc {rel[0]}.{rel[1]} failed on {src} {' '.join(extra)}:\n{r.stderr[-4000:]}")
            os.replace(tmp, dst)
        paths[name] = str(dst)
    return paths


def _cubins() -> tuple[str, str]:
    """The D = 128 forward and backward cubins."""
    p = build_cubins("d128", tuple((Path(n).stem, n, ()) for n in _KERNEL_SOURCES))
    return p["tr_fwd_sm100"], p["tr_bwd_sm100"]


@functools.lru_cache(maxsize=1)
def _ext():
    """The host binding (built by the torch-matched toolkit) with the kernel cubins loaded into it (on the current device, so a
    cubin the driver rejects fails here, inside ``available()``, and the call keeps the Triton path)."""
    ensure_cuda_home()
    ext = load_extension(
        name="transition_fused_sm100a",
        sources=[str(_dir / "transition_sm100.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("100a")],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )
    ext.load(*_cubins())
    return ext


@functools.lru_cache(maxsize=8)
def _is_b200(index: int) -> bool:
    props = torch.cuda.get_device_properties(index)
    # the backward's role split needs whole 2-CTA clusters on both sides of 8 * DW_REPL
    return (props.major, props.minor) == (10, 0) and props.multi_processor_count % 2 == 0


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """The kernels' own requirements: sm_100, bf16 activations, D = 128 / hidden 512 (tile shapes are literals), whole 128-row tiles.
    The weights may be bf16 or fp32 (an fp32 master): the entry casts them to bf16 outside autograd."""
    if os.environ.get("MINIWORLD_TRANSITION_FUSED_SM100A", "1") == "0":
        return False
    if not x.is_cuda or x.dtype is not torch.bfloat16:
        return False
    if not _is_b200(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    if x.shape[-1] != D or wa.shape != (H, D) or ws.shape != (D, H):
        return False
    if wa.dtype not in _WEIGHT_DTYPES or ws.dtype not in _WEIGHT_DTYPES:
        return False
    rows = 1
    for s in x.shape[:-1]:
        rows *= s
    return rows > 0 and rows % ROWS == 0


_BUILD_FAILED = False
_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful (cached) build; a build failure warns once and keeps the existing path."""
    global _BUILD_FAILED
    if _BUILD_FAILED or not supported(x, wa, ws):
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"fused sm100a Transition unavailable, keeping the existing path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor
    return any(isinstance(t, FakeTensor) for t in tensors)


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save):
    """Output structure only: (out, xn, rstd, c1); xn / rstd / c1 are 1-element placeholders unless ``save`` (a compile-time
    argument)."""
    rows = x.shape[0] if save else 1
    f32 = dict(dtype=torch.float32, device=x.device)
    return (torch.empty_like(x), torch.empty_like(x) if save else x.new_empty((1, D)),
            torch.empty((rows,), **f32), torch.empty((rows,), **f32))


@opaque(fake=_fwd_launch_fake, name="transition_fused_fwd_sm100a")
def _fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor,
                eps: float, save: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(out, xn, rstd, c1); xn / rstd / c1 are placeholders unless ``save``."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save)
    return tuple(_ext().transition_fused_fwd(x, gamma, beta, wa, wb, ws, eps, save))


def _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws):
    """Output structure only: the six gradients, each shaped like what it is a gradient of (dgamma / dbeta and the weight
    gradients fp32)."""
    f32 = dict(dtype=torch.float32, device=x.device)
    return (torch.empty_like(x), torch.empty_like(gamma), torch.empty_like(gamma),
            torch.empty(wa.shape, **f32), torch.empty(wb.shape, **f32), torch.empty(ws.shape, **f32))


@opaque(fake=_bwd_launch_fake, name="transition_fused_bwd_sm100a")
def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, rstd: torch.Tensor, c1: torch.Tensor, gamma: torch.Tensor,
                wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dgamma, dbeta, dWa, dWb, dWs); ``dx`` already carries the residual branch."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, rstd, c1, gamma, wa, wb, ws)
    return tuple(_ext().transition_fused_bwd(dy, x, xn, rstd, c1, gamma, wa, wb, ws, _repl()))


class _FusedTransitionSM100A(torch.autograd.Function):
    """``y = transition(x) + x`` with the residual folded into the squeeze epilogue. The forward saves ``xn`` and the LayerNorm
    statistics; the backward's weight role reads ``xn`` directly (the sm_90a measurement: recomputing it costs more than it saves).
    The parameters keep their dtype (bf16, or an fp32 master): the kernels get bf16 casts made here, outside autograd, and the
    parameters get the kernels' fp32 gradients in their own dtype (fp32 ones unrounded)."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        wa, wb, ws = (w.to(x.dtype).contiguous() for w in (wa, wb, ws))
        out, xn, rstd, c1 = _fwd_launch(flat, gf, bf, wa, wb, ws, float(eps), True)
        ctx.save_for_backward(flat, xn, rstd, c1, gf, wa, wb, ws)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, rstd, c1, gf, wa, wb, ws = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        dx, dgam, dbeta, dwa, dwb, dws = _bwd_launch(dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, rstd, c1, gf, wa, wb, ws)
        return (dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt), dwa.to(adt), dwb.to(bwdt), dws.to(sdt), None)


def transition_fused_sm100a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as the Triton ``transition_residual``. Call ``available(x, wa, ws)`` first.
    Inference (no grad needed) runs the forward without writing ``xn`` / statistics."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        shape = x.shape
        out, _, _, _ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), gamma.float().contiguous(), beta.float().contiguous(),
                                   *(w.to(x.dtype).contiguous() for w in (wa, wb, ws)), float(eps), False)
        return out.reshape(shape)
    return _FusedTransitionSM100A.apply(x, gamma, beta, wa, wb, ws, eps)
