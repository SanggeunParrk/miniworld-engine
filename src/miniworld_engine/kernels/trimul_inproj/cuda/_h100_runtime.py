"""Portable compiler / driver support for the selected Anthropic-derived kernels.

All source reads are package-relative; binaries and temporary headers live in a
content-addressed user cache. Module handles are specific to a CUDA device.
"""

from pathlib import Path
from functools import lru_cache, wraps
from contextlib import contextmanager
import ctypes, hashlib, os, shutil, subprocess, json
import threading
import torch
import torch.nn.functional as F

from miniworld_engine import settings

SOURCES = Path(__file__).with_name("h100_sources")
_d256_scope = threading.local()
_d128_scope = threading.local()
_bidi_d256_scope = threading.local()
_d64_scope = threading.local()
_uni_owned_blas_scope = threading.local()

# Launch-scope policies that were environment switches. The switches are gone; these private constants keep the
# alternatives reachable (tests force them with monkeypatch.setattr).
_CUDA_AUX_UNSCOPED = False  # native auxiliary kernels outside the qualified launch scopes
_TRAIN_CUDA_UNSCOPED = True  # D64 native preparation and the L128..1024 D64 buckets outside the qualified scope
_TRAIN_FUSED_PREP_UNSCOPED = True  # D64 fused preparation outside the qualified scope (needs the line above)
_D64_UNI_AFFINE_FORCED = False  # uni outgoing D64: the affine-split backward instead of the affine-register one


def _native_buckets():
    """None: only the qualified cells; True: explicit native buckets; False: every native training cell off."""
    return settings.current().trimul_h100_native_buckets


def _preserved_width(width):
    """Explicit preserved glue: native buckets forced on and the width selected."""
    s = settings.current()
    if width == 128 and s.trimul_h100_native_buckets is None:
        return True  # D128 serves every length 128..1024 natively by default (2026-10-13)
    return s.trimul_h100_native_buckets is True and width in s.trimul_h100_preserve_widths


def d64_training(x, outgoing=False, bidirectional=False):
    """Qualified compact-mask D64 preparation (the bidirectional L384 cell; no single-direction cell is qualified, so
    ``outgoing`` does not change the answer); keep wide-preservation policy separate."""
    if (not bidirectional or x.ndim != 4 or x.shape[1] != x.shape[2]   # any batch: B samples share the B = 1 scope
            or x.shape[-1] != 64 or x.shape[1] != 384):
        return False
    return _native_buckets() is not False


def d64_uni_owned_bmm_enabled(length):
    """Only the qualified training scope selects owned L768 scratch by default."""
    scoped = False if torch.compiler.is_compiling() else getattr(_d64_scope, "enabled", False)
    return scoped and length == 768


def d64_uni_affine_enabled(x, outgoing):
    """The affine-split backward for single-direction outgoing D64, only when forced."""
    return x.shape[-1] == 64 and outgoing and _D64_UNI_AFFINE_FORCED


@contextmanager
def preserved_d64_context(enabled):
    previous = getattr(_d64_scope, "enabled", False)
    _d64_scope.enabled = previous or enabled
    try:
        yield
    finally:
        _d64_scope.enabled = previous


def d128_training(x, outgoing=True, bidirectional=False):
    """Preserved D128 path (``_preserved_width``; the direction does not change the answer)."""
    if x.shape[-1] != 128:
        return False
    if _native_buckets() is False:
        return False
    return _preserved_width(128)


@contextmanager
def preserved_d128_context(enabled):
    previous = getattr(_d128_scope, "enabled", False)
    _d128_scope.enabled = previous or enabled
    try:
        yield
    finally:
        _d128_scope.enabled = previous


def uni_d256_training(x, outgoing):
    """Qualified outgoing L384 default; trimul_h100_native_buckets=False keeps the old diagnostic path."""
    if x.shape[-1] != 256:
        return False
    if _native_buckets() is False:
        return False
    return _preserved_width(256) or (outgoing and x.ndim == 4
                                     and tuple(x.shape[:3]) == (1, 384, 384))


@contextmanager
def preserved_d256_context(enabled):
    """Native launch-local policy, restored even if a build or launch raises."""
    previous = getattr(_d256_scope, "enabled", False)
    _d256_scope.enabled = previous or enabled
    try:
        yield
    finally:
        _d256_scope.enabled = previous


def bidi_training(x):
    """Qualified bidirectional native cells; trimul_h100_native_buckets / preserve_widths retain diagnostic routes."""
    width = x.shape[-1]
    if (x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != x.shape[2]
            or x.shape[1] not in (384, 768) or width not in (256, 384, 512)):
        return False
    if _native_buckets() is False:
        return False
    return _preserved_width(width) or (width == 256 and x.shape[1] == 768)


@contextmanager
def preserved_bidi_context(x):
    """Keep bidirectional D256 register-source selection out of uni scopes."""
    enabled = bidi_training(x)
    previous = getattr(_bidi_d256_scope, "enabled", False)
    _bidi_d256_scope.enabled = previous or (enabled and x.shape[-1] == 256)
    try:
        with preserved_d256_context(enabled):
            yield
    finally:
        _bidi_d256_scope.enabled = previous


def d256_register_source_enabled(selected=False):
    scoped = False if torch.compiler.is_compiling() else getattr(_bidi_d256_scope, "enabled", False)
    return bool(selected or scoped)


def d256_glue_enabled(selected=False):
    scoped = False if torch.compiler.is_compiling() else getattr(_d256_scope, "enabled", False)
    return bool(selected or scoped)


def _upstream():
    return SOURCES / "upstream"


def _launch_module():
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_launch

    return _h100_launch


def preserve_d128_training():
    """Opt-in native glue around the existing, selected D128 B1/B7 kernels."""
    scoped = False if torch.compiler.is_compiling() else getattr(_d128_scope, "enabled", False)
    return scoped or _preserved_width(128)


def preserve_d256_training():
    """Keep selected wide CUDA plans; use deterministic native auxiliary work."""
    scoped = False if torch.compiler.is_compiling() else getattr(_d256_scope, "enabled", False)
    return scoped or _preserved_width(256)


def preserve_wide_training(width):
    """Opt-in D384/D512 native glue around the selected Hopper compute kernels."""
    scoped = False if torch.compiler.is_compiling() else getattr(_d256_scope, "enabled", False)
    return (width in (384, 512)
            and (scoped or _preserved_width(width)))


def uni_training(x, outgoing):
    if uni_d256_training(x, outgoing):
        return True
    bucket = (x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == x.shape[2]
              and x.shape[1] in (384, 768))
    if preserve_wide_training(x.shape[-1]) and bucket:
        return True
    # Qualified outgoing cells preserve the selected native algorithm per shape.
    # D512 serves both directions (GP_DIR 1/2 in contract_gp); D384 L384 stays outgoing-only.
    return (bucket
            and ((x.shape[-1] == 512)
                 or (outgoing and x.shape[-1] == 384 and x.shape[1] == 384))
            and _native_buckets() is not False)


def preserve_training(width):
    return ((width == 128 and preserve_d128_training())
            or (width == 256 and preserve_d256_training())
            or preserve_wide_training(width))


def wide_owned_blas_enabled():
    scoped = False if torch.compiler.is_compiling() else getattr(_uni_owned_blas_scope, "enabled", False)
    return scoped


@contextmanager
def preserved_uni_blas_context(x, outgoing):
    """Graph-owned scratch defaults only for four qualified uni training cells."""
    bucket = (x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == x.shape[2]
              and ((outgoing and x.shape[-1] in (256, 384) and x.shape[1] == 384)
                   or (x.shape[-1] == 512 and x.shape[1] in (384, 768))))
    previous = getattr(_uni_owned_blas_scope, "enabled", False)
    _uni_owned_blas_scope.enabled = previous or (bucket and uni_training(x, outgoing))
    try:
        yield
    finally:
        _uni_owned_blas_scope.enabled = previous


@contextmanager
def native_context(device):
    """One verified framework CUDA context for all maps within a native call."""
    with torch.cuda.device(device):
        L = _launch_module()
        drv = L._make_context_current(device.index)
        with L.tensor_map_scope(drv):
            yield


def device_cache(fn):
    @lru_cache(None)
    def cached(device, args, kwargs):
        with torch.cuda.device(device):
            return fn(*args, **dict(kwargs))

    @wraps(fn)
    def call(*args, **kwargs):
        return cached(torch.cuda.current_device(), args, tuple(sorted(kwargs.items())))

    return call


@lru_cache(None)
def read_config(relative_path):
    """Read packaged, process-constant configuration once; callers must not mutate it."""
    return json.loads((SOURCES / relative_path).read_text())


def cache_dir():
    root = (
        settings.engine_jit_root() / "trimul_h100"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


@lru_cache(None)
def _compiler():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home

    ensure_cuda_home()
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        from torch.utils.cpp_extension import CUDA_HOME

        nvcc = str(Path(CUDA_HOME) / "bin/nvcc") if CUDA_HOME else None
    if not nvcc:
        raise RuntimeError("H100 native kernels require the CUDA nvcc compiler")
    return nvcc, subprocess.check_output([nvcc, "--version"], text=True)


@lru_cache(None)
def _source_digest():
    h = hashlib.sha256()
    for p in sorted(SOURCES.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(SOURCES)).encode())
            h.update(p.read_bytes())
    return h.digest()


def _display_name(source):
    """How a build names ``source`` in the log: its path under h100_sources, a generated (``compile_text``) file, or its name."""
    p = Path(source)
    try:
        return str(p.relative_to(SOURCES))
    except ValueError:
        pass
    if p.parent == cache_dir() and len(p.stem) == 64:
        return f"generated {p.stem[:12]}.cu"
    return p.name


def compile(source, flags):
    """The cubin of ``source`` built with ``flags``. Keyed by the source, the flags, the nvcc version and every file under
    h100_sources (``_source_digest``): any edit there rebuilds every TriMul cubin."""
    from miniworld_engine.kernels._nvcc import building, flock_wait

    nvcc, version = _compiler()
    key = hashlib.sha256(
        source.read_bytes() + repr(flags).encode() + version.encode() + _source_digest()
    ).hexdigest()
    out = cache_dir() / (key + ".cubin")
    if out.exists():  # the finished cubin only ever appears by rename: no lock needed to reuse it
        return out
    label = f"trimul_h100/{_display_name(source)} [{key[:8]}]"
    with out.with_suffix(".lock").open("w") as lock:
        flock_wait(lock, label)
        if not out.exists():
            with building(label, "cubin"):
                tmp = out.with_suffix(".tmp.cubin")
                result = subprocess.run(
                    [nvcc, *flags, str(source), "-o", str(tmp)],
                    capture_output=True,
                    text=True,
                )
                out.with_suffix(".ptxas.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                tmp.replace(out)
    return out


def compile_text(source, flags):
    key = hashlib.sha256(source.encode()).hexdigest()
    p = cache_dir() / (key + ".cu")
    # Same bytes for every writer; atomically publish before invoking nvcc.
    if not p.exists():
        tmp = p.with_suffix(".%d.tmp" % os.getpid())
        tmp.write_text(source)
        tmp.replace(p)
    return compile(p, flags)


class CooperativePlan:
    def __call__(self):
        L = _launch_module()
        drv = self.k.unit.drv
        args = L._Packed([self.p])
        drv._unwrap(
            "cuLaunchCooperativeKernel",
            drv.d.cuLaunchCooperativeKernel(
                drv.d.CUfunction(int(self.k.handle)),
                self.count,
                1,
                1,
                getattr(self, "threads", 256),
                1,
                1,
                self.smem,
                drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),
                ctypes.addressof(args.array),
            ),
        )
        return self.outputs


def k1_smem(cfg):
    bi, bj, slots, sk, *_ = cfg
    ng = bi * bj // 64
    return (
        bi * bj * 256
        + slots * sk * 8192
        + ng * 8192
        + 1024
        + ((2 + 2 * slots) * 8 + 127) // 128 * 128
    )


def k3_smem(cfg):
    bi, bj, slots, *_ = cfg
    bm = bi * bj
    return (
        bm * 256 * 2
        + bm * 128 * 2
        + slots * 256 * 64
        + 16384
        + 3072
        + ((6 + 2 * slots) * 8 + 127) // 128 * 128
    )


def pack_into(dst, wl, wlg, wr, wrg, backward_stack=None, gate=None):
    # Read live weights once on the current stream, including graph replay.
    # An optional saved stack retains the backward layout from the same launch.
    if backward_stack is not None:
        from ._h100_pack import pack_training_pair
        return pack_training_pair(dst, backward_stack, (gate,wl,wlg,wr,wrg))
    from ._h100_pack import pack_into as pack

    pack(dst, wl, wlg, wr, wrg)


_GLUE_CUDA = True   # eager glue (weight pack, input LN, LN-affine fold) on the CUDA aux kernels instead of Triton / Inductor


def cuda_aux_enabled():
    scoped = False if torch.compiler.is_compiling() else (
        getattr(_d256_scope, "enabled", False) or getattr(_d128_scope, "enabled", False)
        or getattr(_d64_scope, "enabled", False))
    return scoped or _CUDA_AUX_UNSCOPED


def cuda_training_aux_enabled():
    """D64 launch-local preparation (qualified launch scope only)."""
    scoped = False if torch.compiler.is_compiling() else getattr(_d64_scope, "enabled", False)
    return scoped or _TRAIN_CUDA_UNSCOPED


def cuda_training_fused_prepare_enabled():
    scoped = False if torch.compiler.is_compiling() else getattr(_d64_scope, "enabled", False)
    return cuda_training_aux_enabled() and (scoped or _TRAIN_FUSED_PREP_UNSCOPED)


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _normalize_into_compiled(dst, x, g, b):
    dst.copy_(F.layer_norm(x.float(), (x.shape[-1],), g, b, 1e-5).to(x.dtype))


def normalize_into(dst, x, g, b, warp=False):
    if cuda_aux_enabled() or (_GLUE_CUDA and not torch.compiler.is_compiling()):
        from ._h100_pack import normalize_into as native
        return native(dst, x, g, b, warp=warp)
    return _normalize_into_compiled(dst, x, g, b)


@device_cache
def load_unit(path, name):
    L = _launch_module()
    device = torch.cuda.current_device()
    drv = L.BlockDriver(device=device)
    return L.Unit(
        name,
        "sm_90a",
        device,
        drv.drv,
        drv.load(Path(path).read_bytes()),
        {},
        str(path),
    )


def publish_header(path, text):
    """Atomically publish an immutable transformed header for concurrent builds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        return
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)
