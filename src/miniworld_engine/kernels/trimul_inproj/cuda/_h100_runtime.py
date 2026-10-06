"""Portable compiler / driver support for the selected Anthropic-derived kernels.

All source reads are package-relative; binaries and temporary headers live in a
content-addressed user cache. Module handles are specific to a CUDA device.
"""

from pathlib import Path
from functools import lru_cache, wraps
from contextlib import contextmanager
import ctypes, fcntl, hashlib, os, shutil, subprocess, json
import threading
import torch
import torch.nn.functional as F

SOURCES = Path(__file__).with_name("h100_sources")
_d256_scope = threading.local()
_d128_scope = threading.local()


def d128_training(x, outgoing=True, bidirectional=False):
    """Preserved bidirectional L384 default; other cells remain candidates."""
    if x.shape[-1] != 128:
        return False
    if (os.environ.get("MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS") == "0"
            or os.environ.get("MINIWORLD_TRIMUL_H100_PRESERVE_D128") == "0"):
        return False
    explicit = (os.environ.get("MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS") == "1"
                and os.environ.get("MINIWORLD_TRIMUL_H100_PRESERVE_D128") == "1")
    return explicit or (bidirectional and x.ndim == 4
                        and tuple(x.shape[:3]) == (1, 384, 384))


@contextmanager
def preserved_d128_context(enabled):
    previous = getattr(_d128_scope, "enabled", False)
    _d128_scope.enabled = previous or enabled
    try:
        yield
    finally:
        _d128_scope.enabled = previous


def uni_d256_training(x, outgoing):
    """Qualified outgoing L384 default; explicit zero keeps the old diagnostic path."""
    if x.shape[-1] != 256:
        return False
    if (os.environ.get("MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS") == "0"
            or os.environ.get("MINIWORLD_TRIMUL_H100_PRESERVE_D256") == "0"):
        return False
    explicit = (os.environ.get("MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS") == "1"
                and os.environ.get("MINIWORLD_TRIMUL_H100_PRESERVE_D256") == "1")
    return explicit or (outgoing and x.ndim == 4
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


def d256_glue_enabled(name, selected=False):
    scoped = False if torch.compiler.is_compiling() else getattr(_d256_scope, "enabled", False)
    default = "1" if selected or scoped else "0"
    return os.environ.get(name, default) == "1"


def _upstream():
    return SOURCES / "upstream"


def _launch_module():
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_launch

    return _h100_launch


def preserve_d128_training():
    """Opt-in native glue around the existing, selected D128 B1/B7 kernels."""
    scoped = False if torch.compiler.is_compiling() else getattr(_d128_scope, "enabled", False)
    return (scoped or (os.environ.get("MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS") == "1"
                      and os.environ.get("MINIWORLD_TRIMUL_H100_PRESERVE_D128") == "1"))


def preserve_d256_training():
    """Keep selected wide CUDA plans; use deterministic native auxiliary work."""
    scoped = False if torch.compiler.is_compiling() else getattr(_d256_scope, "enabled", False)
    return (scoped
            or (os.environ.get("MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS") == "1"
                and os.environ.get("MINIWORLD_TRIMUL_H100_PRESERVE_D256") == "1"))


def preserve_wide_training(width):
    """Opt-in D384/D512 native glue around the selected Hopper compute kernels."""
    scoped = False if torch.compiler.is_compiling() else getattr(_d256_scope, "enabled", False)
    return (width in (384, 512)
            and (scoped or (os.environ.get("MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS") == "1"
                           and os.environ.get("MINIWORLD_TRIMUL_H100_PRESERVE_WIDE") == "1")))


def uni_training(x, outgoing):
    if uni_d256_training(x, outgoing):
        return True
    bucket = (x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == x.shape[2]
              and x.shape[1] in (384, 768))
    if preserve_wide_training(x.shape[-1]) and bucket:
        return True
    # D512 outgoing: both required lengths passed the historical latency bar,
    # IEEE all-gradient/live-graph gates, and project-kernel racecheck.
    return (outgoing and x.shape[-1] == 512 and bucket
            and os.environ.get("MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS") != "0"
            and os.environ.get("MINIWORLD_TRIMUL_H100_PRESERVE_WIDE") != "0")


def preserve_training(width):
    return ((width == 128 and preserve_d128_training())
            or (width == 256 and preserve_d256_training())
            or preserve_wide_training(width))


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
        Path(
            os.environ.get(
                "engine_jit_root",
                str(Path.home() / ".cache/miniworld_engine_jit"),
            )
        )
        / "trimul_h100"
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


def compile(source, flags):
    nvcc, version = _compiler()
    key = hashlib.sha256(
        source.read_bytes() + repr(flags).encode() + version.encode() + _source_digest()
    ).hexdigest()
    out = cache_dir() / (key + ".cubin")
    with out.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists():
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


def cuda_aux_enabled():
    scoped = False if torch.compiler.is_compiling() else (
        getattr(_d256_scope, "enabled", False) or getattr(_d128_scope, "enabled", False))
    return (scoped
            or os.environ.get("MINIWORLD_TRIMUL_H100_CUDA_AUX") == "1")


def cuda_training_aux_enabled():
    """Explicit D64 training candidate; inference qualification stays independent."""
    return os.environ.get("MINIWORLD_TRIMUL_H100_TRAIN_CUDA") == "1"


def cuda_training_fused_prepare_enabled():
    return (cuda_training_aux_enabled()
            and os.environ.get("MINIWORLD_TRIMUL_H100_TRAIN_FUSED_PREP") == "1")


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _normalize_into_compiled(dst, x, g, b):
    dst.copy_(F.layer_norm(x.float(), (x.shape[-1],), g, b, 1e-5).to(x.dtype))


def normalize_into(dst, x, g, b, warp=False):
    if cuda_aux_enabled():
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
