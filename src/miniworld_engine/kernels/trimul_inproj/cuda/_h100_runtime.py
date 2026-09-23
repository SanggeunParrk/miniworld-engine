"""Portable compiler / driver support for the selected Anthropic-derived kernels.

All source reads are package-relative; binaries and temporary headers live in a
content-addressed user cache. Module handles are specific to a CUDA device.
"""

from pathlib import Path
from functools import lru_cache, wraps
import ctypes, fcntl, hashlib, os, shutil, subprocess
import torch
import torch.nn.functional as F

SOURCES = Path(__file__).with_name("h100_sources")


def _upstream():
    return SOURCES / "upstream"


def _launch_module():
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_launch

    return _h100_launch


def device_cache(fn):
    @lru_cache(None)
    def cached(device, args, kwargs):
        with torch.cuda.device(device):
            return fn(*args, **dict(kwargs))

    @wraps(fn)
    def call(*args, **kwargs):
        return cached(torch.cuda.current_device(), args, tuple(sorted(kwargs.items())))

    return call


def cache_dir():
    root = (
        Path(
            os.environ.get(
                "MINIWORLD_ENGINE_JIT_ROOT",
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


# A single packing operation, same interleaving as the measured training route.
# Keeping the copies in torch also makes live weights visible to CUDA graphs.
@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def pack_into(dst, wl, wlg, wr, wrg):
    d = wl.shape[-1]
    dst.copy_(
        torch.stack(
            (
                torch.cat((wlg, wrg)).reshape(-1, 32, d),
                torch.cat((wl, wr)).reshape(-1, 32, d),
            ),
            1,
        ).reshape(4 * wl.shape[0], d)
    )


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def normalize_into(dst, x, g, b):
    dst.copy_(F.layer_norm(x.float(), (x.shape[-1],), g, b, 1e-5).to(x.dtype))


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
