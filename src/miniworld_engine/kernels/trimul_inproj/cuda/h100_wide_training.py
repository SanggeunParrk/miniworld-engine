"""H100 bidirectional TriMul training for D=256/384/512 at L=384/768.

Flattened port of the qualified research selections (September 2026):
D256 ``d256_pool_checkpoint``, D384 ``wide_checkpoint23``, D512 ``wide_checkpoint24``
(L384 with a 4-way input-weight split). Every CUDA kernel is a frozen source under
``h100_sources/wide_train``; dense GEMMs are cuBLASLt (explicit algorithm selection,
see ``lt_selection.json``) or torch.mm/bmm exactly where the research plan used them.

Forward returns ``y`` and the activations the backward reads. Each call owns its
buffers: there is no cached activation, workspace or weight pack between calls.
B=1, BF16 pair and projection weights, FP32 LayerNorm affine, LN eps 1e-5, H=2D.
"""

from functools import lru_cache
import ctypes
import fcntl
import json
import os
import re
import shutil
import struct
import subprocess
import warnings
from pathlib import Path

import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

R = T.SOURCES / "wide_train"
pack_into, normalize_into = T.pack_into, T.normalize_into
WORKSPACE = 64 << 20
#: "config" matches the frozen algorithm (minus its runtime-version word) in the
#: current heuristic list; "index" reuses the frozen heuristic position (tests
#: comparing against the research plan under the same library); "heuristic" always
#: takes cuBLASLt's first recommendation.
LT_SELECTION = "config"


def supports(width, length):
    # Tuple membership compares values: it also works for symbolic sizes under Dynamo.
    return width in (256, 384, 512) and length in (384, 768)


# --------------------------------------------------------------------------- build
@lru_cache(None)
def forward_headers():
    """Inference headers with the wide-forward K1 occupancy/streaming overrides."""
    source = T.SOURCES / "inference"
    dest = T.cache_dir() / ("wide_train_headers_" + T._source_digest().hex()[:16])
    for name in ("tmn_kernels.cuh", "tmn_ptx.cuh", "common/tmn_math.cuh"):
        text = (source / name).read_text()
        if name == "tmn_kernels.cuh":
            for old, new in (
                ("static constexpr int MINB = NCWG == 1 ? 2 : 1;",
                 "static constexpr int MINB = MW_MINB;"),
                ("W_RESIDENT || NSLOT >= 2 * SPB",
                 "W_RESIDENT || (SCHED == 1 && (MW_K1_STREAM || NSLOT >= SPB)) || NSLOT >= 2 * SPB"),
            ):
                if text.count(old) != 1:
                    raise RuntimeError("wide training header anchor changed: " + old)
                text = text.replace(old, new)
        T.publish_header(dest / name, text)
    return dest


def _include(which):
    return str(T._upstream() / "csrc") if which == "upstream" else str(forward_headers())


def _defines(**values):
    return tuple(f"-D{k}={v}" for k, v in values.items())


@T.device_cache
def _unit(source, include, defines, pool=None):
    flags = ["-std=c++17", "-O3", "-arch=sm_90a", "--cubin", "-lineinfo", "-Xptxas=-v",
             "-I" + _include(include), *defines]
    cubin = T.compile(R / source, flags)
    if pool is not None:
        cubin = _register_pool(cubin, *pool)
    return T.load_unit(str(cubin), source)


@T.device_cache
def _kernel(source, entry, include, defines, smem=0, pool=None):
    k = _unit(source, include, defines, pool).kernel(entry)
    if smem:
        k.set_max_dynamic_smem(smem)
    if pool is not None and _occupancy(k, 256, smem) < 2:
        warnings.warn(f"{entry}: {_occupancy(k, 256, smem)} resident CTA(s); the lowered "
                      "register pool did not take effect", RuntimeWarning)
    return k


def _occupancy(k, threads, smem):
    drv = k.unit.drv
    return int(drv._unwrap(
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            drv.d.CUfunction(int(k.handle)), threads, smem)))


@T.device_cache
def _resident_grid(source, entry, include, defines, threads, smem):
    """Persistent grid: every SM times the kernel's measured residency."""
    k = _kernel(source, entry, include, defines, smem)
    occupancy = _occupancy(k, threads, smem)
    if occupancy < 1:
        raise RuntimeError(f"{entry}: no resident CTA with {smem} B shared memory")
    return _sms() * occupancy


@T.device_cache
def _sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _cuobjdump():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home

    ensure_cuda_home()
    tool = shutil.which("cuobjdump")
    if tool is None and os.environ.get("CUDA_HOME"):
        candidate = Path(os.environ["CUDA_HOME"]) / "bin/cuobjdump"
        tool = str(candidate) if candidate.exists() else None
    return tool


def _register_pool(cubin, kernel, initial, consumer):
    """Lower the initial CTA register pool of a setmaxnreg kernel (D256/L384 source).

    The kernel allocates dynamically: a 32-register producer and ``consumer``-register
    workers after ``setmaxnreg``. ptxas records ``consumer`` as the launch count, which
    admits one CTA per SM. Only the ``.nv.info`` launch count changes, to ``initial``;
    the machine code is unchanged. The patch is applied only when a conservative
    control-flow walk of the SASS shows every register operand below the live cap in
    each allocation state. Otherwise the original cubin is returned (correct, one
    resident CTA) with a warning.
    """
    out = cubin.with_name(cubin.stem + f".pool{initial}.cubin")
    if out.exists():
        return out
    try:
        tool = _cuobjdump()
        if tool is None:
            raise RuntimeError("cuobjdump not found")
        sass = subprocess.check_output([tool, "--dump-sass", str(cubin)], text=True)
        _check_register_pool(sass, kernel, initial, consumer)
        data = bytearray(cubin.read_bytes())
        _patch_register_count(data, consumer, initial)
    except Exception as error:  # noqa: BLE001 - any failure keeps the verified original
        warnings.warn(f"{kernel}: initial register pool not lowered ({error}); "
                      "running with one resident CTA per SM", RuntimeWarning)
        return cubin
    with out.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists():
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(bytes(data))
            tmp.replace(out)
    return out


def _check_register_pool(sass, kernel, initial, consumer):
    if sass.count("Function : ") != 1 or "Function : " + kernel not in sass:
        raise RuntimeError("unexpected kernel set")
    inst = {}
    for line in sass.splitlines():
        m = re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;", line)
        if m:
            inst[int(m[1], 16)] = m[2]
    if sum("TRY_ALLOC.CTAPOOL" in x for x in inst.values()) != 1 or \
            sum("DEALLOC.CTAPOOL" in x for x in inst.values()) != 1:
        raise RuntimeError("expected one CTA-pool allocation and release")
    queue, seen, caps = [(0, initial)], set(), set()
    while queue:
        pc, cap = queue.pop()
        if (pc, cap) in seen:
            continue
        seen.add((pc, cap))
        op = inst[pc]
        caps.add(cap)
        if any(int(r) >= cap for r in re.findall(r"\bR(\d+)\b", op)):
            raise RuntimeError(f"register above cap {cap} at {pc:#x}: {op}")
        if "USETMAXREG" in op:
            cap = int(re.search(r"0x([0-9a-f]+)$", op)[1], 16)
        if any(s in op for s in ("BRX", "JMX", "CALL", "RET ")):
            raise RuntimeError(f"indirect control flow at {pc:#x}")
        branch = re.search(r"\bBRA\s+0x([0-9a-f]+)", op)
        if branch:
            queue.append((int(branch[1], 16), cap))
            if not op.startswith("@"):
                continue
        if "EXIT" in op and not op.startswith("@"):
            continue
        if pc + 16 in inst:
            queue.append((pc + 16, cap))
    if 32 not in caps or consumer not in caps or initial * 256 < 128 * (32 + consumer):
        raise RuntimeError(f"unexpected allocation states {sorted(caps)}")


def _patch_register_count(data, old, new):
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    if header[0][:6] != b"\x7fELF\x02\x01":
        raise RuntimeError("not a 64-bit ELF cubin")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * header[11])
                for i in range(header[12])]
    names = sections[header[13]]
    changed = 0
    for s in sections:
        name = bytes(data[names[4] + s[0]:]).split(b"\0", 1)[0]
        if name != b".nv.info":
            continue
        pos, end = s[4], s[4] + s[5]
        while pos < end:
            fmt, kind, size = struct.unpack_from("<BBH", data, pos)
            if fmt != 4:
                raise RuntimeError("unexpected .nv.info attribute format")
            if kind == 0x2F:  # EIATTR_REGCOUNT
                _, count = struct.unpack_from("<II", data, pos + 4)
                if count != old:
                    raise RuntimeError(f"register count {count}, expected {old}")
                struct.pack_into("<I", data, pos + 8, new)
                changed += 1
            pos += 4 + size
    if changed != 1:
        raise RuntimeError(f"{changed} register-count attributes")


# --------------------------------------------------------------------------- launch
def _map(t, box, dims, strides, swizzle="128B", l2="128B"):
    return T._launch_module().tensor_map(t, box, dims=dims, strides_bytes=strides,
                                         swizzle=swizzle, l2=l2)


def _rows(t, cols, rows, box_rows=64):
    """Row-major [rows, cols] BF16 operand, [64, box_rows] boxes."""
    return _map(t, [64, box_rows], [cols, rows], [cols * 2])


def _params(key, fields):
    return T._launch_module().Struct.fixed("h100_wide:" + key, fields)


def _launch(k, grid, threads, smem, params):
    k.launch(grid if isinstance(grid, tuple) else (grid, 1, 1), (threads, 1, 1), [params], smem)


def _launch_cooperative(k, grid, threads, smem, params):
    L = T._launch_module()
    drv = k.unit.drv
    args = L._Packed([params])
    drv._unwrap("cuLaunchCooperativeKernel", drv.d.cuLaunchCooperativeKernel(
        drv.d.CUfunction(int(k.handle)), grid, 1, 1, threads, 1, 1, smem,
        drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),
        ctypes.addressof(args.array)))


@T.device_cache
def _side_stream():
    return torch.cuda.Stream()


# --------------------------------------------------------------------------- cuBLASLt
class _Algorithm(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heuristic(ctypes.Structure):
    _fields_ = [("algo", _Algorithm), ("workspaceSize", ctypes.c_size_t),
                ("state", ctypes.c_int), ("wavesCount", ctypes.c_float),
                ("reserved", ctypes.c_int * 4)]


@lru_cache(None)
def _lt_library():
    # The torch wheel has already loaded libcublasLt; the soname resolves to that
    # process-wide copy, so the selection below is made for the library in use.
    lib = ctypes.CDLL("libcublasLt.so.12")
    P, I, Z = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    for name, args in {
        "cublasLtCreate": [ctypes.POINTER(P)],
        "cublasLtMatmulDescCreate": [ctypes.POINTER(P), I, I],
        "cublasLtMatrixLayoutCreate": [ctypes.POINTER(P), I, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64],
        "cublasLtMatrixLayoutSetAttribute": [P, I, P, Z],
        "cublasLtMatmulPreferenceCreate": [ctypes.POINTER(P)],
        "cublasLtMatmulPreferenceSetAttribute": [P, I, P, Z],
        "cublasLtMatmulAlgoGetHeuristic": [P, P, P, P, P, P, P, I, ctypes.POINTER(_Heuristic), ctypes.POINTER(I)],
        "cublasLtMatmulAlgoCheck": [P, P, P, P, P, P, ctypes.POINTER(_Algorithm), ctypes.POINTER(_Heuristic)],
        "cublasLtMatmul": [P, P, P, P, P, P, P, P, P, P, P, P, ctypes.POINTER(_Algorithm), P, Z, P],
        "cublasLtGetVersion": [],
    }.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_size_t if name == "cublasLtGetVersion" else I
    return lib


def _lt_check(status, call):
    if status:
        raise RuntimeError(f"{call}: cuBLASLt status {status}")


@T.device_cache
def _lt_handle():
    handle = ctypes.c_void_p()
    _lt_check(_lt_library().cublasLtCreate(ctypes.byref(handle)), "cublasLtCreate")
    return handle


def _layout_key(t):
    return tuple(t.shape), tuple(t.stride()), t.dtype


class _LtProblem:
    """Descriptors and heuristics of one strided-batch GEMM layout (no pointers)."""

    def __init__(self, a, b, out):
        lib = _lt_library()
        P = ctypes.c_void_p
        self.desc = P()
        # CUBLAS_COMPUTE_32F (68), FP32 alpha/beta; no transposes: layouts carry order.
        _lt_check(lib.cublasLtMatmulDescCreate(ctypes.byref(self.desc), 68, 0), "MatmulDescCreate")
        self.layouts = []
        for t in (a, b, out):
            row = t.stride(2) == 1
            if not row and t.stride(1) != 1:
                raise ValueError("cuBLASLt operand needs one unit-stride matrix dimension")
            layout = P()
            _lt_check(lib.cublasLtMatrixLayoutCreate(
                ctypes.byref(layout), 14 if t.dtype == torch.bfloat16 else 0,
                t.shape[1], t.shape[2], t.stride(1) if row else t.stride(2)), "MatrixLayoutCreate")
            for attr, value in ((1, ctypes.c_int(1 if row else 0)),  # ORDER
                                (5, ctypes.c_int(t.shape[0])),  # BATCH_COUNT
                                (6, ctypes.c_int64(t.stride(0)))):  # STRIDED_BATCH_OFFSET
                _lt_check(lib.cublasLtMatrixLayoutSetAttribute(
                    layout, attr, ctypes.byref(value), ctypes.sizeof(value)), "LayoutSetAttribute")
            self.layouts.append(layout)
        preference = P()
        _lt_check(lib.cublasLtMatmulPreferenceCreate(ctypes.byref(preference)), "PreferenceCreate")
        size = ctypes.c_uint64(WORKSPACE)
        _lt_check(lib.cublasLtMatmulPreferenceSetAttribute(
            preference, 1, ctypes.byref(size), ctypes.sizeof(size)), "PreferenceSetAttribute")
        self.heuristics = (_Heuristic * 16)()
        count = ctypes.c_int()
        ad, bd, dd = self.layouts
        _lt_check(lib.cublasLtMatmulAlgoGetHeuristic(
            _lt_handle(), self.desc, ad, bd, dd, dd, preference, 16, self.heuristics,
            ctypes.byref(count)), "AlgoGetHeuristic")
        self.indices = [i for i in range(count.value) if self.heuristics[i].state == 0]
        if not self.indices:
            raise RuntimeError("no supported cuBLASLt algorithm")
        self.chosen = {}

    def algorithm(self, name, frozen, donor=None):
        """Selected algorithm: frozen configuration, validated for this runtime."""
        key = (LT_SELECTION, name, None if frozen is None else tuple(frozen[1]),
               None if donor is None else tuple(donor.data))
        if key not in self.chosen:
            self.chosen[key] = self._select(name, frozen, donor)
        return self.chosen[key]

    def _select(self, name, frozen, donor):
        first = self.heuristics[self.indices[0]].algo
        if donor is not None and LT_SELECTION != "heuristic":
            # Reuse the full-size projection's algorithm for its row-compacted replay.
            algo = _Algorithm()
            ctypes.memmove(ctypes.byref(algo), ctypes.byref(donor), ctypes.sizeof(_Algorithm))
            if self._supported(algo):
                return algo
            _warn_once(name, "donor algorithm rejected by cublasLtMatmulAlgoCheck")
            return first
        if frozen is None or LT_SELECTION == "heuristic":
            return first
        index, words = frozen
        if LT_SELECTION == "index":
            return self.heuristics[index].algo if index in self.indices else first
        for i in self.indices:
            data = list(self.heuristics[i].algo.data)
            if data[:4] == words[:4] and data[5:] == words[5:]:
                return self.heuristics[i].algo
        _warn_once(name, "frozen configuration not offered by this cuBLASLt; using its first heuristic")
        return first

    def _supported(self, algo):
        result = _Heuristic()
        ad, bd, dd = self.layouts
        return _lt_library().cublasLtMatmulAlgoCheck(
            _lt_handle(), self.desc, ad, bd, dd, dd, ctypes.byref(algo), ctypes.byref(result)) == 0 \
            and result.workspaceSize <= WORKSPACE


class _Shape:
    """Layout-only stand-in so cached descriptors never retain tensors."""

    def __init__(self, shape, strides, dtype):
        self.shape, self._strides, self.dtype = shape, strides, dtype

    def stride(self, i):
        return self._strides[i]


@T.device_cache
def _lt_problem(a, b, out):
    """Per-device descriptors/heuristics keyed by (shape, strides, dtype) of each operand."""
    return _LtProblem(*(_Shape(*key) for key in (a, b, out)))


_WARNED = set()


def _warn_once(name, message):
    if name not in _WARNED:
        _WARNED.add(name)
        warnings.warn(f"h100_wide_training {name}: {message}", RuntimeWarning)


@lru_cache(None)
def _frozen():
    return json.loads((R / "lt_selection.json").read_text())


def _lt(cell, name, a, b, out, workspace, donor=None):
    """out = a @ b (per batch) with the cell's selected algorithm; returns it."""
    a, b, out = (t.unsqueeze(0) if t.ndim == 2 else t for t in (a, b, out))
    problem = _lt_problem(*(_layout_key(t) for t in (a, b, out)))
    frozen = _frozen()[f"{cell[0]}-{cell[1]}"].get(name)
    algo = problem.algorithm(f"{cell[0]}-{cell[1]} {name}", frozen, donor)
    one, zero = ctypes.c_float(1), ctypes.c_float(0)
    ad, bd, dd = problem.layouts
    _lt_check(_lt_library().cublasLtMatmul(
        _lt_handle(), problem.desc, ctypes.byref(one), a.data_ptr(), ad, b.data_ptr(), bd,
        ctypes.byref(zero), out.data_ptr(), dd, out.data_ptr(), dd, ctypes.byref(algo),
        workspace.data_ptr(), workspace.numel(), torch.cuda.current_stream().cuda_stream),
        "cublasLtMatmul " + name)
    return algo


def _workspace(x):
    return torch.empty(WORKSPACE, device=x.device, dtype=torch.uint8)


# --------------------------------------------------------------------------- forward
def saved_like(x):
    """Empty tensors shaped like ``forward``'s saved list (fake/meta support).

    ab, tri, x_n, output-LN normalized rows, projection and gate products, output-LN
    mean and rstd; then D256 packed input weights, or D384/D512 channel-major
    projection preactivations (plus D512 packed-normalization patch records).
    """
    n, D = x.shape[1], x.shape[-1]
    M, H = n * n, 2 * D
    f32 = dict(dtype=torch.float32)
    saved = [x.new_empty((2 * H, n, n)), x.new_empty((H, n, n)), torch.empty_like(x),
             x.new_empty((M, H)), x.new_empty((M, D)), x.new_empty((M, D)),
             x.new_empty((M,), **f32), x.new_empty((M,), **f32)]
    if D == 256:
        return saved + [x.new_empty((8 * D, D))]
    saved.append(x.new_empty((8 * D, M)))
    if D == 512:
        saved += [x.new_empty((max(1, M * H // 4096),), dtype=torch.int64),
                  x.new_empty((1,), dtype=torch.int32), x.new_empty((M // 64,), dtype=torch.int32)]
    return saved


def _front_params(x, operand, w1, ab, mask, gi, bi, stats, xn, cfg, pre=None):
    """K1 parameters (engine ``K1Params``), optionally with the saved-preactivation map."""
    n, D = x.shape[1], x.shape[-1]
    a, b = cfg[0], cfg[1]
    tj = (n + b - 1) // b
    tiles = ((n + a - 1) // a) * tj
    fields = [
        _map(operand, [64, b, a], [D, n, n], [D * 2, n * D * 2]),
        _map(w1, [64, 64], [D, 8 * D], [D * 2]),
        _map(ab, [64, 1, 32], [n, n, 4 * D], [n * 2, n * n * 2]),
        mask, gi, bi, ab, stats, xn,
        n, n, tj, tiles, 1, n, 1, 1e-5, n * D, D, 0, 0,
    ]
    if pre is not None:
        fields.insert(9, _map(pre, [64, 64], [pre.shape[1], pre.shape[0]], [pre.shape[1] * 2]))
    return fields, tiles, 128 * (a * b // 64 + 1)


def _k1_smem(D, cfg, stage_extra=0):
    a, b, slots, sk = cfg[:4]
    return (a * b * D * 2 + slots * sk * 8192 + (a * b // 64) * 8192 + 8 * D
            + ((2 + 2 * slots) * 8 + 127) // 128 * 128 + stage_extra)


# K1 tiles (BI, BJ, slots, K steps, min blocks[, shared/stream]) of the qualified plans.
_K1 = {256: (2, 64, 4, 4, 1), 384: (2, 64, 2, 6, 1), 512: (2, 64, 4, 2, 1, 2)}


def _front(x, w1, ab, mask, gi, bi, xn, stats, pre):
    D = x.shape[-1]
    cfg = _K1[D]
    fw = "forward"
    common = _defines(MW_MINB=cfg[4], MW_K1_STREAM=int(len(cfg) > 5 and cfg[5] == 2),
                      TMN_SIGMOID_TANH=1, TMN_WSKIP=1, TMN_MASK_TEMPLATE=1)
    if D == 256:
        smem = _k1_smem(D, cfg)
        k = _kernel("front_d256.cu", "mw_d256_front_save_stats", fw, common, smem)
        fields, tiles, threads = _front_params(x, x, w1, ab, mask, gi, bi, stats, xn, cfg)
    else:
        # Saved channel-major preactivations use a separate 4 KB staging slot per consumer.
        smem = _k1_smem(D, cfg, cfg[0] * cfg[1] // 64 * 4096)
        source = "front_pre_d384.cu" if D == 384 else "front_pre_d512.cu"
        k = _kernel(source, "mw_wide_front_pre_overlap", fw, common, smem)
        normalize = D != 512
        if not normalize:
            normalize_into(xn, x, gi, bi)
        fields, tiles, threads = _front_params(
            x, x if normalize else xn, w1, ab, mask, gi, bi, None, xn if normalize else None,
            cfg, pre)
    _launch(k, min(tiles, _sms() * cfg[4]), threads, smem, _params(f"front{D}", fields))


def _triangle(cell, ab, tri, workspace):
    """tri = [outgoing | incoming] contraction of the saved left/right planes."""
    D = cell[0]
    H = 2 * D
    frozen = _frozen()[f"{D}-{cell[1]}"]
    ops = (("fw0", ab[:D], ab[H:H + D].transpose(-1, -2), tri[:D]),
           ("fw1", ab[D:H].transpose(-1, -2), ab[H + D:], tri[D:]))
    for name, a, b, out in ops:
        if name in frozen:
            _lt(cell, name, a, b, out, workspace)
        else:
            torch.bmm(a, b, out=out)


def _output_norm(cell, tri, norm, go, bo, mu, rs, delta):
    D, n = cell
    M, H = n * n, 2 * D
    tri_rows = lambda rows, swizzle: _map(tri, [rows, 64], [M, H], [M * 2], swizzle)
    fw = "forward"
    if D == 256:
        defines = _defines(MW_MINB=1, MW_K1_STREAM=0, TMN_SIGMOID_TANH=1, WIDTH=D, GROUPS=1, KCHUNK=1)
        smem = 128 * H + 128
        k = _kernel("output_dense.cu", "mw_wide_dense_norm", fw, defines, smem)
        grid = _resident_grid("output_dense.cu", "mw_wide_dense_norm", fw, defines, 128, smem)
        fields = [_map(tri, [64, 64], [M, H], [M * 2]), _rows(norm, H, M), go, bo, mu, rs, M]
        _launch(k, grid, 128, smem, _params("dense_norm", fields))
        return
    defines = _defines(MW_MINB=1, MW_K1_STREAM=0, WIDTH=D, OUTPUT_ROWS=32, OUTPUT_THREADS=128)
    maps = [tri_rows(32, "64B"), _map(norm, [64, 32], [H, M], [H * 2])]
    smem = 32 * H * 2 + 128 + H * 8
    if D == 384:
        k = _kernel("forward_norm_d384.cu", "mw_wide_cached_forward_norm", fw, defines, smem)
        grid = _resident_grid("forward_norm_d384.cu", "mw_wide_cached_forward_norm", fw, defines, 128, smem)
        _launch(k, grid, 128, smem, _params("forward_norm", [*maps, go, bo, mu, rs, M]))
        return
    # D512: packed normalization, recording rows where it differs from the scalar form.
    patches, count, changed = delta
    count.zero_()
    changed.zero_()
    k = _kernel("delta_norm_d512.cu", "mw_wide_lowreg_delta_norm", fw, defines, smem)
    grid = _resident_grid("delta_norm_d512.cu", "mw_wide_lowreg_delta_norm", fw, defines, 128, smem)
    _launch(k, grid, 128, smem, _params("delta_norm", [
        *maps, patches, count, patches.numel(), changed, go, bo, mu, rs, M]))


def _pair_mask(mask, n):
    """BF16 [n, n] pair mask read by K1 and the contraction kernels (no copy if already so)."""
    return mask.reshape(n, n).to(torch.bfloat16).contiguous()


def forward(leaves, mask, ds):
    """Return ``[y, *saved]``; ``saved`` matches :func:`saved_like`."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n, D = x.shape[1], x.shape[-1]
    cell = (D, n)
    if not supports(D, n):
        raise ValueError(f"wide TriMul training covers D256/384/512 at L384/768, not {cell}")
    M, H = n * n, 2 * D
    mask = _pair_mask(mask, n)
    saved = saved_like(x)
    ab, tri, xn, norm, proj, gate, mu, rs = saved[:8]
    w1 = saved[8] if D == 256 else x.new_empty((8 * D, D))
    pack_into(w1, wl, wlg, wr, wrg)
    stats = x.new_empty((M, 2), dtype=torch.float32) if D == 256 else None
    _front(x, w1, ab, mask, gi, bi, xn, stats, None if D == 256 else saved[8])
    workspace = _workspace(x)
    _triangle(cell, ab, tri, workspace)
    _output_norm(cell, tri, norm, go, bo, mu, rs, saved[9:] if D == 512 else None)
    _lt(cell, "proj", norm, wp.t(), proj, workspace)
    _lt(cell, "gate", xn.reshape(M, D), wg.t(), gate, workspace)
    y = torch.empty_like(x)
    defines = _defines(MW_MINB=1, MW_K1_STREAM=0, TMN_SIGMOID_TANH=1, WIDTH=D, GROUPS=1, KCHUNK=1)
    k = _kernel("output_dense.cu", "mw_wide_dense_output_epi", "forward", defines)
    _launch(k, 1056, 256, 0, _params("epilogue", [proj, gate, x, ds, y, M * D, n * D]))
    return [y, *saved]


# --------------------------------------------------------------------------- backward
def _gate(cell, proj, gate, dy, ds, dp, prefix, affine):
    """dp = dy*ds*sigmoid(gate); gate gradient into the dX GEMM prefix (+ zero LN sums)."""
    D, n = cell
    M = n * n
    defines = _defines(MW_MINB=1, MW_K1_STREAM=0, WIDTH=D)
    if D != 512:
        defines += _defines(TMN_SIGMOID_TANH=1)
    smem = 49152 + 128
    fields = [_rows(proj, D, M), _rows(gate, D, M), _rows(dy, D, M), _rows(ds, D, n),
              _rows(dp, D, M), _map(prefix, [64, 64], [M, D], [M * 2]), M, n]
    if affine is None:
        k = _kernel("gate.cu", "mw_prefix_gate_epi", "forward", defines, smem)
        key = "gate"
    else:
        k = _kernel("gate_affine.cu", "mw_early_both_affine_gate", "forward", defines, smem)
        fields += affine
        key = "gate_affine"
    _launch(k, _sms() * 4, 128, smem, _params(key, fields))


def _output_ln(cell, tri, dt, dn, mu, rs, go, dgo, dbo):
    """Output-LN backward: dt from dn; dgamma/dbeta sums. L384 kernels do not zero them."""
    D, n = cell
    M, H = n * n, 2 * D
    rows = 32 if D == 256 or (D, n) == (384, 768) else 16
    minblocks = 3 if D == 256 else 2
    defines = _defines(MW_MINB=1, MW_K1_STREAM=0, WIDTH=D, LN_ROWS=rows, LN_DN_TMA=1, LN_FENCE=0,
                       LN_MINBLOCKS=minblocks)
    source = f"output_ln_d{D}_l{n}.cu"
    tail = [dn, mu, rs, go, dgo, dbo, M]
    if n == 384:
        # One 3D map per tensor: [M, 64 channels, H/64 groups] for tri/dt, rows for dn.
        swizzle = "64B" if rows == 32 else "32B"
        maps = [_map(t, [rows, 64, H // 64], [M, 64, H // 64], [M * 2, M * 128], swizzle)
                for t in (tri, dt)]
        maps.append(_map(dn, [64, rows, H // 64], [64, M, H // 64], [H * 2, 128]))
        smem = {256: 67968, 384: 101760, 512: 70528}[D]
        k = _kernel(source, "mw_independent_output_ln", "upstream", defines, smem)
        # Qualified grids (SMs x the replaced cooperative kernel's residency, D256 doubled).
        _launch(k, _sms() * {256: 6, 384: 2, 512: 3}[D], 128, smem, _params(source, [*maps, *tail]))
        return
    swizzle = "64B" if rows == 32 else "32B"
    maps = [_map(t, [rows, 64], [M, H], [M * 2], swizzle) for t in (tri, dt)]
    maps.append(_map(dn, [64, rows], [H, M], [H * 2]))
    threads = 256 if D == 512 else 128
    buffers = 3 if D == 512 else 2
    smem = buffers * rows * H * 2 + 128 + H * 4 + rows * 8
    k = _kernel(source, "mw_wide_stats_ln", "upstream", defines, smem)
    grid = _resident_grid(source, "mw_wide_stats_ln", "upstream", defines, threads, smem)
    _launch_cooperative(k, grid, threads, smem, _params(source, [*maps, *tail]))


def _contract_gp(cell, dt, ab, pre, mask, gp):
    """Fused incoming/outgoing contraction and gate/projection gradient (D384/D512)."""
    D, n = cell
    M, H = n * n, 2 * D
    defines = _defines(WIDTH=D)
    if n == 768:
        source, entry = "contract_gp_l768.cu", "mw_wide_fixed_length_gp"
    elif D == 384:
        source, entry = "contract_gp_d384_l384.cu", "mw_wide_staged_epilogue_gp"
    else:
        source, entry = "contract_gp_d512_l384.cu", "mw_wide_loader_warp_gp"
    threads = 288 if entry == "mw_wide_loader_warp_gp" else 256
    smem = 98304 + 128
    k = _kernel(source, entry, "upstream", defines, smem)
    if entry == "mw_wide_loader_warp_gp":
        # Whole 4D maps: [64 cols, 64 rows, n/64, channels*n/64] for every operand.
        def whole(t, channels, box):
            return _map(t, box, [64, 64, n // 64, channels * n // 64], [n * 2, 128, n * 128])
        # A operands: only A1 (transposed dt) takes 2x1 boxes; B operands: all but B2.
        maps = [whole(t, D, (64, 64, 2, 1) if i == 1 else (64, 64, 1, 2)) for i, t in enumerate(
            (dt[:D], dt[:D], ab[H + D:], ab[D:H]))] + [
            whole(t, D, (64, 64, 2, 1) if i != 2 else (64, 64, 1, 2)) for i, t in enumerate(
                (ab[H:H + D], ab[:D], dt[D:], dt[D:]))]
        tail = [whole(pre, 8 * D, (64, 64, 2, 2)), whole(mask, 1, (64, 64, 2, 2)),
                *[whole(g, H, (64, 64, 2, 2)) for g in gp]]
    else:
        def plane(t, channels):
            return _map(t, [64, 64], [n, channels * n], [n * 2])
        maps = [plane(t, D) for t in (dt[:D], dt[:D], ab[H + D:], ab[D:H],
                                      ab[H:H + D], ab[:D], dt[D:], dt[D:])]
        tail = [plane(pre, 8 * D), plane(mask, 1), *[plane(g, H) for g in gp]]
    # Pointer fields of this parameter block are not read by these kernels.
    fields = [*maps, pre, mask, *gp, None, None, *tail, n]
    _launch(k, 4 * D * (n // 128) ** 2, threads, smem, _params(source, fields))


def _contract_d256(cell, dt, ab, dl, dr):
    """D256/L384 native N384 contraction tile with ordered K accumulation."""
    D, n = cell
    H = 2 * D
    defines = _defines(ROW_GROUPS=2, GRID_ORDER=1, MIN_BLOCKS=1, CONTRACT_SLOTS=3)
    smem = (2 + 6) * 8192 * 3 + 128
    k = _kernel("contract_d256_l384.cu", "mw_d256_full_spatial_contract", "upstream", defines, smem)

    def operand(t, transposed, count):
        if transposed:
            return _map(t, [64, 64, count], [64, n * D, n // 64], [n * 2, 128], l2="256B")
        return _map(t, [64, 64, count], [n, 64, n * D // 64], [n * 2, n * 128], l2="256B")
    a = (dt[:D], dt[:D], ab[H + D:], ab[D:H])
    b = (ab[H:H + D], ab[:D], dt[D:], dt[D:])
    out = lambda t: _map(t, [64, 64, 6], [64, n * H, n // 64], [n * 2, 128])
    fields = [*[operand(t, i == 1, 2) for i, t in enumerate(a)],
              *[operand(t, i != 2, 6) for i, t in enumerate(b)], out(dl), out(dr)]
    _launch(k, 4 * 256 * (n // 128), 256, smem, _params("contract_d256", fields))


def _source_d256(cell, xn, w1, dl, dr, gp, mask, partial):
    """D256 gate/projection derivatives from dl/dr and split-K input-weight partials."""
    D, n = cell
    M, H = n * n, 2 * D
    splits = 8 if n == 384 else 16
    defines = _defines(WEIGHT_SPLITS=splits)
    smem = 114816 + 256
    if n == 384:
        # 208-register workers, 32-register producer: pool of 120 admits two CTAs.
        k = _kernel("source_d256_l384.cu", "mw_d256_register_budget_source", "upstream", defines,
                    smem, pool=("mw_d256_register_budget_source", 120, 208))
    else:
        k = _kernel("source_d256_l768.cu", "mw_d256_mask_source", "upstream", defines, smem)
    rows = lambda t: _map(t, [64, 32], [M, H], [M * 2])
    fields = [_rows(xn.reshape(M, D), D, M), _rows(w1, D, 8 * D), rows(dl), rows(dr),
              *[rows(g) for g in gp], mask, *gp, partial, M]
    _launch(k, 32 * splits, 256, smem, _params(f"source_d256_{n}", fields))


def _input_ln(cell, x, dxn, dy, dx, gi, dgi, dbi, partial, dw, dwg, cooperative, master=False):
    """Input-LN backward and residual into dx; reduce input (and gate) weight partials."""
    D, n = cell
    M = n * n
    defines = _defines(WIDTH=D, INPUT_ROWS=16, INPUT_THREADS=128, INPUT_MINBLOCKS=4, MASTER_FP32=int(master))
    smem = max(3 * 16 * D * 2 + 128, 2 * 4 * D * 4) + D * 4
    maps = [_map(t, [64, 16], [D, M], [D * 2]) for t in (x, dxn, dy, dx)]
    fields = [*maps, gi, dgi, dbi, partial, *dw]
    if dwg is not None:
        fields.append(dwg)
    fields.append(M)
    if n == 384:
        source, entry = f"input_ln_d{D}_l384.cu", "mw_independent_input_ln"
    elif D == 512:
        source, entry = "input_reduce_d512_l768.cu", "mw_wide_joint_input_reduce"
    else:
        source, entry = "input_reduce_l768.cu", "mw_wide_cached_input"
    k = _kernel(source, entry, "upstream", defines, smem)
    if cooperative:
        grid = _resident_grid(source, entry, "upstream", defines, 128, smem)
        _launch_cooperative(k, grid, 128, smem, _params(source, fields))
    else:
        # Qualified grid of the cooperative reduction this independent kernel replaced.
        _launch(k, _sms() * {256: 6, 384: 5, 512: 4}[D], 128, smem, _params(source, fields))


def _split_input_dw(cell, dxin, xn, partial, workspace, joint):
    """Split-K FP32 input (and, if ``joint``, output-gate) weight partials."""
    D, n = cell
    M = n * n
    splits = {(384, 384): 8, (384, 768): 16, (512, 384): 4, (512, 768): 16}[cell]
    step = M // splits
    rows = 9 * D if joint else 8 * D
    first = 0 if joint else D
    a = dxin[first:].as_strided((splits, rows, step), (step, M, 1))
    b = xn.as_strided((splits, step, D), (step * D, D, 1))
    out = partial.reshape(-1)[(2 if joint else 3) * D * D:].as_strided(
        (splits, rows, D), (11 * D * D, D, 1))
    _lt(cell, "input_dw", a, b, out, workspace)


def _delta_projection(cell, saved, wp, proj, go, bo, workspace, donor):
    """D512: patch packed-normalization rows and re-project exactly the changed rows."""
    D, n = cell
    M, H = n * n, 2 * D
    tri, norm, mu, rs = saved[1], saved[3], saved[6], saved[7]
    patches, count, changed = saved[9:12]
    capacity = patches.numel()
    fw = "forward"
    defines = _defines(MW_MINB=1, MW_K1_STREAM=0, WIDTH=D, OUTPUT_ROWS=32, OUTPUT_THREADS=128)
    norm_smem = 32 * H * 2 + 128 + H * 8
    grid = _resident_grid("delta_norm_d512.cu", "mw_wide_lowreg_delta_norm", fw, defines, 128, norm_smem)
    apply = _kernel("delta_norm_d512.cu", "mw_wide_apply_delta_norm", fw, defines)
    _launch(apply, grid, 256, 0, _params("delta_apply", [patches, count, capacity, norm]))
    smem = 32 * H * 2 + 128
    fallback = _kernel("delta_norm_fallback_d512.cu", "mw_wide_lowreg_delta_norm_fallback", fw,
                       defines, smem)
    maps = [_map(tri, [32, 64], [M, H], [M * 2], "64B"), _map(norm, [64, 32], [H, M], [H * 2])]
    _launch(fallback, grid, 128, smem, _params("delta_fallback", [
        *maps, go, bo, mu, rs, M, count, capacity]))
    rows_capacity = M // 64
    seen = torch.zeros(M + 1, device=norm.device, dtype=torch.int32)
    rows = torch.empty(rows_capacity, device=norm.device, dtype=torch.int32)
    compact = norm.new_empty((rows_capacity, H))
    projected = norm.new_empty((rows_capacity, D))
    defines = _defines(MW_MINB=1, MW_K1_STREAM=0, WIDTH=D)
    params = _params("compact", [patches, count, capacity, seen, seen[M:], rows, rows_capacity,
                                 norm, compact, projected, proj])
    for entry in ("mw_wide_compact_changed_rows", "mw_wide_gather_changed_norm"):
        _launch(_kernel("compact_projection_d512.cu", entry, fw, defines), _sms() * 4, 256, 0, params)
    _lt(cell, "compact", compact, wp.t(), projected, workspace, donor=donor)
    _launch(_kernel("compact_projection_d512.cu", "mw_wide_scatter_changed_proj", fw, defines),
            _sms() * 4, 256, 0, params)
    smem = 49152 + 128
    k = _kernel("compact_projection_fallback_d512.cu", "mw_wide_compact_projection_fallback", fw,
                defines, smem)
    fields = [_rows(norm, H, M), _rows(wp, H, D), _rows(proj, D, M), changed, M, count, capacity,
              seen[M:], rows_capacity]
    _launch(k, (M // 64, D // 128, 1), 128, smem, _params("compact_fallback", fields))


def _weight_reduce_d512(cell, partial, dwp):
    """Reduce the 32 FP32 dWproj splits (widths.cu parameter block; only f[7], t[21] read)."""
    D, n = cell
    L = T._launch_module()
    empty = L.TensorMap(bytes(128))
    tensors = [None] * 24
    tensors[21] = dwp
    floats = [None] * 13
    floats[7] = partial
    k = _kernel("weight_reduce_d512_l768.cu", "mw_wide512_dwp_reduce", "upstream",
                _defines(MASTER_FP32=int(dwp.dtype == torch.float32), WIDTH=D, WEIGHT_SPLITS=32))
    _launch(k, _sms() * 2, 256, 0, _params("weight_reduce", [
        *([empty] * 16), *tensors, *floats, n * n, n]))


def backward(leaves, mask, ds, saved, dy, master=False):
    """Return the eleven leaf gradients (dx, dW*, dLN affine) in ``leaves`` order."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n, D = x.shape[1], x.shape[-1]
    cell = (D, n)
    M, H = n * n, 2 * D
    mask = _pair_mask(mask, n)
    ds = ds.reshape(n, D)
    dy = dy.contiguous()
    ab, tri, xn, norm, proj, gate, mu, rs = saved[:8]
    f32 = dict(device=x.device, dtype=torch.float32)
    dx = torch.empty_like(x)
    dp = x.new_empty((M, D))
    dn = x.new_empty((M, H))
    dt = torch.empty_like(tri)
    dxin = x.new_empty((9 * D, M))  # [gate-gradient prefix; four gate/projection gradients]
    gp = list(dxin[D:].view(4, H, M).unbind())
    dxn = x.new_empty((M, D))
    partial = torch.empty((32, 11 * D * D), **f32)
    dgi, dbi = torch.empty(D, **f32), torch.empty(D, **f32)
    dgo, dbo = torch.empty(H, **f32), torch.empty(H, **f32)
    dw = [torch.empty_like(w, dtype=torch.float32 if master else w.dtype) for w in (wl, wlg, wr, wrg)]
    dwg, dwp = (torch.empty_like(w, dtype=torch.float32 if master else w.dtype) for w in (wg,wp))
    wcat = x.new_empty((9 * D, D))
    workspace = _workspace(x)
    main = torch.cuda.current_stream()
    side = side_workspace = None

    if D == 512:
        donor = _lt_problem(*(_layout_key(t.unsqueeze(0)) for t in (norm, wp.t(), proj))).algorithm(
            f"{D}-{n} proj", _frozen()[f"{D}-{n}"].get("proj"))
        _delta_projection(cell, saved, wp, proj, go, bo, workspace, donor)
    affine = [dgo, dbo, dgi, dbi] if n == 384 else None
    _gate(cell, proj, gate, dy, ds, dp, dxin[:D], affine)
    _lt(cell, "dn", dp, wp, dn, workspace)
    if D == 256 and n == 384:
        _output_ln(cell, tri, dt, dn, mu, rs, go, dgo, dbo)
        dl, dr = torch.empty_like(tri), torch.empty_like(tri)
        _contract_d256(cell, dt, ab, dl, dr)
        # Output weight gradients overlap the source kernel on a side stream.
        side = _side_stream()
        side.wait_stream(main)
        side_workspace = _workspace(x)
        with torch.cuda.stream(side):
            _lt(cell, "dwp", dp.t(), norm, dwp, side_workspace)
            _lt(cell, "dwg", dxin[:D], xn.reshape(M, D), dwg, side_workspace)
        _source_d256(cell, xn, saved[8], dl, dr, gp, mask, partial)
    elif D == 256:
        _lt(cell, "dwp", dp.t(), norm, dwp, workspace)
        _lt(cell, "dwg", dxin[:D], xn.reshape(M, D), dwg, workspace)
        _output_ln(cell, tri, dt, dn, mu, rs, go, dgo, dbo)
        dl, dr = torch.empty_like(tri), torch.empty_like(tri)
        _lt(cell, "bc0", dt[:D], ab[H:H + D], dl[:D], workspace)
        _lt(cell, "bc1", dt[:D].transpose(-1, -2), ab[:D], dr[:D], workspace)
        _lt(cell, "bc2", ab[H + D:], dt[D:].transpose(-1, -2), dl[D:], workspace)
        _lt(cell, "bc3", ab[D:H], dt[D:], dr[D:], workspace)
        _source_d256(cell, xn, saved[8], dl, dr, gp, mask, partial)
    else:
        pre = saved[8]
        if cell == (512, 768):
            # 32 FP32 split intervals of dWproj, reduced in order by a native kernel.
            step = M // 32
            _lt(cell, "dwp", dp.as_strided((32, D, step), (step * D, 1, D)),
                norm.as_strided((32, step, H), (step * H, H, 1)),
                partial.as_strided((32, D, H), (11 * D * D, H, 1)), workspace)
            _weight_reduce_d512(cell, partial, dwp)
        else:
            torch.mm(dp.t(), norm, out=dwp, **({"out_dtype": torch.float32} if master else {}))
        joint = cell in ((384, 384), (512, 768))
        if not joint:
            _lt(cell, "dwg", dxin[:D], xn.reshape(M, D), dwg, workspace)
        _output_ln(cell, tri, dt, dn, mu, rs, go, dgo, dbo)
        _contract_gp(cell, dt, ab, pre, mask, gp)
        if cell != (512, 768):
            _split_input_dw(cell, dxin, xn, partial, workspace, joint)
    torch.cat((wg, wl, wlg, wr, wrg), out=wcat)
    if cell == (512, 768):
        # The joint input/gate dW overlaps the dX GEMM; both only read dxin.
        side = _side_stream()
        side.wait_stream(main)
        side_workspace = _workspace(x)
        with torch.cuda.stream(side):
            _split_input_dw(cell, dxin, xn, partial, side_workspace, True)
    _lt(cell, "dx", dxin.t(), wcat, dxn, workspace)
    if cell == (512, 768):
        main.wait_stream(side)  # the input reduction reads the side stream's partials
    joint = cell in ((384, 384), (512, 768))
    _input_ln(cell, x, dxn, dy, dx, gi, dgi, dbi, partial, dw, dwg if joint else None,
              cooperative=n == 768, master=master)
    if cell == (256, 384):
        main.wait_stream(side)  # dWproj/dWgate join at the end, as qualified
    del side_workspace
    return [dx, *dw, dwg, dwp, dgi, dbi, dgo, dbo]
