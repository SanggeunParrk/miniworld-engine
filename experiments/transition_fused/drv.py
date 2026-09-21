"""Minimal CUDA driver surface (cuda.bindings): load a cubin, encode 2-D TMA descriptors, launch with by-value CUtensorMap / pointer / scalar arguments
on torch's current stream (capture-safe).  Mirrors the argument conventions of the Anthropic esm_t16 loader (ef2_t16_nvjit)."""
import ctypes
import struct
import torch
from cuda.bindings import driver as cu

_INIT = False


def _chk(res, what):
    err, *vals = res if isinstance(res, tuple) else (res,)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {err}")
    return vals[0] if len(vals) == 1 else (vals if vals else None)


def init():
    global _INIT
    if not _INIT:
        torch.zeros(1, device="cuda")
        _chk(cu.cuInit(0), "cuInit")
        _INIT = True


class TensorMap:
    """2-D tiled map: dims innermost-first, one outer stride in bytes, box innermost-first, 128-B swizzle by default (bf16)."""
    def __init__(self, tensor, dims, stride_bytes, box, swizzle=128):
        init()
        self.keep = tensor
        sw = {0: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE, 32: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
              64: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B, 128: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B}[swizzle]
        self.tm = _chk(cu.cuTensorMapEncodeTiled(cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, tensor.data_ptr(),
                                                 [cu.cuuint64_t(d) for d in dims], [cu.cuuint64_t(stride_bytes)], [cu.cuuint32_t(b) for b in box],
                                                 [cu.cuuint32_t(1), cu.cuuint32_t(1)], cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                                                 cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                                 cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled")
        words = [int(w) for w in self.tm.opaque]
        raw = struct.pack(f"<{len(words)}Q", *words)
        assert len(raw) == 128, len(raw)
        self.buf = (ctypes.c_ubyte * 192)()          # 64-B aligned copy of the 128-B descriptor
        base = ctypes.addressof(self.buf)
        self.addr = (base + 63) & ~63
        ctypes.memmove(self.addr, raw, 128)


class Kernel:
    def __init__(self, cubin_path, func_name, smem_bytes, cluster=None):
        init()
        data = open(cubin_path, "rb").read()
        self.module = _chk(cu.cuModuleLoadData(data), "cuModuleLoadData")
        self.func = _chk(cu.cuModuleGetFunction(self.module, func_name.encode()), "cuModuleGetFunction")
        if smem_bytes > 48 * 1024:
            _chk(cu.cuFuncSetAttribute(self.func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes), "smem attr")
        if cluster:
            _chk(cu.cuFuncSetAttribute(self.func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1), "cluster attr")
        self.smem = smem_bytes
        self.cluster = cluster
        self.regs = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func), "regs")
        self.lmem = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.func), "lmem")

    def __call__(self, grid, block, *args, stream=None):
        holders, ptrs = [], []
        for a in args:
            if isinstance(a, TensorMap):
                ptrs.append(a.addr)
            elif isinstance(a, torch.Tensor):
                h = ctypes.c_uint64(a.data_ptr()); holders.append(h); ptrs.append(ctypes.addressof(h))
            elif isinstance(a, bool):
                h = ctypes.c_int32(int(a)); holders.append(h); ptrs.append(ctypes.addressof(h))
            elif isinstance(a, int):
                h = ctypes.c_int32(a); holders.append(h); ptrs.append(ctypes.addressof(h))
            elif isinstance(a, float):
                h = ctypes.c_float(a); holders.append(h); ptrs.append(ctypes.addressof(h))
            elif a is None:
                h = ctypes.c_uint64(0); holders.append(h); ptrs.append(ctypes.addressof(h))
            else:
                raise TypeError(type(a))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        st = torch.cuda.current_stream().cuda_stream if stream is None else stream
        if self.cluster:
            cfg = cu.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid[0], grid[1], grid[2]
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block[0], block[1], block[2]
            cfg.sharedMemBytes = self.smem
            cfg.hStream = cu.CUstream(st)
            at = cu.CUlaunchAttribute()
            at.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = self.cluster, 1, 1
            cfg.attrs = [at]; cfg.numAttrs = 1
            _chk(cu.cuLaunchKernelEx(cfg, self.func, ctypes.addressof(arr), 0), "cuLaunchKernelEx")
        else:
            _chk(cu.cuLaunchKernel(self.func, grid[0], grid[1], grid[2], block[0], block[1], block[2], self.smem, cu.CUstream(st),
                                   ctypes.addressof(arr), 0), "cuLaunchKernel")
        self._keep = (holders, arr)
