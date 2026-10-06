"""The kernels a captured CUDA graph launches, read from the graph itself.

Tests that count the kernels of a replay used to profile ``graph.replay()`` with the torch profiler. In a long pytest
process that is unreliable: after a profiled replay, later profiling windows in the same process lose kernel records
(a replay reported 0 kernels; an eager call lost its first kernel or all of them), so the tests failed or passed
depending on what ran before them. The graph's own nodes do not depend on the profiler: capture with
``torch.cuda.CUDAGraph(keep_graph=True)`` and list its kernel nodes through the driver API."""
from __future__ import annotations

import ctypes
import functools

import torch

_KERNEL_NODE = 0  # CU_GRAPH_NODE_TYPE_KERNEL


class _KernelParams(ctypes.Structure):  # CUDA_KERNEL_NODE_PARAMS_v2
    _fields_ = [("func", ctypes.c_void_p), ("grid", ctypes.c_uint * 3), ("block", ctypes.c_uint * 3),
                ("shared_mem_bytes", ctypes.c_uint), ("kernel_params", ctypes.c_void_p), ("extra", ctypes.c_void_p),
                ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]


@functools.cache
def _driver() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


def graph_kernels(graph: torch.cuda.CUDAGraph) -> list[str]:
    """Names of the kernel nodes of ``graph`` (captured with ``keep_graph=True``), one entry per launch, in node order."""
    cu = _driver()
    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    count = ctypes.c_size_t(0)
    assert cu.cuGraphGetNodes(handle, None, ctypes.byref(count)) == 0
    nodes = (ctypes.c_void_p * count.value)()
    assert cu.cuGraphGetNodes(handle, nodes, ctypes.byref(count)) == 0
    names = []
    for node in nodes:
        kind = ctypes.c_int()
        assert cu.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind)) == 0
        if kind.value != _KERNEL_NODE:
            continue
        params = _KernelParams()
        assert cu.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(node), ctypes.byref(params)) == 0
        name = ctypes.c_char_p()
        assert cu.cuFuncGetName(ctypes.byref(name), ctypes.c_void_p(params.func)) == 0
        names.append((name.value or b"").decode())
    return names


def launched_kernels(fn) -> list[str]:
    """The kernels one call of ``fn`` launches, read from a CUDA graph capture of that call (after one eager warm-up call on
    a side stream). The profiler is not used: in long pytest processes it drops the records of kernels launched through
    the driver API (cuLaunchKernelEx with programmatic dependent launch) -- the engine's own kernels -- while it keeps
    cuBLAS and ATen ones. ``fn`` must be capturable; the capture serves the caches as any capture does
    (``kernels._capture``)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph_kernels(graph)
