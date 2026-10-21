"""The sm_100a driver loader (``kernels/augmented_attention/cuda/sm100/driver.py``) on a thread with no current CUDA context.

A thread's first runtime call makes the device's primary context current; ``cudaGetDevice`` alone does not, and torch's device guard
skips ``cudaSetDevice`` when the device is already the one ``cudaGetDevice`` reports (0 on a fresh thread). Autograd's backward thread
for device 0 can therefore reach a cubin load (a bias-only DiT block's fused backward, built on first use) with no context current:
``cuModuleLoadData`` failed with CUDA_ERROR_INVALID_CONTEXT and the block kept its per-step launches for the rest of the process.
"""

import threading

import pytest
import torch

pytestmark = pytest.mark.gpu

_SRC = 'extern "C" __global__ void ctx_probe(float* p) { p[threadIdx.x] = 1.0f + threadIdx.x; }\n'


@pytest.fixture(scope="module", autouse=True)
def _sm100_only():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("sm_100 only")


def test_a_cubin_loads_and_launches_on_a_thread_without_a_current_context():
    from cuda.bindings import driver as cu

    from miniworld_engine.kernels import _nvcc
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import _NVCC_FLAGS, driver

    if torch.cuda.current_device() != 0:
        pytest.skip("the context-less case is device 0 (a fresh thread's cudaGetDevice)")
    path = _nvcc.cubin("augattn_sm100/ctx_probe", _NVCC_FLAGS, text=_SRC, toolchain="torch-nohost")
    driver.init()  # the process is initialised on the main thread, as in a training run
    out = torch.zeros(32, device="cuda")
    torch.cuda.synchronize()
    seen: dict = {}

    def worker():
        try:
            err, ctx = cu.cuCtxGetCurrent()
            seen["ctx_on_entry"] = int(ctx) if err == cu.CUresult.CUDA_SUCCESS else -1
            with torch.cuda.device(0):  # what the families do around a load: a no-op on device 0 here
                k = driver.Kernel(str(path), "ctx_probe", 0)
                k((1, 1, 1), (32, 1, 1), out)
            seen["ok"] = True
        except Exception as exc:  # noqa: BLE001 -- reported below
            seen["error"] = repr(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    if seen.get("ctx_on_entry", 0) != 0:
        pytest.skip("this runtime made a context current on the new thread already")
    assert seen.get("ok"), seen.get("error")
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), torch.arange(32, dtype=torch.float32) + 1)
