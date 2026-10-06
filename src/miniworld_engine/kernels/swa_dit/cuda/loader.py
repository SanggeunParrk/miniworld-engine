"""JIT build of the three sm_90a bf16-wgmma kernels of the fused SWA atom DiT block.

Moved from team-gm ``swa_fused_triton._cuda_ext`` (commit 14f2c73). Same sources, same nvcc flags (``-O3``,
``-gencode=arch=compute_90a,code=sm_90a``, ``--use_fast_math``); what changed is where and how they build:

* under ``$settings.engine_jit_root/<extension>`` (default ``~/.cache/miniworld_engine_jit``), one directory per
  extension, instead of torch's extension cache or team-gm's ``SWA_CUDA_BUILD``. Set the root per checkout/GPU so
  concurrent builds do not collide;
* through ``kernels._nvcc.load_extension``, which reclaims a lock left by a killed build (NFS-safe: it trusts the
  server's ctime, not an NFS create verifier) and bounds the wait on a live one;
* with ``kernels._nvcc.host_flags()``, so nvcc is driven by a host g++ it can parse.

``extension(which)`` never raises: a card that is not sm_90, a missing toolkit or a failed build all return None and the
caller takes the Triton kernel. The reason is kept in ``ERRORS[which]``.

The kernels (``swa_qkvg_fwd.cu``, ``swa_ffn_fwd.cu``, ``swa_ffn_bwd.cu``) serve d_atom 128, 4 heads x 32, SwiGLU hidden
256, bf16 operands with fp32 accumulate and an fp32 [B*S, 768] modulation. ``swa_ffn_bwd.cu`` still reads the team-gm
diagnostics ``SWA_FFN_ABL`` (ablation bits) and ``SWA_FFN_PROF`` (clock64 section profile) from the environment; both
default off and change nothing when unset.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import torch
from miniworld_engine import settings

_DIR = Path(__file__).resolve().parent

#: stage -> source file stem (and the python-visible entry point it defines).
SOURCES = {"qkvg": "swa_qkvg_fwd", "fwd": "swa_ffn_fwd", "bwd": "swa_ffn_bwd"}
SOURCES.update(qkvg_small="swa_qkvg_fwd", fwd_small="swa_ffn_fwd", bwd_overwrite="swa_ffn_bwd")

# Separate build identities preserve the legacy kernels and their schedules.
_VARIANT_FLAGS = {
    "qkvg_small": ["-DMW_SWA_QKVG_SP=2", "-DMW_SWA_FWD_STAGES=2"],
    "fwd_small": ["-DMW_SWA_FFN_SP=4", "-DMW_SWA_FWD_STAGES=2"],
    "bwd_overwrite": ["-DMW_SWA_FFN_OVERWRITE=1"],
}

_EXT: dict[str, Any] = {}
#: stage -> why the extension is unavailable (repr of the build exception).
ERRORS: dict[str, str] = {}


def jit_root() -> Path:
    """Where the extensions build: ``$settings.engine_jit_root``, else ``~/.cache/miniworld_engine_jit``."""
    return settings.engine_jit_root()


@functools.lru_cache(maxsize=16)
def _capability(index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(index)


def is_sm90(device: torch.device | None = None) -> bool:
    """True on Hopper (sm_90) exactly: the kernels are sm_90a wgmma/TMA code."""
    if not torch.cuda.is_available():
        return False
    if device is not None and device.type != "cuda":
        return False
    index = (
        device.index
        if device is not None and device.index is not None
        else torch.cuda.current_device()
    )
    return _capability(index) == (9, 0)


def extension(which: str) -> Any:
    """The built extension for stage ``which`` ("qkvg", "fwd" or "bwd"), or None. Never raises.

    Built once per process on first use; a failed build is not retried (``ERRORS`` says why).
    """
    if which in _EXT:
        return _EXT[which]
    _EXT[which] = None
    try:
        if not is_sm90():
            ERRORS[which] = "not an sm_90 device"
            return None
        from miniworld_engine.kernels._nvcc import (
            ensure_cuda_home,
            host_flags,
            load_extension,
        )

        ensure_cuda_home()
        src = SOURCES[which]
        name = f"miniworld_swa_dit_{src}"
        if which in _VARIANT_FLAGS:
            name += f"_{which}"
        build = jit_root() / name
        build.mkdir(parents=True, exist_ok=True)
        _EXT[which] = load_extension(
            name,
            [str(_DIR / f"{src}.cu")],
            build_directory=str(build),
            extra_include_paths=[str(_DIR)],
            extra_cuda_cflags=[
                *host_flags(),
                "-O3",
                "-gencode=arch=compute_90a,code=sm_90a",
                "--use_fast_math",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception as ex:  # noqa: BLE001 -- fall back to Triton
        ERRORS[which] = repr(ex)
    return _EXT[which]
