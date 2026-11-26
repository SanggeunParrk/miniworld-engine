"""Environment-variable names: ``FOLDJET_ENGINE_*`` since 1.0.0; the old ``MINIWORLD_*`` / ``MINIWORLD_ENGINE_*`` names still work.

Every engine module that reads the environment imports this module first. On the first import it copies each old name to its
new name when the new one is unset (one ``FutureWarning`` names them), and copies the new name back to the old one for the
switches the compiled kernels read themselves (``getenv`` in the CUDA sources keeps the old spelling: those sources are part of
the prebuilt kernel keys, so they are not edited). ``sync()`` can be called again after the environment changes.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import MutableMapping

OLD, NEW = "MINIWORLD_", "FOLDJET_ENGINE_"

#: Read by ``getenv`` inside CUDA sources under the old name (token_dit_pack.cu, the B200 TriAttn sources).
KERNEL_GETENV = ("A100_CLEANUP", "TRIATTN_B200_PDL", "TRIATTN_B200_L128_GB")


def new_name(old: str) -> str:
    """``MINIWORLD_X`` / ``MINIWORLD_ENGINE_X`` -> ``FOLDJET_ENGINE_X``."""
    rest = old[len(OLD) :]
    return NEW + (rest.removeprefix("ENGINE_"))


def sync(environ: MutableMapping[str, str] = os.environ) -> list[str]:
    """Copy old names to unset new names and the kernel-read new names back; return the old names that were copied."""
    copied = []
    for key in list(environ):
        if key.startswith(OLD):
            new = new_name(key)
            if new not in environ:
                environ[new] = environ[key]
                copied.append(key)
    for rest in KERNEL_GETENV:
        if NEW + rest in environ and OLD + rest not in environ:
            environ[OLD + rest] = environ[NEW + rest]
    return copied


_copied = sync()
if _copied:
    warnings.warn(
        "foldjet-engine 1.0.0 renamed its environment variables MINIWORLD_* -> FOLDJET_ENGINE_*; read as the new names: "
        + ", ".join(sorted(_copied)),
        FutureWarning,
        stacklevel=2,
    )
