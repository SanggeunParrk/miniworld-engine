from __future__ import annotations

import subprocess
import sys


def test_kernel_package_does_not_import_backends_eagerly() -> None:
    code = """
import sys
from miniworld_kernels import kernels

assert "triton_tm1" in dir(kernels)
assert "miniworld_kernels.kernels.adaln.triton.inference" not in sys.modules
assert "miniworld_kernels.kernels.transition.triton.fused" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
