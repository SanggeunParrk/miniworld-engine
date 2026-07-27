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


def test_module_package_does_not_import_optional_baselines_eagerly() -> None:
    code = """
import sys
from miniworld_kernels import modules

assert "TriangleMultiplication" in dir(modules)
assert "miniworld_kernels.modules.triangle_multiplication" not in sys.modules

import torch
from miniworld_kernels.kernels.mpnn_edge_layernorm import edge_layer_norm
from miniworld_kernels.modules.mpnn import ProteinMPNN, ProteinMPNNConfig

assert ProteinMPNN.__name__ == "ProteinMPNN"
model = ProteinMPNN(
    ProteinMPNNConfig(
        message_backend="triton_memory",
        edge_norm_backend="memory",
        edge_dropout_backend="bitpack",
        encoder_node_w1_recompute="checkpoint",
        transition_recompute="update",
    )
)
with torch.no_grad():
    edge_layer_norm(
        torch.randn(2, 128),
        torch.ones(128),
        torch.zeros(128),
        1e-5,
        backend="memory",
    )
    model.encoder.layers[0].edge_message.dropout(torch.randn(2, 128))
assert "triton" not in sys.modules
assert "miniworld_kernels.kernels.mpnn_edge_layernorm.triton.main" not in sys.modules
assert "miniworld_kernels.kernels.mpnn_edge_dropout.triton.main" not in sys.modules
assert "miniworld_kernels.modules.triangle_multiplication" not in sys.modules
assert "cuequivariance_torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
