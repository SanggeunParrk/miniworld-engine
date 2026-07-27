"""Direct parity check against the checked-out ProteinMPNN_CSSB source.

This is an integration test for developers working beside the upstream checkout.
It skips cleanly when that repository is absent from the machine.
"""

from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import torch

from miniworld_kernels.modules.mpnn import NaiveProteinMPNN

_DEFAULT_UPSTREAM = Path("/home/psk6950/practice/ProteinMPNN_CSSB")
_UPSTREAM = Path(os.environ.get("PROTEINMPNN_CSSB_ROOT", _DEFAULT_UPSTREAM))
_UPSTREAM_REV = "4870bcaf4f55c45b5d7ee5ff8097a3ce3d020ac0"

pytestmark = pytest.mark.skipif(
    not (_UPSTREAM / ".git").exists(),
    reason="ProteinMPNN_CSSB git checkout is unavailable",
)


def _materialize_frozen_revision(tmp_path: Path) -> Path:
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(_UPSTREAM),
            "archive",
            "--format=tar",
            _UPSTREAM_REV,
            "fullmoon_initial_package",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        source.extractall(tmp_path, filter="data")
    return tmp_path / "fullmoon_initial_package"


def _upstream_model_class(package: Path):
    sys.path.insert(0, str(package))
    try:
        return importlib.import_module("model.model").ProteinMPNN
    finally:
        sys.path.remove(str(package))


def _randomize_parameters(module: torch.nn.Module, seed: int = 123) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            values = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            )
            parameter.copy_(values.mul_(0.05))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("packed", [False, True])
def test_forward_and_backward_match_frozen_upstream_revision(
    device: str, packed: bool, tmp_path: Path
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    upstream_cls = _upstream_model_class(_materialize_frozen_revision(tmp_path))
    kwargs = dict(
        node_features=16,
        edge_features=16,
        hidden_dim=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=4,
        augment_trans=0,
        augment_rot=0,
        dropout=0,
    )
    upstream = upstream_cls(**kwargs).eval()
    _randomize_parameters(upstream)
    reference = NaiveProteinMPNN(**kwargs).eval()
    reference.load_state_dict(upstream.state_dict(), strict=True)
    upstream, reference = upstream.to(device), reference.to(device)

    generator = torch.Generator().manual_seed(99)
    length = 12
    xyz_cpu = torch.randn(1, length, 4, 3, generator=generator)
    seq = torch.randint(0, 21, (1, length), generator=generator).to(device)
    mask = torch.ones(1, length, device=device)
    if packed:
        mask[:, 3] = 0
        mask[:, 10] = 0
    residue_idx = torch.arange(length, device=device).unsqueeze(0)
    chain_idx = torch.zeros(1, length, dtype=torch.long, device=device)
    decoding_order = torch.randperm(length, generator=generator).to(device).unsqueeze(0)
    patch_index = torch.arange(length, device=device).unsqueeze(0)
    len_tensor = torch.tensor([5, 7] if packed else [length], device=device)
    common = (
        seq,
        mask,
        residue_idx,
        chain_idx,
        decoding_order,
        patch_index,
        mask,
        len_tensor,
    )

    xyz_upstream = xyz_cpu.to(device).requires_grad_(True)
    xyz_reference = xyz_cpu.to(device).requires_grad_(True)
    output_upstream = upstream(xyz_upstream, *common)
    output_reference = reference(xyz_reference, *common)
    torch.testing.assert_close(output_reference, output_upstream, atol=0, rtol=0)

    output_upstream.square().mean().backward()
    output_reference.square().mean().backward()
    # Repeated-index CUDA backward reductions use atomics, so two otherwise
    # identical executions can differ in their final few bits.  CPU remains a
    # bitwise oracle; CUDA keeps a tolerance far below the optimized-path
    # numerical budget while avoiding a flaky exact-equality assertion.
    grad_atol = 0.0 if device == "cpu" else 2e-6
    grad_rtol = 0.0 if device == "cpu" else 2e-5
    torch.testing.assert_close(
        xyz_reference.grad,
        xyz_upstream.grad,
        atol=grad_atol,
        rtol=grad_rtol,
    )
    for (name_upstream, parameter_upstream), (name_ref, parameter_ref) in zip(
        upstream.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name_upstream == name_ref
        torch.testing.assert_close(
            parameter_ref.grad,
            parameter_upstream.grad,
            atol=grad_atol,
            rtol=grad_rtol,
        )
