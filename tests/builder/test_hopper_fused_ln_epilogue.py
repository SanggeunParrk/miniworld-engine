"""Device-produced LN statistics must survive quack's host-side epilogue setup."""
from types import SimpleNamespace

import pytest

pytest.importorskip("quack")
pytest.importorskip("cutlass.cute")

from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    GemmLNLFusedSm90,
    SmemColVec,
)


def test_internal_stats_survive_none_argument_filter(monkeypatch):
    gemm = object.__new__(GemmLNLFusedSm90)
    args = GemmLNLFusedSm90.EpilogueArguments()
    params = gemm.epi_to_underlying_arguments(args)
    # No CUDA or CuTe IR context: exercise the actual host filter/dictionary builder.
    monkeypatch.setattr(SmemColVec, "get_smem_tensor", lambda op, *args: op.name)
    tensors = gemm.epi_get_smem_tensors(params, SimpleNamespace(epi=None))
    assert tensors["mRstd"] == "mRstd"
    assert tensors["mC1"] == "mC1"
    assert {op.name for op in gemm._epi_ops} == {"mRstd", "mC1"}
    assert params.mRstd is None
    assert params.mC1 is None


def test_filter_still_removes_absent_external_operands():
    from cutlass import Float32
    gemm = object.__new__(GemmLNLFusedSm90)
    gemm._filter_epi_ops(GemmLNLFusedSm90.EpilogueArguments(alpha=Float32(1.0)))
    assert {op.name for op in gemm._epi_ops} == {"alpha", "mRstd", "mC1"}
    gemm._filter_epi_ops(GemmLNLFusedSm90.EpilogueArguments())
    assert {op.name for op in gemm._epi_ops} == {"mRstd", "mC1"}


@pytest.mark.parametrize("tile_m", [64, 128, 192])
def test_stage_budget_reserves_both_stats_and_pingpong_halves(tile_m):
    args = GemmLNLFusedSm90.EpilogueArguments()
    budget = GemmLNLFusedSm90.epi_smem_bytes(args, (tile_m, 128, 64), (64, 64))
    assert budget.unstaged >= 2 * 2 * tile_m * 4
    assert budget.c_stage == 0
    assert budget.d_stage == 0
