"""CPU contracts for fused CUDA tuning; these tests never compile or launch CUDA."""
import pytest
import torch

from miniworld_engine.autotune import native, native_compile
from miniworld_engine.autotune.fused_config import transition_candidates, trimul_candidates, unpack_trimul, validator


@pytest.mark.parametrize("sms", [114, 132])
@pytest.mark.parametrize("backward", [False, True])
def test_transition_grid_roundtrips_and_keeps_previous_default(sms, backward):
    grid = transition_candidates(sms, backward=backward)
    assert grid[0] == {"ctas": sms, "dw_repl": 8}
    assert len(grid) > 1
    assert all(c["ctas"] > 8 * c["dw_repl"] for c in grid)
    op = "transition_bwd_residual_sm90_cuda" if backward else "transition_fwd_residual_sm90_cuda"
    key = native.tensor_key(torch.empty(128, 128, dtype=torch.bfloat16), extra=(True, None, sms))
    assert native.candidates_for(op, key) == grid


@pytest.mark.parametrize("length", [128, 256, 384, 512, 640, 768])
@pytest.mark.parametrize("direction", [0, 1, 2])
def test_trimul_inference_grid_covers_existing_lengths_and_directions(length, direction):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_infer_kernel import TILE_TABLE, lookup
    for (arch, cz, ch, form), _ in TILE_TABLE.items():
        if arch != "sm_90a" or form != "b":
            continue
        grid = trimul_candidates(cz, ch, length, direction)
        default = lookup(arch, cz, ch, form, length, "incoming" if direction == 2 else "outgoing", True)
        assert unpack_trimul(grid[0])["k1"] == default["k1"]
        assert unpack_trimul(grid[0])["k3"] == default["k3"]
        assert len(grid) == len({tuple(sorted(c.items())) for c in grid})


def test_transition_precompile_uses_the_exact_runtime_flags(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm90a
    calls = []
    monkeypatch.setattr(fused_sm90a, "_ext", lambda *args: calls.append(args))
    key = native.tensor_key(torch.empty(128, 128), extra=(False, 1e-5, 114))
    cfg = {"ctas": 114, "dw_repl": 4}
    task = native_compile.task_for("transition_fwd_residual_sm90_cuda", cfg, key)
    native_compile.compile_task(task)
    assert calls == [(114, 4, False)]


def test_candidate_validation_rejects_wrong_results_and_ignores_unwritten_inference_saves():
    expected = torch.tensor([1.0, 2.0])
    check = validator(lambda _: (expected, torch.full((1,), float("nan"))), {}, output_only=True)
    check({}, (expected.clone(), torch.full((1,), float("nan"))))
    with pytest.raises(AssertionError):
        check({}, (expected + 1, torch.zeros(1)))
    gradients = validator(lambda _: (expected, expected), {})
    with pytest.raises(AssertionError):
        gradients({}, (expected, expected + 1))


def test_native_inference_cli_is_explicit():
    from miniworld_engine.cli import build_parser
    args = build_parser().parse_args(["build", "all", "--backend", "native", "--mode", "eval"])
    assert args.backend == "native" and args.mode == "eval" and args.config_type == "default"
