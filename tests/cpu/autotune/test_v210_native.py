"""CPU contracts for fused CUDA tuning; these tests never compile or launch CUDA."""
import pytest
import torch

from miniworld_engine.autotune import native, native_compile
from miniworld_engine.autotune.fused_config import (
    transition_candidates,
    trimul_candidates,
    unpack_trimul,
    validator,
)


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
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_infer_kernel import (
        TILE_TABLE,
        lookup,
    )
    for (arch, cz, ch, form) in TILE_TABLE:
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
    args = build_parser().parse_args(["build", "all", "--kernels", "native", "--mode", "eval"])
    assert args.kernels == "native"
    assert args.mode == "eval"
    assert args.config_type == "default"


@pytest.mark.parametrize("sms", [114, 132])
@pytest.mark.parametrize("backward", [False, True])
def test_transition_n2_grid_has_four_slices_and_roundtrips(sms, backward):
    """Hidden 2 D (256 rows at D128): four 64-unit slices, ~60 weight-role CTAs = 4 x 15 (measured: 14-15 best, 16 is 4 % slower)."""
    grid = transition_candidates(sms, backward=backward, n=2)
    assert grid[0] == {"ctas": sms, "dw_repl": 15}
    assert all(c["ctas"] > 4 * c["dw_repl"] for c in grid)
    assert len(grid) == len({tuple(sorted(c.items())) for c in grid})
    op = "transition_bwd_residual_sm90_cuda" if backward else "transition_fwd_residual_sm90_cuda"
    small, wa, ws = torch.empty(4, 128), torch.empty(256, 128), torch.empty(128, 256)   # x [rows, D]: n = hidden / D
    tensors = [small] * 6 + [wa, wa, ws] if backward else [small] * 3 + [wa, wa, ws]
    assert native.candidates_for(op, native.tensor_key(*tensors, extra=(True, None, sms))) == grid


def test_transition_n2_precompile_builds_the_hidden_2d_variant(monkeypatch):
    from miniworld_engine.kernels.transition.cuda import fused_sm90a
    calls = []
    monkeypatch.setattr(fused_sm90a, "_ext", lambda *args: calls.append(args))
    small, wa, ws = torch.empty(4), torch.empty(256, 128), torch.empty(128, 256)
    key = native.tensor_key(*([small] * 3 + [wa, wa, ws]), extra=(True, 1e-5, 114))
    task = native_compile.task_for("transition_fwd_residual_sm90_cuda", {"ctas": 114, "dw_repl": 4}, key)
    native_compile.compile_task(task)
    assert calls == [(114, 4, True, 2)]


def test_precompile_workers_key_their_builds_for_the_measuring_arch(monkeypatch, tmp_path):
    """The workers run with no device; the extension key's arch must still be the measuring process's, or every
    candidate is compiled twice (once under "cpu" in the worker, once more inside the unit)."""
    from miniworld_engine.kernels import _codegraph
    seen = []
    monkeypatch.setattr(_codegraph, "target_arch", lambda: "sm90")
    monkeypatch.setattr(native_compile, "_run_one",
                        lambda task, directory, timeout, env: (seen.append(env), {"task": task, "status": "ok", "seconds": 0.0})[1])
    native_compile.run_tasks([{"op": "x", "config": {}, "tensors": [], "extra": []}], jobs=1, directory=tmp_path)
    assert seen[0]["CUDA_VISIBLE_DEVICES"] == "" and seen[0]["MINIWORLD_ENGINE_TARGET_ARCH"] == "sm90"


def test_the_compile_plumbing_is_not_part_of_the_measured_identity():
    """Where candidates are precompiled does not change what is measured: editing native_compile.py keeps the caches."""
    import inspect
    assert "native_compile.py" not in inspect.getsource(native.source_identity).split('"""')[-1]


def test_every_trimul_candidate_is_instantiated_in_the_unit():
    """A candidate the grid offers but the width's unit does not instantiate fails at load (no_kernel), so the shape is
    never tuned: c_z 384 offered two K1 tiles its unit did not build."""
    import re
    from pathlib import Path

    from miniworld_engine.kernels.trimul_inproj.cuda._h100_infer_kernel import TILE_TABLE
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_runtime import SOURCES
    text = re.sub(r"\s+", "", (Path(SOURCES) / "inference" / "tmn_sm90_unit.cuh").read_text())

    def built(macros, cz, ch, cfg):
        args = ",".join(str(v) for v in (cz, ch, *cfg))
        loose = ",".join(str(v) for v in cfg)
        return any(f"{m}_SET_BF16({args})" in text or f"{m}({cz},{ch},false,b,{loose}," in text for m in macros)

    missing = []
    for (arch, cz, ch, form) in TILE_TABLE:
        if arch != "sm_90a" or form != "b":
            continue
        for length in (128, 256, 384, 512, 640, 768):
            for direction in (0, 1, 2):
                for config in trimul_candidates(cz, ch, length, direction):
                    k = unpack_trimul(config)
                    k3, wide = (k["k3"][:4], True) if len(k["k3"]) == 5 else (k["k3"], False)
                    if not built(("TMN_K1",), cz, ch, k["k1"]):
                        missing.append((cz, ch, "k1", k["k1"]))
                    if not built(("TMN_K3W",) if wide else ("TMN_K3",), cz, ch, k3):
                        missing.append((cz, ch, "k3", k["k3"]))
    assert not sorted(set(missing))


def test_candidate_validation_allows_one_bf16_step_at_the_tensor_scale_only():
    """A reordered bf16 weight-gradient sum lands one rounding step from the default (2.0 at ~256); a wrong one does not pass."""
    torch.manual_seed(0)
    expected = (torch.randn(512, 128) * 100).bfloat16()
    expected[0, 0] = 256.0
    check = validator(lambda _: (expected,), {})
    reordered = expected.clone()
    reordered[0, 0] = 258.0                                     # the next bf16 value above 256
    reordered[1, 1] = expected[1, 1] + 1.0                      # near-zero elements move by the same absolute step
    check({}, (reordered,))
    wrong = expected.clone()
    wrong[2, 2] = expected[2, 2] + 16.0
    with pytest.raises(AssertionError):
        check({}, (wrong,))
    fp32 = torch.tensor([300.0])
    with pytest.raises(AssertionError):
        validator(lambda _: (fp32,), {})({}, (fp32 + 2.0,))     # fp32 outputs keep the fixed 5e-3
