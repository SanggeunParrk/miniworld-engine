"""Native tuning must measure candidates, retain coverage, and fail visibly."""
from dataclasses import replace

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.autotune import capture, native


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    from miniworld_engine.autotune import native_compile
    monkeypatch.setattr(native_compile, "precompile", lambda *args: {})
    monkeypatch.setattr(settings, "_ACTIVE", replace(settings.current(), run_autotune=True))
    monkeypatch.setattr(native, "source_identity", lambda: "source-v1")
    monkeypatch.setattr(native, "gpu_key", lambda _: "test-sm90")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(capture, "_bench_lock_acquire", lambda: None)
    monkeypatch.setattr(capture, "_bench_lock_release", lambda: None)
    capture.reset()
    yield
    capture.reset()


def test_build_searches_records_and_reuses_exact_workload(monkeypatch):
    import triton.testing
    calls = []
    def run(c):
        calls.append(c["tile_m"])
        if c["tile_m"] == 192:
            raise ValueError("unsupported tile")
        return 1 / c["tile_m"]
    monkeypatch.setattr(triton.testing, "do_bench", lambda fn, **_: fn())
    grid = [{"tile_m": t} for t in (64, 128, 192)]
    args = {"dtype": "bf16", "bucket": "M17|K128", "run": run}
    assert native.choose_config("test", grid, **args) == {"tile_m": 128}
    assert set(calls) == {64, 128, 192}
    slot = capture._CAPTURE["test"]
    assert len(slot["searched"][("bf16", "M17|K128")]) == 3
    assert len(slot["entries"][("bf16", "M17|K128")]) == 2
    assert capture._UNUSABLE["test"] == 1
    calls.clear()
    native.choose_config("test", grid, **args)
    assert not calls
    native.choose_config("test", grid, **{**args, "bucket": "M18|K128"})
    assert calls


def test_all_failed_round_is_not_a_success(monkeypatch):
    def fail(_):
        raise RuntimeError("compilation failed")
    with pytest.raises(RuntimeError, match="every native configuration failed"):
        native.choose_config("bad", [{"tile_m": 64}], dtype="bf16", bucket="x", run=fail)
    assert not capture._CAPTURE["bad"]["entries"]
    assert not native._WINNERS


def test_precompile_failure_is_recorded_without_launching_candidate(monkeypatch):
    import triton.testing

    from miniworld_engine.autotune import native_compile
    events = []
    grid = [{"tile_m": 64}, {"tile_m": 128}]

    def precompile(*_):
        events.append("compile")
        assert not capture._NATIVE_LOCK_HELD
        return {0: {"status": "timeout", "log": "candidate.log"}, 1: {"status": "ok"}}

    monkeypatch.setattr(native_compile, "precompile", precompile)
    monkeypatch.setattr(capture, "_bench_lock_acquire", lambda: events.append("lock"))
    monkeypatch.setattr(triton.testing, "do_bench", lambda fn, **_: fn())

    def run(config):
        assert config == grid[1]
        assert capture._NATIVE_LOCK_HELD
        return 1.0

    assert native.choose_config("compile-failure", grid, dtype="bf16", bucket="x", run=run) == grid[1]
    assert events == ["compile", "lock"]
    assert capture._UNUSABLE["compile-failure"] == 1
    assert len(capture._CAPTURE["compile-failure"]["searched"][("bf16", "x")]) == 2
    assert not capture._NATIVE_LOCK_HELD


def test_runtime_uses_cache_without_benchmark(monkeypatch):
    settings.configure(run_autotune=False)
    received = {}
    def select(*args, **kwargs):
        received.update(kwargs)
        return {"kwargs": {"tile_m": 128}}
    monkeypatch.setattr(native, "select_config", select)
    assert native.choose_config("x", [{"tile_m": 64}, {"tile_m": 128}],
                                dtype="bf16", bucket="shape", device_index=3,
                                run=lambda _: pytest.fail("runtime must not time")) == {"tile_m": 128}
    assert received["device_index"] == 3
    assert received["op_id"] == "source-v1"


@pytest.mark.parametrize(("stored", "current"), [(0, 1), (1, 2), (2, 1)])
def test_native_runtime_rejects_changed_measurement_revision(monkeypatch, stored, current):
    from miniworld_engine.autotune import cache
    settings.configure(run_autotune=False)
    configs = [{"tile_m": 64}, {"tile_m": 128}]
    data = {"build_rev": stored, "op_identity": "source-v1", "env_identity": "env",
            "entries": {"bf16|shape": [cache.config_to_dict({"kwargs": configs[1]}, 1.0)]}}
    monkeypatch.setattr(cache, "gpu_key", lambda *_: "test-sm90")
    monkeypatch.setattr(cache, "env_identity", lambda: "env")
    monkeypatch.setattr(cache, "_load", lambda *_: data)
    monkeypatch.setattr(cache, "build_rev", lambda _: current)
    monkeypatch.setattr(cache, "_scheme_stale", lambda *_: False)
    assert native.choose_config("revision-test", configs, dtype="bf16", bucket="shape") == configs[0]
    data["build_rev"] = current
    assert native.choose_config("revision-test", configs, dtype="bf16", bucket="shape") == configs[1]


def test_native_timing_honors_paired_budget_and_retimes_when_it_changes(monkeypatch):
    from types import SimpleNamespace

    import triton.testing
    driver = SimpleNamespace(get_empty_cache_for_benchmark=None)
    monkeypatch.setattr(capture, "_bench_driver", lambda: driver)
    seen = []

    def bench(fn, **kwargs):
        seen.append(kwargs)
        return fn()

    monkeypatch.setattr(triton.testing, "do_bench", bench)
    args = {"dtype": "bf16", "bucket": "shape", "run": lambda _: 1.0}
    settings.configure(bench_clear_mb=16, bench_rep_ms=40)
    native.choose_config("budget-test", [{"tile_m": 64}], **args)
    assert driver.get_empty_cache_for_benchmark is capture._bench_clear_buffer
    assert seen == [{"warmup": 10, "rep": 40, "quantiles": None, "return_mode": "median"}]
    settings.configure(bench_rep_ms=8)
    native.choose_config("budget-test", [{"tile_m": 64}], **args)
    assert len(seen) == 2
    assert seen[-1]["warmup"] == 2
    assert seen[-1]["rep"] == 8
    monkeypatch.setattr(native, "build_rev", lambda _: 99)
    native.choose_config("budget-test", [{"tile_m": 64}], **args)
    assert len(seen) == 3


def test_layout_and_reduction_widths_have_distinct_keys():
    x = torch.empty(8, 8)
    assert native.tensor_key(x) != native.tensor_key(x.t())
    assert native.tensor_key(x) != native.tensor_key(x.float(), extra=(True,))
    assert native.tensor_key(x) != native.tensor_key(x.to(torch.bfloat16))


def test_cute_space_excludes_unused_swap_and_respects_reduction_contract():
    from miniworld_engine.autotune.cute_config import (
        lnbwd_candidates,
        plain_sm90_candidates,
    )
    assert all(not c.swap_ab for c in plain_sm90_candidates())
    assert {c.tile_m for c in lnbwd_candidates(256)} == {64}
    assert all(c.tile_n == 256 and not c.is_dynamic_persistent for c in lnbwd_candidates(256))
    assert not lnbwd_candidates(512)


def test_cuda_defines_are_validated_and_change_with_config():
    from miniworld_engine.autotune.hopper_cuda_config import candidates, defines
    grid = candidates("expand_gate", 128)
    assert defines("expand_gate", 128, grid[0]) != defines("expand_gate", 128, grid[1])
    with pytest.raises(ValueError, match="invalid Hopper"):
        defines("expand_gate", 128, {"bn": 17})


def test_layernorm_ragged_mask_gradients_match_autograd():
    from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda
    x = torch.randn(7, 125, requires_grad=True)
    weight = torch.randn(125, requires_grad=True)
    bias = torch.randn(125, requires_grad=True)
    scale = torch.rand(7)
    dy = torch.randn_like(x)
    y = torch.nn.functional.layer_norm(x, (125,), weight, bias, 0.03) * scale[:, None]
    expected = torch.autograd.grad(y, (x, weight, bias), dy)
    mean = x.detach().mean(-1)
    rstd = torch.rsqrt(x.detach().var(-1, unbiased=False) + 0.03)
    actual = layer_norm_bwd_cuda(dy, x.detach(), weight.detach(), mean, rstd, scale)
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=2e-6)


def test_native_units_survive_triton_config_directory_filter(tmp_path):
    from miniworld_engine.autotune.builder import op_units
    units = op_units(only={"transition_swiglu_fwd_sm90_cute"}, config_dir=tmp_path)
    assert units
    assert len({u.length for u in units}) > 1
    assert {u.side for u in units} == {"pair", "token"}


def test_build_all_requests_native_drivers_even_if_module_reaches_them(monkeypatch):
    from miniworld_engine.autotune import derive
    monkeypatch.setattr(derive, "kernel_rows", lambda _: [{"kernel": k} for k in native.BUILD_OPS])
    assert derive.uncovered_kernels("sm90") >= native.BUILD_OPS
    assert "transition_swiglu_fwd_sm90_cute" not in derive.uncovered_kernels("sm86")


def test_runtime_reader_accepts_native_grid_and_invalidates_source(monkeypatch):
    from miniworld_engine.autotune import cache
    settings.configure(run_autotune=False)
    configs = [{"tile_m": 64}, {"tile_m": 128}]
    data = {"op_identity": "source-v1", "env_identity": "env",
            "entries": {"bf16|shape": [cache.config_to_dict({"kwargs": configs[1]}, 0.1)]}}
    monkeypatch.setattr(cache, "gpu_key", lambda _: "test-sm90")
    monkeypatch.setattr(cache, "env_identity", lambda: "env")
    monkeypatch.setattr(cache, "_load", lambda *_: data)
    monkeypatch.setattr(cache, "_scheme_stale", lambda *_: False)
    assert native.choose_config("x", configs, dtype="bf16", bucket="shape") == configs[1]
    data["op_identity"] = "old-source"
    with pytest.warns(UserWarning, match="source/environment changed"):
        assert native.choose_config("x", configs, dtype="bf16", bucket="shape") == configs[0]


def test_separate_trimul_inference_masks_projections_and_preserves_eps(monkeypatch):
    import inspect

    from miniworld_engine.kernels.trimul_inproj.cute import inference
    pair = torch.randn(1, 3, 3, 4)
    gamma, beta = torch.rand(4), torch.rand(4)
    mask = torch.rand(9)
    def ln(x, w, b, eps):
        return torch.nn.functional.layer_norm(x, (4,), w, b, eps)
    normalized = ln(pair, gamma, beta, 0.03)
    def front(x, *args, **kwargs):
        torch.testing.assert_close(x, normalized)
        value = x.permute(0, 3, 1, 2)
        return value, value, None
    def back(tri, xn, wp, wg, lw, lb, eps, residual):
        torch.testing.assert_close(xn, normalized)
        masked = normalized.permute(0, 3, 1, 2) * mask.view(1, 1, 3, 3)
        torch.testing.assert_close(tri, torch.einsum("bdik,bdjk->bdij", masked, masked))
        assert eps == 0.07
        return residual.clone()
    monkeypatch.setattr(inference, "triton_layernorm", ln)
    monkeypatch.setattr(inference, "trimul_inproj_cute_forward", front)
    monkeypatch.setattr(inference, "trimul_back_triton", back)
    w = torch.eye(4)
    inspect.unwrap(inference.trimul_inproj_inference)(
        pair, w, w, w, w, w, w, gamma, beta, gamma, beta, 0.03, w, mask, 0.07)


def test_back_half_honors_legacy_config_dictionary(monkeypatch):
    from miniworld_engine.kernels.trimul_inproj.cute import back_split
    seen = []
    def lnl(view, *args, **kwargs):
        seen.append(kwargs["config"])
        return torch.zeros_like(view)
    monkeypatch.setattr(back_split, "layernorm_linear_cute", lnl)
    monkeypatch.setattr(back_split.dispatch, "pick", lambda *args: torch.zeros(9, 4))
    pair, w = torch.zeros(1, 3, 3, 4), torch.eye(4)
    back_split.trimul_back_split(pair.permute(0, 3, 1, 2), pair, w, w,
                                torch.ones(4), torch.zeros(4), pair,
                                lnl_config={"tile_m": 64, "tile_n": 64,
                                            "cluster_m": 1, "cluster_n": 1, "pingpong": True})
    assert seen[0].tile_m == seen[0].tile_n == 64


def test_portable_cuda_tuning_does_not_require_quack(monkeypatch):
    import builtins

    import triton.testing
    original = builtins.__import__
    def without_quack(name, *args, **kwargs):
        if name == "quack.cache":
            raise ImportError("CuTe is not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_quack)
    monkeypatch.setattr(triton.testing, "do_bench", lambda fn, **_: fn())
    assert native.choose_config("portable", [{"block": 128}], dtype="bf16",
                                bucket="shape", run=lambda _: 1.0) == {"block": 128}


@pytest.mark.parametrize("op", sorted(native.BUILD_OPS))
def test_native_measurements_survive_publication_and_runtime_lookup(op, tmp_path, monkeypatch):
    import triton.testing

    from miniworld_engine.autotune import cache, cache_status

    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(cache_status, "_DATA", tmp_path / "cache")
    monkeypatch.setattr(cache, "gpu_key", lambda *_: "test-sm90")
    monkeypatch.setattr(capture, "gpu_key", lambda *_: "test-sm90")
    monkeypatch.setattr(triton.testing, "do_bench", lambda fn, **_: fn())
    configs = [{"tile_m": 64}, {"tile_m": 128}]
    key = native.tensor_key(torch.empty(17, 128))
    winner = native.choose_config(op, configs, dtype="torch.bfloat16", bucket=key,
                                  run=lambda c: 1 / c["tile_m"])
    path = tmp_path / "unit.json"
    capture.dump_shard(str(path))
    assert capture.merge_shards([path], gpu="test-sm90", only_ops={op})
    assert not capture._MERGE_SKIPPED
    settings.configure(run_autotune=False)
    assert native.choose_config(op, configs, dtype="torch.bfloat16", bucket=key) == winner
    monkeypatch.setattr(cache_status, "configs_for",
                        lambda *_: pytest.fail("native status must not register a Triton CSV"))
    assert cache_status.scan()[0].verdict == "OK"
    monkeypatch.setattr(native, "source_identity", lambda: "source-v2")
    assert cache_status.scan()[0].verdict == "STALE"


def test_native_launchers_use_registered_measurement_names():
    """A driver op filter must retain every measurement emitted by its launcher."""
    import ast
    from pathlib import Path
    root = Path(native.__file__).resolve().parents[1] / "kernels"
    seen = set()
    for family in ("layernorm", "layernorm_linear", "transition", "tm2"):
        for path in (root / family).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in ("choose_config", "resolve_config")
                        and node.args and isinstance(node.args[0], ast.Constant)):
                    op = node.args[0].value
                    assert op in native.BUILD_OPS, (path, op)
                    seen.add(op)
    assert len(seen) == len(native.BUILD_OPS) - 3  # CUDA transition constructs its three names


def test_m2_rejects_unused_architecture_fields_before_compilation(monkeypatch):
    from miniworld_engine.autotune.cute_config import fused_lnl_candidates
    from miniworld_engine.kernels.layernorm_linear.cute import (
        gemm_layernorm_linear_fused as m2,
    )
    monkeypatch.setattr(m2, "get_device_capacity", lambda _: (9, 0))
    x = torch.empty(8, 128)
    for changes in ({"tile_k": 64}, {"num_warps": 4}, {"device_capacity": 10},
                    {"cluster_k": 2}, {"use_tma_gather": True}):
        with pytest.raises(ValueError, match="Hopper"):
            m2.gemm_lnl_fused(x, x, x, x, x,
                              config=replace(fused_lnl_candidates()[0], **changes))


@pytest.mark.parametrize("width", [0, 8, 24, 264])
def test_tm2_rejects_unsafe_output_layout_before_compiler(width):
    from miniworld_engine.kernels.tm2.cute.tm2_cute_kernel import TM2DualKernel
    with pytest.raises(ValueError, match="SW32/STSM"):
        TM2DualKernel(N=width, K=64, tile_m=64)


def test_tm2_pads_all_tails_and_crops_original_shape(monkeypatch):
    import inspect
    from types import SimpleNamespace

    from miniworld_engine.kernels.tm2.cute import tm2_cute_kernel as tm2
    launch = inspect.unwrap(tm2.tm2_dual_from_scratch)
    x1, x2 = torch.randn(7, 19).bfloat16(), torch.randn(7, 19).bfloat16()
    w1, w2 = torch.randn(23, 19).bfloat16(), torch.randn(23, 19).bfloat16()
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(shared_memory_per_block_optin=232448))
    def padded(a, b, wg, wp, *, tile_m):
        assert a.shape == b.shape == (64, 64)
        assert wg.shape == wp.shape == (32, 64)
        assert tile_m == 64
        return (torch.sigmoid(a.float() @ wg.float().t()) * (b.float() @ wp.float().t())).bfloat16()
    monkeypatch.setattr(tm2, "tm2_dual_from_scratch", padded)
    actual = launch(x1, x2, w1, w2, tile_m=64)
    reference = (torch.sigmoid(x1.float() @ w1.float().t()) * (x2.float() @ w2.float().t())).bfloat16()
    assert actual.shape == (7, 23)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, reference)


def test_cuda_layernorm_driver_uses_requested_width(monkeypatch):
    import importlib.util

    from miniworld_engine.kernels.drivers import layernorm, layernorm_linear
    from miniworld_engine.kernels.layernorm import cuda
    monkeypatch.setattr(layernorm_linear, "_D", 256)
    spec = importlib.util.spec_from_file_location("ln_driver_under_test", layernorm.__file__)
    assert spec is not None
    assert spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, "_M", 7)
    monkeypatch.setattr(driver, "rows2d", lambda m, n: torch.empty(m, n))
    monkeypatch.setattr(driver, "vec", lambda n: torch.empty(n))
    monkeypatch.setattr(driver, "_ln_stats", lambda x: (torch.empty(7), torch.empty(7)))
    widths = []
    monkeypatch.setattr(cuda, "layer_norm_bwd_cuda", lambda dy, x, *args: widths.append(x.shape[-1]))
    driver.layer_norm_bwd_main_kernel()
    assert widths == [256]


def test_native_policy_changes_invalidate_resume_generation(monkeypatch):
    from miniworld_engine.autotune import builder, plan, shard
    monkeypatch.setattr(plan, "source_identity", lambda: "same-python-dispatch")
    monkeypatch.setattr(shard, "provenance", lambda: {"gpu": "H100", "env_identity": "env"})
    before = builder._generation_for_work(None)
    monkeypatch.setattr(native, "source_identity", lambda: "new-cuda-or-config")
    assert builder._generation_for_work(None) != before


def test_one_native_cache_entry_cannot_skip_other_workloads():
    from miniworld_engine.autotune import builder
    op = "layernorm_linear_fwd_foldstats_sm90_cute"
    unit = builder.OpUnit(op=op, length=384, dtype="bfloat16", side="pair", width=256)
    assert not builder._cache_answers(unit, {op})


@pytest.mark.parametrize("layout", ["broadcast", "transpose"])
@pytest.mark.parametrize("full_backward", [False, True])
def test_trimul_gate_backward_materializes_strided_inputs(monkeypatch, layout, full_backward):
    import inspect

    from miniworld_engine.kernels.trimul_inproj.triton import gate_elem

    m, n, length = 9, 4, 3
    dy = (torch.ones(1).expand(m, n) if layout == "broadcast"
          else torch.randn(n, m).t())
    proj, gate = torch.randn(n, m).t(), torch.rand(n, m).t()
    dropscale = torch.rand(1, n).expand(length, n)
    class Kernel:
        def __getitem__(self, grid):
            def run(grad, p, g, dp, dg, ds, seq, rows, **kwargs):
                assert all(t.is_contiguous() for t in (grad, p, g, dp, dg, ds))
                scaled = grad * ds[torch.arange(rows) % seq]
                dp.copy_(scaled * g)
                dg.copy_(scaled * p * g * (1 - g))
            return run
    monkeypatch.setattr(gate_elem, "_gate_elem_bwd_ew_kernel", Kernel())
    xn, wg = torch.randn(m, n), torch.randn(n, n)
    if full_backward:
        got = inspect.unwrap(gate_elem.gate_elem_bwd)(dy, xn, proj, gate, wg, dropscale, length)
        fake = gate_elem._gate_elem_bwd_fake(dy, xn, proj, gate, wg, dropscale, length)
    else:
        got = inspect.unwrap(gate_elem.gate_elem_bwd_ew)(dy, proj, gate, dropscale, length)
        fake = gate_elem._gate_elem_bwd_ew_fake(dy, proj, gate, dropscale, length)
    scaled = dy * dropscale[torch.arange(m) % length]
    dg = scaled * proj * gate * (1 - gate)
    expected = ((scaled * gate, dg @ wg.t(), xn.t() @ dg) if full_backward
                else (scaled * gate, dg))
    for actual, reference in zip(got, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    assert [t.stride() for t in got] == [t.stride() for t in fake]
