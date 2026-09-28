"""Execution/reporting contracts independent of GPU speed or kernel numerics."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from miniworld_engine import cli

REPO = Path(cli.__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("plot_csv_contract", REPO / "benchmarks/runners/plot_csv.py")
assert spec is not None
assert spec.loader is not None
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def row(**changes):
    return {"target": "layernorm_bwd", "level": "kernel", "mode": "training", "metric": "time", "unit": "ms",
                "implementation": "pytorch", "seq_len": "128", "d_pair": "384", "value": "1.0",
                "precision": "bf16", "compiled": "True", "cudagraph": "manual", "device": "A6000",
                "compile_requested": "True", "cudagraph_requested": "manual", "run_name": "test",
                "sweep_axis": "seq_len", "torch_version": "2.10", "cuda_version": "12.9", "measurement_schema": "2"} | changes


def test_plot_refuses_duplicate_points_and_alias_overwrites():
    with pytest.raises(ValueError, match="duplicate measurement"):
        plot.series_by_impl([row(implementation="ours"), row(implementation="miniworld")], "seq_len")


def test_legacy_csv_requires_explicit_unverified_mode():
    legacy = row(measurement_schema="")
    with pytest.raises(ValueError, match="measurement_schema=2"):
        plot.series_by_impl([legacy], "seq_len")
    legacy["_allow_legacy_unverified"] = "true"
    assert plot.series_by_impl([legacy], "seq_len")
    assert "LEGACY UNVERIFIED" in plot.caption([legacy])


def test_legacy_and_verified_cannot_share_a_plot_even_with_override():
    with pytest.raises(ValueError, match="mixed measurement schemas"):
        plot.series_by_impl([row(_allow_legacy_unverified="true"),
                             row(seq_len="256", measurement_schema="", _allow_legacy_unverified="true")], "seq_len")


@pytest.mark.parametrize(("field", "value"), [("precision", "fp32"), ("compile_requested", "False"),
                                        ("run_name", "other"), ("d_pair", "512")])
def test_plot_refuses_mixed_requested_conditions(field, value):
    with pytest.raises(ValueError, match="mixed benchmark conditions"):
        plot.series_by_impl([row(), row(implementation="miniworld", **{field: value})], "seq_len")


def test_plot_labels_actual_regimes_per_implementation():
    rows = [row(parameter_dtype="float32", compile_scope="callable:fullgraph"),
            row(implementation="miniworld", compiled="False", parameter_dtype="bfloat16",
                compile_scope="none")]
    assert len(plot.series_by_impl(rows, "seq_len")) == 2
    assert "compiled=False" in plot.series_label("miniworld", rows)
    assert "parameter_dtype=bfloat16" in plot.series_label("miniworld", rows)
    assert "see implementation labels" in plot.caption(rows)


def test_failed_rows_do_not_make_actual_metadata_unknown_for_successes():
    rows = [row(), row(implementation="miniworld", value="", compiled="", cudagraph="")]
    assert list(plot.series_by_impl(rows, "seq_len")) == ["pytorch"]
    assert "compiled=True" in plot.caption(rows)


def test_vendor_label_discloses_actual_pytorch_reference_execution():
    rows = [row(execution_path="module.reference.torch"),
            row(implementation="cuequivariance", execution_path="module.reference.torch"),
            row(implementation="cuequivariance", seq_len="256", execution_path="module.reference.torch"),
            row(implementation="cuequivariance", seq_len="384", value="", execution_path="")]
    label = plot.series_label(plot.canonical("cuequivariance"), rows)
    assert "execution=pytorch reference" in label
    assert "execution=pytorch reference" not in plot.series_label("pytorch", rows)
    rows[2]["execution_path"] = "cuequivariance_torch.attention"
    # A legitimate shape-dependent dispatch change remains valid; do not claim
    # that every point executed the reference when one used the vendor backend.
    assert plot.series_by_impl(rows, "seq_len")
    assert "execution=pytorch reference" not in plot.series_label(plot.canonical("cuequivariance"), rows)


def test_plot_does_not_mix_actual_regimes_within_a_series():
    with pytest.raises(ValueError, match="mixed actual compiled"):
        plot.series_by_impl([row(), row(seq_len="256", compiled="False")], "seq_len")


def test_plot_filenames_preserve_regime_and_repetition_provenance():
    names = {plot.output_stem([r], None) for r in
             [row(), row(compiled="False"), row(value="1.1"), row(precision="fp32")]}
    assert len(names) == 4


@pytest.mark.parametrize(("visible", "device", "expected"), [
    ("3,5", 1, "5"), ("GPU-aaa,GPU-bbb", 0, "GPU-aaa"),
    ("MIG-aaa,MIG-bbb", 1, "MIG-bbb"), (None, 2, "2"),
])
def test_child_gpu_mask_preserves_parent_mapping(monkeypatch, visible, device, expected):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert cli._bench_visible_device(device) == expected
    assert cli._worker_env(device)["CUDA_VISIBLE_DEVICES"] == expected


@pytest.mark.parametrize(("visible", "device"), [("", 0), ("-1", 0), ("3", 1), ("3", -1)])
def test_invalid_visibility_fails_before_launch(monkeypatch, visible, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match=r"device|CUDA_VISIBLE_DEVICES"):
        cli._bench_visible_device(device)


def test_each_gpu_queue_stays_serial_when_another_gpu_finishes_early(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setattr(cli, "_resolve_gpus", lambda _spec: [0, 1])
    monkeypatch.setattr(cli, "_bench_cmd", lambda _args, target, *_rest: ([target], None))
    monkeypatch.setattr(cli, "_report_coverage", lambda *args, **kwargs: 0)
    lock, release = threading.Lock(), threading.Event()
    active, overlaps, launched = {}, [], []

    def fake_run(cmd, **kwargs):
        gpu, target = kwargs["env"]["CUDA_VISIBLE_DEVICES"], cmd[0]
        with lock:
            active[gpu] = active.get(gpu, 0) + 1
            if active[gpu] != 1:
                overlaps.append((gpu, target))
            launched.append((gpu, target))
        if target == "t0":
            assert release.wait(3), "GPU 1 must independently drain its queue"
        if target == "t3":
            release.set()
        with lock:
            active[gpu] -= 1
        return SimpleNamespace(returncode=7 if target == "t2" else 0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    rc = cli._run_bench(argparse.Namespace(gpus="all"), ("t0", "t1", "t2", "t3"), tmp_path, level="module")
    assert not overlaps
    assert set(launched) == {("3", "t0"), ("5", "t1"), ("3", "t2"), ("5", "t3")}
    assert rc == 7


@pytest.mark.parametrize("script", sorted(REPO.glob("*.sbatch")), ids=lambda p: p.name)
def test_sbatch_propagates_phase_pipeline_and_postprocess_failures(script):
    source = script.read_text()
    assert "set -Eeuo pipefail" in source
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
    # Exercise the exact options and ERR trap from the script, with harmless
    # failing stand-ins for a benchmark, one loop phase, and postprocessing.
    setup = "\n".join(line for line in source.splitlines()
                      if line.startswith("set -") or (line.startswith("trap ") and line.endswith(" ERR")))
    for body in (
        "bash -c 'exit 17' | cat\necho false-success",
        "for phase in 1 2; do bash -c 'exit 19'; echo swallowed; done\necho false-success",
        "true | bash -c 'exit 23'\necho false-success",
        "bash -c 'exit 29' &\nwait $!\necho false-success",
    ):
        done = subprocess.run(["bash", "-c", setup + "\n" + body], capture_output=True, text=True)
        assert done.returncode != 0
        assert "false-success" not in done.stdout


@pytest.mark.parametrize("old_probability", ["", "0.0"])
def test_plot_refuses_to_mix_dropout_workloads(old_probability):
    with pytest.raises(ValueError, match="mixed benchmark conditions for dropout"):
        plot.series_by_impl([row(dropout=old_probability),
                             row(implementation="miniworld", dropout="0.25")], "seq_len")
    assert "dropout=0.25" in plot.caption([row(dropout="0.25")])
    assert "dropout=unrecorded" in plot.caption([row()])
