"""The benchmark plotter (``measure.plot``) on a tiny synthetic set of speed records: figures, per-class tables and the index."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("matplotlib")

from miniworld_engine.measure import plot


def _rec(backend: str, length: int, ms: float | None, *, mode: str = "eval", status: str = "ok", **extra) -> dict:
    return {"row": 1, "module": "LayerNorm", "model": "", "stream": "token_pair", "kwargs": {"normalized_shape": 128, "bias": True},
            "length": length, "depth": None, "samples": 1, "dtype": "master", "mode": mode, "backend": backend, "timing": "graph",
            "gpu": "h100_sm90", "status": status, "median_ms": ms, "p10_ms": ms and ms * 0.95, "p90_ms": ms and ms * 1.05, **extra}


@pytest.fixture
def records(tmp_path):
    path = tmp_path / "speed.jsonl"
    recs = [_rec(b, length, t * length / 128, mode=mode)
            for mode in ("eval", "train") for length in (128, 256, 384) for b, t in (("pytorch", 1.0), ("cuda", 0.4), ("triton", 0.6))]
    recs += [_rec("anthropic", 128, None, status="unsupported", reason="no"),
             _rec("cueq", 256, 0.9, timed_as="eager"),
             _rec("cuda", 128, 1.0, gpu="a100_sm80")]   # another card: not plotted
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


def test_load_reads_records_of_the_card(records):
    points, what = plot.load([str(records)], "h100_sm90")
    assert len(points) == 2 * 3 * 3 + 2
    assert "speed records" in what
    assert sum(p["eager"] for p in points) == 1


def test_render_one_class(records, tmp_path):
    out = tmp_path / "out"
    assert plot.main(["--source", str(records), "--module", "LayerNorm", "--out", str(out)]) == 0
    mdir = out / "h100_sm90" / "LayerNorm"
    svgs = sorted(mdir.glob("*.svg"))
    assert len(svgs) == 1 and svgs[0].name.startswith("L__")   # one L sweep slice (both modes are panels of it)
    readme = (mdir / "README.md").read_text()
    assert "L sweep" in readme
    assert "| 128 |" in readme   # the table of the sweep: one line per length


def test_render_with_summary(records, tmp_path):
    out = tmp_path / "out"
    assert plot.main(["--source", str(records), "--out", str(out), "--model-summaries"]) == 0
    card = out / "h100_sm90"
    assert (card / "summary.svg").is_file()
    index = (card / "README.md").read_text()
    assert "[LayerNorm](LayerNorm/README.md)" in index
    stats = plot.summary_stats(plot.load([str(records)], "h100_sm90")[0], plot.FIGURE_BACKENDS)
    assert stats[("LayerNorm", "eval")]["cuda"] == pytest.approx((2.5, 3))   # pytorch / cuda = 1 / 0.4 at every length


def test_no_points_is_exit_1(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert plot.main(["--source", str(empty), "--out", str(tmp_path / "o")]) == 1
