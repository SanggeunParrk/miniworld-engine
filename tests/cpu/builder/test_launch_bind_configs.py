"""An autotune axis is supplied by Config; an omitted runtime input is still an error."""
from miniworld_engine.build import launch_bind


def test_config_factory_does_not_hide_a_missing_runtime_argument(tmp_path, monkeypatch):
    source = tmp_path / "miniworld_engine"
    source.mkdir()
    (source / "example.py").write_text("""
import triton

def configs():
    unrelated = {"EPS": 1e-5}
    return [triton.Config({"BR": rows}) for rows in (1, 2, 4)]

@triton.autotune(configs=configs(), key=["N"])
@triton.jit
def kernel(X, N, BR, EPS):
    pass

def bad(x):
    kernel[(1,)](x, N=32)

def good(x):
    kernel[(1,)](x, N=32, EPS=1e-5)
""")
    monkeypatch.setattr(launch_bind, "SRC", tmp_path)
    findings, _, checked, _ = launch_bind.audit()
    assert checked == 2
    assert len(findings) == 1
    assert findings[0][2:] == ("kernel", ["EPS"])
