"""No undefined name anywhere in the package, including the vendored kernel bodies.

`pyproject.toml` excludes `kernels/**/{triton,cute,cuda}/` and the `baseline_dtv1*` modules from
ruff with `= ["ALL"]`. That is the right call for STYLE -- they are faithful ports and a diff
against upstream is worth more than local consistency (see docs/library-standards.md F2). But
`["ALL"]` cannot be un-ignored per rule, so it also switched off `F821 undefined-name`, which is
not a style rule: it is a guaranteed `NameError` at runtime.

It was hiding one. `triangle_attention/triton/atomic.py`'s `backward` called `token_key(L)` where
`L` existed only inside einops pattern STRINGS -- so every backward of that kernel raised
NameError. Nothing caught it: the module imports fine, the op registers fine, and the failure needs
the kernel to actually run. It surfaced in a GPU numerical run as
`checker raised NameError: name 'L' is not defined`, with the traceback swallowed by `check_one`.

So F821 is checked here, over the whole package, with the one known false positive declared rather
than ignored: ruff reads a jaxtyping shape string (`Float[torch.Tensor, "d"]`) as a forward
reference and reports the shape name as undefined. That is the same incompatibility the global
config already ignores `F722` for. A finding on a line that is NOT a jaxtyping annotation is a real
undefined name and fails.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "miniworld_engine"

#: A jaxtyping annotation. Its shape names live in a string ruff parses as a forward reference, so
#: `Float[torch.Tensor, "d"]` reports `d` undefined. Nothing else may.
JAXTYPING = re.compile(r"(?:Float|Int|Bool|UInt|Shaped|Num|Complex)\d*\[")


def _f821() -> list[dict]:
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff not on PATH")
    # `--isolated`: the point is to bypass this repo's per-file-ignores, which is what hid the bug.
    done = subprocess.run(
        [ruff, "check", "--no-cache", "--isolated", "--select", "F821",
         "--output-format", "json", str(PKG)],
        capture_output=True, text=True, check=False)
    assert done.returncode in (0, 1), f"ruff failed: {done.stderr[:400]}"
    return json.loads(done.stdout or "[]")


def test_no_undefined_name_outside_a_jaxtyping_annotation() -> None:
    real = []
    for f in _f821():
        path = Path(f["filename"])
        line = path.read_text().splitlines()[f["location"]["row"] - 1]
        if JAXTYPING.search(line):
            continue          # the declared, understood false positive
        real.append(f"{path.relative_to(REPO)}:{f['location']['row']}: {f['message']} -- {line.strip()}")
    assert not real, (
        "undefined name(s) -- each of these is a NameError the moment that line runs:\n  "
        + "\n  ".join(real))


def test_the_check_is_not_vacuous() -> None:
    """If ruff ever stops reporting the jaxtyping shape names, the filter above is silently
    accepting nothing, and a future real finding might be filtered by a broken regex instead. This
    asserts the mechanism still sees something."""
    findings = _f821()
    assert findings, (
        "F821 now reports nothing at all over the package. Either every jaxtyping annotation is "
        "gone (good -- delete the JAXTYPING filter and this test) or the ruff invocation stopped "
        "working (bad).")
