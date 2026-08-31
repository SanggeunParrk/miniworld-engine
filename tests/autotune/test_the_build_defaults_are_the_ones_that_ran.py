"""`build` with no flags is the build that gets run.

Three options were opt-in while they were being trusted -- `--predict-unusable`, `--pin-cores`,
and a `--bench-rep-ms` that is not triton's 100. Every build since has set all three, so the flags
that had to be remembered were the ones that made the build correct and cheap, and forgetting one
was silent: the run finished, it just cost more or measured worse. Off is the thing you ask for.

What each is worth, from the runs that set them:

  --predict-unusable   9-29% of a searched space could not run on the card it was searched for
                       (shared memory, registers). Probing a slice first skips those before the
                       compile rather than after it.
  --pin-cores          a unit that is MEASURING otherwise shares cores with every other unit's
                       compile workers, and the measurement is the only thing a build produces.
  --bench-rep-ms 25    bench is ~97% of a unit's wall time (4,560 s against 123 s of compile on a
                       measured unit), so this number is most of what a build costs.

This does not claim the values are optimal. It keeps the default and the practice from drifting
apart again: whatever gets run, a bare `build` should be that.
"""
from __future__ import annotations

import ast

from paths import ROOT

CLI = ROOT / "src" / "miniworld_engine" / "cli.py"

#: flag -> the value a bare `build` must produce.
EXPECTED = {"--predict-unusable": True, "--pin-cores": True, "--bench-rep-ms": 25}
#: The two booleans must stay overridable. A default that cannot be turned off is a constant, and
#: these are choices -- so they take BooleanOptionalAction rather than flipping to store_false.
OVERRIDABLE = ("--predict-unusable", "--pin-cores")


def _build_flags() -> dict[str, dict]:
    """Every `--flag` the build subparser declares, with its default and action."""
    out: dict[str, dict] = {}
    for node in ast.walk(ast.parse(CLI.read_text())):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "add_argument":
            continue
        if not node.args or not isinstance(getattr(node.args[0], "value", None), str):
            continue
        name = node.args[0].value
        if not name.startswith("--"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        out.setdefault(name, {
            "default": getattr(kw.get("default"), "value", None),
            "action": ast.unparse(kw["action"]) if "action" in kw else None,
        })
    return out


def test_a_bare_build_runs_what_the_operator_runs() -> None:
    flags = _build_flags()
    wrong = {}
    for name, want in EXPECTED.items():
        spec = flags.get(name)
        got = True if spec and spec["action"] == "argparse.BooleanOptionalAction" and \
            spec["default"] is True else spec and spec["default"]
        if got != want:
            wrong[name] = (got, want)
    assert not wrong, (
        "`build` with no flags no longer matches the build that gets run. Each of these was "
        "measured to be worth having and is not something a caller turns off by accident -- if "
        f"one is changing on purpose, change it here too (got, expected): {wrong}")


def test_each_default_on_option_can_still_be_turned_off() -> None:
    flags = _build_flags()
    pinned = [n for n in OVERRIDABLE
              if flags.get(n, {}).get("action") != "argparse.BooleanOptionalAction"]
    assert not pinned, (
        f"{pinned} default on but offer no --no- form, so a caller who needs one off cannot say so")
