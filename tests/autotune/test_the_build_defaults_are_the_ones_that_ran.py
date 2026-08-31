"""`build all` with no flags is the build that has been running.

Three options were opt-in while they were being trusted -- `--predict-unusable`, `--pin-cores`,
and a `--bench-rep-ms` that is not triton's 100. Every build since has passed all three, so the
flags that had to be remembered were the ones that made the build correct and cheap, and
forgetting one was silent: the run finished, it just cost more or measured worse. Off is now the
thing you ask for.

What each is worth, from the runs that set them:

  --predict-unusable   9-29% of a searched space could not run on the card it was searched for
                       (shared memory, registers). Probing a slice first skips those before the
                       compile rather than after it.
  --pin-cores          a unit that is MEASURING otherwise shares cores with every other unit's
                       compile workers, and the measurement is the only thing a build produces.
  --bench-rep-ms 25    bench is ~97% of a unit's wall time (4,560 s against 123 s of compile on a
                       measured unit), so this number is most of what a build costs.

This test is not about the values being optimal. It is about the default and the practice not
drifting apart again: whatever the operator runs, `build` with no flags should be that.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

#: option -> the value a bare `build` must produce.
EXPECTED = {"predict_unusable": True, "pin_cores": True, "bench_rep_ms": 25}


def _defaults() -> dict:
    """Parse `build all` with no other flags and read the namespace back."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, json; sys.path.insert(0, %r);"
         "from miniworld_engine import cli;"
         "ns = cli.build_parser().parse_args(['build', 'all']) if hasattr(cli, 'build_parser')"
         " else cli._parser().parse_args(['build', 'all']);"
         "print(json.dumps({k: v for k, v in vars(ns).items() if isinstance(v, (bool, int))}))"
         % str(ROOT / "src")],
        capture_output=True, text=True, check=False)
    if out.returncode != 0:
        import json

        # No exported parser factory: build the namespace the way __main__ does instead.
        src = (ROOT / "src" / "miniworld_engine" / "cli.py").read_text()
        assert "add_argument" in src
        ns = {}
        for name, want in EXPECTED.items():
            flag = "--" + name.replace("_", "-")
            if f'"{flag}", action=argparse.BooleanOptionalAction, default=True' in src:
                ns[name] = True
            elif f'"{flag}", action="store_true"' in src:
                ns[name] = False
            else:
                import re
                m = re.search(rf'"{re.escape(flag)}", type=int, default=(\d+)', src)
                ns[name] = int(m.group(1)) if m else None
        return ns
    import json

    return json.loads(out.stdout)


def test_a_bare_build_runs_what_the_operator_runs() -> None:
    got = _defaults()
    wrong = {k: (got.get(k), v) for k, v in EXPECTED.items() if got.get(k) != v}
    assert not wrong, (
        "`build` with no flags no longer matches the build that gets run. Each of these was "
        "measured to be worth having and is not something a caller has a reason to turn off by "
        "accident -- if one is being changed on purpose, change it here too "
        f"(got, expected): {wrong}")


def test_each_default_on_option_can_still_be_turned_off() -> None:
    """A default that cannot be overridden is a constant, and these are choices."""
    src = (ROOT / "src" / "miniworld_engine" / "cli.py").read_text()
    for name in ("predict_unusable", "pin_cores"):
        flag = "--" + name.replace("_", "-")
        assert f'"{flag}", action=argparse.BooleanOptionalAction' in src, (
            f"{flag} defaults on but offers no --no-{flag[2:]}, so a caller who needs it off has "
            f"no way to say so")
