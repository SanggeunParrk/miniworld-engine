"""The README's first page has to work, or it is the worst kind of documentation.

A quickstart is the one section a newcomer runs verbatim, and the one most likely to rot: it
names commands, paths and expected output, none of which the compiler checks. So the commands
marked `# cpu` are executed here, and their failure is this test's failure.

`# gpu` blocks are collected but not run -- CI has no device. They are still checked for shape,
so a typo in one is caught even though the command is not.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
README = REPO / "README.md"

#: Commands the test refuses to run even when marked cpu: they mutate the environment this suite
#: is running in. `pip install -e .` is in the quickstart because a reader needs it; running it
#: here would reinstall the package mid-suite.
NOT_EXECUTED = ("pip install",)


def blocks() -> list[tuple[str, str]]:
    """(marker, command) for every fenced bash block in the Quickstart section."""
    text = README.read_text()
    start = text.index("## Quickstart")
    end = text.index("\n## ", start + 1)
    return [(m.group(2), m.group(3).strip())
            for m in re.finditer(r"```bash\n(# (cpu|gpu))\n(.*?)```", text[start:end], re.DOTALL)]


def test_the_quickstart_has_runnable_blocks() -> None:
    """Guard the guard: a reorganised section must not leave this checking nothing."""
    found = blocks()
    assert found, "no ```bash blocks with a `# cpu`/`# gpu` marker under ## Quickstart"
    assert any(k == "cpu" for k, _ in found), "no cpu block to execute"
    assert any(k == "gpu" for k, _ in found), "no gpu block -- did the run_all step go?"


def test_every_cpu_command_succeeds() -> None:
    failures = []
    for kind, body in blocks():
        # A block is what the reader selects and pastes, so it runs as one script. Splitting it
        # per line breaks every `python -c "..."` that spans lines, which is most of them.
        if kind != "cpu" or any(skip in body for skip in NOT_EXECUTED):
            continue
        # `set -e`: without it the script's exit code is the LAST command's, so a block whose
        # first line dies still reports success. That is exactly how this test first passed
        # against a README command that raised AttributeError.
        script = "set -e\n" + body.replace("python ", f"{sys.executable} ")
        proc = subprocess.run(script, shell=True, cwd=REPO, capture_output=True,
                              text=True, timeout=300, check=False)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            failures.append(f"{body}\n      exit {proc.returncode}: "
                            f"{tail[-1][:160] if tail else '(no output)'}")
    assert not failures, (
        "a command the README tells a newcomer to run does not work:\n    "
        + "\n    ".join(failures))


def test_the_gpu_command_is_the_one_the_page_describes() -> None:
    """The expected-output line quotes run_all's summary. If the command stops being run_all, the
    quoted line is describing something else."""
    gpu = [b for k, b in blocks() if k == "gpu"]
    assert any("run_all" in b for b in gpu), f"the gpu block no longer runs run_all: {gpu}"
    text = README.read_text()
    start = text.index("## Quickstart")
    section = text[start:text.index("\n## ", start + 1)]
    for word in ("declared", "skipped"):
        assert word in section, (
            f"the quickstart no longer shows {word!r} from run_all's summary, so the expected "
            f"output it promises has drifted from what the command prints")
