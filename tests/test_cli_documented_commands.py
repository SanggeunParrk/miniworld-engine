"""Every `miniworld-engine ...` line in the docs has to be a command the CLI accepts.

The CLI's own module docstring -- the text `--help` prints -- advertised
`miniworld-engine bench all`. The subcommands are `bench_kernel` and `bench_module`; there has
never been a `bench`. Nine README/doc lines told the reader to run `sbatch submits/run_*.sbatch`
against a tree deleted in 511d905, including the documented way to build the autotune cache.

Documentation drifts silently because nothing executes it. This does: it reads the command lines
back out of the markdown and parses each one. A renamed subcommand or a dropped flag fails here,
in the CPU suite, instead of in someone's terminal.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from miniworld_engine.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "operations" / "dispatch-cache.md",
        ROOT / "src" / "miniworld_engine" / "cli.py"]

#: `<placeholder>` arguments cannot be parsed as themselves; substitute something real.
PLACEHOLDERS = {"<shard-dir>": "/tmp/shards", "<config-set>": "grid", "<op>": "layernorm",
                "<module>": "transition", "<name>": "transition"}


def _commands() -> list[tuple[str, str]]:
    """Command lines, not prose.

    A line counts only if it BEGINS with `miniworld-engine` (after indentation and an optional
    shell prompt), which is what a copy-pasteable command looks like in a fenced block or in the
    CLI docstring. Prose that merely names the tool -- "miniworld-engine is the bottom layer of a
    three-layer stack" -- is not a command and must not be parsed as one.
    """
    out = []
    for doc in DOCS:
        if not doc.is_file():
            continue
        for line in doc.read_text().splitlines():
            m = re.match(r"\s*(?:\$ )?miniworld-engine\s+([a-z_]+[^\n#`'\"]*)", line)
            if not m:
                continue
            out.append((doc.name, m.group(1).strip()))
    return out


def test_the_docs_actually_show_commands():
    """A regex that matches nothing would make every test below vacuously pass."""
    assert len(_commands()) >= 6


@pytest.mark.parametrize(("doc", "cmd"), _commands(), ids=lambda v: v.replace(" ", "_"))
def test_a_documented_command_parses(doc: str, cmd: str):
    argv = [PLACEHOLDERS.get(tok, tok) for tok in cmd.split()]
    parser = build_parser()
    try:
        parser.parse_args(argv)
    except SystemExit as exc:      # argparse exits 2 on an unknown subcommand or flag
        pytest.fail(f"{doc}: `miniworld-engine {cmd}` does not parse (argparse exit {exc.code})")


def test_an_undefined_subcommand_would_be_caught():
    """The guard above only means something if a bad command really fails to parse."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bench", "all"])


def test_build_parser_returns_a_parser():
    assert isinstance(build_parser(), argparse.ArgumentParser)
