"""The package declares requires-python >= 3.10 and ships cp310 wheels, so every shipped file must parse on 3.10.

Developed on 3.12, where PEP 701 allows a quote inside an f-string replacement field to repeat the f-string's own quote
(`f"{", ".join(x)}"`), and PEP 695 / `except*` parse. The 2.4.2 wheel carried two such f-strings and `import miniworld_engine.modules`
raised SyntaxError on 3.10. `ast.parse(feature_version=(3, 10))` rejects the newer statements; the nested quotes it cannot see (the 3.12
parser accepts them whatever the feature version), so they are found from the token stream.
"""
from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SRC = sorted((REPO / "src" / "miniworld_engine").rglob("*.py"))


def _quote(start: str) -> str:
    body = start.lstrip("rRbBuUfF")
    return body[:3] if body[:3] in ('"""', "'''") else body[:1]


def _nested_same_quote(text: str) -> list[int]:
    if sys.version_info < (3, 12):
        return []   # this interpreter already refuses them
    stack: list[str] = []
    bad: list[int] = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.FSTRING_START:
            q = _quote(tok.string)
            if q in stack:
                bad.append(tok.start[0])
            stack.append(q)
        elif tok.type == tokenize.STRING and stack and _quote(tok.string) == stack[-1]:
            bad.append(tok.start[0])   # a plain string inside a replacement field, spelled with the f-string's own quote
        elif tok.type == tokenize.FSTRING_END and stack:
            stack.pop()
    return bad


def test_there_is_source_to_check() -> None:
    assert len(SRC) > 100


@pytest.mark.parametrize("path", SRC, ids=lambda p: str(p.relative_to(REPO / "src")))
def test_a_source_file_parses_on_python_310(path: Path) -> None:
    text = path.read_text()
    ast.parse(text, filename=str(path), feature_version=(3, 10))
    assert not _nested_same_quote(text), (
        f"{path.relative_to(REPO)}: an f-string repeats its own quote inside a replacement field "
        f"(lines {_nested_same_quote(text)}): Python 3.12-only syntax, a SyntaxError on 3.10")
