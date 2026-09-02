"""Find every numeric literal that decides something in a kernel launcher.

Regenerates the table in `thresholds.md`. Run from the repo root:

    python docs/kernels/audit_thresholds.py

Comparisons INSIDE a ``@triton.jit`` body are skipped: there a constexpr comparison is
tile algebra, not policy. Values under 8, and 0/1/2/-1/100/1000, are skipped as noise.
Classification into architecture predicate / threshold / bound is NOT done here -- it
needs reading the site, and the verdicts live in thresholds.md.
"""
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path("src/miniworld_engine/kernels")
SKIP = {0, 1, 2, -1, 100, 1000}

class Finder(ast.NodeVisitor):
    def __init__(self, src, rel):
        self.src, self.rel, self.hits, self.jit = src, rel, [], 0
    def visit_FunctionDef(self, node):
        dec = any("jit" in ast.unparse(d) for d in node.decorator_list)
        self.jit += dec
        self.generic_visit(node)
        self.jit -= dec
    def visit_Compare(self, node):
        if not self.jit:
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)) \
                   and not isinstance(operand.value, bool) \
                   and operand.value not in SKIP and abs(operand.value) >= 8:
                    self.hits.append((self.rel, node.lineno, operand.value,
                                      ast.unparse(node)))
        self.generic_visit(node)

rows = []
for path in sorted(ROOT.rglob("*.py")):
    rel = str(path.relative_to(ROOT))
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        continue
    f = Finder(path.read_text(), rel); f.visit(tree)
    rows.extend(f.hits)
print(json.dumps(rows, indent=0))
print(len(rows), file=sys.stderr)
