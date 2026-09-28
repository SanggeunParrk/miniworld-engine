"""Import the pinned inference release, retaining attribution and a file inventory.

Run on a compute node. The checkout must include common/, every kit's opt/,
and the top-level/kit license and notice files. Stock models are not imported.
"""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

REVISION = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
URL = "https://github.com/anthropics/uplifting-biomolecular-modeling"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    rev = subprocess.check_output(["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise SystemExit(f"Expected {REVISION}, got {rev}")
    entries = subprocess.check_output(["git", "-C", str(a.source), "ls-tree", "-r", "HEAD"], text=True).splitlines()
    files, kernels = [], []
    for entry in entries:
        head, name = entry.split("\t", 1)
        mode, kind, blob = head.split()
        src = a.source / name
        if not src.is_file() or "stock" in Path(name).parts:
            continue
        if mode == "120000":
            raise RuntimeError(f"Review symlink before import: {name}")
        data = src.read_bytes()
        got = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if got != blob:
            raise RuntimeError(f"Modified upstream file: {name}")
        dst = a.destination / "upstream" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rec = {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "git_blob": blob, "bytes": len(data)}
        files.append(rec)
        if src.suffix not in {".py", ".cu", ".cuh", ".cpp", ".h"}:
            continue
        text = data.decode("utf-8", errors="replace")
        if not any(t in text for t in ("@triton.jit", "@triton.autotune", "__global__", "@cute.jit", "@cute.kernel", "pallas_call", "pl.pallas_call", "@gl.jit")):
            continue
        symbols = []
        if src.suffix == ".py":
            try:
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        decorators = [ast.unparse(d) for d in node.decorator_list]
                        if any(re.search(r"(jit|kernel|autotune)", d) for d in decorators):
                            symbols.append({"name": node.name, "line": node.lineno, "decorators": decorators})
            except SyntaxError:
                pass
        kernels.append({**rec, "symbols": symbols, "backend": "cuda" if src.suffix != ".py" else ("pallas" if "pallas_call" in text else "cute" if "@cute." in text else "triton"),
                        "backward_text_present": bool(re.search(r"backward|\bbwd\b|custom_vjp", text)),
                        "status": "imported_unqualified"})
    a.destination.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": URL, "revision": rev, "files": files, "kernel_sources": kernels,
                "scope": "All checked-out optimization kits and common; stock models excluded. Kernel detection is syntactic, not a complete call graph.",
                "counts": {"files": len(files), "kernel_source_files": len(kernels), "unique_kernel_source_sha256": len({r['sha256'] for r in kernels}), "backends": dict(collections.Counter(r['backend'] for r in kernels))}}
    (a.destination / "UPSTREAM.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Imported kernel source inventory", "", f"Upstream: [{rev}]({URL}/tree/{rev})", "", "A source inventory, not proof of execution or full training support. Duplicate SHA-256 values identify identical copies. Per-source GPU qualification is recorded separately.", "", "| Source | Backend | GPU entries | SHA-256 |", "|---|---|---:|---|"]
    for r in kernels:
        lines.append(f"| [{r['path']}](upstream/{r['path']}) | {r['backend']} | {len(r['symbols'])} | `{r['sha256'][:16]}` |")
    (a.destination / "INVENTORY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(manifest['counts']))


if __name__ == "__main__":
    main()
