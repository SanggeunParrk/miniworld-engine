"""Record an sm80 local rebuild without changing the upstream release manifest.

Run after upstream pkg/v11/tools/build_prebuilt.py in a scratch copy. This checks
the compiler's source/binary hashes and the sources against the pinned checkout,
then writes a separate, ABI-local SHA256SUMS consumed by the upstream loader.
The loader's independent GPU byte gate still runs on first use.
"""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from miniworld_engine.integrations.anthropic import REVISION

NATIVE = Path("common/opt_core/opt_core/kernels/triattn/triattn_native")
PAYLOAD = NATIVE / "pkg/v11/triattn_pkg"
EXT = "triattn_sm80_ext"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_build(root: Path, source: Path, stack: str):
    root, source = root.resolve(), source.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("Use a separate scratch copy, outside the upstream checkout")
    if Path(stack).name != stack or stack in ("", ".", ".."):
        raise ValueError("stack must be one ABI directory name")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != REVISION:
        raise ValueError(f"Expected upstream {REVISION}, got {commit}")
    build = root / PAYLOAD / "prebuilt" / stack
    rec = json.loads((build / f"{EXT}.json").read_text())
    if rec["stack_tag"] != stack or rec["ext"] != EXT or rec["arch"] != ["arch=compute_80,code=sm_80"]:
        raise ValueError("Build record is not this ABI's sm80 extension")
    kernel = root / PAYLOAD / "cuda_80"
    actual = {str(p.relative_to(kernel / "csrc")): sha256(p)
              for p in sorted((kernel / "csrc").rglob("*")) if p.is_file()}
    if actual != rec["source_sha256"]:
        raise ValueError("Kernel sources do not match the build record")
    if rec["module_sha256"] != {"triattn_sm80.py": sha256(kernel / "triattn_sm80.py")}:
        raise ValueError("Build module does not match the build record")
    # Check all tracked source, loader and test-vector bytes in the copied core;
    # only new, untracked local build files may differ from the pinned tree.
    prefix = "common/opt_core"
    verified = 0
    with subprocess.Popen(["git", "-C", str(source), "archive", REVISION, prefix],
                          stdout=subprocess.PIPE) as proc:
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    original = archive.extractfile(member)
                    if (root / member.name).read_bytes() != original.read():
                        raise ValueError(f"Scratch copy differs from pinned upstream: {member.name}")
                    verified += 1
            if proc.wait():
                raise RuntimeError("Could not read the pinned upstream archive")
        except BaseException:
            proc.kill()
            raise
    digest = sha256(build / f"{EXT}.so")
    if digest != rec["so_sha256"]:
        raise ValueError("Binary does not match the build record")
    sums = f"{digest}  {EXT}.so\n"
    manifest = build / "SHA256SUMS"
    if manifest.exists() and manifest.read_text() != sums:
        raise ValueError("ABI directory already has a different local manifest; preserve it before rebuilding")
    report = {"kind": "local rebuild, not an upstream-certified binary", "upstream_revision": commit,
              "upstream_manifest_sha256": sha256(root / NATIVE / "SHA256SUMS"),
              "verified_upstream_files": verified, "build_record": rec,
              "gpu_validation": "pending: triattn_native.install() must pass unchanged upstream vectors"}
    (build / "LOCAL_BUILD.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest.write_text(sums)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Scratch upstream root")
    parser.add_argument("--source", type=Path, required=True, help="Pinned upstream Git checkout")
    parser.add_argument("--stack", required=True, help="ABI directory produced by build_prebuilt.py")
    args = parser.parse_args()
    print(json.dumps(record_build(args.root, args.source, args.stack), indent=2))


if __name__ == "__main__":
    main()
