"""Assemble and build the K1/K3 inference payload: Anthropic native v5 `trimul_native` (byte-identical upstream base) + this experiment's
overlay, compiled with the recorded TMN_* switches into <out>/{csrc,python,build,testvectors}.

    python build_payload.py --upstream <dir holding csrc/ python/ VERSION of pkg/v5> [--out payload] [--jobs 4] [--probe] [--no-vectors]

The upstream directory is the `common/opt_core/opt_core/kernels/trimul/native/pkg/v5` tree of anthropics/uplifting-biomolecular-modeling at the
revision named in OVERLAY.json (default: $MINIWORLD_ANTHROPIC_ROOT/<source_prefix>).  Base files are hash-checked before the overlay is applied.
Serve the result through `trimul_native.face` with TRIMUL_NATIVE_BUILD_DIR=<out>/build (the face refuses stale test vectors, so the vectors
step needs an H100; build and vectors must run on a compute node, not a login node).  --probe adds the per-CTA timestamp hooks (measurement only)."""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = json.loads((HERE / "OVERLAY.json").read_text())


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default=None, help="pkg/v5 directory (csrc/, python/, VERSION)")
    ap.add_argument("--out", default=str(HERE / "payload"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--probe", action="store_true", help="also define TMN_CTA_TS=1 (per-CTA clock probes for probe_cta_timeline.py)")
    ap.add_argument("--no-vectors", action="store_true", help="skip `vectors make` (the face will refuse to serve until vectors exist)")
    ap.add_argument("--grid", default="r2", help="test-vector grid for `vectors make` (r2 = the release grid, smoke = 4 cases)")
    ap.add_argument("--force", action="store_true", help="apply the overlay even if the upstream base hashes differ")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--define", action="append", default=[], help="extra K=V switch on top of the recorded ones (e.g. TMN_BIDIR_TILES=1)")
    ap.add_argument("--unit", default=OVERLAY["unit"], help="unit(s) to compile, comma-separated (default: the recorded %s; tmn90_z128_h256 is the "
                    "bidirectional c_hidden=256 shape, and one payload can carry both)" % OVERLAY["unit"])
    a = ap.parse_args()
    up = a.upstream or (os.environ.get("MINIWORLD_ANTHROPIC_ROOT") and os.path.join(os.environ["MINIWORLD_ANTHROPIC_ROOT"], OVERLAY["upstream"]["source_prefix"]))
    if not up or not (Path(up) / "csrc").is_dir() or not (Path(up) / "python" / "trimul_native").is_dir():
        sys.exit("need --upstream <pkg/v5 dir with csrc/ and python/trimul_native/> (or MINIWORLD_ANTHROPIC_ROOT)")
    up = Path(up).resolve()
    bad = [f for f, h in OVERLAY["base_files"].items() if not (up / f).is_file() or sha256(up / f) != h]
    if bad and not a.force:
        sys.exit("upstream base differs from OVERLAY.json (revision %s) for: %s  (use --force to overlay anyway)" % (OVERLAY["upstream"]["revision"][:12], ", ".join(bad)))
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit("refusing to overwrite %s (remove it first)" % out)
    out.mkdir(parents=True)
    shutil.copytree(up / "csrc", out / "csrc")
    shutil.copytree(up / "python", out / "python", ignore=shutil.ignore_patterns("__pycache__"))
    for extra in ("VERSION", "CELLS.json", "README.md", "LICENSE", "NOTICE"):
        if (up / extra).is_file():
            shutil.copy2(up / extra, out / extra)
    for f, h in OVERLAY["overlay_files"].items():
        src = HERE / "overlay" / f
        if sha256(src) != h:
            sys.exit("overlay file %s does not match OVERLAY.json" % f)
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / f)
    (out / "OVERLAY.json").write_text(json.dumps(OVERLAY, indent=1))
    defines = dict(OVERLAY["defines"])
    if a.probe:
        defines.update(OVERLAY["probe_define"])
    for d in a.define:
        k, _, v = d.partition("=")
        defines[k] = v or "1"
    env = dict(os.environ, PYTHONPATH=str(out / "python") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    cmd = [a.python, "-m", "trimul_native.build", "--archs", ",".join(OVERLAY["archs"]), "--units", a.unit, "--src", str(out / "csrc"),
           "--out", str(out / "build"), "--lineinfo", "--jobs", str(a.jobs)]
    for k, v in defines.items():
        cmd += ["--define", "%s=%s" % (k, v)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=out, env=env, check=True)
    man = json.loads((out / "build" / "manifest.json").read_text())
    for name in a.unit.split(","):
        unit = man["units"]["%s/%s" % (name.strip(), OVERLAY["archs"][0])]
        spills = {k: (v.get("spill_stores"), v.get("spill_loads")) for k, v in unit["kernels"].items() if v.get("spill_stores") or v.get("spill_loads")}
        print("%s: built %d kernels; cubin %s; spilling kernels: %s" % (name.strip(), len(unit["kernels"]), unit.get("cubin_sha256", "?")[:12],
                                                                        ", ".join(sorted(spills)) if spills else "none"), flush=True)
    if not a.no_vectors:
        (out / "testvectors").mkdir(exist_ok=True)
        cmd = [a.python, "-m", "trimul_native.vectors", "make", "--out", str(out / "testvectors"), "--grid", a.grid, "--no-ref", "--no-isolated"]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=out, env=dict(env, TRIMUL_NATIVE_BUILD_DIR=str(out / "build")), check=True)
    print("\nexport TRIMUL_NATIVE_BUILD_DIR=%s" % (out / "build"))
    print("PYTHONPATH must NOT contain another trimul_native; the face resolves csrc/ and testvectors/ next to python/ and verifies the manifest digests.")


if __name__ == "__main__":
    main()
