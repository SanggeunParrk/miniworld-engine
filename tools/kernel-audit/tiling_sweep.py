"""Run every checker under one tile configuration and record the numbers.

``MINIWORLD_CONFIG_DIR`` must already be in the environment: ``autotune.configs`` selects the
set when it loads, which is necessarily before any kernel module imports, so the choice cannot
be made from inside this script after the fact.

Only triton kernels are swept. The config sets feed ``configs_for`` and nothing else, so a CUDA
or CUTE kernel's tiling is fixed in its source and re-running it per set measures nothing while
paying a multi-minute nvcc build each time.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

# Do NOT shadow PYTHONPATH: an inserted path takes precedence over it, so a caller that points
# PYTHONPATH at its own copy of the package would still silently import the repo's. Append, so
# "src" is only the fallback for a plain invocation from the repo root.
if "src" not in sys.path:
    sys.path.append("src")



def check_and_hash(check: str, *, seed: int = 0, dirty: bool = False) -> tuple[bool, str, str]:
    """``run_all.check_one``'s comparison, plus a hash of the raw bytes the kernel produced.

    The rel value alone cannot tell a real tile-size effect from bf16 output quantization: rel is
    ``max|a-e| / max|e|``, and a bf16 result rounds a small accumulation-order difference away, so
    two genuinely different tilings routinely report the identical rel to three significant
    figures. Hashing the actual tensor answers the question the rel value cannot -- whether the
    config change reached the kernel at all.
    """
    import hashlib

    import torch

    from miniworld_engine.autotune.run_all import _resolve

    fn = _resolve(check)
    if dirty:
        # Isolation removes the cascade but replaces it with a false negative: a fresh process
        # reads zeros past the end of a tensor, so `x_extra * w_extra` vanishes and an
        # out-of-bounds read produces the right answer. Dirtying the allocator first puts non-zero
        # bytes where the next allocation will land, which is what made the trimul bug visible.
        scratch = torch.full((1 << 26,), 3.7, device="cuda", dtype=torch.float32)
        scratch.uniform_(-2.0, 2.0)
        del scratch
    if seed:
        # The checkers draw fresh randn inputs on every call, so without a fixed seed two runs
        # differ by RNG and the hash compares nothing. Seed immediately before the call, not once
        # per process: the number of draws per kernel varies, so a single global seed would leave
        # every kernel after the first offset between runs.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        got = fn()
    except Exception as exc:                                     # noqa: BLE001
        return False, f"checker raised {type(exc).__name__}: {str(exc).strip().splitlines()[0][:150]}", ""
    pairs = got if isinstance(got, dict) else {"out": got}
    worst, detail, h = 0.0, [], hashlib.sha256()
    for name in sorted(pairs):
        actual, expected = pairs[name]
        a, e = actual.float(), expected.float()
        if a.shape != e.shape:
            return False, f"{name}: shape {tuple(a.shape)} vs reference {tuple(e.shape)}", ""
        num = (a - e).abs().max().item()
        den = e.abs().max().item() or 1.0
        rel = num / den
        worst = max(worst, rel)
        detail.append(f"{name}={rel:.2e}")
        h.update(name.encode())
        # numpy has no bfloat16; widening to fp32 is lossless for bf16/fp16/fp32, so the
        # hash still distinguishes any change in the bits the kernel actually wrote.
        h.update(actual.detach().contiguous().float().cpu().numpy().tobytes())
    ok = worst < 5e-2 and worst == worst
    return ok, ("rel " + " ".join(detail)) if detail else "checker returned nothing", h.hexdigest()[:16]



def run_isolated(args, rows) -> int:
    """One subprocess per kernel, then merge. Needed because CUDA errors are sticky: after an
    illegal address or misaligned access the context is dead, and every subsequent launch in the
    same process raises the *first* kernel's error. A single-process sweep therefore reports one
    real failure as N failures and silently loses the verdict for every kernel after it.
    """
    import subprocess
    import tempfile

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged, fields = [], ["kernel", "family", "phase", "ok", "worst", "sha", "detail"]
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(rows, 1):
            k = r["kernel"]
            cmd = [sys.executable, __file__, "--label", "one", "--out", tmp,
                   "--ops", k, "--backends", args.backends, "--seed", str(args.seed),
                   "--repeat", str(args.repeat)]
            if args.check_override:
                cmd += ["--check-override", args.check_override]
            if args.dirty:
                cmd.append("--dirty")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            f = Path(tmp) / "one.csv"
            got = list(csv.DictReader(f.open())) if f.is_file() else []
            if f.is_file():
                f.unlink()
            if got:
                merged.extend(got)
                rec = got[0]
                # a no-driver row carries ok="" and is not counted as a failure in the summary;
                # printing it as FAIL made the two disagree
                mark = 'ok  ' if rec['ok'] == '1' else ('----' if rec['ok'] == '' else 'FAIL')
                print(f"  [{mark}] {rec['phase']:11s} "
                      f"{k:46s} {rec['detail'][:80]}", flush=True)
            else:
                tail = (proc.stderr or proc.stdout).strip().splitlines()
                why = tail[-1][:150] if tail else f"no output, rc={proc.returncode}"
                merged.append(dict(kernel=k, family=r["family"], phase="crashed", ok="0",
                                   worst="", sha="", detail=f"subprocess produced no row: {why}"))
                print(f"  [FAIL] crashed     {k:46s} {why[:80]}", flush=True)
    dest = out / f"{args.label}.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)
    bad = [r for r in merged if r["ok"] == "0"]
    print(f"\n{args.label}: {len(merged)} kernels   isolated   failing {len(bad)}   -> {dest}")
    for r in bad:
        print(f"   FAIL {r['phase']:11s} {r['kernel']:46s} {r['detail'][:110]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="name for the output CSV")
    ap.add_argument("--out", default=".bench/tiling")
    ap.add_argument("--backends", default="triton")
    ap.add_argument("--ops", default="", help="comma list of kernel names; default all")
    ap.add_argument("--isolate", action="store_true",
                    help="run every kernel in its own subprocess. A CUDA error poisons the whole "
                         "context, so without this ONE bad kernel makes every later kernel report "
                         "the same error and the run measures nothing after the first failure")
    ap.add_argument("--dirty", action="store_true",
                    help="fill and free a large device buffer before each checker, so freshly "
                         "allocated memory is NOT zero. Without this, --isolate hides every "
                         "out-of-bounds read whose garbage term is multiplied by fresh memory: "
                         "the products come out zero and the kernel looks correct")
    ap.add_argument("--check-override", default="",
                    help="TSV of 'kernel<TAB>module:function' lines; applies these checkers on top "
                         "of registry.csv without editing it")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each checker N times with the SAME seed and report whether every "
                         "run produced identical bytes. A kernel reading only its operands must; "
                         "layernorm_fwd_cuda did not, and a single measurement per kernel could "
                         "not see it")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed torch before each checker so inputs are identical across runs; "
                         "required for the sha column to compare anything")
    args = ap.parse_args()

    cfg_dir = os.environ.get("MINIWORLD_CONFIG_DIR", "")
    if not cfg_dir:
        print("MINIWORLD_CONFIG_DIR is unset -- refusing to run: every kernel would be stranded")
        return 2

    from miniworld_engine.autotune import devices
    from miniworld_engine.autotune.run_all import run_one

    backends = {b for b in args.backends.split(",") if b}
    only = {o for o in args.ops.split(",") if o}
    rows = [r for r in devices.registry()
            if r["backend"] in backends and (not only or r["kernel"] in only)]
    override = {}
    if args.check_override:
        for line in Path(args.check_override).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, fn = line.partition("\t")
            if not fn:
                k, _, fn = line.partition(" ")
            override[k.strip()] = fn.strip()
    for r in rows:
        if r["kernel"] in override:
            r["check"] = override[r["kernel"]]
    missing = only - {r["kernel"] for r in devices.registry()}
    if missing:
        print(f"unknown kernel names, not in registry.csv: {sorted(missing)}")
        return 2
    if args.isolate:
        return run_isolated(args, rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.label}.csv"

    recs = []
    for r in rows:
        drv, chk = (r.get("driver") or "").strip(), (r.get("check") or "").strip()
        if not drv:
            recs.append(dict(kernel=r["kernel"], family=r["family"], phase="no-driver",
                             ok="", worst="", sha="", detail=""))
            continue
        ok, detail = run_one(drv)
        phase, sha = "launch", ""
        if ok and chk:
            phase = "check"
            ok, detail, sha = check_and_hash(chk, seed=args.seed, dirty=args.dirty)
            if ok and args.repeat > 1:
                # Differing bytes across repeats is NOT by itself a defect: a kernel that
                # accumulates with floating-point atomics reorders its adds every run by design.
                # What is a defect is a repeat that is numerically WRONG. So fail only on that,
                # and report the hash spread alongside the rel instead of replacing it -- an
                # earlier version overwrote the rel and asserted "reads memory it does not own",
                # which mislabelled augmented_attention_bwd_atomic_triton.
                shas = {sha}
                for _ in range(args.repeat - 1):
                    ok2, d2, s2 = check_and_hash(chk, seed=args.seed, dirty=args.dirty)
                    shas.add(s2)
                    if not ok2:
                        ok, detail = False, f"WRONG ON REPEAT: {d2}"
                        break
                if ok and len(shas) > 1:
                    detail += f"  [{len(shas)}/{args.repeat} distinct hashes, all in band]"
        elif ok:
            phase = "launch-only"
        worst = ""
        if phase == "check" and detail.startswith("rel "):
            vals = []
            for tok in detail[4:].split():
                _, _, v = tok.partition("=")
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
            if vals:
                worst = f"{max(vals):.3e}"
        recs.append(dict(kernel=r["kernel"], family=r["family"], phase=phase,
                         ok="1" if ok else "0", worst=worst, sha=sha, detail=detail[:300]))
        print(f"  [{'ok  ' if ok else 'FAIL'}] {phase:11s} {r['kernel']:46s} {detail[:80]}",
              flush=True)

    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kernel", "family", "phase", "ok", "worst", "sha", "detail"])
        w.writeheader()
        w.writerows(recs)

    n_chk = sum(1 for r in recs if r["phase"] == "check")
    bad = [r for r in recs if r["ok"] == "0"]
    print(f"\n{args.label}: {len(recs)} kernels   checked {n_chk}   failing {len(bad)}   -> {dest}")
    for r in bad[:40]:
        print(f"   FAIL {r['phase']:11s} {r['kernel']:46s} {r['detail'][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
