"""Run every kernel the repo declares, and record which ones ran.

    python -m miniworld_engine.autotune.run_all

No capability scanning, no error-string matching, no reading old artifacts. Import the driver
``registry.csv`` names for a kernel, call it, and write down what happened:

    ok        the driver ran to completion
    failed    it raised -- the exception is the reason, verbatim
    untested  the kernel has no driver yet

``untested`` is a hole in coverage, not a verdict. The denominator is the registry, so a kernel
nobody wrote a driver for stays visible instead of vanishing from both sides of the ratio.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import sys
import traceback

from miniworld_engine.autotune import devices
from miniworld_engine.autotune.cache import gpu_key


def _resolve(path: str):
    """``pkg.mod:func`` -> the callable."""
    mod_name, _, fn_name = path.partition(":")
    if not fn_name:
        raise ValueError(f"driver must be 'module:function', got {path!r}")
    return getattr(importlib.import_module(mod_name), fn_name)


def check_one(check: str) -> tuple[bool, str]:
    """Run a checker and compare what the kernel produced against a torch reference.

    A checker returns ``(actual, expected)`` -- one tensor pair, or a dict of them. Launching a
    kernel proves it runs; it does not prove the number is right. 56 kernels reached by a driver
    had no reference at all, so "ok" meant nothing more than "did not raise".
    """
    try:
        got = _resolve(check)()
    except Exception as exc:
        return False, f"checker raised {type(exc).__name__}: {str(exc).strip().splitlines()[0][:150]}"
    pairs = got if isinstance(got, dict) else {"out": got}
    worst, detail = 0.0, []
    for name, pair in pairs.items():
        actual, expected = pair
        a, e = actual.float(), expected.float()
        if a.shape != e.shape:
            return False, f"{name}: shape {tuple(a.shape)} vs reference {tuple(e.shape)}"
        num = (a - e).abs().max().item()
        den = e.abs().max().item() or 1.0
        rel = num / den
        worst = max(worst, rel)
        detail.append(f"{name}={rel:.2e}")
    # bf16 carries ~3 decimal digits; 5e-2 is the band the bench already accepts for these ops
    ok = worst < 5e-2 and worst == worst
    return ok, ("rel " + " ".join(detail)) if detail else "checker returned nothing"


def run_one(driver: str) -> tuple[bool, str]:
    import torch
    try:
        _resolve(driver)()
        torch.cuda.synchronize()
    except Exception as exc:
        # Keep the line that names the cause, not just the first line. A build error's first line
        # is the nvcc command line -- truncating there hid "unsupported gpu architecture" and
        # "__is_array is undefined" behind a wall of -isystem flags and cost three round trips.
        lines = [ln.strip() for ln in str(exc).split("\n") if ln.strip()]
        # "Error building extension" is torch's wrapper, and the line after it is the nvcc command
        # line. The useful line is a compiler diagnostic further down, so match those specifically
        # and skip the wrapper -- an earlier version matched "error" and kept re-picking line one.
        marks = ("error:", "fatal", "undefined", "no such file", "cannot open",
                 "not supported", "unsupported")
        signal = next((ln for ln in lines
                       if any(m in ln.lower() for m in marks)
                       and "building extension" not in ln.lower()
                       and not ln.lstrip().startswith("/")), "")
        if not signal:
            signal = next((ln for ln in lines
                           if any(m in ln.lower() for m in marks)
                           and "building extension" not in ln.lower()), "")
        head = lines[0][:120] if lines else ""
        detail = f"{type(exc).__name__}: {head}"
        # Compare whole lines, not a prefix. A 40-char prefix check suppressed the diagnostic
        # because it began with the same absolute path as the nvcc command line above it.
        if signal and signal != (lines[0] if lines else ""):
            detail += f" | {signal[:220]}"
        return False, detail
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma list of families to run")
    ap.add_argument("--verbose", action="store_true", help="print the traceback for each failure")
    args = ap.parse_args(argv)

    want = {f for f in args.only.split(",") if f}
    rows = [r for r in devices.registry() if not want or r["family"] in want]
    results: dict[str, tuple[bool, str]] = {}
    for r in rows:
        drv = (r.get("driver") or "").strip()
        if not drv:
            continue
        ok, detail = run_one(drv)
        chk = (r.get("check") or "").strip()
        if ok and chk:
            ok, cdetail = check_one(chk)
            detail = cdetail if ok else f"WRONG NUMBERS: {cdetail}"
        elif ok:
            detail = "launched (no reference -- numbers unverified)"
        results[r["kernel"]] = (ok, detail)
        print(f"  [{'ok  ' if ok else 'FAIL'}] {r['kernel']:46s} {detail[:90]}", flush=True)
        if not ok and args.verbose:
            traceback.print_exc()

    key = gpu_key()
    path = devices.record(key, results)
    have = sum(1 for v in results.values() if v[0])
    checked = sum(1 for r in rows if (r.get("check") or "").strip()
                  and results.get(r["kernel"], (False,))[0])
    print(f"\n{path}")
    print(f"  declared {len(devices.registry())}   driven {len(results)}   "
          f"ok {have}   failed {len(results) - have}   "
          f"no driver {len(devices.registry()) - len(results)}")
    print(f"  numbers checked against a reference: {checked}   "
          f"launched but unverified: {have - checked}")
    by = collections.Counter(r["family"] for r in devices.registry()
                             if not (r.get("driver") or "").strip())
    if by:
        print("  드라이버 없는 커널 (계열별):",
              ", ".join(f"{k}:{v}" for k, v in sorted(by.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
