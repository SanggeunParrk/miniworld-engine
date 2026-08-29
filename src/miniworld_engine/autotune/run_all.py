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
import math
import pathlib
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


#: Seed set before every checker call. The value does not matter; that it is the same one every
#: time does.
CHECKER_SEED = 0


def run_checker(check: str):
    """Resolve a checker and run it on a fixed RNG stream.

    Seeded here, not in the checkers: `checks._fixed()` promises a reproducible failure but only
    two of the fourteen checker modules call it, so ~100 checkers built their inputs from unseeded
    `torch.randn`. That made run_all failures irreproducible, and made two calls to one checker see
    different inputs -- which is what `test_determinism_gpu` was really measuring. One seed at the
    one invocation point; a checker written tomorrow cannot forget.
    """
    import torch

    # Seeds every device, not just the CPU generator -- the inputs are built on the GPU.
    torch.manual_seed(CHECKER_SEED)
    return _resolve(check)()


#: The band a kernel is held to when its registry row does not name one. bf16 carries ~3 decimal
#: digits and this is what the bench already accepts for these ops.
#:
#: It used to be the ONLY band, applied to all 99 checkers -- which is the weakest kernel's
#: tolerance applied to a transpose, a mask fold and a gate multiply. A reduction-order change
#: costing 1e-3 is invisible under it, and 1e-3 on a residual accumulated over 48 blocks is not
#: invisible in the model. A kernel that needs this much slack should say so in `registry.csv`,
#: where a reviewer sees it; `rtol` is that column.
DEFAULT_RTOL = 5e-2
#: How far above a kernel's MEASURED worst relative error its declared band sits. Chosen from the
#: data rather than picked: the same kernel's rel across the two cards in `autotune/manifests/`
#: varies by median 1.07x, p90 1.47x, max 1.95x, so 4x leaves two more doublings of headroom than
#: anything observed. Calibration may only TIGHTEN -- a kernel whose measurement is already close
#: to DEFAULT_RTOL keeps DEFAULT_RTOL rather than being given a looser band than it has today.
RTOL_MARGIN = 4.0


def check_one(check: str, rtol: float | None = None) -> tuple[bool, str]:
    """Run a checker and compare what the kernel produced against a torch reference.

    A checker returns ``(actual, expected)`` -- one tensor pair, or a dict of them. Launching a
    kernel proves it runs; it does not prove the number is right. 56 kernels reached by a driver
    had no reference at all, so "ok" meant nothing more than "did not raise".

    `rtol` is the kernel's declared band (`registry.csv`'s `rtol` column); None means
    :data:`DEFAULT_RTOL`. The band is reported in the detail string either way, so a failure says
    what it was measured against instead of leaving the reader to find the constant.
    """
    try:
        got = run_checker(check)
    except Exception as exc:
        return False, f"checker raised {type(exc).__name__}: {str(exc).strip().splitlines()[0][:150]}"
    pairs = got if isinstance(got, dict) else {"out": got}
    worst, detail, nonfinite = 0.0, [], []
    for name, pair in pairs.items():
        actual, expected = pair
        a, e = actual.float(), expected.float()
        if a.shape != e.shape:
            return False, f"{name}: shape {tuple(a.shape)} vs reference {tuple(e.shape)}"
        num = (a - e).abs().max().item()
        den = e.abs().max().item() or 1.0
        rel = num / den
        # Non-finite is checked HERE, per pair, and not by testing `worst` afterwards. `max()`
        # returns the first argument when the comparison is false, and `nan > 0.0` is false -- so
        # `max(0.0, nan)` is 0.0 and the NaN vanishes. The previous `worst == worst` guard read
        # like a NaN check and could never fire: a kernel writing NaN scored 0.0 and passed every
        # band, including a declared 0.
        if not math.isfinite(rel):
            nonfinite.append(f"{name}={rel}")
        else:
            worst = max(worst, rel)
        detail.append(f"{name}={rel:.2e}")
    if not detail:
        return False, "checker returned nothing"
    band = DEFAULT_RTOL if rtol is None else rtol
    suffix = f" (band {band:.0e}{'' if rtol is None else ' declared'})"
    if nonfinite:
        return False, f"NON-FINITE {' '.join(nonfinite)} | rel {' '.join(detail)}{suffix}"
    return worst <= band, f"rel {' '.join(detail)}{suffix}"


#: Exception text that means "this kernel refuses to run on THIS card", for kernels whose gate is
#: not (or not yet) declared in registry.csv. A backstop, not the primary check -- see
#: :func:`meets_arch`, which decides from the declaration and never launches the kernel at all.
_ARCH_GATED_MARKERS = (
    "expects arch to be one of", "unsupported gpu architecture", "requires sm",
    "not supported on this", "expects arch to be",
    # checkers assert their own gate, e.g. "SM90 (H100) only"
    "sm90", "sm100", "h100 only", "b200 only", "only on sm",
)


def is_arch_gated(detail: str) -> bool:
    """Does this failure mean "wrong card" rather than "wrong answer"?

    An sm100 CuTe kernel raises on an A6000 at every shape. Counting that as a failure makes the
    report red on hardware the kernel was never meant for, and a report that is always red is one
    nobody reads. Lives here rather than in the test that first needed it: `run_all` is what
    produces the verdict, so it is what has to classify it, and `tests/numerics/test_numerical.py` imports
    this instead of keeping a second copy.
    """
    d = detail.lower()
    return any(m in d for m in _ARCH_GATED_MARKERS)


def device_arch() -> str:
    """This card's compute capability as `registry.csv`'s `arch` spells it, e.g. ``"sm86"``."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def meets_arch(row: dict, device: str | None = None) -> bool:
    """Can the card run what this row declares it needs?

    Reads the DECLARED minimum (`registry.csv`'s `arch`, see docs/library-standards.md A5) rather
    than matching an exception after the fact, so a kernel that cannot run here is never launched
    and never compiled. `is_arch_gated` stays as a backstop for a row whose declaration is wrong or
    missing -- and a row that passes this check and then fails with an arch error is exactly that,
    which `main` reports rather than swallows.
    """
    want = (row.get("arch") or "").strip()
    if not want:
        return True
    dev = device if device is not None else device_arch()
    return _sm(dev) >= _sm(want)


def _sm(arch: str) -> int:
    """``"sm86"`` -> 86. Unrecognised text sorts lowest, so it never blocks a launch."""
    digits = arch.lower().removeprefix("sm").strip()
    return int(digits) if digits.isdigit() else -1


def _running_dtype() -> str:
    """The precision the drivers and checkers are building at, for pricing the band."""
    from miniworld_engine.kernels.drivers import DTYPE_MODE

    return DTYPE_MODE


def declared_rtol(row: dict, dtype: str | None = None) -> float | None:
    """The band this registry row declares for ``dtype``, or None for :data:`DEFAULT_RTOL`.

    Blank means "the default applies", not "no band": every kernel with a checker is held to
    something. A malformed value is an error rather than a silent fall back to the default -- a
    typo that widens a kernel's tolerance is exactly what this column exists to prevent.

    TWO SPELLINGS. A bare float is one band for every precision the kernel declares. That is only
    honest when the kernel runs at one precision: bf16 carries ~3.9e-3 of machine epsilon against
    an fp32 reference and fp32 carries ~1e-7, so a single number is either four orders too tight
    for bf16 or four orders too loose for fp32. `bf16=5e-3|fp32=4e-7` gives each its own, and is
    what a kernel declaring `dtypes=bf16|fp32` needs -- before this, those rows carried the bf16
    band alone and their fp32 runs were being checked against a tolerance a real regression could
    hide inside. (Nothing had noticed because no fp32 unit was ever built; see the driver dtype
    fix.) An unknown key, or a dtype the row does not price, is an error rather than a fallback.
    """
    raw = (row.get("rtol") or "").strip()
    if not raw:
        return None
    kernel = row.get("kernel")
    if "=" in raw:
        bands: dict[str, float] = {}
        for part in raw.split("|"):
            key, _, value = part.partition("=")
            key = key.strip()
            if key not in ("bf16", "fp32", "fp16"):
                msg = (f"{kernel}: rtol names precision {key!r}, which is not one of "
                       f"bf16/fp32/fp16")
                raise ValueError(msg)
            bands[key] = _band(kernel, value)
        if dtype is None:
            return max(bands.values())      # no precision asked for: the widest it ever allows
        short = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}.get(dtype, dtype)
        if short not in bands:
            msg = (f"{kernel}: rtol prices {sorted(bands)} but the run is {short}. Add it, or drop "
                   f"{short} from the dtypes column.")
            raise ValueError(msg)
        return bands[short]
    return _band(kernel, raw)


def _band(kernel: str | None, raw: str) -> float:
    try:
        band = float(raw)
    except ValueError:
        msg = (f"{kernel}: rtol={raw!r} in registry.csv is not a number. Leave it blank for the "
               f"default ({DEFAULT_RTOL:.0e}), give a float, or price each precision as "
               f"'bf16=5e-3|fp32=4e-7'.")
        raise ValueError(msg) from None
    if band < 0:
        msg = f"{kernel}: rtol={band} is negative"
        raise ValueError(msg)
    return band


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
    ap.add_argument("--json", default="", metavar="PATH",
                    help="also write the counts as JSON, for a release verdict (plan.md B1)")
    args = ap.parse_args(argv)

    want = {f for f in args.only.split(",") if f}
    rows = [r for r in devices.registry() if not want or r["family"] in want]
    results: dict[str, tuple[bool, str]] = {}
    here = device_arch()
    skipped: dict[str, str] = {}
    mislabelled: list[str] = []
    for r in rows:
        drv = (r.get("driver") or "").strip()
        if not drv:
            continue
        # Declared gate first: a kernel this card cannot run is not launched at all, so it costs no
        # compile and cannot be reported as a failure. Six of these on an A6000 were counted as
        # failures before, which is the "failure vs absence" mistake `is_bad_unit` already avoids
        # on the build side.
        if not meets_arch(r, here):
            skipped[r["kernel"]] = f"needs {r['arch']}, this card is {here}"
            print(f"  [skip] {r['kernel']:46s} {skipped[r['kernel']]}", flush=True)
            continue
        ok, detail = run_one(drv)
        chk = (r.get("check") or "").strip()
        if ok and chk:
            ok, cdetail = check_one(chk, declared_rtol(r, _running_dtype()))
            detail = cdetail if ok else f"WRONG NUMBERS: {cdetail}"
        elif ok:
            detail = "launched (no reference -- numbers unverified)"
        if not ok and is_arch_gated(detail):
            # It passed the declared check and still refused on arch grounds: the DECLARATION is
            # wrong, which is worth saying out loud rather than absorbing into a skip.
            mislabelled.append(f"{r['kernel']}: declared {r.get('arch') or '(blank)'}, refused on "
                               f"{here} -- {detail[:80]}")
            skipped[r["kernel"]] = f"arch-gated at runtime: {detail[:80]}"
            print(f"  [skip] {r['kernel']:46s} {skipped[r['kernel']]}", flush=True)
            continue
        results[r["kernel"]] = (ok, detail)
        print(f"  [{'ok  ' if ok else 'FAIL'}] {r['kernel']:46s} {detail[:90]}", flush=True)
        if not ok and args.verbose:
            traceback.print_exc()

    key = gpu_key()
    path = devices.record(key, results)
    have = sum(1 for v in results.values() if v[0])
    checked = sum(1 for r in rows if (r.get("check") or "").strip()
                  and results.get(r["kernel"], (False,))[0])
    declared = len(devices.registry())
    no_driver = sum(1 for r in devices.registry() if not (r.get("driver") or "").strip())
    print(f"\n{path}")
    # Three categories, not two. "skipped (this card)" is a permanent, correct answer -- the same
    # distinction `is_bad_unit` draws on the build side -- and folding it into either `failed` or
    # `no driver` is what made an A6000 run report six failures it could not have avoided.
    print(f"  declared {declared}   driven {len(results)}   "
          f"ok {have}   failed {len(results) - have}   "
          f"skipped (this card is {here}) {len(skipped)}   no driver {no_driver}")
    accounted = len(results) + len(skipped) + no_driver
    if accounted != declared:
        print(f"  ACCOUNTING: {accounted} of {declared} classified -- "
              f"{declared - accounted} kernel(s) fell through every branch", file=sys.stderr)
    print(f"  numbers checked against a reference: {checked}   "
          f"launched but unverified: {have - checked}")
    if mislabelled:
        # A row that passed the declared gate and then refused on arch grounds anyway: the
        # registry's `arch` column is wrong for it. Reported, never absorbed into the skip count.
        print(f"  registry `arch` disagrees with the device for {len(mislabelled)} kernel(s):",
              file=sys.stderr)
        for m in mislabelled:
            print(f"    {m}", file=sys.stderr)
    by = collections.Counter(r["family"] for r in devices.registry()
                             if not (r.get("driver") or "").strip())
    if by:
        print("  kernels with no driver, by family:",
              ", ".join(f"{k}:{v}" for k, v in sorted(by.items())))

    if args.json:
        # The same numbers the summary above prints, as data. A release verdict has to be
        # machine-checkable (plan.md B1); re-parsing the human summary would break the moment
        # its wording changed, which is the kind of check that fails for the wrong reason.
        import json
        payload = {
            "declared": declared, "driven": len(results), "ok": have,
            "failed": len(results) - have, "skipped": len(skipped), "no_driver": no_driver,
            "checked_against_reference": checked, "launched_unverified": have - checked,
            "arch": here, "accounting_ok": accounted == declared,
            "mislabelled_arch": len(mislabelled),
            "failures": {k: v[1] for k, v in sorted(results.items()) if not v[0]},
        }
        pathlib.Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
