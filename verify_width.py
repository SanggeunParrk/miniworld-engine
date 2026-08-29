"""Does the newly-covered token width actually run, and run correctly?

The change made `width` the thing that decides which channel widths a kernel is tuned at, which
uncapped three families at d_single_token=768. Nothing had ever driven them there. Three questions:
  1. does every unit's driver LAUNCH at 768 (the widths were never built, so a smem-limit or a
     mask bug there has never been seen)?
  2. do the modules at krystal's two real combinations (768/384 token, 128/128 atom) still match
     PyTorch?
  3. is the Triton path actually taken at 768, or does something route back to torch?
"""
import os, sys, traceback
sys.path.insert(0, "src")
import torch

from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
from miniworld_engine.modules.conditioned_transition.module import ConditionedTransition
from miniworld_engine.modules.exceptions import ImplementationType as IT

DEV = "cuda"
FAIL = []


def _cmp(name, ref, got, tol):
    d = (ref.float() - got.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    if scale == 0.0:                       # both sides all-zero: the comparison proved nothing
        print(f"  VACUOUS {name}: reference is identically zero")
        FAIL.append(name + " (vacuous)")
        return
    ok = d <= tol
    print(f"  {'OK ' if ok else 'BAD'} {name:52s} max|diff|={d:.3e} tol={tol:.0e}")
    if not ok:
        FAIL.append(name)


def check_modules():
    print("2/3. 모듈 정확도 (krystal의 두 조합)")
    for tag, dh, dc, dt, tol in (("token_dit 768/384", 768, 384, torch.bfloat16, 3e-2),
                                 ("atom_dit  128/128", 128, 128, torch.bfloat16, 3e-2)):
        x = torch.randn(2, 512, dh, device=DEV, dtype=dt)
        c = torch.randn(2, 512, dc, device=DEV, dtype=dt)
        for cls in (AdaptiveLayerNorm, ConditionedTransition):
            ref_m = cls(dh, dc, implementation=IT.PYTORCH).to(DEV).to(dt).eval()
            got_m = cls(dh, dc, implementation=IT.MINIWORLD).to(DEV).to(dt).eval()
            # ConditionedTransition zero-initialises `squeeze`, its output projection (AF3 starts
            # the residual at identity), so a freshly built module outputs EXACTLY zero and a
            # comparison against torch is 0 vs 0 -- it passes without running anything. Give every
            # weight a real value first.
            with torch.no_grad():
                for prm in ref_m.parameters():
                    prm.normal_(0, 0.02)
            got_m.load_state_dict(ref_m.state_dict())
            with torch.no_grad():
                _cmp(f"{cls.__name__} {tag} inference", ref_m(x, c), got_m(x, c), tol)
            xg = x.clone().requires_grad_(True)
            xg2 = x.clone().requires_grad_(True)
            r = ref_m(xg, c); g = got_m(xg2, c)
            _cmp(f"{cls.__name__} {tag} training fwd", r, g, tol)
            r.float().square().mean().backward(); g.float().square().mean().backward()
            _cmp(f"{cls.__name__} {tag} training dx", xg.grad, xg2.grad, tol)


def check_drivers():
    print("1/3. 드라이버가 768에서 실제로 launch 되는가")
    from miniworld_engine.autotune.builder import op_units
    units = [u for u in op_units() if u.width == 768]
    ops = sorted({u.op for u in units})
    print(f"  width=768 유닛 {len(units)}개, 고유 op {len(ops)}개")
    os.environ["MINIWORLD_DRIVER_WIDTH"] = "768"
    import csv
    from pathlib import Path
    reg = Path("src/miniworld_engine/kernels/registry.csv")
    drv = {r["kernel"]: r["driver"] for r in csv.DictReader(reg.open())}
    import importlib
    for op in ops:
        spec = drv.get(op, "")
        if not spec:
            continue
        mod, _, fn = spec.partition(":")
        try:
            m = importlib.import_module(mod)
            getattr(m, fn)()
            torch.cuda.synchronize()
            print(f"  OK  {op}")
        except Exception as e:
            print(f"  BAD {op}: {type(e).__name__}: {str(e)[:160]}")
            FAIL.append(f"driver:{op}")


def check_path_taken():
    print("3/3. 768에서 Triton 경로를 실제로 타는가")
    from miniworld_engine.kernels.conditioned_transition.triton import dispatch
    seen = []
    for d in (128, 768):
        seen.append((d, "b2b(atom)" if d <= dispatch.ATOM_D_MAX else "composed(token)"))
    for d, p in seen:
        print(f"  d_hidden={d:4d} -> {p}")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    for f in (check_drivers, check_modules, check_path_taken):
        try:
            f()
        except Exception:
            traceback.print_exc(); FAIL.append(f.__name__)
        print()
    print("=" * 70)
    print(f"실패 {len(FAIL)}건" + (": " + ", ".join(FAIL) if FAIL else " — 전부 통과"))
    sys.exit(1 if FAIL else 0)
