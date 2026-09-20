"""Per-CTA timeline of K1 and K3 from a payload built with ``build_payload.py --probe`` (TMN_CTA_TS=1): clock64 at CTA start, at the first
tile's data arrival in the consumers, at the last tile's start and at the end, plus globaltimer at start / end.  Reports the startup
(launch -> first tile ready), the steady per-tile cost, the last tile, the per-CTA total, and the launch / finish skew across the 132 CTAs.
The probe buffers ride on K1Params::stats (unused by the v0 kernels) and K3Params::prof, patched into the cached argument packs.
PDL is switched off for the probe unless TRIMUL_NATIVE_PDL is set: with it on, K1's "startup" includes the griddepcontrol wait behind the
previous launch and the CTAs start staggered (launch skew ~4.5 us instead of 0.1 us).

    TRIMUL_NATIVE_BUILD_DIR=<probe payload>/build python probe_cta_timeline.py --length 384 --output cta-timeline-L384.json"""
import argparse
import json
import os
import statistics

import torch

os.environ.setdefault("TRIMUL_NATIVE_PDL", "0")   # measure the kernels' own startup: with PDL on, slot 1 - slot 0 also holds the wait behind the previous grid

import fixture  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--width", type=int, default=128)
p.add_argument("--output", required=True)
a = p.parse_args()
fx = fixture.make(a.length, a.width, True, "fp32")
F = fixture.face()
cache = {}
NCTA = 132


def walk(c, out):
    for k, v in list(c.items()):
        if isinstance(k, tuple) and k and k[0] in ("tmn.k1", "tmn.k3"):
            out[k[0]] = v
        elif isinstance(v, dict):
            walk(v, out)


with torch.no_grad():
    for _ in range(3):
        F.serve(fx.x, fx.pairmask, direction="outgoing", weights=fx.weights, residual=True, cache=cache, eps=fx.eps, config=None)
    torch.cuda.synchronize()
    ents = {}
    walk(cache, ents)
    assert set(ents) == {"tmn.k1", "tmn.k3"}, list(ents)
    bufs = {n: torch.zeros(NCTA * 8, dtype=torch.int64, device="cuda") for n in ents}
    ents["tmn.k1"][0].set_ptr(0, bufs["tmn.k1"].data_ptr(), field=7)     # K1Params::stats
    ents["tmn.k3"][0].set_ptr(0, bufs["tmn.k3"].data_ptr(), field=11)    # K3Params::prof
    for _ in range(5):
        F.serve(fx.x, fx.pairmask, direction="outgoing", weights=fx.weights, residual=True, cache=cache, eps=fx.eps, config=None)
    torch.cuda.synchronize()
res = {}
for n, b in bufs.items():
    rows = [r for r in b.view(NCTA, 8).cpu().tolist() if r[6] > 0]
    if not rows:
        raise SystemExit("no probe samples for %s: build the payload with --probe" % n)
    f = lambda v: dict(med=statistics.median(v), min=min(v), max=max(v))
    gt0 = min(r[4] for r in rows)
    res[n] = dict(ctas=len(rows), n_iter=sorted({r[6] for r in rows}), startup_cyc=f([r[1] - r[0] for r in rows]),
                  steady_per_tile_cyc=f([(r[2] - r[1]) / (r[6] - 1) for r in rows if r[6] > 1]), last_tile_cyc=f([r[3] - r[2] for r in rows]),
                  total_cyc=f([r[3] - r[0] for r in rows]), launch_skew_us=(max(r[4] for r in rows) - gt0) / 1000,
                  end_spread_us=(max(r[5] for r in rows) - min(r[5] for r in rows)) / 1000, span_us=(max(r[5] for r in rows) - gt0) / 1000)
    c = res[n]
    print("%s: ctas %d n_iter %s | startup med %.0f cyc (min %d max %d) | steady/tile %.0f | last tile %.0f | total med %.0f max %d | launch skew %.2f us, end spread %.2f us, span %.2f us"
          % (n, c["ctas"], c["n_iter"], c["startup_cyc"]["med"], c["startup_cyc"]["min"], c["startup_cyc"]["max"], c["steady_per_tile_cyc"]["med"],
             c["last_tile_cyc"]["med"], c["total_cyc"]["med"], c["total_cyc"]["max"], c["launch_skew_us"], c["end_spread_us"], c["span_us"]), flush=True)
json.dump(dict(length=a.length, res=res), open(a.output, "w"), indent=1)
