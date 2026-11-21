# E4: FP8 path of TriMul / TriAttn (inference)

`fp8_pair.py` measures, in one process and alternating, the default path (`cuda`) and the FP8 path (`engine-fp8`, `miniworld_engine.fp8`)
of the same module on the E2 grid (inference). The FP8 copy is calibrated by two eager calls first; then three rounds alternate the routes
(order flips each round), each round a `measure.speed.measure` (graph replay, L2 flush, warmup 5, 100 iterations). Recorded per case:
both times, the ratio default / FP8, `routes_fired` (`fp8.route_stats()`: did a FP8 route run, or did a length gate keep the default kernels),
`saturated` (`fp8.check()` after calibration and after the replays: the e4m3 saturation check tripped, the route rescaled), the gate setting.
TriAttn FP8 is validated per model: `--model af3` (default; `fp8.TRIATTN_MODELS`).
Two passes (two files): `--gates shipped` (the production gates of `engine-fp8`) and `--gates none` (every length / S gate removed:
`MINIWORLD_TRIMUL_FP8_MINL=0` and the h100_sm90 TriAttn gates cleared, as the 3.0-tree measurements did).
Training is not measured: there is no FP8 backward and `engine-fp8` runs the default path in training.

**Cards.** H100 (sm_90a) and B200 (sm_100a: `fp8.B200_ROUTES` TriMul / TriAttn / Transition, `docs/b200/fp8.md`; the `none` pass also sets `MINIWORLD_TRIATTN_B200_FP8_MINS=0`; run `paper_common/b200_driver.sh`). **RTX 5090: no sm120 FP8 path** (checked: `miniworld_engine/fp8.py` states the
routes are H100 / sm_90a only and no `sm120` kernel under `kernels/` has an e4m3 route; `engine-fp8` resolves to the default `cuda`
kernels there). **RTX 5090: no FP8 route (checked in code: `fp8.py` is sm_90a only); not measured, no E4 CSV for this card** (the smoke / repeat-1 pair records, `routes_fired: false`, ratio ~ 1, are not part of the results). **A100: n/a** (no script).

    cd benchmarks/paper && REP=1 paper_common/chain.sh 6 E4_fp8/h100.sbatch ; python E4_fp8/collect.py --card h100

Output `E4_<card>.csv`: two rows per case and gate setting (`backend` `cuda` = default path, `engine-fp8`), common columns +
`gates,model,ratio_default_over_fp8,fp8_route_fired,saturated,stack,row,variant,samples,repeats`.
