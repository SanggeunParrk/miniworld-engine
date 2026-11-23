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

**Cards.** H100 (sm_90a) and B200 (4th card of the paper set, 3 repeats, no Slurm: 4 lanes on GPUs 0-3 of host `b200` through `paper_common/b200_driver.sh`, ssh ControlMaster, kernels prebuilt for sm100a from main source d967378a2, records in `records/b200/`; the lanes ran concurrently on one node, so timings may include cross-lane load, as the H100 jobs did; the number of timed iterations per round is in each record) (sm_100a: `fp8.B200_ROUTES` TriMul / TriAttn / Transition, `docs/b200/fp8.md`; the `none` pass also sets `MINIWORLD_TRIATTN_B200_FP8_MINS=0`; run `paper_common/b200_driver.sh`). **RTX 5090 (4th-card runs on cssb-rtx, no Slurm, 3 repeats through `~/mwe/gpu.sh`, `rtx5090.sh`):** TriMul only: the opt-in e4m3-plane route `kernels/trimul_inproj/cuda/sm120_fp8.py` (docs/fp8.md "RTX 5090"); TriAttn: the opt-in e4m3 P / V (+ q / k on the fused d_pair-128 kind) core `kernels/triangle_attention/cuda/sm120_fp8.py` (gate L >= 640; the `none` pass sets `MINIWORLD_TRIATTN_SM120_FP8_MINL=0`). Transition has no sm120 FP8 route (not in the E4 module set; see docs/fp8.md "RTX 5090"). **A100: n/a** (no script).

    cd benchmarks/paper && REP=1 paper_common/chain.sh 6 E4_fp8/h100.sbatch ; python E4_fp8/collect.py --card h100

Output `E4_<card>.csv`: two rows per case and gate setting (`backend` `cuda` = default path, `engine-fp8`), common columns +
`gates,model,ratio_default_over_fp8,fp8_route_fired,saturated,stack,row,variant,samples,repeats`.
