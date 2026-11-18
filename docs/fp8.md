# FP8 routes (H100, opt-in, engine 3.0.0 line)

Everything here is opt-in and off by default: FP8 is the backend `engine-fp8`; the default remains the bf16 / fp16 engine. Base: engine 2.5.14 (branch `rel/3.0.0` = 2.5.14 + `fp8/3.0.0-all`; the routes were built on 2.5.12, branch `fp8/3.0.0-on-2512`). With no switch set, every route is bit-identical to 2.5.14 (see "Release line 3.0.0"; before: 169 / 169 against 2.5.12, "Verification on the 2.5.12 base").

## The `engine-fp8` backend (user-facing name, 3.0.0)

FP8 is the opt-in backend **`engine-fp8`**; the default remains the bf16 / fp16 engine (`cuda`), bit-identical to 2.5.14.

```
TriangleMultiplication(128, backend="engine-fp8")    # or module.set_backend("engine-fp8"); a composite passes it to its parts
miniworld_engine.fp8.set_model("af3")                 # the TriAttn FP8 routes validated for that model (or MINIWORLD_FP8_MODEL=af3)
with torch.inference_mode():
    out = model(x)
    if miniworld_engine.fp8.check():                  # guard contract as below
        out = model(x)
```

- `engine-fp8` = the engine `cuda` backend with the `safe` preset's routes on for the calls of the modules that select it (plus the TriAttn part of
  the model `fp8.set_model()` / `MINIWORLD_FP8_MODEL` names, per `fp8.TRIATTN_MODELS`). Every other module, and every module by default, runs the
  bf16 / fp16 engine. `set_backend("engine-fp8")` installs the instance's `_forward_cuda` as `fp8.BackendScope` around the class's
  `_forward_cuda` (switches on, restored after the call), so `forward` and a runtime that calls `_forward_<backend>` directly (FoldForge's guard) both
  get it; a module on another backend called through its `forward` inside it runs with the process's own switches. The default route never enters the scope.
- Names: the public name is `engine-fp8`; where `-` cannot stand (Python identifiers, config keys) it is `engine_fp8` (`fp8.BACKEND_ID`), which
  `set_backend` also takes. It is **not** an entry of any class's `backends` tuple and has no `_forward_engine_fp8` / `_supports_engine_fp8`:
  `set_backend("engine-fp8")` resolves to `cuda` (`fp8.resolve_backend`) on every class with a `cuda` backend and leaves `module.backend == "cuda"`,
  `module.engine_fp8 is True`, `module.backend_name == "engine-fp8"` (also in `repr`). `Class.has_backend("engine-fp8")` says whether a class takes it.
  Its range is `cuda`'s (`_supports_cuda`); calls the FP8 routes do not take (fp32 modules, cards other than H100) run the default kernels.
- **Inference only.** `engine-fp8` accelerates inference; training under `engine-fp8` runs the default bf16 path: every FP8 route is a forward path with
  no FP8 backward, so a call in train mode (`module.training`, dropout live) or with autograd on (`torch.is_grad_enabled()`) runs exactly the `cuda`
  backend (same bytes: output, input and parameter gradients; `tests/gpu/test_engine_fp8_backend.py`). Run inference in `eval()` under
  `torch.no_grad()` / `torch.inference_mode()`.
- Outside the judge: no class lists it, no registry row / tolerance / `backend_versions` entry, the measurers and the judge cycles never select it
  (`tests/cpu/test_fp8_switch.py::test_engine_fp8_is_not_a_judged_backend`).
- A runtime that picks backends by name (FoldForge's `Config.backend` / `engine.ops`) selects it by passing `"engine-fp8"` to `set_backend` of the
  engine modules it would put on `cuda`, calling `fp8.set_model(<model>)` once, and `fp8.check()` after every forward. Until a runtime knows the name,
  the lower-level switch below does the same process wide.

## The lower-level switch (`miniworld_engine.fp8`, process wide)

```
MINIWORLD_FP8=safe                                   # environment, before the engine's kernels are imported: every route but TriAttn
MINIWORLD_FP8=safe MINIWORLD_FP8_MODEL=af3           # + the TriAttn FP8 routes validated for that model
miniworld_engine.fp8.enable("safe")                  # or at run time; fp8.disable(), fp8.status()
miniworld_engine.fp8.enable("safe", model="af3")     # + the TriAttn FP8 routes validated for that model
```

`safe` is the set validated end to end ("fp8safe2", AF3 FoldBench below): `MINIWORLD_TRIMUL_FP8=calib`, `MINIWORLD_OPM_FP8=bound`, `MINIWORLD_PWA_FP8=1`,
`MINIWORLD_TRANSITION_FP8=calib`, `MINIWORLD_TOKEN_DIT_FP8=expand` (not `qkvg`: the one placement that moved FoldBench), `MINIWORLD_ATOMDIT_FP8=1` for every
model, and TriAttn FP8 (PV on the fused and the unfused / front-made sites, e4m3 q / k on the fused d_pair-128 route) **only for a validated model**.
The per-route variables keep working; with `MINIWORLD_FP8` set, a per-route variable set explicitly (also to "") wins over the preset. Guard contract: after
every forward, outside any graph, `if miniworld_engine.fp8.check(): redo the forward` (at most 3 times): one host sync per static-scale route (TriAttn V / q k,
TriMul, Transition, PWA); the routes' eager `sync` guards (default) already redo an overflowing call themselves, under graph capture they record it for
`check()`. Calibration: the static scales are taken on each site's first eager call in a process and kept (two runs with a different first target are not
byte-identical).

### Decisions for the 3.0.0 release (owner, 2026-11-17)

1. **TriAttn FP8 only for validated models.** The model is named once: `MINIWORLD_FP8_MODEL=<model>` next to `MINIWORLD_FP8` (read when the TriAttn kernels
   are imported) or `fp8.enable(preset, model=<model>)` (run time; `enable()` does not read `MINIWORLD_FP8_MODEL`, and `enable("safe")` without `model=`
   turns the TriAttn part off). Without a model, or for a model not in the table, TriAttn runs the default fp16 core (bit-identical to FP8 off) while the
   other `safe` routes are on. The table is `miniworld_engine.fp8.TRIATTN_MODELS` (`h100_sm90.FP8_PV_WHITELIST` is derived from it):

   | model | TriAttn FP8 PV call sites | e4m3 q / k | end-level evidence |
   |---|---|---|---|
   | `af3` | fused + unfused | yes | FoldBench 27 targets with the whole `safe` set incl. q / k (0 flagged, lDDT +0.0000); PV: 4 targets x 4 trunk seeds |
   | `protenix1` | fused | no | PV, 9 + 3 targets x 4 trunk seeds (no q / k run) |
   | `opendde` | unfused | no | PV, 16 + 3 targets x 4 trunk seeds (no q / k run) |

   Not listed: Boltz-2 (9c90 bistable, below), OpenFold3 (PV 4 / 4 inside, not added), every other model.
2. **e4m3 q / k stays in `safe`** (af3 only): its module error in the judge form is above the module tolerance (0.95 / 0.85 % at L512 / 1024 against
   tol 8e-3), but the end-level AF3 FoldBench run with it on flagged 0 / 27 targets.
3. **No FP8 registry rows / tolerances.** FP8 is an opt-in path outside the judge: the registry dtype vocabulary is `float32|master` only, the FP8 planes /
   e4m3 V are internal intermediates of the `master` route, no `backend_versions` entry exists, and the judge tokens cover the default bf16 route only
   (`tests/cpu/test_fp8_switch.py::test_fp8_is_outside_the_judge` keeps the judge / measure / requirements code free of FP8). Do not run the judge or
   `miniworld_engine.measure` with an FP8 switch set. Several routes are outside the module tolerance; the recommendation rests on the end-level evidence.
4. **Experiment harnesses are not in the release tree.** `benchmarks/fp8_e2e/` (end-level FoldForge runs: `probe.py`, `e2e2.py`, `an3.py`, `bz.py`,
   `scales.py`, `results.txt`) and `benchmarks/opm_fp8/` (OPM FP8 `stages.py`, `speed.py`, `err.py`, `lead.py`, `bit.py`, raw rows) were removed from
   `rel/3.0.0`; they live on branch `fp8/3.0.0-all` (e5ce31d0c; added in bd6fe26ef / 5c3db813a) and in the `rel/3.0.0` history up to 80ecfe71f
   (`git show 80ecfe71f:benchmarks/fp8_e2e/results.txt`). No shipped test imported them. The paths below that name them refer to that commit.

## Release line 3.0.0: module time and error against the 2.5.14 default (base 2.5.14, 2026-11-17)

Tree `rel/3.0.0` (2.5.14 9dd18e507 + `fp8/3.0.0-all`; with every switch off its outputs are the 2.5.14 bytes: 64 / 64 registry and direct cases,
sha1, separate processes, see "Release line 3.0.0: verification"). So the default route measured in the same process is the 2.5.14 default.
Every route re-measured with the protocol and shapes of its own section below (module, master weights, CUDA graph of one call per route, one
process, interleaved with a same-cubin control; `cool / sustained`; FP8 with every gate removed unless a row says otherwise) plus the error
against fp64 next to it. Raw rows and scripts: `~/miniworld-engine-scratch/rel300/speed/{ta_tm,opm_pwa_tr,dit}/` (`res/`, `scripts/`).
Summary: 2.5.13's faster default (TriAttn TA_RC: 64:4:16 -5 .. -8 % cool, 32-channel heads -2 .. -4 % from L512; TriMul D64 / D128 LT_UNI
-1 .. -4 % from L1024) raises the FP8 ratios at mid L by 0.01-0.03 (64:4:16 L1024 0.850 -> 0.877; TriMul D128 out L2048 0.739 -> 0.754);
L1536-2048 and the OPM / PWA / Transition / token DiT / atom DiT ratios are unchanged within the control; every error is the 2.5.12-base
number to the printed digit; no gate moves (each gate's region re-measured: below).

### TriAttn / TriMul FP8 routes, base 2.5.14 (rel/3.0.0 tree wt-300, 77cd55eb0 = origin/main 2.5.14 9dd18e507 + FP8 routes), 2026-11-17

Protocol = docs/fp8.md "Base 2.5.12": H100 80GB HBM3 (node02), one process per group, module (`master_weights=True`, one sample unless the case says B, all-ones mask)
in three CUDA graphs of one call each: 2.5.14 default (every FP8 switch off: bit-identical to 2.5.14), the default again (same-cubin control), the FP8 route
(hi plane / FP8 planes, every S / min-L gate removed, static scales from two eager calls, lazy guard; no overflow in any case). cool = 2 s idle + one interleaved
~10 ms burst, 3 reps, median; sustained = 2 s idle, 1 s back-to-back interleaved, 3 interleaved timed blocks, mean. Cells `cool / sustained`.
Scripts: copies of fp8/w300/scripts/{speed.py,err.py}, fp8x-trimul/scripts/{speed2.py,err3.py,variants.py}, fp8x-triattn/scripts/{sp.py,tqk.py} in `scripts/`,
changed only in tree (namespace pin `scripts/pin.py` -> wt-300/src) and output path (`res/`), plus ONE protocol fix in `speed.py`: the FP8 route also sets
`h100_sm90.FP8_PV_FUSED_MIN_S = 0` (that gate was added after the 2.5.12 table was measured; without it 128:4:32 L128 would measure the fp16 core, not FP8).
`sp.py` (PV + QK) already did the same. `err.py` / `tqk.py` unchanged (the docs' error protocol). Jobs: `scripts/jobs.txt`, logs `logs/`, raw rows `res/*.jsonl`,
tables regenerated by `scripts/make_table.py` (-> `TABLE_tables.md`).

TriAttn (`d_pair:heads:channels`, starting node), FP8 PV hi plane with no gate (incl. `FP8_PV_FUSED_MIN_S` = 0):

| config | | L128 | L256 | L512 | L768 | L1024 | L1536 | L2048 | same-cubin control |
|---|---|---|---|---|---|---|---|---|---|
| 128:4:32 | 2.5.14 default ms | 0.031 / 0.030 | 0.102 / 0.101 | 0.430 / 0.454 | 1.127 / 1.222 | 2.348 / 2.613 | 6.846 / 8.241 | 15.12 / 17.07 | |
| | FP8 / default | 1.064 / 1.067 | 0.965 / 0.958 | 0.907 / 0.912 | 0.883 / 0.877 | 0.868 / 0.852 | 0.847 / 0.832 | 0.837 / 0.838 | 0.984 .. 1.000 |
| 256:8:32 | 2.5.14 default ms |  | 0.231 / 0.239 | 0.993 / 1.093 | 2.593 / 2.835 | 5.316 / 5.978 | 15.18 / 17.50 | 32.97 / 36.80 | |
| | FP8 / default |  | 1.010 / 1.036 | 0.972 / 0.970 | 0.953 / 0.971 | 0.931 / 0.948 | 0.917 / 0.916 | 0.965 / 0.869 | 0.984 .. 1.039 |
| 384:12:32 | 2.5.14 default ms |  | 0.402 / 0.426 | 1.712 / 1.952 | 4.442 / 5.001 | 8.985 / 10.27 | 25.25 / 28.30 | 57.81 / 61.79 | |
| | FP8 / default |  | 0.997 / 1.003 | 1.007 / 0.978 | 0.969 / 0.956 | 0.941 / 0.926 | 0.908 / 0.884 | 0.905 / 0.871 | 0.986 .. 1.066 |
| 64:4:16 | 2.5.14 default ms | 0.028 / 0.027 | 0.087 / 0.084 | 0.382 / 0.378 | 1.009 / 1.066 | 2.092 / 2.220 | 6.111 / 6.697 | 13.56 / 15.60 | |
| | FP8 / default | 1.052 / 1.056 | 0.957 / 0.958 | 0.928 / 0.930 | 0.889 / 0.875 | 0.877 / 0.879 | 0.855 / 0.842 | 0.834 / 0.841 | 0.982 .. 1.004 |
| 64:4:32 | 2.5.14 default ms | 0.032 / 0.031 | 0.106 / 0.105 | 0.460 / 0.480 | 1.200 / 1.269 | 2.381 / 2.669 | 6.953 / 8.016 | 15.21 / 18.52 | |
| | FP8 / default | 1.030 / 1.033 | 0.910 / 0.916 | 0.880 / 0.874 | 0.866 / 0.880 | 0.878 / 0.891 | 0.852 / 0.876 | 0.835 / 0.849 | 0.994 .. 1.012 |
| 64:2:32 | 2.5.14 default ms | 0.024 / 0.022 | 0.064 / 0.063 | 0.270 / 0.272 | 0.669 / 0.703 | 1.307 / 1.423 | 3.691 / 4.205 | 7.960 / 9.179 | |
| | FP8 / default | 1.063 / 1.049 | 0.954 / 0.947 | 0.936 / 0.932 | 0.908 / 0.916 | 0.913 / 0.909 | 0.889 / 0.879 | 0.865 / 0.853 | 0.990 .. 1.017 |

TriMul D128 / D256 (one direction / the bidirectional block), FP8 planes with no gate:

| config | | L256 | L512 | L768 | L1024 | L1536 | L2048 | same-cubin control |
|---|---|---|---|---|---|---|---|---|
| D128 out | 2.5.14 default ms | 0.069 / 0.068 | 0.244 / 0.257 | 0.523 / 0.578 | 0.958 / 1.093 | 2.478 / 2.800 | 5.096 / 5.783 | |
| | FP8 / default | 0.907 / 0.913 | 0.848 / 0.836 | 0.852 / 0.863 | 0.838 / 0.882 | 0.798 / 0.801 | 0.754 / 0.752 | 0.991 .. 1.017 |
| D128 in | 2.5.14 default ms | 0.069 / 0.069 | 0.243 / 0.257 | 0.517 / 0.579 | 0.953 / 1.132 | 2.584 / 2.932 | 5.160 / 5.989 | |
| | FP8 / default | 0.882 / 0.884 | 0.855 / 0.841 | 0.861 / 0.875 | 0.848 / 0.851 | 0.772 / 0.788 | 0.748 / 0.730 | 0.987 .. 1.015 |
| D128 bidir | 2.5.14 default ms | 0.116 / 0.115 | 0.397 / 0.431 | 0.884 / 0.992 | 1.652 / 1.916 | 4.393 / 5.081 | 9.112 / 10.80 | |
| | FP8 / default | 0.964 / 0.967 | 0.918 / 0.935 | 0.884 / 0.925 | 0.865 / 0.870 | 0.811 / 0.801 | 0.762 / 0.763 | 0.974 .. 1.019 |
| D256 out | 2.5.14 default ms | 0.154 / 0.161 | 0.554 / 0.611 | 1.228 / 1.367 | 2.253 / 2.503 | 5.736 / 6.702 | 11.58 / 13.47 | |
| | FP8 / default | 0.957 / 0.966 | 0.940 / 0.986 | 0.915 / 0.962 | 0.911 / 0.959 | 0.870 / 0.910 | 0.818 / 0.820 | 0.986 .. 1.011 |
| D256 in | 2.5.14 default ms | 0.155 / 0.162 | 0.557 / 0.599 | 1.215 / 1.327 | 2.252 / 2.518 | 5.851 / 6.965 | 11.53 / 14.30 | |
| | FP8 / default | 0.954 / 0.962 | 0.938 / 0.966 | 0.924 / 0.976 | 0.913 / 0.945 | 0.853 / 0.851 | 0.822 / 0.806 | 0.982 .. 1.004 |
| D256 bidir | 2.5.14 default ms | 0.259 / 0.269 | 0.941 / 1.016 | 2.132 / 2.426 | 4.021 / 4.675 | 10.29 / 11.87 | 20.62 / 23.74 | |
| | FP8 / default | 1.010 / 1.032 | 0.983 / 1.040 | 0.945 / 0.981 | 0.925 / 0.923 | 0.879 / 0.870 | 0.845 / 0.830 | 0.987 .. 1.007 |

TriMul below L256 (the min-L gates; L256 column from the table above):

| config | | L128 | L144 | L160 | L176 | L192 | L224 | L256 | same-cubin control |
|---|---|---|---|---|---|---|---|---|---|
| D128 out | 2.5.14 default ms | 0.023 / 0.023 | 0.039 / 0.038 | 0.041 / 0.039 | 0.041 / 0.040 | 0.048 / 0.047 | 0.064 / 0.061 | 0.069 / 0.068 | |
| | FP8 / default | 0.988 / 0.990 | 0.921 / 0.921 | 0.888 / 0.882 | 0.874 / 0.869 | 0.937 / 0.933 | 0.928 / 0.929 | 0.907 / 0.913 | 0.991 .. 1.012 |
| D128 in | 2.5.14 default ms | 0.024 / 0.023 | 0.035 / 0.034 | 0.036 / 0.035 | 0.037 / 0.036 | 0.048 / 0.047 | 0.064 / 0.062 | 0.069 / 0.069 | |
| | FP8 / default | 0.996 / 1.000 | 1.047 / 1.046 | 1.003 / 0.998 | 1.008 / 1.004 | 0.913 / 0.907 | 0.912 / 0.912 | 0.882 / 0.884 | 0.991 .. 1.008 |
| D128 bidir | 2.5.14 default ms |  |  | 0.066 / 0.064 |  | 0.075 / 0.073 | 0.104 / 0.103 | 0.116 / 0.115 | |
| | FP8 / default |  |  | 0.937 / 0.929 |  | 1.046 / 1.047 | 0.988 / 0.997 | 0.964 / 0.967 | 0.993 .. 1.007 |
| D256 out | 2.5.14 default ms | 0.044 / 0.043 | 0.073 / 0.073 |  | 0.079 / 0.082 | 0.107 / 0.107 | 0.147 / 0.151 | 0.154 / 0.161 | |
| | FP8 / default | 0.935 / 0.932 | 1.008 / 0.995 |  | 1.007 / 0.993 | 1.010 / 1.009 | 0.949 / 0.947 | 0.957 / 0.966 | 0.980 .. 1.007 |
| D256 in | 2.5.14 default ms | 0.043 / 0.042 | 0.083 / 0.082 |  | 0.089 / 0.090 | 0.109 / 0.109 | 0.148 / 0.151 | 0.155 / 0.162 | |
| | FP8 / default | 0.944 / 0.939 | 0.891 / 0.899 |  | 0.914 / 0.916 | 0.981 / 0.984 | 0.930 / 0.931 | 0.954 / 0.962 | 0.993 .. 1.000 |

TriMul D64 / D384 one direction (new coverage; `spd1` + the L2048 repeat of `spd2` is not merged):

| config | | L128 | L256 | L512 | L768 | L1024 | L1536 | L2048 | same-cubin control |
|---|---|---|---|---|---|---|---|---|---|
| D64 out | 2.5.14 default ms | 0.016 / 0.015 | 0.036 / 0.035 | 0.133 / 0.132 | 0.290 / 0.298 | 0.500 / 0.540 | 1.279 / 1.420 | 2.589 / 2.972 | |
| | FP8 / default | 1.144 / 1.148 | 0.984 / 0.983 | 0.901 / 0.905 | 0.874 / 0.915 | 0.887 / 0.936 | 0.827 / 0.880 | 0.780 / 0.789 | 0.959 .. 1.016 |
| D64 in | 2.5.14 default ms |  | 0.036 / 0.035 | 0.133 / 0.133 |  | 0.498 / 0.542 |  | 2.616 / 2.955 | |
| | FP8 / default |  | 0.979 / 0.977 | 0.892 / 0.895 |  | 0.891 / 0.934 |  | 0.779 / 0.784 | 0.986 .. 1.003 |
| D384 out | 2.5.14 default ms | 0.075 / 0.077 | 0.286 / 0.307 | 1.113 / 1.201 | 2.425 / 2.737 | 4.591 / 5.175 | 11.23 / 13.00 | 21.95 / 24.41 | |
| | FP8 / default | 0.956 / 0.962 | 0.932 / 0.940 | 0.908 / 0.932 | 0.916 / 0.940 | 0.894 / 0.918 | 0.864 / 0.878 | 0.838 / 0.835 | 0.987 .. 1.009 |
| D384 in | 2.5.14 default ms |  | 0.287 / 0.307 | 1.104 / 1.182 |  | 4.564 / 5.207 |  | 22.01 / 24.92 | |
| | FP8 / default |  | 0.931 / 0.927 | 0.923 / 0.942 |  | 0.897 / 0.895 |  | 0.887 / 0.831 | 0.989 .. 1.000 |

TriMul bidirectional D64 / D384 and B > 1 (`spd2`; FP8 / FP8 + FASTACC against the default):

| case | 2.5.14 default ms (cool / sus) | FP8 / default | FP8 + FASTACC / default | control |
|---|---|---|---|---|
| bidirectional D64 L512 / 1024 / 2048 | 0.200 / 0.205, 0.843 / 0.862, 4.645 / 4.845 | 1.033 / 1.040, 0.922 / 1.053, 0.779 / 0.873 | 1.015 / 1.029, 0.903 / 1.011, 0.755 / 0.863 | 0.987 .. 1.007 |
| bidirectional D384 L512 / 1024 / 2048 | 1.903 / 2.093, 8.079 / 9.172, 39.35 / 43.17 | 0.968 / 0.986, 0.925 / 0.930, 0.913 / 0.875 | 0.980 / 0.978, 0.909 / 0.912, 0.961 / 0.865 | 0.985 .. 1.079 |
| D64 out B4 L512 / L1024 | 0.501 / 0.500, 1.966 / 2.137 | 0.948 / 0.980, 0.901 / 0.956 | 0.934 / 0.956, 0.885 / 0.922 | 0.991 .. 0.998 |
| D128 out B2 L512 / L1024 | 0.460 / 0.487, 1.900 / 2.154 | 0.900 / 0.911, 0.848 / 0.884 | 0.900 / 0.907, 0.830 / 0.846 | 0.997 .. 1.008 |

TriAttn FP8 PV + QK, 128:4:32 (`sp.py`, `qk.jsonl`):

| L | 128 | 256 | 512 | 768 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|---|---|
| 2.5.14 default ms | 0.032 / 0.031 | 0.101 / 0.100 | 0.430 / 0.466 | 1.134 / 1.232 | 2.349 / 2.515 | 6.839 / 8.049 | 15.10 / 17.67 |
| FP8 PV / default | 1.086 / 1.086 | 0.959 / 0.970 | 0.906 / 0.872 | 0.877 / 0.869 | 0.870 / 0.871 | 0.852 / 0.842 | 0.836 / 0.831 |
| FP8 PV + QK / default | 1.055 / 1.051 | 0.947 / 0.956 | 0.848 / 0.824 | 0.818 / 0.847 | 0.811 / 0.828 | 0.794 / 0.809 | 0.827 / 0.792 |
| FP8 PV + QK / FP8 PV | 0.971 / 0.967 | 0.988 / 0.985 | 0.936 / 0.945 | 0.933 / 0.975 | 0.932 / 0.950 | 0.932 / 0.961 | 0.990 / 0.953 |
| same-cubin control | 0.985 / 0.984 | 0.991 / 0.996 | 0.998 / 0.971 | 0.997 / 0.983 | 0.999 / 0.998 | 0.999 / 0.999 | 1.001 / 1.000 |

Error against fp64 PyTorch (update rms / max relative to the fp64 update; std_max = judge form max|y - y64| / max|y64|):

| case | default 2.5.14 rms / max (update) | default std_max | FP8 hi rms / max | FP8 hi std_max | FP8 hi + lo rms / max | FP8 hi + lo std_max | overflow / peak |
|---|---|---|---|---|---|---|---|
| ta:128:4:32:256 | 1.6% / 1.5% | 0.29% | 3.6% / 2.9% | 0.57% | 2.9% / 2.3% | 0.44% | False |
| ta:128:4:32:512 | 1.9% / 2.1% | 0.29% | 3.3% / 4.7% | 0.66% | 2.8% / 2.7% | 0.38% | False |
| ta:256:8:32:256 | 1.7% / 2.1% | 0.29% | 3.7% / 3.6% | 0.5% | 3% / 2.6% | 0.37% | False |
| ta:256:8:32:512 | 1.9% / 2.2% | 0.27% | 3.4% / 3.5% | 0.43% | 2.9% / 2.9% | 0.36% | False |
| ta:384:12:32:256 | 1.7% / 2.3% | 0.32% | 1.7% / 2.3% | 0.32% | 1.7% / 2.3% | 0.32% | False |
| ta:384:12:32:512 | 1.9% / 2.3% | 0.28% | 3.4% / 3.8% | 0.46% | 2.9% / 3.1% | 0.38% | False |
| ta:64:4:16:256 | 1.6% / 1.7% | 0.29% | 3.3% / 3.4% | 0.55% | 2.7% / 2.6% | 0.42% | False |
| ta:64:4:16:512 | 1.8% / 1.6% | 0.28% | 3% / 3.1% | 0.55% | 2.5% / 1.9% | 0.34% | False |
| ta:64:4:32:256 | 1.2% / 1.3% | 0.3% | 3.1% / 2.7% | 0.64% | 2.5% / 2.1% | 0.5% | False |
| ta:64:4:32:512 | 1.3% / 1.5% | 0.3% | 2.7% / 3.2% | 0.65% | 2.2% / 2.2% | 0.45% | False |
| ta:64:2:32:256 | 1.6% / 1.7% | 0.31% | 3.3% / 4% | 0.75% | 2.7% / 2.3% | 0.42% | False |
| ta:64:2:32:512 | 1.8% / 1.9% | 0.29% | 3% / 3.8% | 0.58% | 2.5% / 2.7% | 0.4% | False |
| tm:128:1:256 | 0.66% / 0.7% | 0.5% | 3.8% / 4.2% | 3% | - | - | False |
| tm:128:1:512 | 0.66% / 0.69% | 0.44% | 3.7% / 4.2% | 2.7% | - | - | False |
| tm:128:1:1024 | 0.65% / 0.7% | 0.49% | 3.7% / 4.4% | 3.1% | - | - | False |
| tm:128:2:256 | 0.66% / 0.72% | 0.47% | 3.8% / 3.9% | 2.5% | - | - | False |
| tm:128:2:512 | 0.66% / 0.75% | 0.54% | 3.7% / 3.9% | 2.8% | - | - | False |
| tm:128:2:1024 | 0.64% / 0.78% | 0.52% | 3.7% / 4% | 2.6% | - | - | False |
| tm:128:0:256 | 0.65% / 0.75% | 0.48% | 3.7% / 3.8% | 2.4% | - | - | False |
| tm:128:0:512 | 0.65% / 0.84% | 0.52% | 3.7% / 4.2% | 2.6% | - | - | False |
| tm:128:0:1024 | 0.64% / 0.72% | 0.5% | 3.7% / 3.9% | 2.7% | - | - | False |
| tm:256:1:256 | 0.65% / 0.78% | 0.5% | 3.8% / 4% | 2.6% | - | - | False |
| tm:256:1:512 | 0.65% / 0.78% | 0.51% | 3.7% / 4.3% | 2.8% | - | - | False |
| tm:256:1:1024 | 0.65% / 0.77% | 0.52% | 3.7% / 4% | 2.7% | - | - | False |
| tm:256:2:256 | 0.65% / 0.82% | 0.46% | 3.7% / 4.1% | 2.3% | - | - | False |
| tm:256:2:512 | 0.65% / 0.77% | 0.51% | 3.7% / 4% | 2.6% | - | - | False |
| tm:256:2:1024 | 0.65% / 0.71% | 0.52% | 3.6% / 4% | 2.9% | - | - | False |
| tm:256:0:256 | 0.65% / 0.73% | 0.51% | 3.7% / 3.6% | 2.5% | - | - | False |
| tm:256:0:512 | 0.65% / 0.78% | 0.5% | 3.7% / 3.5% | 2.3% | - | - | False |
| tm:256:0:1024 | 0.65% / 0.72% | 0.47% | 3.7% / 3.6% | 2.4% | - | - | False |
| tm:64:1:256 | 0.65% / 0.94% | 0.63% | 3.7% / 5.2% | 3.5% | - | - | False |
| tm:64:2:300 | 0.66% / 0.83% | 0.51% | 3.7% / 4.9% | 3% | - | - | False |
| tm:64:1:1024 | 0.63% / 0.73% | 0.5% | 3.5% / 4.1% | 2.8% | - | - | False |
| tm:64:2:1024 | 0.64% / 0.79% | 0.53% | 3.6% / 4.1% | 2.7% | - | - | False |
| tm:384:1:256 | 0.65% / 0.74% | 0.44% | 3.7% / 4% | 2.4% | - | - | False |
| tm:384:2:384 | 0.65% / 0.61% | 0.42% | 3.7% / 3.3% | 2.2% | - | - | False |
| tm:384:1:300 | 0.65% / 0.72% | 0.51% | 3.7% / 3.4% | 2.4% | - | - | False |
| tm:384:1:1024 | 0.64% / 0.75% | 0.54% | 3.7% / 3.6% | 2.6% | - | - | False |
| tm:128:1:512 | 0.66% / 0.69% | 0.44% | 3.7% / 4.2% | 2.7% | - | - | False |
| tm:256:2:512 | 0.65% / 0.77% | 0.51% | 3.7% / 4% | 2.6% | - | - | False |
| tm:64:0:512 | 0.65% / 0.72% | 0.53% | 3.7% / 3.7% | 2.7% | - | - | 18.442607879638672 |
| tm:384:0:512 | 0.65% / 0.78% | 0.46% | 3.7% / 3.9% | 2.3% | - | - | 19.07242774963379 |
| tm:64:1:512:3 | 0.65% / 0.77% | 0.54% | 3.6% / 4.3% | 3% | - | - | 18.907920837402344 |
| tm:384:2:256:2 | 0.65% / 0.7% | 0.49% | 3.7% / 3.5% | 2.5% | - | - | 18.23284912109375 |

Error, FP8 PV + QK (`tqk.py`, `qk1.jsonl`; weights 128^-0.5, 3 masked keys):

| case | route | update rms / max | std_max | nan | overflow |
|---|---|---|---|---|---|
| ta:128:4:32:512 | def | 1.9% / 2.1% | 0.29% | False | False |
| ta:128:4:32:512 | f8 | 3.3% / 4.7% | 0.66% | False | False |
| ta:128:4:32:512 | f8qk | 4.4% / 6.8% | 0.95% | False | False |
| ta:128:4:32:1024 | def | 2.1% / 2.1% | 0.28% | False | False |
| ta:128:4:32:1024 | f8 | 3.1% / 3.4% | 0.43% | False | False |
| ta:128:4:32:1024 | f8qk | 4.1% / 6.6% | 0.85% | False | False |

#### Gates against the 2.5.14 default (data above; per-cell control in parentheses)

TriAttn (`h100_sm90`):
- `FP8_PV_FUSED_MIN_S` = 256 (128:4:32): L128 +6.4 / +6.7 % (control 0.984 / 0.987), L256 -3.5 / -4.2 % (control 0.991 / 0.990; sp.py repeat -4.1 / -3.0 %).
  The faster default shrank the L256 gain (2.5.12: -6.4 / -5.8 %) but did not move the crossover. Keep 256.
- `FP8_PV_UNFUSED_FRONT_MIN_S` d_pair 64 (256 for 64:4:32 / 64:2:32 / 64:4:16): L128 +3 .. +6 %, L256 -9 % / -5 % / -4 % (controls 0.99-1.01). Keep 256.
- `FP8_PV_UNFUSED_FRONT_MIN_S` 256:8:32 = 768: L512 -2.8 / -3.0 % (control 0.994 / 0.984), L768 -4.7 / -2.9 %. Margin over the control 1.4-2.2 %, one run
  (2.5.12: -2.4 / -3.8 % with control 0.968). Not clear; keep 768 (a repeat would decide 512).
- `FP8_PV_UNFUSED_MIN_S[32]` = 1024 (standalone pass, 384:12:32): L512 +0.7 / -2.2 % (not clear), L768 -3.1 / -4.4 % (control 1.000 / 0.992; 2.5.12: -1.8 / -1.2 %).
  L768 now leans to FP8, but run-to-run spread between the 2.5.12 and this run is as large as the gain, and the gate is shared by every 32-channel
  standalone site. Not clear; keep 1024 (candidate 768 after a repeat; matches the OpenDDE 768 evaluation override).
- `FP8_PV_UNFUSED_MIN_S[16]` = 512: no standalone 16-channel case in the tables (64:4:16 takes front-made planes). Not re-measured.

TriMul (`h100_b200style_fp8.MIN_L`):
- D128 out 144: L128 0.988 / 0.990 (control 0.994 / 0.999: not clear), L144 -8 %. Keep.
- D128 in 192: L176 +0.8 / +0.4 % (2.5.12: -2 %), L192 -9 %. Keep.
- D256 out 224: L144-192 +0 .. +1 %, L224 -5 %; L128 -6.5 % (isolated, as on 2.5.12). Keep.
- D256 in 144: L128 -5.6 / -6.1 % (as on 2.5.12; lowering needs L < 128 data). Keep.
- bidirectional D128 256: L160 -6.3 / -7.1 %, L192 +4.6 %, L224 -1.2 / -0.3 %: not monotone. Keep.
- bidirectional D256 1024: L512 -1.7 / +4.0 %, L768 -5.5 / -1.9 % (control 0.998 / 1.006), L1024 -7.5 / -7.7 %. Not clear below 1024. Keep.
- D64 512 (both directions): L128 +14 %, L256 -1.6 / -1.7 % out, -2.1 / -2.3 % in (row control 0.959 .. 1.016). Not clear. Keep.
- D384 128: every L gains (L128 -4.4 / -3.8 %). Keep.
- bidirectional D64 2048: L1024 -7.8 % cool but +5.3 % sustained. Keep.
- bidirectional D384 1024: L512 -3.2 / -1.4 % (control 0.998 / 0.993). Not clear. Keep.

Verdict: the faster 2.5.13 / 2.5.14 default moves no gate; no change recommended. Two candidates for a repeat run only: 256:8:32 front gate 768 -> 512
and `FP8_PV_UNFUSED_MIN_S[32]` 1024 -> 768.

#### Against the 2.5.12-base numbers in docs/fp8.md

- 2.5.14 default TriAttn faster (cool): 64:4:16 -5 .. -8 % (L512 0.410 -> 0.382, L768 1.088 -> 1.009, L1024 2.240 -> 2.092, L2048 14.21 -> 13.56 ms),
  128:4:32 / 64:4:32 / 64:2:32 -2 .. -4 % from L512, 256:8:32 -1.5 .. -2 %, 384:12:32 about 0-1 %. FP8 PV ratios moved up at mid L
  (64:4:16 L512 0.899 -> 0.928, L768 0.856 -> 0.889, L1024 0.850 -> 0.877; 128:4:32 L256 0.936 -> 0.965; 64:2:32 L512 0.917 -> 0.936); L1536-2048 unchanged (0.83-0.87).
- 2.5.14 default TriMul D128 (one direction and bidirectional) -1 .. -4 % from L1024 (out L1024 0.981 -> 0.958, L1536 2.549 -> 2.478), D64 one direction
  -2 .. -3 % from L1024; FP8 ratios up by about 0.01-0.025 there (D128 out L1536 0.774 -> 0.798, L2048 0.739 -> 0.754; D64 out L1024 0.863 -> 0.887).
  D256 and D384 defaults unchanged within noise; their ratios too (D256 bidir L2048 0.878 -> 0.845 cool is a noisy cell, sustained 0.824 -> 0.830).
- FP8 PV + QK: unchanged (0.81-0.85 of the default from L512; + QK / PV 0.93-0.95).
- Error: every FP8 number equals the 2.5.12-base table to the printed digit; defaults the same (TriMul 0.64-0.66 % update rms). No overflow / NaN.

### OPM / PWA / Transition FP8 routes on the 3.0.0 tree (wt-300), base 2.5.14

Tree: `rel/3.0.0` (wt-300) pinned by namespace; default = every FP8 switch off (= 2.5.14 bytes), measured in the same process.
Scripts: copies of the fp8x-* originals in `scripts/` (only output paths changed; `pin.py` pins the tree), raw rows in `res/`.
Cells `cool / sustained`; FP8 / default; control = default captured twice (same cubin).

#### OuterProductMean forward, base 2.5.14

`opm_speed.py` (= fp8x-opm speed.py), routes def / ctl / f8b (bound scales), every gate removed, lazy guard. Ragged MSA. Same-cubin control (cool and sustained, all rows): 0.944 .. 1.081.

d_msa 64 / d_pair 128, Boltz order (2.5.14 default ms cool / sustained in brackets):

| L \ S | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|
| 128 | 0.94 / 0.91 (0.0608 / 0.0590) | 1.35 / 1.36 (0.0493 / 0.0493) | 1.07 / 1.04 (0.0784 / 0.0796) | 0.89 / 0.88 (0.135 / 0.142) | 0.75 / 0.73 (0.256 / 0.273) |  | 0.65 / 0.65 (0.908 / 0.980) |
| 192 |  |  | 0.48 / 0.48 (0.312 / 0.316) |  |  |  |  |
| 256 | 0.98 / 0.99 (0.149 / 0.150) | 1.12 / 1.07 (0.155 / 0.163) | 0.91 / 0.85 (0.254 / 0.276) | 0.74 / 0.71 (0.460 / 0.501) | 0.65 / 0.62 (0.855 / 0.959) | 0.63 / 0.61 (1.619 / 1.815) | 0.61 / 0.57 (3.146 / 3.551) |
| 384 |  | 1.08 / 0.99 (0.316 / 0.348) | 0.86 / 0.79 (0.531 / 0.595) | 0.70 / 0.68 (0.967 / 1.061) | 0.62 / 0.59 (1.829 / 2.094) | 0.63 / 0.58 (3.376 / 3.867) | 0.60 / 0.58 (6.678 / 7.243) |
| 768 |  | 1.05 / 0.95 (1.156 / 1.308) | 0.82 / 0.74 (1.975 / 2.217) | 0.68 / 0.63 (3.670 / 4.107) | 0.60 / 0.57 (6.843 / 7.791) | 0.60 / 0.56 (12.93 / 14.66) | 0.59 / 0.52 (25.67 / 29.98) |
| 1024 |  |  |  |  | 0.48 / 0.48 (14.89 / 16.32) |  | 0.57 / 0.53 (46.86 / 53.01) |
| 2048 |  |  |  |  | 0.52 / 0.52 (54.23 / 59.44) |  | 0.58 / 0.55 (189.99 / 210.86) |

AF3 order and the wide widths (FP8 / default cool / sus, default ms cool):

| case (d_msa:d_pair:L:S[:a]) | default ms cool / sus | f8b / default | control |
|---|---|---|---|
| 128:256:384:1024:a | 0.657 / 0.740 | 0.81 / 0.75 | 1.044 / 0.994 |
| 128:256:384:4096:a | 1.953 / 2.177 | 0.65 / 0.64 | 1.052 / 1.020 |
| 128:256:384:16384:a | 7.157 / 8.070 | 0.60 / 0.56 | 1.021 / 1.019 |
| 128:256:768:1024:a | 2.399 / 2.746 | 0.78 / 0.72 | 1.022 / 0.985 |
| 128:256:768:16384:a | 26.98 / 30.74 | 0.58 / 0.53 | 1.000 / 0.993 |
| 128:384:384:1024:a | 0.728 / 0.810 | 0.82 / 0.78 | 1.027 / 1.011 |
| 128:384:384:16384:a | 7.334 / 8.055 | 0.60 / 0.57 | 1.046 / 1.057 |
| 128:384:768:4096:a | 7.504 / 8.738 | 0.64 / 0.59 | 1.034 / 0.975 |
| 256:512:384:1024 | 0.854 / 0.950 | 0.85 / 0.80 | 1.010 / 0.988 |
| 256:512:384:16384 | 7.957 / 9.088 | 0.64 / 0.60 | 1.037 / 1.006 |
| 256:512:768:4096 | 8.121 / 9.378 | 0.65 / 0.61 | 1.048 / 0.991 |
| 64:128:768:16384:a | 25.66 / 29.30 | 0.58 / 0.54 | 1.001 / 1.000 |
| 64:128:384:2048:a | 0.993 / 1.084 | 0.68 / 0.66 | 1.014 / 1.022 |

Saturation peak over all rows: max 26.0 of 448.

Error against fp64 PyTorch (`opm_err.py`; rms / max relative; tol = max(err_pytorch, 8e-3)):

| case | PyTorch bf16 max | default rms / max | FP8 bound rms / max (/ tol) | FP8 calib max | bound + fast acc. max (/ tol) |
|---|---|---|---|---|---|
| row:55:384:1024 | 4.44e-03 | 2.42e-03 / 4.44e-03 | 9.93e-03 / 1.37e-02 (1.72) | 1.37e-02 | 1.37e-02 (1.72) |
| row:57:384:1024 | 5.76e-03 | 2.39e-03 / 5.76e-03 | 5.66e-03 / 1.05e-02 (1.32) | 1.05e-02 | 1.05e-02 (1.32) |
| row:63:384:1024 | 3.13e-03 | 2.42e-03 / 3.08e-03 | 2.42e-03 / 3.08e-03 (0.38) | 3.08e-03 | 3.08e-03 (0.38) |
| 64:128:384:1024 | 5.25e-03 | 2.59e-03 / 5.25e-03 | 1.11e-02 / 1.15e-02 (1.44) | 1.15e-02 | 1.15e-02 (1.44) |
| 64:128:384:4096:a | 5.50e-03 | 4.66e-03 / 4.36e-03 | 3.01e-02 / 8.20e-03 (1.03) | 8.20e-03 | 1.34e-02 (1.68) |
| 64:128:768:16384 | 6.25e-03 | 2.25e-03 / 5.71e-03 | 3.60e-03 / 6.92e-03 (0.86) | 6.92e-03 | 3.68e-02 (4.59) |
| 64:128:768:16384:a | 6.29e-03 | 4.43e-03 / 4.69e-03 | 2.22e-02 / 6.31e-03 (0.79) | 6.31e-03 | 4.46e-02 (5.58) |
| 64:128:256:16384 | 6.21e-03 | 2.26e-03 / 5.78e-03 | 3.60e-03 / 6.22e-03 (0.78) | 6.22e-03 | 3.46e-02 (4.32) |
| 64:128:768:2048 | 6.18e-03 | 2.42e-03 / 5.92e-03 | 8.14e-03 / 9.46e-03 (1.18) | 9.46e-03 | 9.81e-03 (1.23) |
| 128:256:384:4096:a | 6.13e-03 | 4.84e-03 / 4.24e-03 | 3.48e-02 / 1.01e-02 (1.26) | 1.01e-02 | 1.19e-02 (1.49) |
| 128:384:384:4096:a | 7.13e-03 | 4.83e-03 / 4.72e-03 | 3.49e-02 / 1.01e-02 (1.26) | 1.01e-02 | 1.17e-02 (1.46) |
| 256:512:384:2048 | 5.49e-03 | 2.58e-03 / 5.09e-03 | 9.11e-03 / 1.33e-02 (1.66) | 1.33e-02 | 1.31e-02 (1.64) |
| 64:128:1024:8192:a | 5.37e-03 | 4.59e-03 / 4.90e-03 | 2.73e-02 / 7.21e-03 (0.90) | 7.21e-03 | 2.21e-02 (2.76) |

#### PWA forward 8 x 32, base 2.5.14

`pwa_speed.py`: def / ctl / f8 (lazy guard, MIN_N = MIN_S = 0). Same-cubin control: 0.986 .. 1.017. Overflow in any row: False.

FP8 / default `cool / sustained`, d_pair 128:

| S \ N | 128 | 256 | 384 | 512 | 640 | 768 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|---|---|---|---|
| 128 |  |  | 1.096 / 1.092 |  |  | 0.926 / 0.928 |  |  | 0.859 / 0.864 |
| 256 |  |  | 1.089 / 1.091 |  |  | 0.841 / 0.840 | 0.841 / 0.846 |  |  |
| 512 |  |  | 1.010 / 1.011 | 0.855 / 0.858 |  | 0.816 / 0.815 |  |  |  |
| 1024 | 1.206 / 1.206 | 1.093 / 1.093 | 0.957 / 0.972 | 0.819 / 0.818 | 0.814 / 0.825 | 0.797 / 0.796 | 0.782 / 0.772 | 0.749 / 0.729 | 0.714 / 0.706 |
| 2048 |  |  | 0.926 / 0.938 |  |  | 0.783 / 0.779 |  |  |  |
| 4096 |  |  | 0.912 / 0.922 |  |  | 0.780 / 0.774 | 0.765 / 0.741 |  |  |
| 16384 |  |  | 0.910 / 0.901 |  |  | 0.785 / 0.780 |  |  |  |

2.5.14 default ms (cool / sustained):

| case (S:N[:d_pair]) | default ms | FP8 / default |
|---|---|---|
| 1024:128 | 0.115 / 0.113 | 1.206 / 1.206 |
| 1024:256 | 0.237 / 0.237 | 1.093 / 1.093 |
| 1024:384 | 0.401 / 0.402 | 0.957 / 0.972 |
| 1024:512 | 0.628 / 0.631 | 0.819 / 0.818 |
| 1024:768 | 1.047 / 1.076 | 0.797 / 0.796 |
| 1024:1024 | 1.556 / 1.633 | 0.782 / 0.772 |
| 1024:1536 | 2.880 / 3.064 | 0.749 / 0.729 |
| 1024:2048 | 4.639 / 5.085 | 0.714 / 0.706 |
| 4096:384 | 1.513 / 1.530 | 0.912 / 0.922 |
| 4096:768 | 3.937 / 4.107 | 0.780 / 0.774 |
| 4096:1024 | 5.829 / 6.192 | 0.765 / 0.741 |
| 16384:384 | 5.997 / 6.196 | 0.910 / 0.901 |
| 16384:768 | 15.53 / 16.05 | 0.785 / 0.780 |
| 1024:768:256 | 1.103 / 1.128 | 0.809 / 0.810 |
| 2048:768 | 2.001 / 2.075 | 0.783 / 0.779 |
| 512:384 | 0.215 / 0.215 | 1.010 / 1.011 |
| 128:768 | 0.205 / 0.205 | 0.926 / 0.928 |
| 256:768 | 0.335 / 0.335 | 0.841 / 0.840 |
| 512:768 | 0.569 / 0.582 | 0.816 / 0.815 |
| 2048:384 | 0.774 / 0.774 | 0.926 / 0.938 |
| 1024:640 | 0.819 / 0.839 | 0.814 / 0.825 |
| 128:2048 | 1.029 / 1.039 | 0.859 / 0.864 |
| 256:1024 | 0.488 / 0.491 | 0.841 / 0.846 |
| 128:384 | 0.0863 / 0.0846 | 1.096 / 1.092 |
| 256:384 | 0.124 / 0.124 | 1.089 / 1.091 |
| 512:512 | 0.336 / 0.335 | 0.855 / 0.858 |

Error against fp64 (`pwa_err.py`, cells `upd rms / upd max / judge` in %):

| case (S:N[:d_pair]) | PyTorch bf16 | default 2.5.14 | FP8 |
|---|---|---|---|
| 1024:256 | 2.06 / 3.42 / 0.29 | 2.04 / 3.42 / 0.29 | 3.18 / 4.29 / 0.37 |
| 1024:384 | 2.20 / 4.05 / 0.29 | 2.18 / 3.97 / 0.28 | 3.06 / 4.60 / 0.33 |
| 1024:512 | 2.28 / 3.79 / 0.29 | 2.27 / 3.74 / 0.29 | 2.99 / 4.60 / 0.35 |
| 1024:768 | 2.38 / 4.36 / 0.28 | 2.37 / 4.36 / 0.28 | 2.89 / 5.00 / 0.32 |
| 1024:1024 | 2.43 / 4.47 / 0.28 | 2.42 / 4.47 / 0.28 | 2.84 / 4.95 / 0.31 |
| 4096:768 | 2.38 / 4.26 / 0.28 | 2.37 / 4.21 / 0.28 | 2.90 / 5.03 / 0.33 |
| 1024:768:256 | 2.37 / 4.63 / 0.27 | 2.36 / 4.66 / 0.27 | 2.92 / 5.35 / 0.31 |
| 1024:2048 | 2.53 / 4.67 / 0.28 | 2.52 / 4.59 / 0.28 | 2.75 / 4.91 / 0.30 |
| 2048:384 | 2.20 / 3.88 / 0.28 | 2.18 / 3.88 / 0.28 | 3.06 / 4.35 / 0.31 |

#### PWA wide widths (AF3 family, `forward_wide`), base 2.5.14

`pwaw_speed.py`: WIDE_MIN_N = WIDE_MIN_SN = 0. Same-cubin control: 0.987 .. 1.006. Fired in every row: True; overflow: False.

AF3 width 64 x 8, d_pair 128, FP8 / default `cool / sustained`:

| S \ N | 128 | 192 | 256 | 384 | 400 | 512 | 768 | 784 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 |  |  |  |  |  |  | 1.018 / 1.019 |  |  |  |  |
| 256 |  |  |  | 1.008 / 1.014 |  |  | 0.972 / 0.971 |  |  | 0.941 / 0.944 |  |
| 512 |  |  | 1.026 / 1.037 | 0.974 / 0.969 |  | 0.943 / 0.941 | 0.924 / 0.926 |  |  | 0.885 / 0.886 | 0.857 / 0.864 |
| 1024 | 1.040 / 1.049 | 0.998 / 0.997 | 0.972 / 0.958 | 0.925 / 0.926 | 0.911 / 0.909 | 0.909 / 0.906 | 0.863 / 0.868 | 0.905 / 0.914 | 0.860 / 0.865 | 0.827 / 0.832 | 0.794 / 0.804 |
| 2048 |  |  | 0.915 / 0.916 | 0.895 / 0.887 |  |  | 0.840 / 0.852 |  |  |  |  |
| 4096 | 0.899 / 0.900 |  | 0.894 / 0.894 | 0.860 / 0.866 |  |  | 0.820 / 0.836 |  |  | 0.773 / 0.786 | 0.737 / 0.758 |
| 16384 | 0.878 / 0.878 |  | 0.869 / 0.881 |  |  | 0.824 / 0.848 | 0.809 / 0.832 |  | 0.790 / 0.803 |  |  |

All rows (2.5.14 default ms cool / sus):

| case (D:C:d_pair:S:N) | default ms | FP8 / default |
|---|---|---|
| 64:8:128:1024:768 | 0.404 / 0.404 | 0.863 / 0.868 |
| 64:8:128:1024:384 | 0.186 / 0.186 | 0.925 / 0.926 |
| 64:8:128:1024:256 | 0.122 / 0.118 | 0.972 / 0.958 |
| 64:8:128:1024:512 | 0.246 / 0.241 | 0.909 / 0.906 |
| 64:8:128:1024:1024 | 0.575 / 0.591 | 0.860 / 0.865 |
| 64:8:128:1024:1536 | 1.050 / 1.113 | 0.827 / 0.832 |
| 64:8:128:1024:2048 | 1.660 / 1.761 | 0.794 / 0.804 |
| 64:8:128:512:256 | 0.0782 / 0.0763 | 1.026 / 1.037 |
| 64:8:128:512:384 | 0.120 / 0.119 | 0.974 / 0.969 |
| 64:8:128:512:768 | 0.246 / 0.245 | 0.924 / 0.926 |
| 64:8:128:512:1536 | 0.650 / 0.674 | 0.885 / 0.886 |
| 64:8:128:2048:384 | 0.320 / 0.317 | 0.895 / 0.887 |
| 64:8:128:2048:768 | 0.706 / 0.732 | 0.840 / 0.852 |
| 64:8:128:4096:384 | 0.593 / 0.605 | 0.860 / 0.866 |
| 64:8:128:4096:768 | 1.318 / 1.383 | 0.820 / 0.836 |
| 64:8:128:4096:1536 | 3.419 / 3.722 | 0.773 / 0.786 |
| 64:8:128:16384:256 | 1.435 / 1.445 | 0.869 / 0.881 |
| 64:8:128:16384:512 | 3.034 / 3.134 | 0.824 / 0.848 |
| 64:8:128:16384:768 | 5.016 / 5.255 | 0.809 / 0.832 |
| 64:8:128:16384:1024 | 7.245 / 7.688 | 0.790 / 0.803 |
| 128:8:256:1024:384 | 0.288 / 0.286 | 0.943 / 0.948 |
| 128:8:256:1024:768 | 0.624 / 0.624 | 0.907 / 0.909 |
| 128:8:256:1024:1536 | 1.579 / 1.662 | 0.884 / 0.885 |
| 128:16:256:1024:384 | 0.401 / 0.400 | 0.933 / 0.935 |
| 128:16:256:1024:768 | 0.881 / 0.882 | 0.895 / 0.896 |
| 128:16:256:1024:1536 | 2.285 / 2.346 | 0.851 / 0.869 |
| 128:8:384:1024:768 | 0.711 / 0.707 | 0.919 / 0.921 |
| 64:8:128:1024:400 | 0.214 / 0.214 | 0.911 / 0.909 |
| 64:8:128:1024:784 | 0.415 / 0.424 | 0.905 / 0.914 |
| 64:8:128:1024:128 | 0.0693 / 0.0680 | 1.040 / 1.049 |
| 64:8:128:1024:192 | 0.0987 / 0.0978 | 0.998 / 0.997 |
| 64:8:128:4096:128 | 0.203 / 0.204 | 0.899 / 0.900 |
| 64:8:128:4096:256 | 0.386 / 0.385 | 0.894 / 0.894 |
| 64:8:128:16384:128 | 0.723 / 0.723 | 0.878 / 0.878 |
| 64:8:128:2048:256 | 0.211 / 0.211 | 0.915 / 0.916 |
| 64:8:128:512:512 | 0.154 / 0.154 | 0.943 / 0.941 |
| 64:8:128:256:384 | 0.0829 / 0.0811 | 1.008 / 1.014 |
| 64:8:128:256:768 | 0.171 / 0.171 | 0.972 / 0.971 |
| 64:8:128:256:1536 | 0.456 / 0.462 | 0.941 / 0.944 |
| 64:8:128:128:768 | 0.132 / 0.131 | 1.018 / 1.019 |
| 64:8:128:512:2048 | 1.040 / 1.080 | 0.857 / 0.864 |
| 64:8:128:4096:2048 | 5.278 / 5.570 | 0.737 / 0.758 |
| 128:8:256:512:768 | 0.388 / 0.387 | 0.952 / 0.952 |
| 128:16:256:512:768 | 0.525 / 0.523 | 0.914 / 0.914 |
| 128:16:256:4096:768 | 3.055 / 3.090 | 0.875 / 0.886 |
| 128:8:256:4096:768 | 2.022 / 2.078 | 0.880 / 0.892 |
| 128:16:256:1024:256 | 0.265 / 0.265 | 0.960 / 0.961 |
| 128:8:256:1024:256 | 0.191 / 0.190 | 0.962 / 0.964 |

Error against fp64 (`pwaw_err.py`, `upd rms / upd max / judge` in %):

| case (D:C:d_pair:S:N) | PyTorch bf16 | default 2.5.14 | FP8 |
|---|---|---|---|
| 64:8:128:1024:384 | 2.36 / 3.61 / 0.29 | 2.35 / 3.61 / 0.29 | 3.39 / 4.18 / 0.34 |
| 64:8:128:1024:768 | 2.62 / 4.66 / 0.28 | 2.62 / 4.66 / 0.28 | 3.26 / 5.65 / 0.34 |
| 64:8:128:1024:1536 | 2.81 / 5.00 / 0.28 | 2.80 / 4.93 / 0.28 | 3.17 / 5.43 / 0.31 |
| 64:8:128:4096:768 | 2.63 / 4.33 / 0.28 | 2.62 / 4.37 / 0.28 | 3.27 / 5.39 / 0.35 |
| 64:8:128:1024:256 | 2.17 / 3.39 / 0.29 | 2.16 / 3.36 / 0.29 | 3.46 / 4.39 / 0.38 |
| 64:8:128:512:2048 | 2.86 / 5.93 / 0.29 | 2.86 / 5.82 / 0.28 | 3.14 / 6.14 / 0.30 |
| 64:8:128:1024:400 | 2.38 / 3.66 / 0.30 | 2.37 / 3.66 / 0.30 | 3.38 / 4.27 / 0.35 |
| 128:8:256:1024:768 | 2.45 / 4.27 / 0.28 | 2.44 / 4.21 / 0.28 | 3.01 / 4.78 / 0.31 |
| 128:16:256:1024:768 | 2.64 / 4.52 / 0.28 | 2.63 / 4.52 / 0.28 | 3.22 / 5.24 / 0.32 |
| 128:8:384:1024:768 | 2.47 / 4.27 / 0.28 | 2.47 / 4.27 / 0.28 | 3.04 / 4.96 / 0.33 |
| 64:8:128:16384:512 | 2.48 / 3.58 / 0.30 | 2.47 / 3.42 / 0.28 | 3.33 / 4.18 / 0.35 |

#### Transition forward, base 2.5.14

`tr_f8bench.py` / `tr256_bench.py` speed (MIN_ROWS = 0, calib on two eager calls, lazy guard). Variant names: D128 `f1g1t1s4` full, `f1g0t1s4` G1 only, `f0g1t1s4` G2 only, `f0g0t0s2` bf16 twin; D256 `f1s4` full, `f0s4` h8; wide `f1` full, `f0` h8 (the docs' `w1` / `w0`: the wide G1 switch is `G1` in this tree).

D 128:

| D:n:rows | L | 2.5.14 ms | full | G1 only | G2 only | bf16 twin | control |
|---|---|---|---|---|---|---|---|
| 128:4:16384 | 128 | 0.0175 / 0.0170 | 0.661 / 0.664 | 0.740 / 0.737 | 0.767 / 0.769 | 0.879 / 0.879 | 0.991 / 0.995 |
| 128:4:65536 | 256 | 0.0489 / 0.0511 | 0.744 / 0.749 | 0.841 / 0.844 | 0.880 / 0.872 | 1.036 / 1.021 | 1.002 / 1.006 |
| 128:4:147456 | 384 | 0.107 / 0.113 | 0.719 / 0.735 | 0.829 / 0.840 | 0.850 / 0.858 | 1.046 / 1.066 | 1.000 / 0.999 |
| 128:4:262144 | 512 | 0.184 / 0.192 | 0.705 / 0.745 | 0.823 / 0.865 | 0.847 / 0.885 | 1.041 / 1.120 | 0.991 / 1.010 |
| 128:4:589824 | 768 | 0.388 / 0.423 | 0.698 / 0.726 | 0.818 / 0.840 | 0.850 / 0.855 | 1.038 / 1.067 | 0.999 / 1.001 |
| 128:4:1048576 | 1024 | 0.682 / 0.740 | 0.697 / 0.748 | 0.821 / 0.867 | 0.860 / 0.886 | 1.027 / 1.083 | 0.993 / 1.005 |
| 128:4:2359296 | 1536 | 1.475 / 1.640 | 0.711 / 0.746 | 0.838 / 0.877 | 0.885 / 0.900 | 1.055 / 1.112 | 1.001 / 0.991 |
| 128:4:4194304 | 2048 | 2.621 / 2.928 | 0.710 / 0.752 | 0.834 / 0.872 | 0.886 / 0.892 | 1.059 / 1.071 | 0.993 / 0.997 |
| 128:2:16384 | 128 | 0.0129 / 0.0127 | 0.725 / 0.719 | 0.724 / 0.715 | 0.768 / 0.765 | 0.829 / 0.825 | 0.992 / 0.990 |
| 128:2:65536 | 256 | 0.0333 / 0.0335 | 0.754 / 0.756 | 0.817 / 0.818 | 0.854 / 0.850 | 0.988 / 0.984 | 0.995 / 0.996 |
| 128:2:147456 | 384 | 0.0735 / 0.0751 | 0.701 / 0.700 | 0.767 / 0.766 | 0.803 / 0.806 | 0.990 / 0.985 | 1.005 / 1.004 |
| 128:2:262144 | 512 | 0.123 / 0.126 | 0.682 / 0.695 | 0.754 / 0.771 | 0.799 / 0.813 | 0.995 / 0.999 | 1.001 / 1.007 |
| 128:2:589824 | 768 | 0.252 / 0.274 | 0.680 / 0.690 | 0.762 / 0.759 | 0.800 / 0.798 | 0.991 / 0.976 | 0.991 / 0.969 |
| 128:2:1048576 | 1024 | 0.436 / 0.457 | 0.689 / 0.720 | 0.775 / 0.807 | 0.813 / 0.853 | 0.990 / 1.042 | 0.984 / 0.986 |
| 128:2:2359296 | 1536 | 0.959 / 1.030 | 0.683 / 0.729 | 0.774 / 0.811 | 0.815 / 0.839 | 1.012 / 1.017 | 0.986 / 0.988 |
| 128:2:4194304 | 2048 | 1.676 / 1.818 | 0.689 / 0.759 | 0.781 / 0.850 | 0.830 / 0.866 | 1.018 / 1.047 | 0.973 / 1.004 |

Peak |h sh| over the rows: max 27.8 of 448.

D 256:

| D:n:rows | L | 2.5.14 ms | full | h8 | control |
|---|---|---|---|---|---|
| 256:4:65536 | 256 | 0.178 / 0.196 | 0.719 / 0.744 | 0.911 / 0.909 | 0.997 / 0.994 |
| 256:4:147456 | 384 | 0.383 / 0.412 | 0.708 / 0.765 | 0.912 / 0.931 | 0.998 / 1.042 |
| 256:4:262144 | 512 | 0.669 / 0.767 | 0.699 / 0.683 | 0.913 / 0.867 | 0.994 / 0.958 |
| 256:4:589824 | 768 | 1.426 / 1.683 | 0.697 / 0.716 | 0.914 / 0.906 | 1.002 / 0.998 |
| 256:4:1048576 | 1024 | 2.569 / 2.846 | 0.690 / 0.720 | 0.909 / 0.913 | 0.988 / 0.994 |
| 256:4:2359296 | 1536 | 5.638 / 6.520 | 0.692 / 0.728 | 0.918 / 0.926 | 0.995 / 1.002 |
| 256:4:4194304 | 2048 | 10.00 / 12.03 | 0.691 / 0.721 | 0.919 / 0.924 | 0.998 / 1.002 |
| 256:2:65536 | 256 | 0.110 / 0.117 | 0.746 / 0.751 | 0.906 / 0.895 | 0.997 / 0.994 |
| 256:2:147456 | 384 | 0.228 / 0.239 | 0.749 / 0.785 | 0.917 / 0.941 | 0.998 / 1.016 |
| 256:2:262144 | 512 | 0.391 / 0.408 | 0.740 / 0.792 | 0.917 / 0.966 | 1.005 / 1.042 |
| 256:2:589824 | 768 | 0.827 / 0.968 | 0.739 / 0.753 | 0.922 / 0.928 | 0.995 / 1.003 |
| 256:2:1048576 | 1024 | 1.464 / 1.590 | 0.742 / 0.795 | 0.922 / 0.981 | 1.005 / 1.035 |
| 256:2:2359296 | 1536 | 3.218 / 3.742 | 0.744 / 0.747 | 0.923 / 0.923 | 0.999 / 0.997 |
| 256:2:4194304 | 2048 | 5.685 / 6.526 | 0.746 / 0.749 | 0.929 / 0.924 | 1.000 / 0.998 |

Peak |h sh| over the rows: max 22.3 of 448.

D 384 / 512 (wide chain):

| D:n:rows | L | 2.5.14 ms | full | h8 | control |
|---|---|---|---|---|---|
| 384:4:65536 | 256 | 0.404 / 0.445 | 0.645 / 0.662 | 0.856 / 0.874 | 0.996 / 1.003 |
| 384:4:262144 | 512 | 1.511 / 1.761 | 0.644 / 0.647 | 0.868 / 0.870 | 0.999 / 0.988 |
| 384:4:1048576 | 1024 | 5.853 / 6.542 | 0.642 / 0.669 | 0.871 / 0.905 | 0.996 / 1.006 |
| 384:4:4194304 | 2048 | 23.70 / 26.51 | 0.629 / 0.657 | 0.883 / 0.883 | 0.996 / 0.998 |
| 512:4:65536 | 256 | 0.654 / 0.708 | 0.586 / 0.598 | 0.807 / 0.823 | 0.999 / 1.006 |
| 512:4:262144 | 512 | 2.558 / 2.874 | 0.570 / 0.580 | 0.797 / 0.811 | 0.998 / 0.997 |
| 512:4:589824 | 768 | 5.621 / 6.625 | 0.567 / 0.562 | 0.790 / 0.785 | 0.997 / 0.990 |
| 384:2:65536 | 256 | 0.249 / 0.267 | 0.714 / 0.703 | 0.876 / 0.860 | 0.998 / 0.983 |
| 384:2:262144 | 512 | 0.897 / 0.969 | 0.717 / 0.747 | 0.904 / 0.919 | 0.997 / 1.013 |
| 384:2:1048576 | 1024 | 3.435 / 3.840 | 0.718 / 0.723 | 0.903 / 0.916 | 0.997 / 0.987 |
| 384:2:4194304 | 2048 | 13.52 / 15.05 | 0.713 / 0.747 | 0.904 / 0.949 | 1.002 / 1.012 |
| 512:2:65536 | 256 | 0.369 / 0.399 | 0.643 / 0.640 | 0.843 / 0.830 | 0.998 / 0.996 |
| 512:2:262144 | 512 | 1.404 / 1.546 | 0.620 / 0.631 | 0.835 / 0.845 | 0.994 / 0.998 |
| 512:2:589824 | 768 | 3.050 / 3.517 | 0.622 / 0.622 | 0.837 / 0.831 | 0.999 / 0.992 |

Peak |h sh| over the rows: max 19.4 of 448.

Error against fp64 (`... err`; judge = max|y - y64| / max|y64|, rms, update rms; ranges over the cases):

| route | judge | rms | update rms | cases |
|---|---|---|---|---|
| D128 PyTorch bf16 | 0.0057-0.0068 | 0.0036-0.0038 | 0.0070-0.0072 | 128:4:16384 128:2:16384 128:4:262144 128:2:262144 128:4:1048576 128:2:1048576 |
| D128 2.5.14 default | 0.0037-0.0042 | 0.0027-0.0028 | 0.0052-0.0053 | 128:4:16384 128:2:16384 128:4:262144 128:2:262144 128:4:1048576 128:2:1048576 |
| D128 full | 0.0410-0.0446 | 0.0341-0.0353 | 0.0661-0.0663 | 128:4:16384 128:2:16384 128:4:262144 128:2:262144 128:4:1048576 128:2:1048576 |
| D128 G1 only | 0.0331-0.0447 | 0.0281-0.0292 | 0.0546-0.0546 | 128:4:16384 128:2:16384 128:4:262144 128:2:262144 128:4:1048576 128:2:1048576 |
| D128 G2 only | 0.0263-0.0368 | 0.0195-0.0202 | 0.0378-0.0379 | 128:4:16384 128:2:16384 128:4:262144 128:2:262144 128:4:1048576 128:2:1048576 |
| D128 guard stats (last) | {'calls': 54, 'retries': 0, 'fallbacks': 0, 'capture_fallbacks': 0, 'calibrations': 18} | | | |
| D256 PyTorch bf16 | 0.0044-0.0056 | 0.0036-0.0037 | 0.0070-0.0071 | 256:4:16384 256:2:16384 256:4:128 256:4:38400 256:2:38400 256:4:65536 256:2:65536 256:4:262144 256:2:262144 |
| D256 2.5.14 default | 0.0035-0.0040 | 0.0027-0.0028 | 0.0052-0.0052 | 256:4:16384 256:2:16384 256:4:128 256:4:38400 256:2:38400 256:4:65536 256:2:65536 256:4:262144 256:2:262144 |
| D256 full | 0.0347-0.0403 | 0.0340-0.0350 | 0.0659-0.0660 | 256:4:16384 256:2:16384 256:4:128 256:4:38400 256:2:38400 256:4:65536 256:2:65536 256:4:262144 256:2:262144 |
| D256 h8 | 0.0214-0.0295 | 0.0194-0.0201 | 0.0378-0.0378 | 256:4:16384 256:2:16384 256:4:128 256:4:38400 256:2:38400 256:4:65536 256:2:65536 256:4:262144 256:2:262144 |
| D256 guard stats (last) | {'calls': 54, 'retries': 0, 'fallbacks': 0, 'capture_fallbacks': 0, 'calibrations': 18} | | | |
| D384/512 PyTorch bf16 | 0.0050-0.0057 | 0.0037-0.0037 | 0.0070-0.0071 | 384:4:65536 384:2:65536 512:4:65536 512:2:65536 384:4:262144 512:2:262144 |
| D384/512 2.5.14 default | 0.0035-0.0039 | 0.0027-0.0027 | 0.0052-0.0052 | 384:4:65536 384:2:65536 512:4:65536 512:2:65536 384:4:262144 512:2:262144 |
| D384/512 full | 0.0346-0.0416 | 0.0339-0.0343 | 0.0657-0.0658 | 384:4:65536 384:2:65536 512:4:65536 512:2:65536 384:4:262144 512:2:262144 |
| D384/512 h8 | 0.0203-0.0285 | 0.0195-0.0197 | 0.0377-0.0378 | 384:4:65536 384:2:65536 512:4:65536 512:2:65536 384:4:262144 512:2:262144 |
| D384/512 guard stats (last) | {'calls': 36, 'retries': 0, 'fallbacks': 0, 'capture_fallbacks': 0, 'calibrations': 12} | | | |

### Token DiT / atom DiT FP8 routes on the 3.0.0 release tree — base 2.5.14

Tree: `wt-300` (branch rel/3.0.0, HEAD 77cd55eb0 = merge of fp8/3.0.0-all onto 2.5.14 9dd18e507, plus the uncommitted working-tree
changes present at run time: `fp8.py` one switch etc., 17 paths). `miniworld_engine` pinned to `wt-300/src` (`scripts/wtpin.py`), every
`MINIWORLD_*FP8*` env variable unset. H100 node02, 2026-11-17, jobs 57643 (token DiT) and 57654 (atom DiT).
Scripts: copies of `fp8x-tokendit/scripts` (speed.py, err.py, tdcommon.py) and `fp8x-atomdit/scripts` (speed.py, err.py, common.py,
tab_*.py) with only the tree pin and the output paths changed; drivers `scripts/run_tok.sh`, `scripts/run_atom.sh`. Raw rows in `res/`.

#### Token DiT forward, e4m3 GEMM operands — module time against the 2.5.14 default

Protocol and shapes of docs/fp8.md (`speed2`): one block call per CUDA graph, same process, interleaved (default, default again =
same-cubin control, each placement); cool = 2 s idle + ~10 ms burst, 3 reps, median; sustained = 2 s idle, 1 s back-to-back, 3 blocks,
mean; master weights, shared conditioning, all-ones mask, every row gate removed (`MIN_ROWS = 0`). Cells `cool / sustained`, ratio to the
default. Raw: `res/tok_speed2.jsonl`.

| A:L (M) | 2.5.14 default ms | same-cubin control | qkvg | expand | qkvg + expand | all four |
|---|---|---|---|---|---|---|
| 5:256 (1280) | 0.0740 / 0.0719 | 0.999 / 0.999 | 0.969 / 0.969 | 0.975 / 0.976 | 0.938 / 0.939 | 0.995 / 1.003 |
| 5:384 (1920) | 0.0956 / 0.0916 | 1.002 / 1.001 | 1.011 / 1.019 | 0.964 / 0.966 | 0.973 / 0.976 | 1.035 / 1.025 |
| 5:512 (2560) | 0.1263 / 0.1246 | 0.999 / 1.004 | 0.982 / 0.992 | 0.958 / 0.975 | 0.941 / 0.959 | 0.977 / 0.996 |
| 5:768 (3840) | 0.1811 / 0.1837 | 0.999 / 1.009 | 0.978 / 0.988 | 0.957 / 0.968 | 0.932 / 0.940 | 0.978 / 0.985 |
| 5:1024 (5120) | 0.2437 / 0.2549 | 0.994 / 0.999 | 0.968 / 0.969 | 0.953 / 0.940 | 0.929 / 0.911 | 0.950 / 0.929 |
| 5:1536 (7680) | 0.3987 / 0.4215 | 0.997 / 0.996 | 0.972 / 0.970 | 0.962 / 0.964 | 0.937 / 0.936 | 0.976 / 0.971 |
| 5:2048 (10240) | 0.5703 / 0.6087 | 0.994 / 1.004 | 0.967 / 0.972 | 0.963 / 0.963 | 0.942 / 0.932 | 0.989 / 0.967 |
| 1:768 (768) | 0.0745 / 0.0728 | 0.995 / 0.998 | 1.015 / 1.019 | 0.972 / 0.976 | 0.987 / 0.989 | 1.057 / 1.056 |
| 1:1536 (1536) | 0.1336 / 0.1329 | 0.995 / 0.993 | 1.006 / 1.008 | 0.962 / 0.959 | 0.975 / 0.970 | 1.007 / 1.000 |

##### Error against fp64 PyTorch (docs' err.py protocol and cases; raw `res/tok_err.jsonl`)

Cells: update rms / update max (judge = max|y - y64| / max|y64|); tolerance max(PyTorch bf16 judge, 8e-3); every gate removed; all finite.

| case | default 2.5.14 | PyTorch bf16 | qkvg | out | expand | squeeze | qkvg + expand | qkvg + expand + squeeze | all four |
|---|---|---|---|---|---|---|---|---|---|
| A5 L384 | 0.656 / 0.646 % (3.6e-03) | 0.943 / 1.07 % (6.0e-03) | 1.25 / 1.16 % (6.4e-03) | 0.855 / 0.796 % (4.4e-03) | 5.35 / 5.64 % (3.1e-02) | 3.71 / 4.17 % (2.3e-02) | 5.45 / 5.19 % (2.9e-02) | 6.56 / 7.13 % (4.0e-02) | 6.58 / 7.83 % (4.4e-02) |
| A5 L768 | 0.655 / 0.667 % (4.1e-03) | 0.939 / 1.05 % (6.4e-03) | 1.07 / 1.1 % (6.7e-03) | 0.788 / 0.766 % (4.7e-03) | 5.36 / 5.61 % (3.4e-02) | 3.73 / 3.86 % (2.4e-02) | 5.43 / 5.36 % (3.3e-02) | 6.54 / 6.57 % (4.0e-02) | 6.56 / 7.2 % (4.4e-02) |
| A5 L1536 | 0.655 / 0.733 % (3.7e-03) | 0.937 / 1.14 % (5.8e-03) | 0.941 / 1.04 % (5.3e-03) | 0.744 / 0.761 % (3.9e-03) | 5.39 / 5.51 % (2.8e-02) | 3.73 / 4.19 % (2.1e-02) | 5.44 / 5.74 % (2.9e-02) | 6.56 / 6.62 % (3.4e-02) | 6.57 / 6.39 % (3.2e-02) |
| A1 L768 | 0.655 / 0.58 % (4.0e-03) | 0.938 / 0.973 % (6.8e-03) | 1.06 / 1 % (7.0e-03) | 0.786 / 0.656 % (4.6e-03) | 5.38 / 4.57 % (3.2e-02) | 3.73 / 3.36 % (2.3e-02) | 5.44 / 4.57 % (3.2e-02) | 6.56 / 5.63 % (3.9e-02) | 6.57 / 5.85 % (4.1e-02) |
| A5 L768, input x 0.05 | 0.64 / 0.69 % (7.0e-03) | 0.966 / 1.13 % (1.2e-02) | 3.54 / 3.86 % (3.9e-02) | 1.91 / 2.04 % (2.1e-02) | 5.36 / 6.03 % (6.1e-02) | 3.72 / 4.74 % (4.8e-02) | 6.39 / 6.42 % (6.5e-02) | 7.36 / 7.33 % (7.5e-02) | 7.58 / 8.33 % (8.5e-02) |
| A5 L768, input x 20 | 7.71 / 7.65 % (2.5e-03) | 9.69 / 13.7 % (4.5e-03) | 7.76 / 7.87 % (2.6e-03) | 7.72 / 7.73 % (2.5e-03) | 9.37 / 10.7 % (3.5e-03) | 8.54 / 8.94 % (2.9e-03) | 9.41 / 10.7 % (3.5e-03) | 10.1 / 10.7 % (3.5e-03) | 10.1 / 10.7 % (3.5e-03) |

#### Atom DiT forward, e4m3 row-kernel GEMMs — module time against the 2.5.14 default

Protocol of docs/fp8.md (`speed2`): registry cases (`master`, eval, A = 5, `atom_single_shared_cond`), CUDA graphs of one call captured under
`static_weights()` + `static_inputs()`, same process: default, default again (same-cubin control), FP8 pre + post, pre only, post only; cool /
sustained as above. Cells `cool / sustained`, ratio to the default. Raw: `res/atom_speed2.jsonl`. (Per-sample rows of the docs not re-run:
the route is gated to a shared conditioning, those cases run the default.)

| case | 2.5.14 default us (cool / sus) | FP8 pre+post | FP8 pre only | FP8 post only | same-cubin control |
|---|---|---|---|---|---|
| AtomDiTOpenFold3:shared:1024:5 | 50.3 / 48.8 | 0.920 / 0.918 | 0.943 / 0.938 | 0.977 / 0.974 | 0.998 / 1.000 |
| AtomDiTOpenFold3:shared:2048:5 | 65.3 / 63.3 | 0.943 / 0.948 | 0.951 / 0.955 | 0.957 / 0.959 | 0.975 / 0.975 |
| AtomDiTOpenFold3:shared:4096:5 | 102.2 / 99.5 | 0.909 / 0.913 | 0.982 / 0.986 | 0.928 / 0.932 | 0.997 / 1.001 |
| AtomDiTOpenFold3:shared:8192:5 | 166.1 / 162.9 | 0.968 / 0.974 | 0.989 / 0.993 | 0.971 / 0.975 | 0.994 / 0.997 |
| AtomDiTOpenFold3:shared:12288:5 | 232.8 / 231.9 | 0.962 / 0.956 | 0.979 / 0.967 | 0.989 / 0.975 | 1.003 / 1.002 |
| AtomDiTOpenFold3:shared:16384:5 | 298.3 / 294.2 | 0.953 / 0.956 | 0.966 / 0.970 | 0.985 / 0.990 | 1.000 / 1.003 |
| AtomDiTProtenix:shared:4096:5 | 82.8 / 80.7 | 0.895 / 0.904 | 0.983 / 0.987 | 0.911 / 0.913 | 0.996 / 0.998 |
| AtomDiTProtenix:shared:16384:5 | 275.8 / 272.4 | 0.954 / 0.956 | 0.970 / 0.973 | 0.983 / 0.983 | 0.999 / 1.001 |
| AtomDiTChai1:shared:4096:5:slide | 91.9 / 87.6 | 0.947 / 0.953 | 0.992 / 0.995 | 0.952 / 0.958 | 0.994 / 0.997 |

##### Error against fp64 (docs' err.py, cases of err3; raw `res/atom_err3.jsonl`)

judge = max|y - y64| / max|y64|, tol = max(err_pytorch, 8e-3), rms_rel = ||y - y64|| / ||y64||.

| case | tol | PyTorch master judge | bf16 cuda judge / rms_rel | FP8 pre+post judge / rms_rel / max abs | FP8 pre only judge / rms_rel | FP8 post only judge / rms_rel | finite |
|---|---|---|---|---|---|---|---|
| AtomDiTOpenFold3:shared:1024:5 | 8.0e-03 | 5.79e-03 | 5.79e-03 / 3.85e-03 | 5.73e-02 / 2.87e-02 / 0.364 | 1.25e-02 / 7.86e-03 | 5.24e-02 / 2.78e-02 | True |
| AtomDiTOpenFold3:shared:4096:5 | 8.0e-03 | 7.66e-03 | 6.37e-03 / 3.84e-03 | 4.83e-02 / 2.86e-02 / 0.282 | 2.14e-02 / 7.99e-03 | 4.60e-02 / 2.77e-02 | True |
| AtomDiTOpenFold3:shared:16384:5 | 8.0e-03 | 7.23e-03 | 7.50e-03 / 3.85e-03 | 5.51e-02 / 2.86e-02 / 0.357 | 1.65e-02 / 7.86e-03 | 5.35e-02 / 2.78e-02 | True |
| AtomDiTProtenix:shared:4096:5 | 8.0e-03 | 7.74e-03 | 6.90e-03 / 3.85e-03 | 4.78e-02 / 2.86e-02 / 0.273 | 1.65e-02 / 8.10e-03 | 5.05e-02 / 2.77e-02 | True |
| AtomDiTChai1:shared:4096:5:slide | 9.6e-03 | 9.63e-03 | 9.07e-03 / 6.08e-03 | 9.77e-02 / 5.64e-02 / 1.200 | 9.71e-02 / 3.53e-02 | 6.80e-02 / 4.45e-02 | True |
| AtomDiTOpenFold3:shared:4096:5:slide_qblock | 8.0e-03 | 7.66e-03 | 6.37e-03 / 3.84e-03 | 4.83e-02 / 2.86e-02 / 0.282 | 2.14e-02 / 7.99e-03 | 4.60e-02 / 2.77e-02 | True |

#### Against the 2.5.12-base numbers in docs/fp8.md

- Error: every number is byte-identical to the 2.5.12-base runs (`fp8x-atomdit/res/err3.jsonl` equal as a file; token DiT `err1` + `err2`
  equal field for field; `qkvg + expand + squeeze` is new here, the docs did not tabulate it). Expected: default bit-identical, FP8 kernels unchanged.
- Time: 2.5.14 default times within -1.3 .. +1.2 % of the 2.5.12 defaults, except token DiT 5:384 sustained (0.0940 -> 0.0916 ms, -2.6 %)
  and Chai-1 4096 sustained (89.6 -> 87.6 us, -2.2 %) -- run-to-run of a sustained block, the cool cells agree within 1.2 %. Every FP8 ratio
  within +-0.02 of the docs except token DiT 5:1024 all four sustained (0.960 -> 0.929); next largest: 5:1024 qkvg + expand sustained
  0.927 -> 0.911, 5:512 qkvg + expand sustained 0.948 -> 0.959, atom DiT 12288 pre only sustained 0.977 -> 0.967. No win turned into a
  loss or back (token 5:256 all four 1.006 -> 0.995 cool is a tie either way).

#### Gates against the data

- Token DiT `MIN_ROWS_PER`: expand 640 — the e4m3 expand wins at every M measured (768 .. 10240, 0.94-0.98): matches. qkvg 2560 — qkvg
  loses at M 768 / 1536 / 1920 (1.006-1.019) and wins from 2560 (0.98-0.99 at 2560, 0.97 above): matches (5:256 = 1280 rows wins 0.969,
  the same non-monotonic point the docs had; not worth a gate change). out / squeeze 2560 bound a loss only: `all four` minus
  `qkvg + expand` is +2-7 % everywhere, so out / squeeze stay off.
- Atom DiT: the shared-conditioning gate holds; every shared case wins with pre + post (0.895-0.968), pre only (0.943-0.992), post only
  (0.911-0.989); no case needs a new gate.

## Release line 3.0.0: verification (branch `rel/3.0.0`, 2026-11-17)

- Merge: 2.5.14 (9dd18e507) + `fp8/3.0.0-all` (e5ce31d0c). One conflict (`h100_b200style._contract`: 2.5.13's `table` argument and FP8's
  `_contract_bf16` split; the table is passed through, also on the FP8 route's bf16 fallback); `triattn_h16_sm90.cu` merged cleanly (TA_RC
  next to TA_F8PV / TA_F8QK). No `version` / `backend_versions` change: judge owed 0 (accuracy and speed) on h100_sm90 / a100_sm80 / 5090_sm120.
- Default route against the 2.5.14 tree, separate processes, sha1 of every output (training: + input / parameter gradients): **64 / 64**
  (`~/miniworld-engine-scratch/rel300/scripts/bit.py`, raw `res/bit_*.jsonl`; the 44 cases of the fp8safe2 merge check + TriAttn AF3 64:4:16 /
  128:4:32 / 512:8:64, 64:2:32, 256:8:32, 384:12:32, AF3 training, TriMul D64-D384 incl. training, direct TriMul at L300 / 1104 with and without a
  mask), on the merged tree and again on the final tree.
- AF3 end to end (FoldForge 399d0b4, `~/miniworld-engine-scratch/rel300/scripts/fb_run.py`): every switch off, 8gzl + 7sxp: **10 / 10 CIFs
  byte-identical to 2.5.14**. `MINIWORLD_FP8=safe` and nothing else, 8gzl then 8e4h in one process (guard: `fp8.check()` after every forward):
  every route fired on 8e4h (TriAttn PV 208 / q k 592 calls, TriMul 704, OPM 24, PWA 24, Transition 356, token DiT 9600, atom DiT 4800), 0 redos,
  peaks <= 0.28 of 448, and **10 / 10 CIFs byte-identical to the fp8safe2 FoldBench run** (2.5.12 base, per-route switches, same calibration
  order): the `safe` preset is that run's route set, and the FoldBench result carries over to the 2.5.14 base.
- GPU tests (one 3-GPU queue job): the 13 FP8 files 210 passed; default-route files the merge touches (TriAttn h16 core / tail gate / key tiles /
  fronts / variants / bidirectional, TriMul LT_UNI tables / K3 pipe / workspace) 471 passed, 24 skipped.
- CPU: `test_fp8_switch` (new), `test_capture_api`, `test_init`, `test_imports`, `test_supports_signature`, `test_prebuild_plan`, `test_judge`,
  `test_requirements`, `test_modules_cpu`: 3218 passed, 2 skipped. `ruff check` / `pyright` clean on every changed `src` / `tests` file.
- Prebuild (`_prebuild --check --list`): sm90a 513 -> 560 artifacts (29 new names + 18 TriMul FP8 cubins under existing names; 101 default
  artifacts re-key through FP8 `#if` text in shared sources, bitwise unchanged; no name dropped), every one built (0 failed); sm80 72 / sm120 64:
  same names and keys as 2.5.14; no new `--check` finding (the FP8 loaders and sources are now reached / compiled).
- After decisions 1-4 (TriAttn only for a model in `fp8.TRIATTN_MODELS`; harnesses out): default route 64 / 64 bit-identical to 2.5.14 again
  (`res/bit_rc3.jsonl`). Module level (`rel300/scripts/tri_safe.py`, raw `res/tri_safe.jsonl`; TriAttn AF3 128:4:32 L768, AF3 64:4:16 L512,
  128:4:32 L384, 384:12:32 L1024): `MINIWORLD_FP8=safe` and `fp8.enable("safe")` give the FP8-off bytes with 0 FP8 core calls; with
  `MINIWORLD_FP8_MODEL=af3` / `enable(model="af3")` the FP8 core runs (1-2 calls per case, other bytes). AF3 end to end (`fb_run.py` arms
  `engine_safe_env_af3`, `engine_safe_env_nomodel`): with af3 **10 / 10 CIFs byte-identical** to the `MINIWORLD_FP8=safe` run before the decision
  (same route calls); without a model TriAttn PV / q k 0 calls, TriMul 704, OPM 24, PWA 24, Transition 356, token DiT 9600, atom DiT 4800 (8e4h
  CIFs differ, 8gzl's equal: no TriAttn FP8 site fires on 28 tokens). GPU: the 13 FP8 files + `test_fp8_switch` 222 passed (one 3-GPU queue
  job). `_prebuild --check --list` sm90a / sm80 / sm120: identical to the release-tree lists (560 / 72 / 64).

## TriMul forward: e4m3 planes (`h100_b200style_fp8`, details in `docs/h100/trimul.md`)

Scope: bf16 inference, D64 / D128 / D256 / D384 (hidden = pair), one direction or the bidirectional block (`BidirectionalTriangleMultiplication`, from 2026-11-16, branch `fp8/bidir-tm`), 1-16 samples per call.
D64 / D384 and B > 1 are from the 3.0.0 exploration (branch `fp8x/trimul`, see "TriMul: where else e4m3 can go"); D512 and training stay bf16.

Enable (environment, read at import, set before `import miniworld_engine`):

| env | effect |
|---|---|
| `MINIWORLD_TRIMUL_FP8=calib` | e4m3 a / b planes between K1 and the contraction (cuBLASLt e4m3 batched GEMM, bf16 out), static per-module scales calibrated on the first eager call (16x headroom) |
| `MINIWORLD_TRIMUL_FP8_GUARD=lazy` | no host sync (required with CUDA graphs); `sync` (default) redoes an overflowing call itself; `off` |
| `MINIWORLD_TRIMUL_FP8_FASTACC=1` | cuBLASLt fast accumulation (1.07-1.08x of the contraction at L >= 1024) |
| `MINIWORLD_TRIMUL_FP8_HEADROOM`, `_SCALE` | scale headroom (16) / `tensor` (default) or `channel` scales |
| `MINIWORLD_TRIMUL_FP8_K1=1` | EXPERIMENTAL e4m3 K1 compute (6.9 % update error): never a default |
| `MINIWORLD_TRIMUL_FP8_MINL=<n>` | replaces the per-width / direction min-L gates (below them the bf16 default route runs): D128 out 144, D128 in 192, D256 out 224, D256 in 144, bidirectional D128 256, bidirectional D256 1024; D64 512 (both directions), D384 128, bidirectional D64 2048, bidirectional D384 1024; 0 = no gate |

The K3 pipeline flags are the 2.5.9 bf16 ones (`h100_b200style.K3_EXTRA`); the FP8 route uses that single hook (no private copy).
The experiments `K3_MC` (2-CTA multicast, D256), the K1 `WTMA` / `PIPE` variants and the TriAttn core macros `TA_F8SW` / `SWN` / `QM` / `K` / `TAU10` were removed from the code (no gain): see "Tried, no gain".

### Guard contract (for FoldForge)

K1 reports the max scaled plane value; the e4m3 cast saturates above 448, and unguarded saturation gave 35 % update error (guarded 3.8 %, bf16 floor 0.6 %).
With `GUARD=lazy` the caller must, once per forward and outside any graph:

1. `peak = h100_b200style_fp8.overflow_peak()` (one host sync, device max since the last read);
2. if `peak > 448`: `h100_b200style_fp8.rescale_all(peak)` (in place, captured graphs follow) and redo the forward (RNG restored; at most 3 times).

The patch that wires this into FoldForge is a file only, not applied: `~/miniworld-engine-scratch/fp8/trimul/foldforge_patch/trimul-fp8-guard.patch` (README beside it).
A capture with no calibration takes the bf16 route (the first two eager warm-up calls of FoldForge's `_Graph` calibrate).

## TriMul: where else e4m3 can go (3.0.0 exploration, branch `fp8x/trimul`, 2026-11-17, base `fp8/3.0.0-on-2512` 91a9241fb)

Question: the kernels stay as they are; which further operands / intermediates of the TriMul forward can be e4m3, and what does each pay? Raw rows:
`~/miniworld-engine-scratch/fp8x-trimul/res/` (`stg1.jsonl` stage times, `spd1.jsonl` / `spd2.jsonl` / `spd_tri8.jsonl` module times, `err2_x.jsonl` / `err3_*.jsonl`
errors, `tri8ub.jsonl` upper bounds, `emul.jsonl` accuracy map), scripts beside them (`scripts/`).

### Stages of the 2.5.12 default and of the FP8-plane route (one direction, outgoing, B = 1)

Per-kernel device time from the profiler on CUDA-graph replay (3 replays; the kernel sum equals the graph replay time within 1-2 %), ms. Floors per
pair token (M = L^2, h = D): K1 reads z (2 D B) and writes the planes (4 D B bf16, 2 D B e4m3) and LN_in statistics (8 B), 8 D^2 FLOPs; the
contraction 2 D L FLOPs (reads the planes, writes t 2 D B); K3 reads t (2 D B) and z (2 D B), writes y (2 D B), 4 D^2 FLOPs. %SOL = max(byte floor at
3.05 TB/s, tensor floor at 748 TF bf16 / 1431 TF e4m3, the cuBLAS-measured peaks) / measured.

| D | L | default K1 / contraction / K3 (graph) | %SOL K1 / contr / K3 | FP8 planes K1 / contraction / K3 (graph) | FP8-compute K1 (`MINIWORLD_TRIMUL_FP8_K1`) |
|---|---|---|---|---|---|
| 64 | 1024 | 0.169 / 0.196 / 0.146 (0.515) | 80 / 94 / 92 | 0.166 / 0.124 / 0.147 (0.444) | - |
| 64 | 2048 | 0.666 / 1.407 / 0.559 (2.69) | 81 / 105 / 96 | 0.625 / 0.833 / 0.569 (2.13) | - |
| 128 | 1024 | 0.325 / 0.386 / 0.277 (1.035) | 82 / 95 / 96 | 0.278 / 0.246 / 0.280 (0.841) | K1 0.250 (module 0.805) |
| 128 | 2048 | 1.378 / 2.826 / 1.107 (5.28) | 77 / 104 / 96 | 1.081 / 1.682 / 1.090 (3.90) | K1 0.947 (module 3.91) |
| 256 | 1024 | 0.841 / 0.725 / 0.694 (2.41) | 87 / 101 / 76 | 0.877 / 0.494 / 0.695 (2.20) | K1 0.656 (module 1.876) |
| 256 | 2048 | 3.81 / 6.36 / 3.13 (12.19) | 77 / 92 / 68 | 4.08 / 3.64 / 3.16 (9.79) | K1 2.62 (module 8.75) |
| 384 | 1024 | 2.08 / 1.14 / 1.33 (4.60) | 79 / 97 / 62 | 2.08 / 0.750 / 1.35 (4.37) | - |
| 384 | 2048 | 9.36 / 9.17 / 5.71 (22.49) | 71 / 96 / 58 | 9.24 / 5.15 / 5.66 (18.18) | - |
| 512 | 1024 | 3.93 / 1.53 / 2.28 (7.75) | 75 / 96 / 64 | (not served) | - |
| 512 | 2048 | 17.54 / 11.78 / 9.98 (37.59) | 67 / 100 / 59 | (not served) | - |

(%SOL above 100: the bf16 contraction at D64 / D128 L2048 runs above the 748 TF cuBLAS-measured reference.) Bound, from the floors: K1 is byte-bound at
D64 / D128 (byte floor 2-2.6x its tensor floor) and tensor-bound from D256 (D256 tensor 2.94 ms vs bytes 2.12 ms at L2048; D384 6.6 vs 3.2); K3 is
byte-bound at D64 / D128 (96 % of the byte floor) and at D256-D512 runs at 58-76 % of max(byte floor, tensor floor) with neither floor reached (no removal
ablation of K3's internals was run, so no bound is claimed there); the contraction is tensor-bound everywhere (92-105 % of 748 TF).

### Where e4m3 can go: placements, share, upper bound, result

| stage | e4m3 operand / tensor | share of the default module (L2048, D64 / 128 / 256 / 384 / 512) | upper bound | result |
|---|---|---|---|---|
| contraction | a, b (the planes, K1 output) | 52 / 54 / 52 / 41 / 31 % | e4m3 tensor rate 1.91x: -25 / -26 / -25 / -20 / -15 % of the module | shipped opt-in at D128 / D256; **this branch: D64 and D384** (n64 K1 blocks), bidirectional D64 / D384, B > 1 (tables below). D512: not served (its K1 is the packed `_front`, its K3 `k3reorder.cu` has no `K3_CSCALE`; an n128 F8 K1 does not fit shared memory at D384 / D512) |
| K1 store | the planes as e4m3 bytes (comes with the row above) | K1 25 / 26 / 31 / 42 / 47 % | byte-bound K1 only (D64 / D128): planes are 2/3 of K1's bytes, e4m3 halves them: K1 -33 % = -8 % of the module | measured K1 -6 % (D64 L2048) / -22 % (D128 L2048); D256 / D384 K1 are tensor-bound: no change (D256 +7 %, the n128 F8 tile) |
| K1 compute | LN_in(z) and W_ag, W_a, W_bg, W_b (wgmma e4m3) | K1 tensor-bound from D256: 31 / 42 / 47 % | e4m3 MMA rate: K1 up to -50 % at D256+ | the existing experiment (`TMN_K1_F8C`): K1 -12 % (D128) / -36 % (D256 L2048), module a further -11 % at D256 L2048 (8.75 vs 9.79 ms); error 6.9 % of the update (failed the Protenix pLDDT criterion before). Accuracy map below: the projection operands are the expensive part |
| contraction output | t as e4m3 (cuBLASLt e4m3 D with D scale + amax) | contraction write 2 D B per token | (a) contraction: none measurable: bf16-out 0.254 / 1.701 ms vs e4m3-out 0.259 / 1.704 ms (D128 L1024 / L2048), the GEMM is tensor-bound; (b) K3 reads half the t bytes: bounded by K3 at h / 2 hidden channels (half the t bytes AND half its projection / LN work): -0.044 / -0.14 ms (D128 L1024 / 2048), -0.12 / -0.62 ms (D256) = -3.6 .. -6.3 % of the FP8 module | **implemented and measured, no gain, removed from the code**: K3 widening the e4m3 chunk in shared memory (in place, two warpgroup barriers per chunk) + the guard (two small kernels): module / default 0.736-0.940 against the FP8 planes' 0.739-0.926 cool, 0.780-1.019 against 0.744-0.988 sustained (slower in 15 of 16 cells: D64-D384, L512-2048); update error 3.7 -> 4.4-4.6 % (patch: `res/tri8_experiment.patch`) |
| K3 projection | LN_out(t) and W_o | K3 21 / 21 / 26 / 25 / 27 % | K3 at D64 / D128 is byte-bound: 0; at D256+ the tensor floor (1.5 / 3.3 / 5.9 ms at L2048) is below the measured K3 (3.1 / 5.7 / 10.0): at most the MMA share of an unbounded K3 | not implemented: e4m3 wgmma takes K-major operands only, t is channel-major (MN-major A): a transpose in shared memory per chunk; accuracy map below |
| K3 gate | z (raw, LN folded) and W_g | inside K3 | as above | not implemented: the folded LN_in (`rs (z . Wgf - mu ug) + vg`) cancels mu against an e4m3-rounded z . Wgf |
| training | forward planes / backward gradients | - | - | not re-opened (no new idea; e5m2 no-go and e4m3 backward no gain before) |

### Accuracy map (float64 emulation, e4m3 rounding inserted per placement)

`scripts/emul.py`: the module in float64 (the PyTorch statement, checked against the module's own float64 output to 6e-15), with e4m3 rounding
(power-of-two per-tensor scale, saturating; planes 16x headroom, t 4x, weights and LayerNorm outputs at their own amax) inserted at the named
operands; random weights as in the error tables, 3 masked tokens; update error rms (y - y64) / rms (y64 - z) and the engine form max |y - y64| / max |y64|.
The rows are the same to 0.1 % at D64 L384, D128 L256 (both directions), D256 L256, D384 L192; D128 outgoing shown, the range over the five cases in brackets.
The emulation reproduces the kernels: planes 3.7 % (kernel route 3.6-3.7 %), planes + t 4.5 % (the built e4m3-t route 4.4-4.6 %), planes + all K1
projection operands 6.8 % (the FP8-compute K1, 6.9 %). The bf16 default route is 0.63-0.66 %.

| e4m3 at | update rms | max / max |y64| | alone (without the planes): update rms |
|---|---|---|---|
| planes a, b (shipped opt-in) | 3.72 % (3.63-3.72) | 3.0 % (2.6-3.3) | - |
| planes + K1 gate projections (LN_in(z), W_ag, W_bg) | 4.25 % (4.16-4.25) | 3.3 % (2.8-3.3) | 2.06 % |
| planes + all K1 projections (the FP8-compute K1) | 6.80 % (6.70-6.80) | 4.9 % (4.6-5.3) | K1 value projections alone (LN_in(z), W_a, W_b): 5.3 % |
| planes + t (K3 operand) | 4.54 % (4.43-4.54) | 3.5 % (2.9-3.7) | 2.61 % |
| planes + K3 gate (LN_in(z), W_g) | 4.00 % (3.90-4.00) | 3.0 % (2.6-3.3) | 1.47 % |
| planes + K3 projection (LN_out(t), W_o) | 5.27 % (5.20-5.27) | 3.7 % (3.5-4.0) | 3.74 % |

Reading: the gate operands (K1 gate projections, K3 gate) are the cheap e4m3 placements (the sigmoid's slope damps their rounding: +0.3-0.5 % on top of the
planes); the value paths (K1 projections, t, K3 projection) cost 0.8-3.1 % each. A K1 with the gate projections in e4m3 and the value projections in bf16 would sit
at 4.2 % (FP8-compute K1: 6.8 %), and at D256-D512 (K1 tensor-bound, 31-47 % of the module) halve the MMA time of half of K1: from the FP8-compute K1's K1
-36 % at D256 L2048, about -18 % of K1 = -7 % of the module (upper bound, not built: it needs two MMA chains with different operand types per K1 weight block).


### New coverage: module time (CUDA-graph replay, cool / sustained, interleaved with a same-cubin control; FP8 / default)

One direction, B = 1 (`spd1.jsonl`; outgoing / incoming):

| D | dir | | L128 | L256 | L512 | L768 | L1024 | L1536 | L2048 | control |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | out | default ms | 0.016 / 0.015 | 0.036 / 0.035 | 0.132 / 0.132 | 0.290 / 0.295 | 0.513 / 0.555 | 1.315 / 1.430 | 2.639 / 2.987 | |
| 64 | out | FP8 / default | 1.135 / 1.144 | 0.980 / 0.977 | 0.903 / 0.905 | 0.872 / 0.905 | 0.863 / 0.890 | 0.808 / 0.845 | 0.766 / 0.773 | 0.964 .. 1.004 |
| 64 | in | FP8 / default | | 0.976 / 0.976 | 0.894 / 0.896 | | 0.873 / 0.900 | | 0.773 / 0.788 | 0.990 .. 1.009 |
| 384 | out | default ms | 0.075 / 0.076 | 0.288 / 0.309 | 1.104 / 1.199 | 2.419 / 2.743 | 4.571 / 5.081 | 11.26 / 13.03 | 22.09 / 24.67 | |
| 384 | out | FP8 / default | 0.963 / 0.963 | 0.930 / 0.921 | 0.916 / 0.933 | 0.921 / 0.924 | 0.893 / 0.896 | 0.863 / 0.865 | 0.823 / 0.819 | 0.985 .. 1.008 |
| 384 | in | FP8 / default | | 0.925 / 0.902 | 0.916 / 0.937 | | 0.903 / 0.911 | | 0.869 / 0.801 | 0.972 .. 1.004 |

Bidirectional, B > 1, fast accumulation (`spd2.jsonl`; `FP8` / `FP8 + FASTACC` against the default):

| case | default ms (cool / sus) | FP8 / default | FP8 + FASTACC / default | control |
|---|---|---|---|---|
| bidirectional D64 L512 / 1024 / 2048 | 0.201 / 0.203, 0.838 / 0.852, 4.64 / 5.20 | 1.036 / 1.051, 0.927 / 1.076, 0.780 / 0.816 | 1.017 / 1.033, 0.910 / 1.040, 0.754 / 0.785 | 0.956 .. 1.006 |
| bidirectional D384 L512 / 1024 / 2048 | 1.90 / 2.10, 8.04 / 9.24, 39.0 / 43.2 | 0.966 / 0.988, 0.920 / 0.926, 0.932 / 0.858 | 0.969 / 0.982, 0.906 / 0.906, 0.890 / 0.839 | 0.991 .. 1.085 (L2048 cool: noisy) |
| D64 out B4 L512 / L1024 | 0.501 / 0.498, 1.950 / 2.136 | 0.954 / 0.982, 0.907 / 0.955 | 0.940 / 0.961, 0.893 / 0.923 | 0.987 .. 0.999 |
| D128 out B2 L512 / L1024 | 0.462 / 0.491, 1.895 / 2.252 | 0.898 / 0.909, 0.845 / 0.847 | 0.900 / 0.904, 0.830 / 0.811 | 0.983 .. 0.998 |
| D64 out L2048 / D384 out L2048 (repeat) | 2.618 / 2.921, 21.97 / 24.44 | 0.772 / 0.799, 0.831 / 0.836 | 0.745 / 0.774, 0.909 / 0.819 | 0.996 .. 1.005 |

Gates (`MIN_L`): D64 512 both directions (L128 +13.5 %, L256 -2 % against a control of 1.00-1.01: not clear), D384 128 (every measured L gains, L128 -3.7 %),
bidirectional D64 2048 (L512 +4 %, L1024 -7 % cool but +8 % sustained: not clear; L1536 not measured), bidirectional D384 1024 (L512 -3 % / -1 %: not clear).
B > 1 runs the one-sample route per sample (`run_samples`, one output, shared static scales): the batched default launches once for B samples, so
the per-sample loop gains less at small L (D64 B4 L512 -5 / -2 %).

### Error of the new coverage (`err2_x.jsonl`, `err3_x2.jsonl`; same protocol as the table in "Base 2.5.12": random weights, 3 masked tokens, every gate removed)

| case | default rms / max (update) | default std_max | FP8 rms / max (update) | FP8 std_max | peak scaled plane value |
|---|---|---|---|---|---|
| D64 out L256 / L1024 | 0.65 / 0.94 %, 0.63 / 0.73 % | 0.63 %, 0.50 % | 3.7 / 5.2 %, 3.5 / 4.1 % | 3.5 %, 2.8 % | (no overflow) |
| D64 in L300 / L1024 | 0.66 / 0.83 %, 0.64 / 0.79 % | 0.51 %, 0.53 % | 3.7 / 4.9 %, 3.6 / 4.1 % | 3.0 %, 2.7 % | (no overflow) |
| D384 out L256 / L300 / L1024 | 0.65 / 0.74 %, 0.65 / 0.72 %, 0.64 / 0.75 % | 0.44 %, 0.51 %, 0.54 % | 3.7 / 4.0 %, 3.7 / 3.4 %, 3.7 / 3.6 % | 2.4 %, 2.4 %, 2.6 % | (no overflow) |
| D384 in L384 | 0.65 / 0.61 % | 0.42 % | 3.7 / 3.3 % | 2.2 % | (no overflow) |
| bidirectional D64 / D384 L512 | 0.65 / 0.72 %, 0.65 / 0.78 % | 0.53 %, 0.46 % | 3.7 / 3.7 %, 3.7 / 3.9 % | 2.7 %, 2.3 % | 18.4, 19.1 |
| D64 out L512 B3 / D384 in L256 B2 | 0.65 / 0.77 %, 0.65 / 0.70 % | 0.54 %, 0.49 % | 3.6 / 4.3 %, 3.7 / 3.5 % | 3.0 %, 2.5 % | 18.9, 18.2 |
| (reference) D128 out L512, D256 in L512 | 0.66 / 0.69 %, 0.65 / 0.77 % | 0.44 %, 0.51 % | 3.7 / 4.2 %, 3.7 / 4.0 % | 2.7 %, 2.6 % | |

Engine standard form (max |y - y64| / max |y64|, tolerance max(err_pytorch, 8e-3)): the FP8 route of every new width is 2.2-3.5 % (fails 8e-3, as the
shipped D128 / D256 route does); the bf16 default 0.42-0.63 %. No NaN, inf or saturation (peak scaled plane value 18-19 of 448 at 16x headroom).

### What it means

- The contraction operands remain the one placement that pays: the new widths gain what D128 / D256 gain (D64 -23 % / D384 -18 % at L2048, -10 % / -8 %
  at L512), at the same 3.6-3.7 % update error.
- The next lever by size is the K1 projection compute at D256-D512 (K1 31-47 % of the module, tensor-bound): the existing FP8-compute K1 takes a further
  -11 % at D256 L2048, but its value-projection operands are the most accuracy-expensive e4m3 placement (6.8 %, accuracy map). A gate-projection-only
  variant (LN_in(z), W_ag / W_bg in e4m3, value projections bf16) emulates at 4.2 % (planes alone 3.7 %) with an upper bound of about -7 % of the module at
  D256 L2048 (more at D384 / D512, where K1 is 42-47 % of the module); it needs two MMA chains with different operand types per K1 weight block (a K1
  change, not built here). It is the next candidate.

### Verification (branch `fp8x/trimul`)

- Default route (no FP8 switch) against the base tree 91a9241fb, separate processes, sha1 of every output (training: output, input gradient, every
  parameter gradient): **181 / 181 identical** -- TriMul one direction D64-D512 both directions L256-1536 B1 / 2 (68), training (16), bidirectional
  D64 / D128 / D256 incl. training (44), TriAttn module shapes (31, the header is shared) and the shipped FP8 route at D128 / D256 one direction and
  bidirectional, L256-1024 (22: the n128 F8 K1 is untouched). The header change is under `TMN_K1_F8` (`if constexpr (G::BW == 64)`), so the
  anchor-patched families build from the same text as before except for that block.
- Tests: `tests/gpu/test_trimul_fp8_widths.py` (new: D64 / D384 both directions L100 / 256 / 300, bidirectional D64 / D384, B = 3 equal to
  per-sample calls), `test_trimul_fp8_planes.py`, `test_trimul_fp8_bidir.py`, `test_trimul_fp8_k1.py`: 65 passed (18 new), one job, 1 GPU. CPU: `ruff check`, `ruff format`,
  `pyright` clean on the touched files; `tests/cpu/test_prebuild_plan.py`, `test_imports.py`: 930 passed. The D64 / D384 FP8 K1 builds are not in the
  prebuild plan (exploration; they JIT-compile on first use).
- e4m3 t (K3 operand) was built and measured: the contraction does not get faster (tensor-bound) and K3's byte saving (<= 3.6-6.3 % of the module,
  measured upper bound) is eaten by the widening; removed.
- K3's own MMAs (projection, gate): K3 is byte-bound where it is fast (D64 / D128) and not floor-bound at D256+; e4m3 there needs K-major t
  (a shared-memory transpose) and costs accuracy (map). Not done.

## TriAttn forward: FP8 PV, per-model whitelist (`h100_sm90`)

`triattn_h16_sm90.cu` built with `-DTA_F8PV=1` (P as e4m3 bits straight from the logits, V as e4m3 `V^T` hi plane, or hi + lo with `-DTA_F8HL=1`; row sums from the quantized P).
Not a general route. Use the whitelist API:

```python
from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

# True only for whitelisted models; sets FP8_PV, FP8_PV_FUSED, FP8_VSCALE_AUTO (+ FP8_PV_UNFUSED / the S gate for "unfused" models)
if hs.enable_fp8_pv_for("protenix1"):
    # d_pair 128 inference: the front writes V^T planes (tri_front_vt); other fronts: standalone V^T pass, single pass, static scale
    ...
# after every forward: if hs.take_overflow(): hs.rescale_all(); redo
hs.disable_fp8_pv()
```

`FP8_PV_WHITELIST` maps a model to the call-site kinds whose end-level accuracy was checked: `{"protenix1": {"fused"}, "opendde": {"unfused"}, "af3": {"fused", "unfused"}}` (3.0.0: derived from `miniworld_engine.fp8.TRIATTN_MODELS`, which the one switch uses); Boltz-2 is absent on purpose (9c90, see limits).
`enable_fp8_pv_for(model, hl=False)`: `hl=True` writes the hi + lo planes (output error about half, see below). Default (non-FP8) route: bit-identical to 2.5.9 (10 shapes / orientations / head counts compared as bytes; branch `fp8/vt`, whose default builds of `front_fused3_sm90.cu` and `triattn_mod_sm90.cu` changed in source (the V^T code is `#ifdef` / a template instance): 14 more shapes d_pair 64 / 128 / 256 / 384, start / end, flip, masked, L256..1536, as bytes against main ec11768fc). The FP8 route with front-made planes equals the standalone-pass FP8 route as bytes (hi: 64:4:32 / 64:2:32 / 64:4:16 L256..1536 with a padded key mask, trans; hi + lo L256..1024; 256:8:32 L256..2048, trans).

Three kinds of call sites (branch `fp8/vt`: the first two make the planes in the front, the pass only remains where the front is cuBLAS):
- **fused** (d_pair 128, 4 x 32): the front writes the e4m3 V^T planes itself (static per-weight-pack scale, power of two above 4 x the calibration V absmax, saturation flag).
- **front-made** (d_pair 64: 4 x 32, 2 x 32, 4 x 16; d_pair 256 x 8 x 32; `FP8_FRONT_VT`, default on inside the FP8 route): the one-kernel fronts write the planes the same way. d_pair 64: `front_fused3_sm90.cu` built with `-DMW_F3_VTW=1|2`
  (bf16 front up to L768 / 2048 compact, fp16-exact Y16 front beyond; the V block of the projection is quantized from the fp16-rounded V, a lane pair swaps bytes so each lane stores one 32-bit word of 4 keys,
  then 16-byte global stores); d_pair 256: the group-2 CTA of `tri_front_wide` (`VTW` template, hi plane only; hi + lo keeps the standalone pass). The planes are bit for bit the standalone pass's
  (same fp16 V, same static scale, same cvt): the module output equals the standalone route as bytes. The first eager call of a site still runs the pass (it takes the static scale).
- **unfused with cuBLAS front** (d_pair 384 x 12 x 32, and any shape without a one-kernel front): fp16 V from the cuBLAS q | k | v | g GEMM, one `vt_quant` pass makes the planes. With `FP8_VSCALE_AUTO` the pass uses a per-site static scale
  (calibrated on the first eager call; single pass, no amax pass, same `take_overflow` / `rescale_all` contract). `vt_quant` serves up to 16 heads (12-head d_pair 384 had silently run the slow torch reference).
S gates: fused route (d_pair 128) `FP8_PV_FUSED_MIN_S` 256 (2.5.12 base; x 2 with hi + lo); standalone pass `FP8_PV_UNFUSED_MIN_S` (32-channel heads 1024, 16-channel heads 512; x 2 with hi + lo); front-made planes `FP8_PV_UNFUSED_FRONT_MIN_S` per (d_pair, heads, channels): 64:4:32, 64:2:32, 64:4:16 from L256, 256:8:32 from L768
(x 2 with hi + lo). Below the gate the fp16 core runs (bit-identical to the default). The scale tensors derived from the static scale (1 / scale, scale per sample) are cached and updated in place by `rescale_all`: a call launches no torch glue kernel for them.

### Base 2.5.12 (branch `fp8/3.0.0-on-2512`, 2026-11-17): module time, error and gates against the 2.5.12 default

The FP8 routes sit on the 2.5.12 kernels (TriAttn h16 core with `TA_POLY16` = 8, bidirectional TriAttn fused tails, TriMul launch table v6, bidirectional
TriMul port K3 / `LT_BIDIR`, OPM). The 2.5.12 default got faster than 2.5.9 (TriAttn d_pair 64 x 4 x 16 by 3-11 %, bidirectional TriMul D128 by 1-5 %),
so every FP8 ratio below is against that faster default.

Protocol (H100 80GB HBM3, 700 W cap; `~/miniworld-engine-scratch/fp8/w300/scripts/speed.py`, raw rows `res/speed.jsonl`, `res/speed2.jsonl`): one process per group, the module (`master_weights=True`, one sample, all-ones mask)
captured in three CUDA graphs of one call each -- the 2.5.12 default route, the default again (same-cubin control), the FP8 route (hi plane, every S /
min-L gate removed, static scales calibrated on two eager calls, lazy guard; no overflow in any case). **cool** = 2 s idle, then one interleaved burst
(~10 ms per route), 3 reps, median; **sustained** = 2 s idle, 1 s of back-to-back interleaved replays, then 3 interleaved timed blocks (~20 ms per route),
mean (sustained load throttles the SM clock 15-18 %). Cells: `cool / sustained`. Control = range of the same-cubin control's ratios over the row (noise).

TriAttn (`d_pair:heads:channels`, starting node), FP8 PV hi plane with no gate:

| config | | L128 | L256 | L512 | L768 | L1024 | L1536 | L2048 | same-cubin control |
|---|---|---|---|---|---|---|---|---|---|
| 128:4:32 | 2.5.12 default ms | 0.031 / 0.031 | 0.104 / 0.103 | 0.441 / 0.468 | 1.168 / 1.245 | 2.441 / 2.61 | 7.116 / 8.652 | 15.522 / 17.774 | |
| | FP8 / default | 1.104 / 1.104 | 0.936 / 0.942 | 0.902 / 0.9 | 0.88 / 0.871 | 0.86 / 0.87 | 0.847 / 0.845 | 0.845 / 0.842 | 0.99 .. 1.003 |
| 256:8:32 | 2.5.12 default ms |  | 0.232 / 0.242 | 1.009 / 1.128 | 2.64 / 2.939 | 5.407 / 6.498 | 15.533 / 18.258 | 33.739 / 38.441 | |
| | FP8 / default |  | 0.994 / 0.991 | 0.976 / 0.962 | 0.954 / 0.955 | 0.938 / 0.94 | 0.911 / 0.921 | 0.967 / 0.873 | 0.968 .. 1.057 |
| 384:12:32 | 2.5.12 default ms |  | 0.403 / 0.443 | 1.73 / 2.021 | 4.483 / 5.142 | 9.051 / 10.413 | 25.474 / 28.41 | 56.625 / 62.114 | |
| | FP8 / default |  | 0.999 / 0.987 | 1.014 / 0.964 | 0.982 / 0.988 | 0.958 / 0.95 | 0.957 / 0.912 | 0.955 / 0.886 | 0.964 .. 1.082 |
| 64:4:16 | 2.5.12 default ms | 0.028 / 0.027 | 0.09 / 0.087 | 0.41 / 0.397 | 1.088 / 1.1 | 2.24 / 2.332 | 6.6 / 7.067 | 14.214 / 16.001 | |
| | FP8 / default | 1.052 / 1.053 | 0.948 / 0.945 | 0.899 / 0.902 | 0.856 / 0.85 | 0.85 / 0.852 | 0.834 / 0.839 | 0.834 / 0.823 | 0.993 .. 1.005 |
| 64:4:32 | 2.5.12 default ms | 0.033 / 0.032 | 0.107 / 0.106 | 0.469 / 0.496 | 1.222 / 1.314 | 2.436 / 2.688 | 7.127 / 8.345 | 15.604 / 17.787 | |
| | FP8 / default | 1.049 / 1.048 | 0.907 / 0.905 | 0.877 / 0.878 | 0.869 / 0.866 | 0.878 / 0.895 | 0.857 / 0.864 | 0.856 / 0.846 | 0.982 .. 1.009 |
| 64:2:32 | 2.5.12 default ms | 0.024 / 0.023 | 0.066 / 0.064 | 0.278 / 0.278 | 0.682 / 0.723 | 1.323 / 1.426 | 3.759 / 4.165 | 8.174 / 9.494 | |
| | FP8 / default | 1.058 / 1.057 | 0.94 / 0.94 | 0.917 / 0.914 | 0.909 / 0.911 | 0.922 / 0.917 | 0.895 / 0.9 | 0.875 / 0.885 | 0.994 .. 1.006 |

TriMul (one direction / the bidirectional block), FP8 planes with no gate (L >= 256; the gate region below):

| config | | L256 | L512 | L768 | L1024 | L1536 | L2048 | same-cubin control |
|---|---|---|---|---|---|---|---|---|
| D128 out | 2.5.12 default ms | 0.07 / 0.068 | 0.244 / 0.256 | 0.52 / 0.595 | 0.981 / 1.119 | 2.549 / 2.983 | 5.181 / 5.928 | |
| | FP8 / default | 0.903 / 0.903 | 0.848 / 0.839 | 0.857 / 0.833 | 0.813 / 0.792 | 0.774 / 0.751 | 0.739 / 0.739 | 0.971 .. 1.017 |
| D128 in | 2.5.12 default ms | 0.069 / 0.069 | 0.244 / 0.263 | 0.535 / 0.602 | 0.982 / 1.136 | 2.575 / 3.105 | 5.174 / 6.091 | |
| | FP8 / default | 0.877 / 0.875 | 0.853 / 0.831 | 0.833 / 0.83 | 0.816 / 0.813 | 0.771 / 0.733 | 0.744 / 0.721 | 0.956 .. 1.001 |
| D128 bidir | 2.5.12 default ms | 0.115 / 0.115 | 0.396 / 0.441 | 0.88 / 1.003 | 1.643 / 1.966 | 4.37 / 5.086 | 9.116 / 11.385 | |
| | FP8 / default | 0.963 / 0.965 | 0.924 / 0.909 | 0.887 / 0.922 | 0.863 / 0.879 | 0.809 / 0.782 | 0.757 / 0.744 | 0.973 .. 1.003 |
| D256 out | 2.5.12 default ms | 0.154 / 0.16 | 0.552 / 0.596 | 1.22 / 1.365 | 2.239 / 2.587 | 5.677 / 6.849 | 11.556 / 13.56 | |
| | FP8 / default | 0.954 / 0.969 | 0.937 / 0.985 | 0.917 / 0.991 | 0.917 / 0.966 | 0.875 / 0.859 | 0.814 / 0.822 | 0.979 .. 1.001 |
| D256 in | 2.5.12 default ms | 0.155 / 0.16 | 0.555 / 0.599 | 1.212 / 1.35 | 2.244 / 2.575 | 5.824 / 7.154 | 11.515 / 13.866 | |
| | FP8 / default | 0.95 / 0.954 | 0.935 / 0.977 | 0.924 / 0.994 | 0.914 / 0.968 | 0.853 / 0.839 | 0.826 / 0.805 | 0.992 .. 1.003 |
| D256 bidir | 2.5.12 default ms | 0.258 / 0.269 | 0.94 / 1.022 | 2.122 / 2.413 | 4.007 / 4.549 | 10.245 / 12.813 | 20.534 / 23.533 | |
| | FP8 / default | 1.009 / 1.036 | 0.979 / 1.026 | 0.946 / 1.01 | 0.925 / 0.949 | 0.88 / 0.873 | 0.878 / 0.824 | 0.992 .. 1.01 |

TriMul below L256 (the min-L gates):

| config | | L128 | L144 | L160 | L176 | L192 | L224 | L256 | same-cubin control |
|---|---|---|---|---|---|---|---|---|---|
| D128 out | 2.5.12 default ms | 0.024 / 0.023 | 0.039 / 0.038 | 0.041 / 0.04 | 0.043 / 0.041 | 0.049 / 0.047 | 0.064 / 0.062 | 0.07 / 0.068 | |
| | FP8 / default | 1.003 / 0.983 | 0.919 / 0.92 | 0.887 / 0.883 | 0.864 / 0.859 | 0.899 / 0.895 | 0.924 / 0.922 | 0.903 / 0.903 | 0.995 .. 1.011 |
| D128 in | 2.5.12 default ms | 0.024 / 0.023 | 0.034 / 0.033 | 0.036 / 0.035 | 0.038 / 0.037 | 0.049 / 0.047 | 0.064 / 0.062 | 0.069 / 0.069 | |
| | FP8 / default | 0.998 / 0.999 | 1.059 / 1.053 | 1.015 / 1.009 | 0.98 / 0.976 | 0.893 / 0.887 | 0.921 / 0.917 | 0.877 / 0.875 | 0.993 .. 1.001 |
| D128 bidir | 2.5.12 default ms |  |  | 0.066 / 0.064 |  | 0.075 / 0.073 | 0.104 / 0.102 | 0.115 / 0.115 | |
| | FP8 / default |  |  | 0.939 / 0.933 |  | 1.063 / 1.057 | 0.995 / 0.998 | 0.963 / 0.965 | 0.994 .. 1.003 |
| D256 out | 2.5.12 default ms | 0.044 / 0.043 | 0.074 / 0.072 |  | 0.08 / 0.082 | 0.108 / 0.108 | 0.148 / 0.151 | 0.154 / 0.16 | |
| | FP8 / default | 0.939 / 0.937 | 1.002 / 1.006 |  | 1.018 / 1.008 | 1.016 / 1.013 | 0.947 / 0.948 | 0.954 / 0.969 | 0.988 .. 1.006 |
| D256 in | 2.5.12 default ms | 0.044 / 0.043 | 0.084 / 0.081 |  | 0.09 / 0.091 | 0.109 / 0.109 | 0.148 / 0.152 | 0.155 / 0.16 | |
| | FP8 / default | 0.941 / 0.943 | 0.891 / 0.893 |  | 0.909 / 0.905 | 0.994 / 0.992 | 0.931 / 0.929 | 0.95 / 0.954 | 0.992 .. 1.005 |

Gates against the 2.5.12 default (applied only where the data is clear):

- **New: TriAttn fused route (128:4:32, whitelist kind "fused") `FP8_PV_FUSED_MIN_S = 256`** (x 2 with hi + lo, as the other gates). It had no gate: at L128
  the FP8 route is 10.4 % slower cool and sustained (control 0.99), at L256 6 % faster. Below the gate the fp16 core runs (bit-identical to the default; test
  `test_fused_route_s_gate`).
- Unchanged, confirmed: TriAttn front-made planes d_pair 64 from L256 (L128 +5 %, L256 -5 .. -9 %); 256:8:32 front-made from L768 (L512 -2.4 % cool /
  -3.8 % sustained with a sustained control of 0.968: not clear, gate stays); 384:12:32 standalone pass from L1024 (L768 -1.8 / -1.2 %: not clear).
  384:12:32 at L256 is not served by the FP8 core at all (FP8 bytes = default bytes; the same on the 2.5.9-based branch).
- Unchanged, confirmed: TriMul D128 out 144 (L128 1.00, L144 -8 %), D128 in 192 (L176 -2 %, L192 -11 %), D256 out 224 (L144-192 +0 .. +2 %, L224 -5 %;
  L128 is -6 % but not the L above it), D256 in 144 (L128 -6 %: lowering the gate needs L < 128 data), bidirectional D128 256 (L160 -6 %, L192 +6 %, L224
  0 %: not monotone), bidirectional D256 1024 (L768 -5 % cool but +1 % sustained, L1024 -7.5 / -5 %). The faster 2.5.12 default moved no gate up.

Error against fp64 PyTorch (`~/miniworld-engine-scratch/fp8/w300/scripts/err.py`, raw rows `res/err2_*.jsonl`; the module in float64 on the PyTorch backend, same inputs; random weights, LayerNorm weights
1 +- 0.1, biases +- 0.1, key mask with 3 masked tokens; FP8 with every gate removed; rms / max of (y - y64) relative to rms / max of the update y64 - x).
"2.5.9-based" = the same script on `fp8/bidir-tm` (8bef4e1ab), whose default route is 2.5.9:

| case | default 2.5.12 | FP8 hi, 2.5.12 base | FP8 hi, 2.5.9-based | FP8 hi + lo, 2.5.12 base | FP8 hi + lo, 2.5.9-based | default 2.5.9 |
|---|---|---|---|---|---|---|
| TriAttn 128:4:32 L256 | 1.6% / 1.5% | 3.6% / 2.9% | 3.6% / 2.9% | 2.9% / 2.3% | 2.9% / 2.6% | 1.6% / 1.5% |
| TriAttn 128:4:32 L512 | 1.9% / 2.1% | 3.3% / 4.7% | 3.3% / 4.7% | 2.8% / 2.7% | 2.8% / 2.7% | 1.9% / 2.1% |
| TriAttn 256:8:32 L256 | 1.7% / 2.1% | 3.7% / 3.6% | 3.7% / 3.6% | 3% / 2.6% | 3% / 2.6% | 1.7% / 2.1% |
| TriAttn 256:8:32 L512 | 1.9% / 2.2% | 3.4% / 3.5% | 3.4% / 3.5% | 2.9% / 2.9% | 2.9% / 2.9% | 1.9% / 2.2% |
| TriAttn 384:12:32 L256 | 1.7% / 2.3% | 1.7% / 2.3% | 1.7% / 2.3% | 1.7% / 2.3% | 1.7% / 2.3% | 1.7% / 2.3% |
| TriAttn 384:12:32 L512 | 1.9% / 2.3% | 3.4% / 3.8% | 3.4% / 3.8% | 2.9% / 3.1% | 2.9% / 3.1% | 1.9% / 2.3% |
| TriAttn 64:2:32 L256 | 1.6% / 1.7% | 3.3% / 4% | 3.3% / 4% | 2.7% / 2.3% | 2.7% / 2.3% | 1.6% / 1.7% |
| TriAttn 64:2:32 L512 | 1.8% / 1.9% | 3% / 3.8% | 3% / 3.8% | 2.5% / 2.7% | 2.5% / 2.7% | 1.8% / 1.9% |
| TriAttn 64:4:16 L256 | 1.6% / 1.7% | 3.3% / 3.4% | 3.3% / 3.4% | 2.7% / 2.6% | 2.7% / 2.6% | 1.6% / 1.7% |
| TriAttn 64:4:16 L512 | 1.8% / 1.6% | 3% / 3.1% | 3% / 3.1% | 2.5% / 1.9% | 2.5% / 1.9% | 1.8% / 1.6% |
| TriAttn 64:4:32 L256 | 1.2% / 1.3% | 3.1% / 2.7% | 3.1% / 2.7% | 2.5% / 2.1% | 2.5% / 2.1% | 1.2% / 1.3% |
| TriAttn 64:4:32 L512 | 1.3% / 1.5% | 2.7% / 3.2% | 2.7% / 3.2% | 2.2% / 2.2% | 2.2% / 2.2% | 1.3% / 1.5% |
| TriMul D128 bidir L1024 | 0.6% / 0.7% | 3.7% / 3.9% | 3.7% / 3.9% | - | - | 0.7% / 0.7% |
| TriMul D128 bidir L256 | 0.6% / 0.8% | 3.7% / 3.8% | 3.7% / 3.8% | - | - | 0.7% / 0.8% |
| TriMul D128 bidir L512 | 0.6% / 0.8% | 3.7% / 4.2% | 3.7% / 4.2% | - | - | 0.7% / 0.9% |
| TriMul D128 out L1024 | 0.6% / 0.7% | 3.7% / 4.4% | 3.7% / 4.4% | - | - | 0.6% / 0.7% |
| TriMul D128 out L256 | 0.7% / 0.7% | 3.8% / 4.2% | 3.8% / 4.2% | - | - | 0.7% / 0.7% |
| TriMul D128 out L512 | 0.7% / 0.7% | 3.7% / 4.2% | 3.7% / 4.2% | - | - | 0.7% / 0.7% |
| TriMul D128 in L1024 | 0.6% / 0.8% | 3.7% / 4% | 3.7% / 4% | - | - | 0.6% / 0.8% |
| TriMul D128 in L256 | 0.7% / 0.7% | 3.8% / 3.9% | 3.8% / 3.9% | - | - | 0.7% / 0.7% |
| TriMul D128 in L512 | 0.7% / 0.8% | 3.7% / 3.9% | 3.7% / 3.9% | - | - | 0.7% / 0.8% |
| TriMul D256 bidir L1024 | 0.6% / 0.7% | 3.7% / 3.6% | 3.7% / 3.6% | - | - | 0.6% / 0.7% |
| TriMul D256 bidir L256 | 0.6% / 0.7% | 3.7% / 3.6% | 3.7% / 3.6% | - | - | 0.6% / 0.7% |
| TriMul D256 bidir L512 | 0.6% / 0.8% | 3.7% / 3.5% | 3.7% / 3.5% | - | - | 0.6% / 0.8% |
| TriMul D256 out L1024 | 0.6% / 0.8% | 3.7% / 4% | 3.7% / 4% | - | - | 0.6% / 0.8% |
| TriMul D256 out L256 | 0.6% / 0.8% | 3.8% / 4% | 3.8% / 4% | - | - | 0.6% / 0.8% |
| TriMul D256 out L512 | 0.6% / 0.8% | 3.7% / 4.3% | 3.7% / 4.3% | - | - | 0.6% / 0.8% |
| TriMul D256 in L1024 | 0.6% / 0.7% | 3.6% / 4% | 3.6% / 4% | - | - | 0.6% / 0.7% |
| TriMul D256 in L256 | 0.7% / 0.8% | 3.7% / 4.1% | 3.7% / 4.1% | - | - | 0.7% / 0.8% |
| TriMul D256 in L512 | 0.6% / 0.8% | 3.7% / 4% | 3.7% / 4% | - | - | 0.6% / 0.8% |

The FP8 errors are the 2.5.9-based ones to the printed digit (TriMul identical bytes of the error; TriAttn 128:4:32 L256 hi + lo max 2.3 % vs 2.6 %:
below the new fused gate x 2 the call takes the standalone-pass FP8 core with a per-site scale instead of the fused front's). The defaults differ where
2.5.10-2.5.12 changed them (64:4:16 `TA_POLY16`, bidirectional TriMul D128 port K3: 0.7 -> 0.6 %). In the engine-standard form (max |y - y64| / max |y64|):
TriAttn FP8 0.32-0.75 %, TriMul FP8 2.3-3.1 % (tolerance 8e-3: the TriMul FP8 route would fail a registry accuracy row; no row exists, see "Open
decisions"), defaults 0.27-0.54 %.

### Module time of the merged state (branch `fp8/main`) against the 2.5.9 default (base 2.5.9, history; the 2.5.12 numbers are above)

ms per sample; percent change of the FP8 route against the 2.5.9 default (negative = FP8 faster); interleaved, 9 rounds (median of 8 calls per round), one H100 per job, random weights, flat logits, 2.5.9 default measured twice per round as the
same-cubin control (control column: the control against the default, i.e. the noise of that row; differences below ~3 % are noise). Routes exactly as `enable_fp8_pv_for` sets them: `fused` = whitelist kind "fused" (d_pair 128: the front writes the e4m3 V^T planes),
`shipped` = kind "unfused" with the S gates (below its gate the fp16 core runs, so the row reads 0 +- noise), `no gate` = unfused route with every S gate removed (the cost / gain of the FP8 core where the gate keeps it off). Hi plane. Raw rows: `~/miniworld-engine-scratch/fp8/main/res/mod_*.jsonl`.

| config (d_pair:heads:channels) | route | L256 | L512 | L768 | L1024 | L1536 | L2048 | control range |
|---|---|---|---|---|---|---|---|---|
| 128:4:32 (Protenix v1, Boltz-2, AF3) default ms | | 0.098 | 0.439 | 1.209 | 2.565 | 7.861 | 17.32 | |
| 128:4:32 | fused | -6.8 % | -10.0 % | -11.7 % | -11.2 % | -15.9 % | -17.2 % | -0.3 .. +2.5 % |
| 128:4:32 | shipped (gate 1024) | 0.0 % | +0.3 % | +0.8 % | -5.9 % | -12.0 % | -13.3 % | |
| 128:4:32 | no gate | +1.8 % | 0.0 % | -3.8 % | -6.9 % | -12.2 % | -13.4 % | |
| 256:8:32 default ms | | 0.218 | 1.060 | 2.896 | 6.022 | 17.78 | 38.67 | |
| 256:8:32 | shipped (front-made planes from L768) | +0.6 % | +0.3 % | -4.4 % | -4.9 % | -10.9 % | -14.2 % | -1.2 .. +2.8 % |
| 256:8:32 | no gate | +1.9 % | -1.1 % | -7.0 % | -7.6 % | -11.6 % | -13.6 % | |
| 384:12:32 (OpenDDE) default ms | | 0.395 | 1.833 | 4.963 | 10.07 | 28.94 | 64.01 | |
| 384:12:32 (cuBLAS front, pass) | shipped (gate 1024) | +0.4 % | +1.6 % | +0.3 % | -4.6 % | -9.5 % | -11.5 % | -0.3 .. +0.8 % |
| 384:12:32 | no gate | +0.3 % | +2.2 % | -3.1 % | -5.5 % | -9.3 % | -11.0 % | |
| 64:4:16 default ms | | 0.0875 | 0.404 | 1.095 | 2.385 | 7.187 | 15.91 | |
| 64:4:16 (front-made, N24 core) | shipped (from L256) | -6.3 % | -14.6 % | -18.8 % | -21.4 % | -23.8 % | -21.2 % | -1.9 .. +0.2 % |
| 64:4:16 | no gate | -6.3 % | -14.6 % | -18.8 % | -20.5 % | -22.0 % | -22.0 % | |
| 64:4:32 default ms | | 0.103 | 0.457 | 1.233 | 2.546 | 7.775 | 17.44 | |
| 64:4:32 (front-made) | shipped (from L256) | -10.5 % | -11.1 % | -9.4 % | -8.4 % | -13.5 % | -16.9 % | -0.4 .. +2.5 % |
| 64:2:32 default ms | | 0.0627 | 0.258 | 0.678 | 1.397 | 4.022 | 9.072 | |
| 64:2:32 (front-made) | shipped (from L256) | -4.3 % | -6.0 % | -8.2 % | -7.1 % | -9.3 % | -14.0 % | -0.1 .. +3.0 % |

Where the gain comes from: the F8 core is 24-28 % faster than the 2.5.9 core on real activations (table below); the V^T planes cost a standalone pass (0.01-1 ms, 56-95 % of the byte floor; launch bound below L512) that ate 40-55 % of that gain at module level
until the fronts that are ours (d_pair 64, d_pair 256 x 8, d_pair 128) wrote the planes themselves (the pass removed: 64:4:32 L1024 -4.6 -> -8.4 %, L2048 -13.9 -> -16.9 %). The cuBLAS-front site (384:12:32) keeps one single-pass static-scale `vt_quant` (74 % of the byte floor; a one-kernel K = 384 front
would gain about 2.5 % of the module and is a second numerics path: not done). The S gates (the "shipped" rows) come from the earlier per-round measurements; the "no gate" rows above are the re-measure and agree with them (the gate only matters where the row is above +-noise: 128:4:32 below L1024, 256:8:32 / 384:12:32 below L768 / L1024).
History in one line each: round 2 (V as hi / lo planes, fused d_pair 128 front), round 3 (static per-site scale, `vt_quant` for 16 heads, S gate, whitelist per call-site kind), round 4 (front-made planes for d_pair 64 / 256 x 8; `TA_F8N24` for 16-channel heads: PV as m64n24, bit-identical).
hi + lo costs more (128:4:32 fused +2 .. -10 %, unfused +16 .. -5 % depending on config and L): hence the gate is twice as high for it.

### TriMul FP8 planes, module time against the 2.5.9 default (base 2.5.9, history; 2026-11-15, branch `fp8/tm2`; D128 / D256, one direction, one sample, bf16 inference; `MINIWORLD_TRIMUL_FP8=calib`, lazy guard)

Protocol: the module (`TriangleMultiplication`, `master_weights=True`), 9 interleaved rounds (default, default control, FP8), median; primary column = CUDA-graph replay of 8 calls (what FoldForge runs; GPU time), second block = eager (8 calls back to back). Same-cubin control within +-1 % (graph). H100 SXM, power-capped at ~690 W under sustained load.

Graph replay, FP8 vs default (negative = FP8 faster); L >= 256 is above every gate:

| D | direction | L256 | L512 | L768 | L1024 | L1536 | L2048 | default ms at L2048 |
|---|---|---|---|---|---|---|---|---|
| 128 | outgoing | -9.0 % | -15.3 % | -14.7 % | -18.5 % | -21.3 % | -24.9 % | 6.13 |
| 128 | incoming | -10.1 % | -14.9 % | -16.5 % | -17.4 % | -23.0 % | -23.2 % | 5.97 |
| 256 | outgoing | -3.9 % | -6.6 % | -5.9 % | -5.1 % | -13.6 % | -16.4 % | 13.23 |
| 256 | incoming | -4.1 % | -6.7 % | -4.4 % | -6.1 % | -14.2 % | -16.3 % | 13.33 |

Eager (host-bound below L768: the module call costs 0.31-0.44 ms of host time, the GPU 0.07-0.24 ms; run-to-run +-5 % at L256 / 512): D128 out -6.3 / -7.9 (repeat; first run +9.3) / -14.8 / -19.2 / -20.5 / -28.8 %, D128 in -6.3 / -5.2 / -16.6 / -17.4 / -23.6 / -25.0 %,
D256 out -19.4 / -6.8 / -5.9 / -4.1 / -11.3 / -16.8 %, D256 in -20.2 / -7.0 / -5.3 / -4.0 / -15.9 / -16.5 % (same L columns).

What the earlier table (D128 incoming +7.4 / +8.9 % at L256 / 512, D256 -3 .. -5 % at L512-1024) was: an EAGER measurement.
1. Below L768 the eager module call is host-bound for BOTH routes (stalled-GPU enqueue time 0.33-0.35 ms = the eager wall, GPU 0.07 / 0.24 ms), so the eager column compares host work. The FP8 incoming route did 20-30 us more host work per call than the default
   (the LN-statistics transpose of the incoming direction: `view`/`transpose`/`contiguous`/`copy_`, plus an `aten::zeros` for the amax buffer that `setdefault(.., torch.zeros(1))` evaluates on every call): +17 us at L256 (0.356 vs 0.339 ms).
2. Both are gone (see "Fixes"); host time of the FP8 route is now 0.30-0.33 ms, below the default's 0.33-0.44 ms.
3. In graph replay (no host cost) the FP8 route already won at every L >= 256 before the fix (D128 in -3.9 / -12.0 %, D256 -3.7 .. -7.6 %); the fix gave another -6.5 % (L256) / -2.9 % (L512) of the D128 incoming module.

Fixes (output bytes identical to `fp8/main`: 20 / 20 FP8 cases, 40 / 40 default cases): K1 stores the LN_in statistics at the swapped pair token when it reads the pair transposed (`K1Params.stats_t`, `TMN_K1_F8` builds only; the host transpose of `xs`,
one elementwise kernel of 3.1 / 5.5 / 13.9 us at L256 / 512 / 1024 and four host ops, is gone); the amax buffer is allocated once per module. The remaining extra kernel launches of the FP8 route are the cuBLASLt e4m3 contraction and its 0.8 us workspace `Memset` (the only kernels the default route has no twin for).

Per-stage breakdown (profiler device time, single call; graph = module graph-replay time per call; %SOL = max(byte floor at 3.05 TB/s, tensor floor at the 989 / 1979 TF peak) / measured; in brackets against the cuBLAS-measured peak 748 / 1431 TF; incoming, outgoing is within 1-2 %):

| D | L | stage | default ms (%SOL) | FP8 ms (%SOL) |
|---|---|---|---|---|
| 128 | 256 | K1 / contraction / K3 | 0.0276 (60) / 0.0144 (115) / 0.0274 (61) | 0.0245 (46) / 0.0110 (100) / 0.0227 (73) |
| 128 | 256 | module: kernel sum / graph / %SOL | 0.0694 / 0.0671 / 72 | 0.0589 (+0.0007 other) / 0.0603 / 66 |
| 128 | 512 | K1 / contraction / K3 | 0.0863 (77) / 0.0743 (89) / 0.0855 (78) | 0.0797 (56) / 0.0507 (87) / 0.0753 (89) |
| 128 | 512 | module | 0.2461 / 0.2429 / 81 | 0.2065 / 0.2067 / 75 |
| 128 | 1024 | K1 / contraction / K3 | 0.3405 (78) / 0.3850 (72, 95) / 0.2784 (96) | 0.2837 (63) / 0.2453 (72, 78) / 0.2808 (95) |
| 128 | 1024 | module | 1.0065 / 0.9846 / 81 | 0.8106 / 0.8131 / 77 |
| 128 | 2048 | K1 / contraction / K3 | 1.392 (77) / 2.925 (76, 100) / 1.153 (93) | 1.178 (61) / 1.700 (65, 90) / 1.156 (92) |
| 128 | 2048 | module | 5.473 / 5.971 / 80 (graph 73) | 4.034 / 4.583 / 72 (graph 63) |
| 256 | 256 | K1 / contraction / K3 | 0.0646 (54, 71) / 0.0353 (93) / 0.0513 (65) | 0.0717 (48, 64) / 0.0216 (102) / 0.0531 (62) |
| 256 | 256 | module | 0.1540 / 0.1531 / 66 | 0.1472 / 0.1468 / 61 |
| 256 | 512 | K1 / contraction / K3 | 0.2284 (61, 80) / 0.1460 (90) / 0.1803 (74) | 0.2414 (58, 76) / 0.0979 (90) / 0.1835 (72) |
| 256 | 512 | module | 0.5577 / 0.5596 / 72 | 0.5236 / 0.5223 / 69 |
| 256 | 1024 | K1 / contraction / K3 | 0.8712 (64, 84) / 0.7406 (75, 99) / 0.7274 (73) | 0.9168 (61, 80) / 0.5051 (70, 76) / 0.7231 (73) |
| 256 | 1024 | module | 2.342 / 2.466 / 70 (graph 67) | 2.146 / 2.316 / 67 (graph 62) |
| 256 | 2048 | K1 / contraction / K3 | 2.839 (78, 104) / 6.344 (70, 93) / 3.112 (68) | 3.081 (72, 95) / 3.663 (61, 84) / 3.126 (68) |
| 256 | 2048 | module | 12.30 / 13.33 / 72 (graph 66) | 9.871 / 11.16 / 67 (graph 59) |

(graph minus kernel sum at D256 L1024 / L2048 and D128 L2048: sustained replay runs at the power-capped clock; measured SM clock / power during 60 replays of L1024: D256 default 1390 MHz, FP8 1462-1505 MHz, both 690 W; D128 default 1610-1628, FP8 1488-1500 MHz.)

Reading it, with the removal ablations (CUDA-graph module time, same-cubin control within 0.2 %, 11 rounds, `scripts/abl.py` of the work dir):
- Stages add up: removing the cuBLAS contraction (no call) takes 14.6 / 53.1 / 270 us off the D128 incoming module at L256 / 512 / 1024 (its profiler time 11.1 + 0.8 Memset / 51.9 / 259), 24.5 / 89 / 544 us at D256 L256 / 512 / 1024: the contraction is a serial stage, nothing overlaps with it.
- The overflow guard (K1 amax: `atomicMax` per warp) costs nothing measurable: with the K1 amax pointer removed the module time moves by -1.0 .. +1.6 % at D256 (L256 / 512 / 768 / 1024, noise) and 0.0 .. -0.2 % at D128 (profiler K1 time D256 L768 -9 % does not show in the module time).
- D256 gains little because the contraction is only 30 % of the module (0.73 of 2.34 ms at L1024) and e4m3 speeds it up 1.47x (0.74 -> 0.51 ms; the bf16 contraction already runs at 75 % / 99 % of the cuBLAS peak, 1.39 GHz): the saving is at most 0.23 ms of 2.34 (9.8 %), and the FP8 K1 is 2-6 % SLOWER than the bf16 K1 at D256 (L512 0.241 vs 0.228 ms, L1024 0.917 vs 0.871): the net is -5 .. -7 %.
- D128 gains more because the planes are 1/3 of the K1 write bytes (K1 -6 .. -17 %), the contraction share is larger (38 % at L1024, e4m3 1.57x) and K3 reads the same bytes.
- At small L the whole D128 module is 60-67 us: the GPU time is launch- and latency-bound (K1 / K3 at 46-73 % of their byte floor at L256), the FP8 saving there is the planes' bytes in K1 and the contraction (3.3 us), and 4.6 us of K3 time that comes from the direct K3 (`k3_static`) the FP8 route uses being faster than the packaged D128 `tmn_k3` at L256, not from FP8.

Min-L gates (below them the bf16 default route runs unchanged; `MIN_L` in `h100_b200style_fp8.py`, `MINIWORLD_TRIMUL_FP8_MINL=<n>` overrides, 0 = none). Graph replay, FP8 vs default, same-cubin control within +-1 %:

| L | 64 | 96 | 112 | 128 | 144 | 160 | 176 | 192 | 224 | 300 | 384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D128 out | +6.4 | +2.4 | +1.3 | +1.3 / +0.8 | -5.4 | -8.1 | -10.1 | -7.7 | -6.3 | -15.7 | -13.1 |
| D128 in | +8.1 | +3.4 | +2.7 | +1.7 / +2.6 | +7.0 | +2.6 | +0.3 | -8.9 | -7.6 | -18.6 | -14.2 |
| D256 out | +1.7 | -1.3 | +3.4 | -3.0 / -4.4 | +1.8 | +0.1 | -0.9 | +1.1 | -5.8 | +0.3 | -5.6 |
| D256 in | -1.4 | +0.3 | +3.4 | -1.9 / -2.7 | -10.1 | -9.9 | -9.7 | -0.5 | -7.5 | -0.7 | -4.8 |

Gates: D128 outgoing L >= 144, D128 incoming L >= 192, D256 outgoing L >= 224, D256 incoming L >= 144. D256 outgoing below L224 is within +-3 % on the GPU (20 us more host launch work than the default in eager mode); D256 incoming gains 10 % at L144-176 because the default incoming route is 14 % slower than its outgoing there (FP8 in = out), is flat at L192 and below L144. At L300 (row-padded K1 / K3) D256 is flat (+0.3 / -0.7 %), inside the gated range: no further gate. Two-sample cells (L128, 192) agree to 1 %; the gate lines sit between measured L, not at an interpolated threshold (L between 176 and 192 for D128 incoming, 128 and 144 for D128 outgoing: untested).

### TriMul FP8 planes, bidirectional block (base 2.5.9, history; 2026-11-16, branch `fp8/bidir-tm`, on `fp8/main` + `fp8/tm2`; D128 / D256, one sample, bf16 inference, `MINIWORLD_TRIMUL_FP8=calib`, lazy guard)

Route (`h100_b200style_fp8.run(..., direction=0)`): the block's two directions are two of the single-direction FP8 routes on the matrix halves (rows `[:d]` outgoing, `[d:]` incoming of `to_left`, `to_left_gate`, `to_right`, `to_right_gate`): K1 (the single-direction FP8 tile, the incoming one on the transposed pair with the LN statistics stored at the swapped token) -> cuBLASLt e4m3 batched contraction of `d` channels, each writing its channels of one bf16 `tri [2 d, n, n]` -> one K3 over the 2 d channels (LN_out is joint over them, as in the default) with the per-channel scale product folded into the projection operand (`K3_CSCALE=1`: the two directions have different scales; the single-direction route keeps the scalar `K3_CSCALE=2`). Scales: per direction and per tensor (a, b), calibrated on the first eager call (16x headroom) into `cache["bd"][1 | 2]`; `refresh` / `rescale_all` / the lazy guard (`overflow_peak`, one amax buffer shared by the two K1) follow the same contract as the single direction. A capture with no calibration, and the guard's retry-exhausted fallback, run the bf16 default route of the block (`_update_inference_body`, not the single-direction port). The K1 planes of the two directions are two launches (the default has one K1 over 2 d hidden channels), which is why D256 gains less than D128: the FP8 K1 is tensor-bound and recomputes LN_in and re-reads z for the second direction.

Graph replay of the module, 11 interleaved rounds, median (default, default control, FP8, FP8 control; controls within 1 %); update error = rms(y - y_fp32) / rms(y_fp32 - x) against the fp32 PyTorch module on random inputs, mask with 3 masked tokens:

| D | L | default ms | FP8 ms | FP8 / default | update error FP8 (bf16 default 0.62 % at D128, 0.59 % at D256) | peak scaled plane value |
|---|---|---|---|---|---|---|
| 128 | 144 / 160 / 192 / 224 | 0.061 / 0.063 / 0.072 / 0.101 | 0.061 / 0.060 / 0.075 / 0.100 | 1.00 / 0.95 / 1.04 / 0.99 | 3.78-3.79 % | 17-19 |
| 128 | 256 / 384 / 512 / 768 | 0.113 / 0.232 / 0.407 / 1.012 | 0.107 / 0.216 / 0.369 / 0.859 | 0.949 / 0.932 / 0.907 / 0.848 | 3.78-3.79 % | 19-20 |
| 128 | 1024 / 1536 / 2048 | 1.964 / 5.363 / 10.79 | 1.609 / 3.937 / 7.750 | 0.819 / 0.734 / 0.719 | 3.79 % | 20 |
| 256 | 256 / 384 / 512 / 768 | 0.257 / 0.560 / 1.007 / 2.379 | 0.260 / 0.581 / 1.040 / 2.417 | 1.012 / 1.038 / 1.033 / 1.016 | 3.78-3.79 % | 19-20 |
| 256 | 1024 / 1536 / 2048 | 4.561 / 11.84 / 24.64 | 4.446 / 10.48 / 20.26 | 0.975 / 0.885 / 0.822 | 3.79 % | 20-22 |

Same error and peak as the single direction (3.8 %, scale peak far below 448); eager (8 back-to-back calls): D128 L512 / 1024 / 2048 1.026 / 0.829 / 0.726 of the default, D256 L512 / 1024 / 2048 1.003 / 0.971 / 0.824; host enqueue per call +0.06-0.07 ms at D128 (0.43-0.46 vs 0.37-0.40 ms), +0.00-0.01 ms at D256. Gates (`MIN_L[(d, 0)]`): D128 256 (L144-224 inside +-4 % noise, no gain), D256 1024 (L256-768 are 1-4 % slower: the second K1 launch); the `cuda` backend refuses D256 L % 128 != 0 at B = 1 for the default too, so those lengths never reach the route.

Where the FP8 module time goes (graph replay, per-kernel profiler time, D128 L1024 / L2048, D256 L1024 / L2048): K1 (two launches) 0.64 / 2.42, 1.93 / 8.11 ms; e4m3 contraction (two launches) 0.54 / 3.52, 1.04 / 7.30 ms; K3 (2 d channels) the rest. Removal ablation of the contraction (`res/abl_*.jsonl`): D128 L1024 1.60 -> 1.06 ms, L2048 7.77 -> 4.30; D256 L1024 4.21 -> 3.22, L2048 19.95 -> 12.30 (the contraction is serial: its removal removes its time; removing the amax output of K1 changes nothing within noise: 1.5 % at D128 L1024). Against the e4m3 tensor floor the contraction runs at 85-87 % of 1431 TF (D128 L2048 3.52 ms vs 3.07, D256 L2048 7.30 vs 6.15): nothing left to gain there, the cuBLASLt algorithm sweep of the single direction found no faster pick. K1 runs at 61 % (D128 L2048) .. 73 % (D256 L2048) of its bf16 tensor floor at 748 TF (two launches recompute LN_in).

Scale floor: `pow2_scale` (calibration scale of a channel from its amax) returned a scale from `amax.clamp_min(1e-30)`, i.e. 2^-99 .. 2^-107 for a channel whose amax is a denormal-sized ~0 (exactly 0 already gave 1): such a scale saturates the e4m3 cast on any later input (seen in the OpenDDE end-level runs). It is now floored at 2^-30 (amax 4e-7); channels above are unchanged (the single-direction route's outputs are bit-identical: 20 / 20 FP8 cases against `fp8/tm2`). The TriAttn per-head V scale (`vscale_from_amax`, `h100_sm90.py`, 2^-99 for a head with amax 0) is the same pattern: on the 2.5.12 base it has the same 2^-30 floor, and a head whose calibration absmax is ~0 takes scale 1 (`VSCALE_FLOOR`; tests `test_vscale_floor_and_zero_heads`, `test_static_scale_of_a_zero_head_does_not_saturate_later`).

### The F8 core alone on real activations

The FP8 core alone on REAL activations (q, k, v, bias of the dumps of Protenix v1 / Boltz-2 / AF3 / OpenDDE, 6 layers x start / end of 5I28, tiled to L1024 / L2048; ms per sample, mean over 12 layers):

| model (config) | L | default | hi core | hi core + V^T pass | hi + lo core | output error vs fp16 default, hi / hi + lo (mean / max) |
|---|---|---|---|---|---|---|
| Protenix v1 / Boltz-2 / AF3 (128:4:32) | 1024 | 1.79 | -24.4 .. -24.7 % | -11.4 % (dynamic pass) | -19 % | 2.1-2.4 / 2.7 %, 1.0-1.2 / 1.5 % |
| same | 2048 | 13.95 | -27.8 .. -28.2 % | -21.7 % (dynamic pass) | -21 % | 2.3-2.4 / 2.9 %, 1.2-1.5 / 1.8 % |
| OpenDDE (384:12:32) | 1024 | 5.50 | -25.3 % | -10.5 % (dynamic pass) | -19.5 % | 2.4 / 2.6 %, 0.95 / 1.2 % |
| OpenDDE | 2048 | 41.6 | -26.3 % | -18.5 % (dynamic pass) | -16.9 % | 2.6 / 2.8 %, 1.2 / 1.7 % |

(the real-activation cores gain more than the flat random ones: -25..-28 % vs -13..-16 %.) Error vs fp32 (reference of the 16 sampled rows, all 24 layer-targets per model): fp16 default 0.17-0.18 %, hi 2.0-2.3 %, hi + lo 0.85-1.2 % (mean).


### FP8 core, where the time goes and the 16-channel PV (2026-11-15, branch `fp8/core`)

Removal ablation of the F8 core (`TA_ABL` bits, time only, results wrong; one part removed per variant, interleaved rounds, 7 rounds, same-cubin control within +-2 %; percent of the F8 core time saved,
H100): the F8 core has NO MUFU at all (P is made by two HFMA2 + one HMNMX2 + a PRMT per four probabilities: `F8conv`), so there is no exp share and `TA_POLY` does not apply to it.

| removed | hv 4:32 L1024 / L2048 | flat 4:32 L1024 / L2048 | real (Protenix v1 / AF3 / Boltz-2) 4:32 L1024 / L2048 | hv 4:16 L1024 / L2048 |
|---|---|---|---|---|
| PV wgmma (4 x m64n40k32 per sub-step) | 28.5 / 27.1 % | 26.2 / 30.7 % | 23-27 / 25-29 % | 23.2 / 21.4 % |
| QK^T wgmma | 19.3 / 19.3 % | 10.1 / 14.6 % | 4.5-7.4 / 2.5-5.2 % | 11.4 / 11.1 % |
| e4m3 conversion (HFMA2 x 2, HMNMX2, PRMT) | 7.8 / 3.8 % | 0.8 / 6.8 % | 2.7-7.6 / 3.5-7.6 % | 8.2 / 11.5 % |
| the whole X phase (tree, vote, offsets, conversion) | 27.9 / 13.3 % | 10.8 / 10.0 % | 12.6-16.9 / 6.7-10.3 % | 26.1 / 20.7 % |
| bias ldmatrix | 6.4 / 4.6 % | 11.0 / 8.8 % | n/a | 7.1 / 1.3 % |
| epilogue stores (computation kept) | 3.3 / -1.9 % | -5.5 / 0.6 % | n/a | 3.1 / -4.5 % |
| K / V^T TMA loads, half the bias tile | within +-7 % (up to 20 % at hv L2048: zeroed operands also lower the power draw) | | | |

Caveats: a removal that zeroes or constants the data (loads, the exact path) also lowers the power draw (the card is power-capped), so those rows are upper bounds. The row-max tree + vote alone (tree computed, exact body never taken) costs
about as little as nothing; what the 17-30 % "no max machinery" ablation removed is the exact path (offset rise: 4 shuffles, `ex2f`, the 20 FMUL rescale of O) AND the data dependence it creates. The clean test of that
(`TA_F8K=7 TA_F8TAU10=15`: offset one octave lower so the exact path fires about half as often, data kept) gains 0-9 % on synthetic logits and -0.1..+3.7 % on real activations, with a 0.4-0.6 % output change: the exact path is not
what limits the real-activation core. PV is the largest part, the e4m3 conversion and the max machinery together are 7-17 % on real data.


Tried on the F8 core and REMOVED from the code (2026-11-15, `fp8/main`; no gain, default off, the removal changes no output byte: see "Verification"). Core-only, interleaved, 7 rounds, high-variance and flat logits, 4:32 / 4:16 / 8:32 at L512 / 1024 / 2048 and four real-activation layers at L1024 / L2048, against the F8 core with `TA_F8N24`:

| removed idea | what it did | core vs the F8 core | bit-identical |
|---|---|---|---|
| `TA_F8SW=1` (software-pipelined sub-step) | QK^T of the next sub-step under the e4m3 conversion | 4:32 -1.8 / -1.8 / -3.4 % hv, -1.2 .. -1.6 % flat; 8:32 +0.3 .. -4.4 %; 4:16 +0.1 .. +7.8 % | yes |
| `TA_F8SW=2` (16-channel heads only) | same, 16-channel heads | 4:16 +0.5 .. +5.4 %; 32-channel -2 .. +2 % | yes |
| `TA_F8SW=1 TA_F8SWN=2 / 4` | part of the conversion before the next QK^T | +5.7 .. +13.6 % | yes |
| `TA_F8QM=1` | row max of the vote in fp16, one vote | hv -1.4 .. -4.1 %, flat -1.3 .. +3.3 %, real +1.7 .. +9.8 % | yes |
| `TA_F8QM=2` | fp16 quad max in the exact path only | hv +1 .. +3.6 %, flat -1.7 .. +2.2 %, real -3.1 .. +0.6 % | yes |
| `TA_F8K=7 TA_F8TAU10=15` / `TA_F8K=6 TA_F8TAU10=25` | lower probability offset (exact path fires about half as often) | hv -3 .. +2 %, flat -3 .. -9 %, real -0.1 .. +3.7 % | no (0.4-0.8 % rel. L2 on real layers, 1.1-1.4 % on hv) |
| TriMul K1 `TMN_K1_WTMA` | per-warp plane TMA stores (no per-block warpgroup barrier), bf16 K1 | no number was kept (default off, never became faster than the shared-barrier store) | yes |
| TriMul K1 `TMN_K1_PIPE` (F8 compute, "schedule 6" split n128 block) | two n64 accumulators, MMAs of the next half under the epilogue | no number was kept (default off; the F8-compute K1 is itself experimental) | n/a |
| TriMul K3 `K3_MC` (D256) | weight chunks TMA-multicast over 2-CTA clusters | committed "untested on GPU", no number | n/a |

Kept: `TA_F8N24` (default on; heads of 16: PV as m64n24k32, V^T box of 16 rows, ones at rows 16-23): core -3.2 .. -4.9 % on 4 x 16, module 64:4:16 unfused hi -1.7 .. -2.8 %, bit-identical. The ablation bits `TA_ABL` of main's core stay (one mask, also used for main's variants).

### End level (FoldForge, 200 steps, TRUNK seeds, same protocol as the bf16 baseline; "inside" = max same-trunk-seed RMSD to bf16 <= max(1.1 x the largest bf16 seed-pair RMSD, 0.5 A))

| model | route | targets (FoldBench picks) | bf16 trunk seeds | FP8 same-seed RMSD vs bf16 | inside |
|---|---|---|---|---|---|
| Protenix v1 | hi fused / hi + lo fused (calibrated, no overflow) | 9 of 18 (the rest not run with FP8) | 4 | max 0.08-1.01 A (hi), 0.05-2.49 A (hi + lo) against bf16 seed-pair max 0.18-2.44 A | 9 / 9, 9 / 9 |
| Boltz-2 | hi fused / hi + lo fused | 7 | 4 | 0.16-4.13 A against bf16 0.21-4.83 A; **9c90: 12.6 A, ipTM 0.41 vs 0.71-0.79** (bf16 seeds agree to 0.2 A) | 6 / 7 (9c90 out, both planes) |
| OpenDDE | hi / hi + lo, dynamic V^T scale, unfused | 16 | 8 (4 on 8 targets) | max 0.03-4.97 A (hi), 0.04-2.04 A (hi + lo) against bf16 seed-pair max 0.12-6.41 A | 16 / 16, 16 / 16 |
| OpenDDE | hi / hi + lo, static per-site scale (the route `enable_fp8_pv_for` sets) | 16 | 8 (4 on 8 targets) | max 0.03-2.03 A (hi), 0.04-4.86 A (hi + lo) | 15 / 16 each (7g1v: 1.60 A against 1.26 A; its bf16 seeds already jump between two modes 1.25 A apart) |

ipTM: OpenDDE max |d ipTM| to bf16 0.000-0.04 (9c90 0.02-0.04, bf16 range 0.708-0.776); Protenix v1 <= 0.013; Boltz-2 <= 0.073 outside 9c90.
Boltz-2 9c90 is bistable under numerical noise (relative noise on the TriAttn output of 0.5 % flips 2 of 4 draws, 1 % 1 of 4, 2 % 4 of 4; the bf16 seeds agree because that path is deterministic), but a strict reading of the
acceptance rule fails there, so Boltz-2 is not whitelisted. Not applicable: Chai-1 (64-channel heads: no H100 CUDA inference core), ESMFold2 (no TriAttn); OpenFold3 / AF3: checked in the widened section below.

### End level, widened scope (2026-11-15, branch `fp8/e2e`; shipped route through `enable_fp8_pv_for`, S gates as shipped)

Harness: `benchmarks/fp8_e2e/` (not in the release tree: branch `fp8/3.0.0-all` e5ce31d0c; `probe.py` call sites, `e2e2.py` runs, `an3.py` / `bz.py` tables, `results.txt` raw tables). Same protocol as above (FoldForge `miniworld` backend, 200 steps, diffusion_seed 0, 1 sample, TRUNK seeds, acceptance = max same-trunk-seed RMSD
of FP8 vs bf16 <= max(1.1 x the largest bf16 trunk-seed-pair RMSD, 0.5 A)). Two differences to the tables above, both on purpose:

- **The bf16 baselines of this section were re-run in the current environment** (FoldForge 52a5877 + this engine). bf16 is bit-deterministic here (9c90 trunk seeds 0-3 rerun in a second job: 4 / 4 bytes equal), but it is NOT the bf16 of the earlier
  rounds (Boltz-2 9c90 seeds 0 and 1: 12.5 A apart from the earlier files; Protenix v1 8chi 0.05 A), so earlier bf16 files must not be mixed with new FP8 files.
- **The earlier end-level runs removed every S gate** (targets of 52-316 residues). With the shipped gates the 384:12:32 (OpenDDE, from L1024) and 256:8:32 (from L768) cores never engage at those sizes, so those runs measured the FP8 core where it is not shipped.
  The runs below use the shipped route, with targets large enough to engage the gates (residue counts 660-1130; the token bucket is the next multiple of 128, the template stacks of every model engage from L256).
- The static scale is calibrated on the first eager call of each site and kept for the whole process (all targets and seeds of a job share it: the stress case of the contract). Targets whose bf16 trunk seeds already differ by > 5 A
  are weak probes (the 1.1 x rule then accepts almost anything); they are listed, but the conclusions rest on the others.

#### Call sites per model (probe: one trunk run of a 316-residue target, bucket 384, every S gate removed so that all sites show; counts are core calls of one `run_trunk`)

| model | sites (d_pair : heads : channels, kind) | calls | engaged by the shipped route |
|---|---|---|---|
| Protenix v1 | 128:4:32 fused; 64:4:32 front-made (templates) | 1040; 160 | whitelist "fused": only the 128 sites; "unfused" adds the templates (gate L256) |
| Protenix v2 | 256:8:32 front-made (trunk; gate L768); 64:2:32 front-made (templates, gate L256) | 1040; 160 | "unfused" only (no fused site in v2) |
| OpenDDE | 384:12:32 cuBLAS front + standalone pass (trunk; gate L1024); 64:2:32 front-made (templates); 12-head sites at ~1.67 x L (8 calls at 1280 for L768) | 1040; 160; 8 | "unfused" |
| Boltz-2 | 128:4:32 fused; 64:4:32 front-made (templates) | 1360; 160 | not whitelisted |
| OpenFold3, AF3 | 128:4:32 fused; 64:4:16 front-made (templates, gate L256) | 1040; 160 | not whitelisted (candidates below) |
| RoseTTAFold3 | 128:4:32 fused only | 1040 | not whitelisted |
| IntelliFold2 | 256:8:32 front-made, 4 sites only (160 calls; no trunk site on this path) | 160 | not whitelisted |
| Chai-1 | 64:4:32 front-made, 160 calls only (trunk not on this path) | 160 | not applicable |
| ESMFold2 | no template path, no TriAttn site | 0 | not applicable |

A model whose template inputs are absent for a target runs bit-identical to bf16 below its trunk gate (Protenix v2 / IntelliFold2 8gn4 / 8ad9 / 8ee9: 0.00 A, nothing engaged).

#### Results per model (n = trunk seeds per target; RMSD in A; "bf16 max" = largest bf16 pair RMSD; "FP8 max" = largest same-seed FP8 vs bf16)

| model, route | targets | bf16 seeds | bf16 max | FP8 max | inside | max abs d ipTM |
|---|---|---|---|---|---|---|
| Protenix v1, fused + unfused (templates, L256 / 384) | 9 small (7g1v 7y37 7yey 8ad9 8chi 8d78 8ee9 8ey7 8fb3) | 4 | 0.20-2.51 | 0.07-1.03 | 9 / 9 | 0.015 |
| Protenix v1, fused + unfused, large | 7zau 8id0 8fxv (L768 / 896 / 1152) | 4 | 4.35-22.69 (7zau an unstable probe) | 0.37-12.83 | 3 / 3 | 0.036 |
| OpenDDE, shipped, L >= 1024 (gate 1024 engaged) | 8d3i 8fxv 8ifx | 4 | 1.30-8.65 | 0.19-6.44 | 3 / 3 | 0.003 |
| OpenDDE, gate 768 for 384:12:32 (see below) | 16 targets of 660-812 residues | 4 (8 on 8scr) | 0.56-29.74 | 0.10-28.93 | 16 / 16 | 0.112 (8dn7, bf16 spread 29.7 A) |
| Protenix v2, "unfused" (256:8:32 from L768, templates from L256) | 11: 7y37 8ad9 8ee9 8gn4 8id0 8vdn 8x3q 8p7u 8ifx 7zau 8fxv | 4 (12 on 7y37) | 0.24-15.13 | 0.00-12.95 | 11 / 11 | 0.179 (8fxv, unstable) |
| OpenFold3, fused + unfused (templates 64:4:16) | 8ad9 8ee9 7y37 8gn4 | 4 | 0.30-2.18 | 0.33-0.55 | 4 / 4 | 0.008 |
| AF3, same | 8ad9 8ee9 7y37 8gn4 | 4 | 0.06-0.69 | 0.04-0.55 | 4 / 4 | 0.011 |
| RoseTTAFold3, fused | 8ad9 8ee9 7y37 8gn4 | 4 | 1.49-2.46 | 0.46-2.24 | 3 / 4 (7y37: 2.24 against 1.49 x 1.1; its ipTM is 0.10, a failed prediction, bad probe) | 0.002 |
| IntelliFold2, fused + unfused (4 template sites, 256:8:32) | 8id0 8j2w 8vdn 8x3q (+ 8ad9 8ee9 nothing engaged) | 4 | 0.22-4.73 | 0.00-0.76 | 6 / 6 | 0.001 |

Per target rows are in `benchmarks/fp8_e2e/results.txt`. Worth knowing: Protenix v2 7y37 flips once in 12 FP8 trunk seeds (4.26 A at seed 1); its bf16 trunk seeds have a second mode too (4.31 A pair maximum over 12 seeds). With 4 bf16 seeds the same
target reads 0.21 A and the FP8 flip is OUT (4.26 A): the verdict depends on the number of bf16 seeds, which is why 7y37 was run with 12.

#### OpenDDE gate 768 for 384:12:32 (S gate of the standalone V^T pass; `FP8_PV_MIN_S[32]` and `FP8_PV_FRONT_MIN_S[(384, 12, 32)]` = 768, evaluation override only, the shipped gate is unchanged)

16 targets in token buckets 768 / 896 (residues 660-812: 7zau 8cjd 8j2w 8auz 8id0 8bf4 8dn7 8vvl 8vdn 8phb 8scr 8p7u 8avz 7u9i 8x3q 9jf5), 4 trunk seeds, FP8 core engaged on the 12-head sites (1040 calls per trunk) that are bf16-identical under the shipped gate:
16 / 16 inside (8scr: with 4 bf16 seeds 1.38 A against 1.05 x 1.1, OUT; with 8 bf16 / 8 FP8 seeds 2.31 A against 3.04 x 1.1, inside). The nine targets whose bf16 trunk seeds agree to < 4 A
(8id0 0.10 A against 1.14, 8avz 0.30 / 0.56, 8x3q 0.38 / 0.93, 8scr (8 seeds), 7u9i 1.00 / 1.76, 8j2w 0.70 / 2.70, 8p7u 0.80 / 3.05, 8vdn 0.11 / 2.71, 8vvl 2.92 / 3.75) are all inside; no saturation flag, no redo, peak below 0.43 of the e4m3 range.
Decision: the numbers allow 768 (OpenDDE gate 768: 16 / 16 inside, no saturation), but the gain is -2 .. -3 % of the module at L768 and the shipped code has one gate per channel count (`FP8_PV_MIN_S` is keyed by channels, so a 384:12:32-only gate needs a per-config key in `_F8Core`):
not changed here (optional, low priority).

#### Boltz-2 9c90 (data for the whitelist decision; Boltz-2 is NOT whitelisted)

24 trunk seeds, bf16 and FP8 (fused 128:4:32 hi plane, as `enable_fp8_pv_for` would set it); bf16 clusters at 3 A: 5 modes with 8 / 6 / 7 / 2 / 1 seeds, **bf16 itself spans 15.3 A between trunk seeds, ipTM 0.40-0.85** (12 seeds: 15.31 A, 0.407-0.813); the earlier "4 seeds agree
to 0.2 A" does not reproduce in the current environment (seeds 0 and 1 are in the low-ipTM mode now). Same-seed FP8 vs bf16: 9 of 24 trunk seeds (37 %) end in a different bf16 mode (flips, 3.8-14.1 A), 15 of 24 stay (0.08-0.92 A);
by the acceptance rule FP8 is inside (13.68 A max against 1.1 x 15.31 A). Population view: the fraction of trunk seeds in the low-ipTM mode (ipTM < 0.6) is 9 / 24 for bf16 and 8 / 24 for FP8 (modes 0-2 hold 8 / 6 / 7 bf16 and 8 / 8 / 7 FP8 seeds, mode 3 two against one, mode 4 one against zero).
With the templates too (fused + unfused, 12 seeds) 4 / 12 flip. Both routes leave the mode population unchanged, but a given trunk seed does not keep its mode (as with 0.5-2 % noise on the TriAttn output, see limits). 9c90 is the only Boltz-2 target of the 7 that fails the strict reading;
whether a target whose own bf16 trunk seeds fall into five modes is a valid probe is the open decision.

#### Scale / guard contract in these runs

288 FP8 cases (Protenix v1 48, Protenix v2 52, OpenDDE 80 (68 + 12), Boltz-2 36, OpenFold3 16, AF3 16, RoseTTAFold3 16, IntelliFold2 24): `take_overflow()` returned False after every forward, 0 redos (`rescale_all` never called). Per-call peak of the V^T planes
as a fraction of the e4m3 range (448; a calibration call is exactly 1.0, excluded; the calibration leaves a 4 x headroom, i.e. 0.25 at the calibration input), max over all calls / worst p99:

| model | sites | log2(scale) range | max peak | max p99 |
|---|---|---|---|---|
| Protenix v1 | 104 fused + 4 templates | -4 .. -1 | 0.643 (L256) | 0.500 |
| Protenix v2 | 104 (256:8:32) + 4 | -10 .. -2 | **0.857** (256:8:32, L768) | 0.571 |
| OpenDDE | 112-115 (384:12:32) + 4 | -9 .. -1 (and -99, below) | 0.429 | 0.393 |
| Boltz-2 (data) | 136 + 4 | -4 .. -2 | 0.429 | 0.321 |
| OpenFold3 / AF3 | 136 + 4 | -4 .. -1 | 0.393 / 0.429 | 0.321 / 0.357 |
| RoseTTAFold3 | 136 | -4 .. -2 | 0.500 | 0.412 |
| IntelliFold2 | 4 | -4 .. -2 | **0.929** (L896, 256:8:32) | 0.929 |

Findings for the contract: (1) the 4 x headroom held in every run, but two sites reach 0.86-0.93 of the range at a later call of a process-wide scale taken on another target (no flag: the planes are 8-14 % below the cap); a target with a 10 % larger V peak would
have saturated and needed the redo. (2) In the OpenDDE runs at L >= 768 some sites calibrate at **log2(scale) = -99** (35 of 1380 heads of the 12-head sites; probably the 3 extra 12-head sites that appear only at ~1.67 x L, whose V is then zero at the calibration call (not isolated); `vscale_from_amax` clamps amax at 1e-30). No flag, so V stays zero there, but
a head calibrated at exactly zero saturates on any later non-zero V, and `rescale_all(4)` x 3 (the redo limit) cannot recover from 2^-99: a floor on the scale (or re-calibrating a head with amax == 0) is the guard to add; not changed here.

## OuterProductMean forward: e4m3 grouped product (`kernels/integrations/opm_fp8.py`, branch `fp8x/opm`, 2026-11-17)

Scope: inference of the `opm_train` forms (groups 1, d_hidden 32, (d_msa, d_pair) in `opm_train.SHAPES`, batch 1, either normalisation order,
`count_offset`, the pair residual), H100. Training, Chai-1 groups (measured 0.7 % of Chai-1's kernel time at S 1024: see "Chai-1's option block" under Token DiT) and the RoseTTAFold3 biased inference (`composed.biased_wide`) stay bf16.

### Where the time goes and where e4m3 can go (default 2.5.12 route, measured)

Stage times (the `benchmarks/opm_fp8/` scripts and raw rows are not in the release tree: branch `fp8/3.0.0-all` e5ce31d0c): `benchmarks/opm_fp8/stages.py` (raw `benchmarks/opm_fp8/raw/st1.jsonl`): each stage alone on the same operands, captured in a CUDA graph, interleaved,
cool = 2 s / 1 s idle then a ~40 ms burst (median of 3), sustained = after 1 s of back-to-back replays (mean of 3). Bound classes are from
the counters of benchmarks/opm_win (ncu: prologue DRAM 76 % of peak; cuBLAS product ~99 % of the tensor peak per cycle, power-capped) and
from the achieved rates below; the e4m3 product was not put under ncu (cuBLAS).

L 768, S 16384, d_msa 64 / d_pair 128 (AF3 / Boltz widths; default route past depth 4096 = wide prologue + cuBLAS product + count table + epilogue):

| stage | GEMM / work | FLOP share | bytes (bf16 default) | default ms cool / sus | e4m3 operand | upper bound | FP8 ms cool / sus |
|---|---|---|---|---|---|---|---|
| prologue (LN + W_l, W_r on wgmma + mask) | [S N, 64] x [64, 64] | 0.5 % | in 1.6 GB, out A2 + BT 1.6 GB | 1.33 / 1.33 (DRAM-bound) | A2 / BT **stored** as e4m3 (out 0.8 GB) | x 0.75 (bytes) | 0.93 / 0.99 |
| pair counts | fp32-output GEMM of the 0 / 1 mask | - | 25 MB | 0.04 | none (must stay exact) | - | 0.04 |
| grouped product O = A2 BT^T | M = N = 32 L = 24576, K = S = 16384 | 98.7 % | in 1.6 GB, O 1.2 GB | 24.46 / 27.40 (810 TF, tensor / power-bound) | **both operands e4m3**, fp32 accumulation, bf16 O | x 0.5 (e4m3 peak = 2 x bf16) | 13.93 / 15.46 (1420 TF) |
| epilogue (permute, / n, W_o 1024 -> 128, bias) | [L^2, 1024] x [1024, 128] | 0.8 % | O 1.2 GB in, z 151 MB out | 0.46 / 0.48 | O as e4m3 (cuBLASLt D e4m3): -0.6 GB read + write; W_o e4m3: no gain (not tensor-bound) | ~ -0.3 ms (2 %) | not done |
| whole module (speed.py) | | | | 25.97 / 28.83 | | | 15.06 / 15.85 |

Other shapes (cool ms, default -> FP8 piece): L 384 S 4096 product 1.53 -> 0.88, prologue 0.160 -> 0.121, epilogue 0.126 (fused default kernel 1.74);
L 256 S 1024 product 0.190 -> 0.125, prologue 0.031 -> 0.027, epilogue 0.058 (fused 0.228); L 768 S 2048 product 3.12 -> 1.89, epilogue 0.455 (fused 3.68);
d_msa 128 / d_pair 256 L 768 S 4096 product 6.09 -> 3.55, prologue 0.454 -> 0.350, epilogue 0.651; d_msa 256 / d_pair 512 L 384 S 4096 product
1.53 -> 0.87, prologue 0.354 -> 0.342 (its LayerNorm is the cost, not the store), epilogue 0.322.

Placements:

1. **Implemented: the grouped product on e4m3 operands** (the only stage that matters: 99 % of the FLOPs, 94 % of the time at depth 16384,
   and the share grows with S). The FP8 build of the wide prologue (`-DMINIWORLD_OPM_FP8`, `opm_train._ext(fp8=True)`; `#if`-guarded in
   `csrc/opm_epilogue.cu`, so the default builds compile the same code) multiplies the fp32 projection accumulator by a static power-of-two
   `1 / scale` per operand, casts with `__nv_cvt_float2_to_fp8x2(..., __NV_SATFINITE, __NV_E4M3)` into a 128 B-swizzled byte tile and stores it
   with one TMA store per operand (half the default's A2 / BT bytes); it keeps the max |scaled value| of each operand (one `atomicMax` per warp
   per CTA) for the guard. The product is a cuBLASLt e4m3 TN GEMM (`opm_lt_f8`, fp32 accumulation with cuBLAS's periodic promotion, A / B
   scale pointers = the dequantisation scales, so a captured graph follows an in-place rescale), bf16 O exactly where the default product route
   writes it; the count table (exact) and the epilogue are the default's. Below depth 4096 the FP8 route replaces the fused bf16 kernel by
   this composition (an e4m3 product + O round trip beats the fused bf16 kernel from S 1024 / L 256, see the gate).
2. Not done: O as e4m3 (a third quantisation for <= 2 % of the call at depth 16384, more at S 1024-2048 where the epilogue is 15-25 % of the FP8
   route), W_o in e4m3 (the epilogue is not tensor-bound), the fused kernel on e4m3 wgmma (k32; would remove the O round trip at S <= 4096: upper
   bound = the epilogue + O write, 0.06-0.46 ms of 0.24-2.5 ms; operand layouts, TMA boxes and the k-permuted Opair all change -- a kernel
   rewrite, out of this task's scope), the prologue's projections in e4m3 (0.5 % of the FLOPs, DRAM-bound).

### Enable

| env (read at import; the module globals can be set at run time) | effect |
|---|---|
| `MINIWORLD_OPM_FP8=bound` | scales from the weights alone: \|LN(x)_k\| <= sqrt(d_msa - 1) \|gamma_k\| + \|beta_k\| for every input, so max_c sum_k \|W_ck\| (sqrt(d_msa - 1) \|gamma_k\| + \|beta_k\|) (+ \|projection bias\|), x (1 + 2^-6), to the next power of two, bounds every scaled value by 448: the cast cannot saturate, no calibration call, no guard needed. **Recommended.** |
| `MINIWORLD_OPM_FP8=calib` | the docs/fp8.md convention: the first eager call of a module measures max \|A2\|, max \|BT\| (one extra FP8 prologue at scale 1), scale = next power of two x `HEADROOM` (16); weakref-finalized cache keyed by `to_left.weight`; a capture with no calibration takes the default route |
| `MINIWORLD_OPM_FP8_GUARD` | `sync` (default, eager: an overflowing call rescales and redoes itself), `lazy` (no host sync; capturable): `opm_fp8.overflow_peak()` once per forward, if > 448 `opm_fp8.rescale_all(peak)` and redo (same contract as TriMul) |
| `MINIWORLD_OPM_FP8_MINS=<s>` | one depth gate instead of the measured shape rule (`_gate`: padded S >= 2048, or S >= 1024 at padded L >= 256); 0 = no gate |
| `MINIWORLD_OPM_FP8_FASTACC=1` | cuBLASLt fast accumulation: product 4-7 % faster, but max error 3.5-4.5 % at depth 16384 (K = 16384 without promotion): **not recommended** |

With the random-weight modules here `bound` and `calib` chose the same scales (2^-2 at d_msa 64, 2^-1 at 128, 2^0 at 256; scaled amax 21-26,
11, 5-6 of 448) and give identical outputs; `bound` needs no calibration call and no guard. No saturation in any run (peak <= 26).

### Module time vs the 2.5.12 default (H100, CUDA graph, same process, interleaved, same-cubin control)

`benchmarks/opm_fp8/speed.py` (raw `benchmarks/opm_fp8/raw/sp1.jsonl`, `sp2.jsonl`, `sp3.jsonl`): the module (`master_weights=True`, ragged MSA: the last S / 8 rows
cover half the tokens), graphs of def / ctl (default again) / f8b (bound) / f8bf (bound + fast accumulation), every gate removed. cool = 2 s idle,
burst of ~40 ms per route, 3 reps median; sustained = 1 s back-to-back interleaved, then 3 blocks of ~60 ms per route, mean. Cells: FP8 / default,
`cool / sustained`; the same-cubin control's cool ratios are 0.99-1.10 (single-call bursts at the 26-190 ms shapes), sustained 0.93-1.03.

d_msa 64 / d_pair 128, Boltz order (2.5.12 default ms cool in brackets):

| L \ S | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|
| 128 | 0.93 / 0.94 (0.061) | 1.34 / 1.35 (0.050) | 1.05 / 1.04 (0.080) | 0.93 / 0.87 (0.137) | 0.76 / 0.71 (0.259) | | 0.66 / 0.64 (0.928) |
| 256 | 0.98 / 0.98 (0.153) | 1.12 / 1.06 (0.156) | 0.93 / 0.85 (0.256) | 0.75 / 0.70 (0.464) | 0.65 / 0.62 (0.870) | 0.63 / 0.61 (1.64) | 0.61 / 0.57 (3.21) |
| 384 | | 1.10 / 1.01 (0.316) | 0.85 / 0.80 (0.539) | 0.69 / 0.66 (0.991) | 0.62 / 0.58 (1.87) | 0.62 / 0.58 (3.44) | 0.60 / 0.56 (6.79) |
| 768 | | 1.04 / 0.97 (1.17) | 0.82 / 0.74 (2.00) | 0.68 / 0.62 (3.67) | 0.61 / 0.56 (6.92) | 0.59 / 0.54 (13.1) | 0.58 / 0.55 (26.0) |
| 1024 | | | | | 0.49 / 0.48 (15.1) | | 0.57 / 0.53 (47.9) |
| 2048 | | | | | 0.54 / 0.49 (53.8) | | 0.58 / 0.55 (191.8) |

(L 192 S 1024, padded to 256: 0.48 / 0.48 against a 0.317 ms default -- the padded default is slower than L 256's; L 1024 S 4096: the fused
default kernel takes 15.1 ms, 2.2x L 768's, so the FP8 composition is 0.49.)

AF3 order and the wide widths:

| width, order | L | S 1024 | S 2048 | S 4096 | S 16384 |
|---|---|---|---|---|---|
| 64 / 128, AF3 | 384 | | 0.70 / 0.70 | | |
| 64 / 128, AF3 | 768 | | | | 0.59 / 0.55 |
| 128 / 256, AF3 (Protenix-v2) | 384 | 0.82 / 0.78 | | 0.67 / 0.63 | 0.60 / 0.59 |
| 128 / 256, AF3 | 768 | 0.79 / 0.73 | | | 0.58 / 0.51 |
| 128 / 384, AF3 (ESMFold2) | 384 | 0.84 / 0.79 | | | 0.60 / 0.59 |
| 128 / 384, AF3 | 768 | | | 0.65 / 0.59 | |
| 256 / 512, Boltz order (IntelliFold2) | 384 | 0.86 / 0.83 | | | 0.65 / 0.62 |
| 256 / 512 | 768 | | | 0.69 / 0.62 | |

Gate (`_gate`): FP8 from padded S 2048 at every L, from S 1024 at padded L >= 256; below, the default runs (L 128 S 1024 +5 %, S 512 +0..34 %).

### Against Anthropic at large depth (benchmarks/opm_win fair protocol)

`fair.py` (opm_win) run with `MINIWORLD_OPM_FP8=bound` (raw `benchmarks/opm_fp8/raw/fair_fp8.jsonl`); Anthropic and the default from opm_win `raw/v3_kit*` /
`raw/v3_after*` (2.5.11 = 2.5.12 OPM); lead = (PyTorch_engine / CUDA) / (PyTorch_kit / Anthropic), > 1: the engine is faster (`benchmarks/opm_fp8/lead.py`):

| row | L | S | Anthropic ms | default ms | FP8 ms | lead default cool / sus | lead FP8 cool / sus |
|---|---|---|---|---|---|---|---|
| 55 (AF3) | 256 | 8192 | 1.73 | 1.69 | 1.03 | 1.016 / 1.052 | 1.680 / 1.657 |
| 55 | 256 | 16384 | 3.32 | 3.29 | 1.99 | 1.004 / 1.050 | 1.675 / 1.687 |
| 55 | 384 | 8192 | 3.67 | 3.52 | 2.12 | 1.033 / 1.093 | 1.721 / 1.711 |
| 55 | 384 | 16384 | 7.08 | 6.90 | 4.14 | 1.021 / 1.040 | 1.708 / 1.797 |
| 55 | 512 | 8192 | 6.46 | 6.14 | 3.66 | 1.048 / 1.086 | 1.765 / 1.885 |
| 55 | 512 | 16384 | 12.51 | 12.09 | 7.13 | 1.027 / 1.035 | 1.749 / 1.802 |
| 55 | 768 | 8192 | 13.93 | 13.26 | 7.93 | 1.044 / 1.046 | 1.754 / 1.760 |
| 55 | 768 | 16384 | 27.14 | 26.30 | 15.54 | 1.024 / 1.119 | 1.746 / 1.866 |
| 57 (Boltz) | 256 | 8192 | 2.01 | 1.67 | 1.03 | 1.195 / 1.179 | 1.956 / 1.867 |
| 57 | 256 | 16384 | 3.65 | 3.25 | 1.99 | 1.120 / 1.150 | 1.841 / 1.818 |
| 57 | 384 | 8192 | 4.14 | 3.47 | 2.12 | 1.181 / 1.244 | 1.946 / 1.938 |
| 57 | 384 | 16384 | 7.59 | 6.82 | 4.14 | 1.103 / 1.170 | 1.835 / 1.925 |
| 57 | 512 | 8192 | 7.13 | 6.06 | 3.66 | 1.170 / 1.220 | 1.953 / 2.008 |
| 57 | 512 | 16384 | 13.16 | 11.94 | 7.15 | 1.094 / 1.139 | 1.842 / 1.926 |
| 57 | 768 | 8192 | 15.50 | 13.17 | 7.92 | 1.170 / 1.242 | 1.965 / 2.107 |
| 57 | 768 | 16384 | 28.49 | 26.16 | 15.53 | 1.087 / 1.176 | 1.838 / 1.942 |

(The Anthropic kits' products are bf16 cuBLAS: this is FP8 against bf16, not a like-for-like kernel comparison.)

### Error against fp64 PyTorch

`benchmarks/opm_fp8/err.py` (raw `benchmarks/opm_fp8/raw/er1.jsonl`): the module in float64 on the PyTorch backend, same inputs; rms = rms(y - y64) / rms(y64),
max = max|y - y64| / max|y64| (the engine judge metric; tol = max(err_pytorch, 8e-3)). `row:R` = the requirements row's construction
(`measure.config`); the others random weights (W ~ N(0, 1/fan_in), LN weight 1 +- 0.1, biases +- 0.1) with a ragged MSA. All outputs finite,
no saturation.

| case | PyTorch bf16 max | default rms / max | FP8 bound rms / max (/ tol) | FP8 calib | FP8 bound + fast acc. max (/ tol) |
|---|---|---|---|---|---|
| row 55, L 384, S 1024 | 4.4e-3 | 2.4e-3 / 4.4e-3 | 9.9e-3 / 1.37e-2 (1.72) | = bound | 1.37e-2 (1.72) |
| row 57, L 384, S 1024 | 5.8e-3 | 2.4e-3 / 5.8e-3 | 5.7e-3 / 1.05e-2 (1.32) | = bound | 1.05e-2 (1.32) |
| row 63 (count_offset), L 384, S 1024 | 3.1e-3 | 2.4e-3 / 3.1e-3 | 2.4e-3 / 3.1e-3 (0.38) | = bound | (0.38) |
| 64 / 128 Boltz, L 384, S 1024 | 5.2e-3 | 2.6e-3 / 5.2e-3 | 1.11e-2 / 1.15e-2 (1.44) | = bound | (1.44) |
| 64 / 128 AF3, L 384, S 4096 | 5.5e-3 | 4.7e-3 / 4.4e-3 | 3.01e-2 / 8.2e-3 (1.03) | = bound | 1.34e-2 (1.68) |
| 64 / 128 Boltz, L 768, S 2048 | 6.2e-3 | 2.4e-3 / 5.9e-3 | 8.1e-3 / 9.5e-3 (1.18) | = bound | 9.8e-3 (1.23) |
| 64 / 128 Boltz, L 256, S 16384 | 6.2e-3 | 2.3e-3 / 5.8e-3 | 3.6e-3 / 6.2e-3 (0.78) | = bound | 3.46e-2 (4.32) |
| 64 / 128 Boltz, L 768, S 16384 | 6.2e-3 | 2.3e-3 / 5.7e-3 | 3.6e-3 / 6.9e-3 (0.86) | = bound | 3.68e-2 (4.59) |
| 64 / 128 AF3, L 768, S 16384 | 6.3e-3 | 4.4e-3 / 4.7e-3 | 2.22e-2 / 6.3e-3 (0.79) | = bound | 4.46e-2 (5.58) |
| 64 / 128 AF3, L 1024, S 8192 | 5.4e-3 | 4.6e-3 / 4.9e-3 | 2.73e-2 / 7.2e-3 (0.90) | = bound | 2.21e-2 (2.76) |
| 128 / 256 AF3, L 384, S 4096 | 6.1e-3 | 4.8e-3 / 4.2e-3 | 3.48e-2 / 1.01e-2 (1.26) | = bound | 1.19e-2 (1.49) |
| 128 / 384 AF3, L 384, S 4096 | 7.1e-3 | 4.8e-3 / 4.7e-3 | 3.49e-2 / 1.01e-2 (1.26) | = bound | 1.17e-2 (1.46) |
| 256 / 512 Boltz, L 384, S 2048 | 5.5e-3 | 2.6e-3 / 5.1e-3 | 9.1e-3 / 1.33e-2 (1.66) | = bound | 1.31e-2 (1.64) |

Reading: e4m3 rounds each A2 / BT value with 2.7 % rms (measured; one e4m3 rounding: the same rms as torch's cast of the bf16 A2 / BT, the kernel casting the fp32 accumulator directly, no
double rounding), O with 3.4 % rms per element; the projection and the mean over the depth average it down. Judge metric 0.6-1.4 %:
inside tol at depth >= 8192 (the depth averaging wins), 1.03-1.72 x tol at depth 1024-4096. The AF3-order rms is 2-3.5 % because there the
bias is divided by n too, so y64's rms is small (the max metric stays 0.6-1 %). Row 63: the FP8 route ran (scales cached, 9 calls) but its
output equals the default's to 1e-8 rms: in that row's construction the product term is negligible against the bias / n. Fast accumulation:
max error 3.5-5.6 % at depth 8192-16384 (not recommended).

### Bit-identity of the default and tests

- Default route (FP8 off) bit-identical to the base (91a9241fb = 2.5.12 + the FP8 routes): `benchmarks/opm_fp8/bit.py` (sha1 of the output, and
  in training of y and every gradient), base tree and this tree in separate processes, 13 / 13 cases identical: d_msa 64 / d_pair 128 at
  (L, S) = (256, 1024) fused + TMA prologue, (384, 2048), (768, 4096), (256, 8192) product route, (200, 3000) padded + pair residual, (384, 1024)
  AF3 order + count_offset, (128, 512), (256, 8192) AF3 + residual; 128 / 256 (256, 1024), 128 / 384 (256, 2048), 256 / 512 (128, 1024);
  training forward + backward at 64 / 128 (256, 1024) and 128 / 256 (128, 512). The FP8 kernel code is `#if defined(MINIWORLD_OPM_FP8)`
  only (the default builds get the same code; their cache keys change with the source text).
- `tests/gpu/test_opm_fp8.py` (new, 16 cases: off by default, operands = one e4m3 rounding of the default A2 / BT, module error budget for both
  policies / both orders / the row-chunked product / the residual / three widths, sync and lazy guard + `rescale_all`, graph replay = eager):
  16 / 16 pass. `tests/gpu/test_opm_deep_product_route.py`: pass. `tests/cpu/test_capture_api.py`: the three `opm_fp8` dicts listed; the
  test still fails on the TriAttn / TriMul FP8 dicts already unlisted on the base branch (not touched here). ruff check / format, pyright: clean
  on the touched Python files. (The `dev code-graph` CLI is not in this tree; the test files were chosen by import of `opm_train` / `opm_fp8`.)

### Recommendation

- `MINIWORLD_OPM_FP8=bound` is the OPM FP8 route to offer: 1.7-2.0x faster than 2.5.12 at depth 4096-16384 on every width (FP8 / default
  0.49-0.65), 1.07-1.6x at depth 1024-2048 from L 256 (L 128 from S 2048); against Anthropic at depth 8192 / 16384 the lead goes from 1.00-1.24 to 1.66-2.11.
  No calibration, no guard, graph-safe. Error: judge metric 0.6-1.4 % (default 0.3-0.6 %), inside the engine tol at depth >= 8192, up to
  1.72 x tol at depth 1024-4096: the owner's call (end-level check on a deep-MSA model before any default).
- Not worth it: fast accumulation (accuracy), FP8 at depth <= 512 (slower), e4m3 O / W_o (<= 2 %).
- Open: an e4m3 version of the fused kernel for S <= 2048 (removes the O round trip; the remaining gain there is the epilogue's share, 15-25 %).

## PWA forward: e4m3 key contraction (`kernels/integrations/pwa_fp8.py`, branch `fp8x/pwa`, 2026-11-17)

Scope: `MSAPairWeightedAveraging`, H100 `pwa_train` inference (d_msa 64, 8 heads x 32: Boltz-2 / RF3 / MiniWorld at d_pair 128, Chai-1 at d_pair 256),
one sample. The `pwa_wide` widths (8 x 8 / 8 x 16: AF3 family) have their own route, below; not served: BIG 256 x 8 x 32, training, B200. Off by default; with it off the
module output is the 2.5.12 bytes (below).

Route: `pair_fwd3` (default pair kernel, w bf16) -> `w_quant` (e4m3 of 256 w) -> `ln_vt8` (y = `ln_vg`'s bytes, v as e4m3 **K-major** `vT [H][S*C][N]`)
-> cuBLASLt e4m3 batched GEMM (`o^T = vT^T w^T`, bf16 o head-major, fp32 promotion) -> `pwa_gateout` (default gate / out pass, unchanged).
Hopper's e4m3 MMA takes K-major operands only and the contraction runs over the key index j, so v must be written j-contiguous: `ln_vt8`
(`csrc/pwa_vt8.cu`, new) is `ln_vg` with the projection turned around (`vT tile = Wv . y^T`, one m64n64k16 wgmma per 64 hc, the [8 h][32 c][64 n]
e4m3 tile out by one TMA store) -- no transpose pass. Scales: w is a probability, fixed scale 2^8 (no calibration, cannot saturate); v takes a
static power-of-two scale (`vscale_from_amax` of the TriAttn route, 16 x headroom over the first eager call's absmax, keyed by the value weight,
weakref-dropped). Both scales are device scalars passed as cuBLASLt's A / B scale pointers (no rescale pass; `rescale_all` updates them in
place, captured graphs follow). Guard: `ln_vt8` records max |v / scale|; same contract as TriAttn / TriMul (`take_overflow()` -> `rescale_all()`
-> redo; `GUARD=sync` does it per call, `lazy` for CUDA graphs).

| env | effect |
|---|---|
| `MINIWORLD_PWA_FP8=1` (or `pwa_fp8.ENABLED = True`) | the route above |
| `MINIWORLD_PWA_FP8_GUARD` | `sync` (default: one host sync per call, redo on saturation), `lazy` (graphs; the caller checks), `off` |
| `MINIWORLD_PWA_FP8_MIN_N` / `_MIN_S` | gates (default N >= 384, and S >= 1024 below N 512); 0 = no gate |
| `MINIWORLD_PWA_FP8_HEADROOM` | v scale headroom (16) |
| `MINIWORLD_PWA_FP8_FASTACC=1` | cuBLASLt fast accumulation (module 1-3 % faster at N >= 768) |

### Where FP8 can go: stages of the 2.5.12 H100 inference route (S1024 N768, d_pair 128)

Times: CUDA graph per stage, median (`fp8x-pwa/scripts/stageprof.py`, `res/prof.jsonl`); counters: ncu (`res/ncu_1024:768_v1.csv`, base clocks).
Module 1.064 ms; N >= 512 runs the split route (cuBLAS contraction + `pwa_gateout`), N < 512 the fused `pwa_fwd3`.

| stage | GEMM / work | GFLOP | DRAM R / W (MB) | time ms | DRAM % / tensor % | bound | e4m3 placement | upper bound | measured |
|---|---|---|---|---|---|---|---|---|---|
| `pair_fwd3` | LN_z, [N^2, 128] x [128, 8], softmax | 1.2 | 152 / 8.5 | 0.066 (6 %) | 70 / 4 | memory (z read) | z is the module input (bf16 from the pair stack); w out as e4m3 = `w_quant`'s 7 us | ~1 % | - |
| `ln_vg` | LN_m, [S N, 64] x [64, 256] | 25.8 | 101 / 484 | 0.226 (21 %) | 80 / 16 | memory (v write = 83 % of bytes) | v stored e4m3 (and K-major) | 0.65x of the stage (bytes) | `ln_vt8` 0.157 (0.70x), R/W 101 / 281 |
| contraction | per head [N, N] x [N, S C] (cuBLAS bf16, 645 TFLOP/s) | 309 | 413 / 387 | 0.479 (45 %) | 46 / 81 | tensor | w and v e4m3 | 0.5x (2x MMA rate; bytes floor 0.18 ms) | 0.304 (0.63x), fast acc 0.282 |
| `pwa_gateout` | [S N, 64] x [64, 256] gate, [S N, 256] x [256, 64] out | 51.5 | 604 / 95 | 0.322 (30 %) | 61 / 20 (SM 43) | neither (latency) | o as e4m3 (halves the o read) | o read is 2/3 of the bytes, but not bandwidth-bound | tried: no gain (below) |

Sum of the upper bounds: ~0.73x of the module; measured 0.795x (cool) at this shape. What is left after FP8 is mostly `pwa_gateout` (37 % of the FP8 route).

### Module time against the 2.5.12 default (H100 80GB HBM3, 700 W cap)

Protocol as the TriAttn / TriMul tables above (`fp8x-pwa/scripts/speed.py`, raw rows `res/v1.jsonl`): one process per group, module (`master_weights=True`,
one sample, all-ones mask) captured in three CUDA graphs -- 2.5.12 default, default again (same-cubin control), FP8 (lazy guard, every gate removed,
static scale from two eager calls; no saturation in any case). Cells `cool / sustained`, FP8 / default. Same-cubin control over all rows 0.987 .. 1.007.

| S \ N | 128 | 256 | 384 | 512 | 640 | 768 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|---|---|---|---|
| 128 | | | 1.084 / 1.082 | | | 0.927 / 0.929 | | | 0.861 / 0.866 |
| 256 | | | 1.084 / 1.087 | | | 0.837 / 0.839 | 0.844 / 0.848 | | |
| 512 | | | 1.006 / 1.001 | 0.855 / 0.858 | | 0.811 / 0.814 | | | |
| 1024 | 1.207 / 1.205 | 1.082 / 1.085 | 0.950 / 0.957 | 0.815 / 0.814 | 0.810 / 0.833 | 0.795 / 0.779 | 0.779 / 0.768 | 0.747 / 0.727 | 0.714 / 0.707 |
| 2048 | | | 0.924 / 0.931 | | | 0.780 / 0.768 | | | |
| 4096 | | | 0.911 / 0.920 | | | 0.778 / 0.769 | 0.764 / 0.741 | | |
| 16384 | | | 0.902 / 0.893 | | | 0.782 / 0.784 | | | |

Default ms (cool / sustained): S1024: N128 0.116 / 0.114, N256 0.237 / 0.237, N384 0.402 / 0.401, N512 0.631 / 0.633, N768 1.046 / 1.107, N1024 1.557 / 1.654,
N2048 4.643 / 5.001; S4096 N768 3.928 / 4.098; S16384 N768 15.52 / 16.16. d_pair 256 (Chai-1), S1024 N768: 0.808 / 0.804 (default 1.103 / 1.145 ms).
Fast accumulation (`res/fast.jsonl`): S1024 N768 0.781 / 0.790, N2048 0.692 / 0.693, S4096 N768 0.755 / 0.746.

Gates (default `MIN_N` 384, `MIN_S` 1024 below N 512): N <= 256 loses at every depth (the fused `pwa_fwd3` default beats split + FP8: N128 +21 %, N256 +8 %);
N384 gains only from S1024 (S256 +8 %, S512 0 %, S1024 -5 %); from N512 every measured depth gains (S128 N768 -7 %).

### Error against fp64 PyTorch (`fp8x-pwa/scripts/err.py`, raw rows `res/err_*.jsonl`)

Module in float64 on the PyTorch backend, same inputs; random weights (fan-in^-0.5), LayerNorm weights 1 +- 0.1, biases +- 0.1, key mask with 3 masked
tokens; FP8 with every gate removed. upd = rms / max of (y - y64) relative to rms / max of the update y64 - x; judge = max |y - y64| / max |y64|
(engine metric, tol = max(err_pytorch, 8e-3)). Cells `upd rms / upd max / judge`.

| case (S:N[:d_pair]) | PyTorch bf16 | default 2.5.12 | FP8 | FP8 o-e4m3 (tried, not kept) |
|---|---|---|---|---|
| 1024:256 | 2.06 / 3.42 / 0.29 % | 2.04 / 3.42 / 0.29 % | 3.18 / 4.29 / 0.37 % | |
| 1024:384 | 2.20 / 4.05 / 0.29 % | 2.18 / 3.97 / 0.28 % | 3.06 / 4.60 / 0.33 % | 4.06 / 5.84 / 0.41 % |
| 1024:512 | 2.28 / 3.79 / 0.29 % | 2.27 / 3.74 / 0.29 % | 2.99 / 4.60 / 0.35 % | |
| 1024:768 | 2.38 / 4.36 / 0.28 % | 2.37 / 4.36 / 0.28 % | 2.89 / 5.00 / 0.32 % | 3.93 / 6.03 / 0.38 % |
| 1024:1024 | 2.43 / 4.47 / 0.28 % | 2.42 / 4.47 / 0.28 % | 2.84 / 4.95 / 0.31 % | |
| 4096:768 | 2.38 / 4.26 / 0.28 % | 2.37 / 4.21 / 0.28 % | 2.90 / 5.03 / 0.33 % | 3.93 / 5.70 / 0.38 % |
| 1024:768:256 | 2.37 / 4.63 / 0.27 % | 2.36 / 4.66 / 0.27 % | 2.92 / 5.35 / 0.31 % | |
| 1024:2048 | 2.53 / 4.67 / 0.28 % | 2.52 / 4.59 / 0.28 % | 2.75 / 4.91 / 0.30 % | |
| 2048:384 | 2.20 / 3.88 / 0.28 % | 2.18 / 3.88 / 0.28 % | 3.06 / 4.35 / 0.31 % | |

Fast accumulation (`res/err_fast.jsonl`): 1024:768 2.90 / 5.00 / 0.32 %, 1024:2048 2.79 / 5.03 / 0.30 % (no-fast 2.89 / 5.00 / 0.32, 2.75 / 4.91 / 0.30).

The update error of the bf16 routes (~2.4 %) is mostly the bf16 rounding of y = msa + update (the update is small next to the residual); the FP8
route adds ~1.6 % rms in quadrature. At the contraction itself (head 0, against an fp32 product of the same bf16 w, v): bf16 0.17 % rms, e4m3
3.8 % rms (5-8 % max), fast accumulation the same to three digits. The judge metric of the FP8 route (0.30-0.37 %) is inside the engine tolerance
(8e-3) at every measured case -- unlike TriMul's FP8 planes. No NaN / inf, no saturation (peak |v / scale| 22-24 of 448). Random weights only:
the end-level effect (real MSA, trained weights) is not measured.

### Tried, no gain

- **o as e4m3 too** (cuBLASLt e4m3 D with D scale = 1 / v scale -- |o| <= max |v|, so v's scale serves o; `pwa_gateout` built with
  `-DPWA_O8=1` reading e4m3 tiles; commit df188a291, removed after): module 0.793 / 0.803 at S1024 N768 (main route 0.795 / 0.779), N2048
  0.705 / 0.742 (0.714 / 0.707), N384 0.925 / 0.948 (0.950 / 0.957), error +1 % rms (table). ncu at S1024 N768: the contraction 354 vs 343 us
  (DRAM 33 %: not write-bound), the gate pass 359 vs 342 us although it reads 403 instead of 604 MB (it is not bandwidth-bound; the e4m3
  unpack and the unswizzled tile reads cost more than the bytes save). The removal also keeps `pwa_gateout.cu` byte-identical to 2.5.12 (no re-key).
- **`ln_vt8` occupancy**: 3 blocks / SM is 20-24 % slower than 2 (0.193 vs 0.157 ms at S1024 N768); stages 2 / 3 / 4 within 2 %.
- Not attempted: gate / out projections in e4m3 (51 GFLOP in a kernel at 20 % tensor use: no MMA-bound part to speed up), y as e4m3 (100 of the
  604 MB the gate pass reads, and it feeds the gate's sigmoid), z (the module's input), the backward (dv = w^T do, dw = do v^T are the same
  contraction shapes and could take the same placement; training forward was not measured to pay first).

Verification: default route bit-identical to the 2.5.12 base tree (sha1 of the outputs, separate processes; `res/bit_wt.jsonl` = `res/bit_base.jsonl`):
10 cases -- inference d_pair 128 at N256 (fused) / 384 / 512 / 768 / 2048, d_pair 256 N768, the wide widths 128:256:8 / 128:256:16, and the
native training forward + backward at S1024 N384 / N768 (y, dmsa, dpair, every parameter gradient). Tests: `tests/gpu/test_pwa_fp8.py` (new: `ln_vt8`
vs `ln_vg`, `w_quant`, route accuracy against fp64 within the engine tolerance at d_pair 128 / 256, default unchanged after the route ran, graph
replay = eager, the guard fires on a forced saturation and `rescale_all` clears it), `test_pwa_long_inference.py`,
`test_i64_offsets.py::test_pwa_train_pair_past_2_31`: 18 passed; after the gate change `test_pwa_fp8.py` again (with `test_gates`): 6 passed, and the bit-identity check repeated on the final tree. CPU: `test_capture_api` (the new module dicts listed; the remaining failure is the
base branch's TriAttn / TriMul FP8 dicts, unchanged), `test_init`, `test_requirements`, `test_prebuild_plan`: 1444 passed. `ruff check` / `format`,
`pyright` clean on the touched Python files. The new extension `miniworld_pwa_vt8` is JIT-built on first use (not in the prebuild plan).

### The wide widths: AF3 family (`pwa_fp8.forward_wide`, branch `fp8x/pwa-wide`, 2026-11-17)

Scope: `pwa_wide` inference on H100, 8 heads x 8 / 16 -- (d_msa, C) = (64, 8) AF3 / OpenFold3 / Protenix-v1 (d_pair 128), (128, 8) Protenix-v2
(d_pair 256) / OpenDDE (384), (128, 16) ESMFold2 (256): every non-BIG wide row of `requirements_module.csv`. Not served: the BIG build
(256 x 8 x 32, IntelliFold2), training, B200. Why it did not fire before: AF3's 64 x 8 is not `pwa_train`'s width, so the module runs `pwa_wide`,
which had no FP8 hook (and the 8 x 32 route also asks N % 128; FoldBench lengths are arbitrary). Same switch `MINIWORLD_PWA_FP8=1`, same v scale
(static power of two, `HEADROOM` 16, keyed by the value weight), w scale (2^8), guard (`take_overflow` / `rescale_all`, `GUARD` sync / lazy) and
cuBLASLt contraction as the 8 x 32 route.

Route: `pair_fwd3` (pwa_wide's pair kernel, w bf16) -> `w_quant` -> `ln_vt8` built at the width (`pwa_vt8.cu` with `-DVT_D` / `-DVT_C`:
d_msa 128 as two 64-column swizzled blocks per tile and Wv, HC / 64 m64 blocks; v as e4m3 K-major `vT [H][S*C][N]`, **no y** -- the wide gate
pass normalises m again in inference) -> the cuBLASLt e4m3 batched contraction (per head M = S*C, N = N_i, K = N_j: the heads are the batch, the
MSA rows x channels the long M, so C = 8 still gives a full-size GEMM) -> `gate_out_w` in its LayerNorm mode (pwa_wide's default inference gate /
out pass, unchanged). N need only be a multiple of 16 (pwa_wide pads to 16): the last token block of a row is computed from the next row's
(or TMA zero-filled) tokens, clipped by the vT store and left out of amax. `ln_vt8` launch (blocks / SM, stages), measured best of
{1, 2, 3, 4, 6, 8} x {2, 3, 4} (`res/tune.jsonl`): 64 x 8 (3, 3), 128 x 8 (3, 2), 128 x 16 (2, 3); the 8 x 32 build keeps (2, 4).

| env | effect |
|---|---|
| `MINIWORLD_PWA_FP8_WIDE_MIN_N` | shortest N served on the wide widths (256); 0 = no gate |
| `MINIWORLD_PWA_FP8_WIDE_MIN_SN` | smallest S * N served (3 * 2^16 = 196608); 0 = no gate |

Scripts and raw rows: `~/miniworld-engine-scratch/fp8x-pwa-wide/{scripts,res}`.

Stages of the 2.5.12 wide route at AF3 width (64 x 8, d_pair 128), CUDA graph per stage, median (`res/prof.jsonl`); removal = module chain minus
that stage. e4m3 contraction = the cuBLASLt route on PyTorch-built vT (prototype, before the kernel).

| S:N | module ms | `pair_fwd3` | `ln_vg_w` | contraction (cuBLAS bf16) | `gate_out_w` (LN mode) | e4m3 contraction | `ln_vt8` (no y) |
|---|---|---|---|---|---|---|---|
| 1024:768 | 0.405 | 0.065 (16 %) | 0.084 (21 %) | 0.122 (30 %) | 0.127 (31 %) | 0.079 (0.65x) | 0.057 (0.68x, `res/tune.jsonl`) |
| 1024:384 | 0.185 | 0.018 | 0.043 | 0.044 | 0.067 | 0.035 | |
| 4096:768 | 1.336 | 0.065 | 0.313 | 0.453 | 0.503 | 0.300 | |
| 1024:1536 | 1.076 | 0.232 | 0.164 | 0.411 | 0.253 | 0.253 | |
| 16384:512 | 3.086 | 0.033 | 0.803 | 0.923 | 1.296 | 0.627 | |
| 128x8 1024:768 (d_pair 256) | 0.631 | 0.122 | 0.115 | 0.122 | 0.257 | 0.080 | 0.091 |
| 128x16 1024:768 (d_pair 256) | 0.891 | 0.122 | 0.154 | 0.232 | 0.374 | 0.153 | 0.134 |

The contraction is 77 GFLOP at S1024 N768 (a quarter of the 8 x 32 one) and the e4m3 GEMM is 0.65x, not 0.5x: its o write (bf16, 100 MB) is
the larger part of its bytes. What is left after FP8 is mostly `gate_out_w` (~37 % of the FP8 route), which reads o, m twice and writes the
output. Not attempted here: o as e4m3 into the gate pass (a new gate kernel next to `pwa_wide.cu`, whose bytes stay the 2.5.12 ones; the 8 x 32
attempt gained nothing, below).

Module time, FP8 / default, `cool / sustained` (`scripts/speed.py`, `res/v2.jsonl`; protocol as the 8 x 32 table: one process, three CUDA graphs
-- default, default again = same-cubin control, FP8 lazy guard with gates removed -- interleaved). Control over all 48 rows 0.982 .. 1.006; no
saturation in any row; the route fired in every row. AF3 width (64 x 8, d_pair 128):

| S \ N | 128 | 192 | 256 | 384 | 512 | 768 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|---|---|---|---|
| 128 | | | | | | 1.016 / 1.018 | | | |
| 256 | | | | 1.008 / 1.013 | | 0.967 / 0.968 | | 0.939 / 0.944 | |
| 512 | | | 1.033 / 1.033 | 0.974 / 0.967 | 0.949 / 0.952 | 0.926 / 0.927 | | 0.886 / 0.896 | 0.857 / 0.864 |
| 1024 | 1.052 / 1.057 | 1.015 / 1.013 | 0.963 / 0.967 | 0.919 / 0.922 | 0.915 / 0.912 | 0.865 / 0.866 | 0.858 / 0.864 | 0.826 / 0.831 | 0.794 / 0.806 |
| 2048 | | | 0.914 / 0.915 | 0.887 / 0.890 | | 0.840 / 0.859 | | | |
| 4096 | 0.903 / 0.904 | | 0.886 / 0.889 | 0.861 / 0.874 | | 0.826 / 0.843 | | 0.772 / 0.786 | 0.739 / 0.759 |
| 16384 | 0.876 / 0.879 | | 0.870 / 0.877 | | 0.827 / 0.851 | 0.811 / 0.821 | 0.791 / 0.812 | | |

N not a multiple of 128 (S1024): N400 0.910 / 0.909, N784 0.905 / 0.911. Default ms (cool): S1024 N256 0.123, N768 0.404, N2048 1.661; S4096 N768
1.315; S16384 N1024 7.243. Other widths at d_pair 256 (S1024): 128 x 8 N256 0.972, N384 0.945, N768 0.910, N1536 0.882; S512 N768 0.952, S4096
N768 0.880; 128 x 16 N256 0.957, N384 0.933, N768 0.894, N1536 0.849; S512 N768 0.917, S4096 N768 0.872; 128 x 8 at d_pair 384 (OpenDDE)
S1024 N768 0.918.

Gates (default `WIDE_MIN_N` 256, `WIDE_MIN_SN` 3 * 2^16): every loss measured is below them -- N128 / N192 at S1024 (+5 / +2 %), S512 N256 (+3 %),
S256 N384 (+1 %), S128 N768 (+2 %); every served cell measured gains (worst: 128 x 8 S1024 N256 0.97). Not served although measured to gain:
N128 at S >= 4096 (0.88-0.90).

Error against fp64 PyTorch (`scripts/err.py`, `res/err_v2.jsonl`; method as the 8 x 32 table: random weights, 3 masked keys, gates removed).
Cells `upd rms / upd max / judge` in %:

| case (D:C:d_pair:S:N) | PyTorch bf16 | default 2.5.12 | FP8 |
|---|---|---|---|
| 64:8:128:1024:256 | 2.17 / 3.39 / 0.29 | 2.16 / 3.36 / 0.29 | 3.46 / 4.39 / 0.38 |
| 64:8:128:1024:384 | 2.36 / 3.61 / 0.29 | 2.35 / 3.61 / 0.29 | 3.39 / 4.18 / 0.34 |
| 64:8:128:1024:400 | 2.38 / 3.66 / 0.30 | 2.37 / 3.66 / 0.30 | 3.38 / 4.27 / 0.35 |
| 64:8:128:1024:768 | 2.62 / 4.66 / 0.28 | 2.62 / 4.66 / 0.28 | 3.26 / 5.65 / 0.34 |
| 64:8:128:1024:1536 | 2.81 / 5.00 / 0.28 | 2.80 / 4.93 / 0.28 | 3.17 / 5.43 / 0.31 |
| 64:8:128:512:2048 | 2.86 / 5.93 / 0.29 | 2.86 / 5.82 / 0.28 | 3.14 / 6.14 / 0.30 |
| 64:8:128:4096:768 | 2.63 / 4.33 / 0.28 | 2.62 / 4.37 / 0.28 | 3.27 / 5.39 / 0.35 |
| 64:8:128:16384:512 | 2.48 / 3.58 / 0.30 | 2.47 / 3.42 / 0.28 | 3.33 / 4.18 / 0.35 |
| 128:8:256:1024:768 | 2.45 / 4.27 / 0.28 | 2.44 / 4.21 / 0.28 | 3.01 / 4.78 / 0.31 |
| 128:16:256:1024:768 | 2.64 / 4.52 / 0.28 | 2.63 / 4.52 / 0.28 | 3.22 / 5.24 / 0.32 |
| 128:8:384:1024:768 | 2.47 / 4.27 / 0.28 | 2.47 / 4.27 / 0.28 | 3.04 / 4.96 / 0.33 |

Judge metric of the FP8 route 0.30-0.38 % at every case, inside the engine tolerance (8e-3); no NaN / inf, no saturation (peak |v / scale|
21.5-25.5 of 448). Contraction alone (head 0 vs fp32 of the same bf16 w, v): bf16 0.17 % rms, e4m3 3.8 % rms (`res/prof.jsonl`). Random weights
only: the end-level effect (real MSA, trained weights, FoldBench) is not measured.

Verification: default route (FP8 off) bit-identical to ffcd465f7 (sha1 of the outputs, separate processes, `res/bit_wt.jsonl` = `res/bit_base.jsonl`),
15 cases: wide inference 64:128:8 at N256 / 768 / 777 (padded) / 2048, 128:256:8, 128:256:16, 128:384:8 N400; wide training forward + backward
64:128:8 N384 S1024, 128:256:16 N256 (y, dmsa, dpair, every parameter gradient); `pwa_train` inference / training 8 x 32; and the **8 x 32 FP8
route** itself at S1024 N768, S2048 N384, d_pair 256 N512 (the generalised `pwa_vt8.cu` gives its bytes unchanged). `pwa_wide.py` gains one
`if not save` branch before `ln_vg_w`; `pwa_wide.cu` / `pair_wide.cu` are untouched. Tests: `tests/gpu/test_pwa_fp8_wide.py` (new: `ln_vt8` at each
wide width vs `ln_vg_w` with S 70 / N 208 -- the clipped token tail --, route accuracy against fp64 within the engine tolerance at the four
widths incl. N 777 padded, the route fired, default unchanged after it ran, graph replay = eager, guard fires and `rescale_all` clears it, gates),
`test_pwa_fp8.py`, `test_pwa_long_inference.py`: 28 passed (final tree). CPU: `test_capture_api`, `test_init`: 274 passed; `ruff check` / `format`, `pyright` clean on the touched Python files. The wide `ln_vt8` builds (`miniworld_pwa_vt8_{d}_{c}`) are JIT-built on first use (not in the
prebuild plan); the 8 x 32 build keeps its extension name (its cache key changes with the source).

## Transition forward: e4m3 GEMMs (`kernels/transition/cuda/fp8_sm90a.py`, branch `fp8x/transition`, 2026-11-17)

Scope: bf16 inference, one H100, D 128 / 256 (n 2 / 4: fused one-kernel forwards; D 256 added on branch `fp8x/transition-d256`, section
"D 256" below) and D 384 / 512 (n 2 / 4: the two-kernel wide chain). Training, D 64, D 768 and fp32 activations stay on the 2.5.12 route. Off by default; with `MINIWORLD_TRANSITION_FP8` unset the dispatcher sees one `POLICY is None`
check and the outputs are the 2.5.12 bytes (below). Scripts and raw rows: `~/miniworld-engine-scratch/fp8x-transition/{scripts,res}` (`res/...` below).

| env (read at import) | effect |
|---|---|
| `MINIWORLD_TRANSITION_FP8=calib` | route the served calls through the e4m3 kernels; h scale calibrated on the first eager call of a module |
| `MINIWORLD_TRANSITION_FP8_GUARD` | `sync` (default, eager: redo an overflowing call) / `lazy` (no host sync; CUDA graphs: `overflow_peak()` + `rescale_all(peak)` + redo, the TriMul contract) / `off` |
| `MINIWORLD_TRANSITION_FP8_G1`, `_G2` | 1 / 1 (default): expand and squeeze in e4m3; `G1=0`: bf16 expand (D 128, D 256 and wide); `G2=0`: bf16 squeeze (D 128 only) |
| `MINIWORLD_TRANSITION_FP8_HEADROOM` | h scale headroom over the calibration amax (16) |
| `MINIWORLD_TRANSITION_FP8_MINROWS` | replaces the row gate `ROW_GATE` (D 128: 16384 rows = pair L128; D 256: 8192 rows; D 384 / 512: 65536 = L256; see "D 256" for the small-row and token-count measurements) |
| `MINIWORLD_TRANSITION_FP8_WIDTHS`, `_SIGT`, `_SLOTS`, `_ABL` | widths routed (128,256,384,512); sigmoid form (1 = one tanh.approx); weight-ring slots (4, D 128 / 256); timing ablations (never a route) |

### Where the time goes (2.5.12 default) and where e4m3 can go

Module time, CUDA graph, cool, master weights, one sample (`res/an1.jsonl`; FLOPs = 2 rows (2 D H + H D), H = n D):

| D, n | kernels of the route | L1024 ms | TFLOP/s | L2048 ms |
|---|---|---|---|---|
| 64, 4 / 2 | `wide_d64_fwd_small_f3` (one fused kernel) | 0.268 / 0.183 | 385 / 282 | 1.05 / 0.70 |
| 128, 4 / 2 | `transition_fwd_fused` (one fused kernel) | 0.680 / 0.430 | 607 / 479 | 2.61 / 1.68 |
| 256, 4 / 2 | `wide_d256_fwd` (one fused kernel) | 2.53 / 1.45 | 651 / 568 | 10.0 / 5.69 |
| 384, 4 / 2 | `wide_lnsg` (LN + expand + SwiGLU, h to HBM) + `wide_squeeze` (+ residual) | 3.80 + 2.04 / 2.12 + 1.29 | 633 / 542 | 16.7 + 8.4 / 9.0 + 5.4 |
| 512, 4 / 2 (L768) | the same two kernels | 3.63 + 1.91 / 1.90 + 1.10 | 665 / 618 | (L768 is the largest row) |

Counters of the D128 kernel at L1024 (ncu, fixed clocks, `res/ncu_n1.csv`): n 4: tensor pipe active 67 % of peak, MUFU (xu) 45 %, FMA 16 %, DRAM 19 %,
issue 33 %; n 2: tensor 54 %, MUFU 36 %, DRAM 31 %. The default is tensor-pipe-dominated with MUFU (ex2 + rcp per sigmoid) second; DRAM is not the limit.
Stage ablation on the same kernel structure (`ABL` builds, `res/abl1.jsonl`, D128 n4 L1024, cool): LayerNorm + x in / y out only (no chunk loop) 0.194 ms
(the HBM floor of x + y: 2.8 TB/s), GEMMs without the SwiGLU math 0.636 ms, full 0.705 ms (the bf16 twin; the 2.5.12 kernel 0.681).

| stage | operand that can be e4m3 | share | honest upper bound |
|---|---|---|---|
| expand GEMM (2/3 of the FLOPs) | xn (A, in registers / the resident tile), [Wa; Wb] (B) | D128: 0.64 - 0.19 ms of 0.68 is GEMM + SwiGLU; wide: `lnsg` 65 % of the module | GEMM part x2 (tensor rate); MUFU / LN / HBM unchanged |
| SwiGLU h (the squeeze's A) | h quantized in the expand epilogue | D128: registers only (no bytes); wide: h is an HBM round trip of 2 rows H bytes | wide: h bytes / 2 (D384 n4 L2048: 12.9 GB -> 6.4 GB each way) |
| squeeze GEMM (1/3) | h (A), Ws (B) | wide `squeeze` 35 % of the module | x2 tensor rate + half the h read |
| LayerNorm, residual, x / y | none (bf16 in / out is the interface) | D128 n4: 0.19 ms floor of 0.68 | none |

cuBLAS(Lt) bf16 vs e4m3 at the same GEMM shapes (`res/ub1.jsonl`, L1024): squeeze 1.46-1.68x faster in e4m3 at every width, expand 1.37-1.50x at D 384 /
512, 1.06-1.14x at D 64-256 (bf16 output bound). The engine's fused kernels beat these cuBLAS GEMMs already; the ratios bound what e4m3 can do per GEMM.

### What was built

- **D 128** (`transition_fp8_fwd_sm90a.cu`): the 2.5.12 D128 inference build (LNFRAG, XFIRST, DECOUPLE, SKEW, unrolled chunk loop) with both GEMMs on
  `wgmma m64n128k32 .e4m3` (fp32 accumulation). xn is quantized in registers from the fp32 LayerNorm (no bf16 rounding first) with the power-of-two scale
  of the LayerNorm bound (|xn| <= sqrt(D - 1) max|gamma| + max|beta|: never saturates, no calibration). The fp32 C fragment of the expand IS the e4m3
  k32 A fragment of the squeeze after a fixed K permutation (columns {2t, 2t+1, 8+2t, 9+2t} of each 16-block in the logical order 4t..4t+3), so the
  packed weights carry K in that order (`_perm_k`) and no shuffle runs. h = silu(a) b sh with silu(a) = (a/2)(1 + tanh(a/2)) (one MUFU op), cast with
  `cvt.rn.satfinite.e4m3x2`, max |h sh| per CTA to the guard buffer. Weights: per-tensor power-of-two scales, packed once per parameter version
  (no per-call prepare kernel). 24 KB weight slots (4 in the ring instead of 2). Switches `F8G1` / `F8G2` give the partial placements; `F8G1 = F8G2 = 0`
  with the 2.5.12 sigmoid reproduces the 2.5.12 output bytes (test `test_bf16_build_reproduces_default`), so the kernel is the default's structure.
- **D 384 / 512** (`wide/fp8_bind.cu`, `wide/kernels/ln_swiglu_gemm_f8.cu`, `wide/kernels/squeeze_gemm_f8.cu`: e4m3 twins of the 2.5.12 files, which
  are untouched): the LayerNorm writes e4m3 xn in place over the resident x tile (G1; 128-byte K slabs, `wgmma m64n128k32` SS), the expand epilogue
  writes h as e4m3 (stmatrix of e4m3 pairs into a 64B-swizzled 4 KB staging tile, TMA store; the stored columns carry the same 16-block permutation,
  the e4m3 Ws is packed to match), the squeeze runs `wgmma m64n{192,256}k32` SS over 128-byte K slabs and adds the residual (`co = 1 / (sh s2)`).
- Scale / guard: the TriMul conventions (static power-of-two scales, 16x headroom, `overflow_peak()` / `rescale_all()` / redo, weakref-finalized caches
  keyed by the expand_a weight); calibration runs one probe call with sh = 1 (the pre-cast peak is exact even when the probe saturates).

### Module time against the 2.5.12 default

Protocol as in "Base 2.5.12" (`scripts/f8bench.py speed`): one process per group, module (`master_weights=True`, one sample) captured in CUDA graphs of one
call each: the 2.5.12 default, the default again (same-cubin control), each FP8 variant (calibrated on two eager calls, lazy guard, no overflow in any
case: peak 15-28 of 448). cool = 2 s idle then one interleaved burst, 3 reps, median; sustained = 1 s of back-to-back replays, then 3 interleaved blocks,
mean. Cells `cool / sustained` (`res/sp3.jsonl`, `res/spw2.jsonl`).

D 128, n 4 (rows = L^2). full = both GEMMs e4m3; G1 = expand only; G2 = squeeze only; twin = the FP8 kernel built as bf16 (`F8G1=F8G2=0`, 2.5.12 sigmoid):

| L | 2.5.12 ms | full / default | G1 only | G2 only | bf16 twin | control |
|---|---|---|---|---|---|---|
| 128 | 0.0174 / 0.0170 | 0.659 / 0.664 | 0.738 / 0.733 | 0.765 / 0.768 | 0.876 / 0.877 | 0.989 / 0.993 |
| 256 | 0.0483 / 0.0509 | 0.745 / 0.749 | 0.843 / 0.847 | 0.889 / 0.875 | 1.043 / 1.023 | 1.010 / 1.013 |
| 384 | 0.108 / 0.114 | 0.714 / 0.728 | 0.823 / 0.838 | 0.845 / 0.868 | 1.043 / 1.048 | 0.998 / 0.996 |
| 512 | 0.185 / 0.192 | 0.702 / 0.733 | 0.816 / 0.852 | 0.844 / 0.888 | 1.044 / 1.124 | 0.993 / 1.008 |
| 768 | 0.390 / 0.427 | 0.698 / 0.730 | 0.820 / 0.842 | 0.856 / 0.862 | 1.030 / 1.069 | 0.994 / 0.988 |
| 1024 | 0.687 / 0.745 | 0.693 / 0.735 | 0.817 / 0.861 | 0.861 / 0.862 | 1.040 / 1.077 | 1.000 / 0.997 |
| 1536 | 1.487 / 1.659 | 0.713 / 0.736 | 0.836 / 0.862 | 0.884 / 0.888 | 1.062 / 1.082 | 0.987 / 0.984 |
| 2048 | 2.607 / 2.934 | 0.721 / 0.739 | 0.843 / 0.856 | 0.900 / 0.891 | 1.075 / 1.103 | 1.000 / 0.994 |

D 128, n 2:

| L | 2.5.12 ms | full / default | G1 only | G2 only | bf16 twin | control |
|---|---|---|---|---|---|---|
| 128 | 0.0130 / 0.0127 | 0.710 / 0.707 | 0.717 / 0.708 | 0.761 / 0.758 | 0.832 / 0.822 | 0.987 / 0.991 |
| 256 | 0.0335 / 0.0334 | 0.756 / 0.754 | 0.818 / 0.812 | 0.856 / 0.843 | 0.989 / 0.976 | 0.996 / 0.997 |
| 384 | 0.0738 / 0.0760 | 0.699 / 0.704 | 0.766 / 0.772 | 0.800 / 0.806 | 0.979 / 0.971 | 1.005 / 1.002 |
| 512 | 0.123 / 0.129 | 0.684 / 0.696 | 0.756 / 0.767 | 0.796 / 0.798 | 0.982 / 0.988 | 1.003 / 0.992 |
| 768 | 0.245 / 0.266 | 0.696 / 0.731 | 0.783 / 0.807 | 0.825 / 0.838 | 1.034 / 1.030 | 1.000 / 0.993 |
| 1024 | 0.435 / 0.457 | 0.690 / 0.732 | 0.777 / 0.820 | 0.821 / 0.860 | 0.984 / 1.041 | 0.990 / 0.998 |
| 1536 | 0.956 / 1.028 | 0.683 / 0.731 | 0.780 / 0.819 | 0.828 / 0.840 | 0.996 / 1.016 | 0.979 / 0.989 |
| 2048 | 1.701 / 1.816 | 0.682 / 0.754 | 0.773 / 0.842 | 0.826 / 0.873 | 0.978 / 1.046 | 0.965 / 1.008 |

The bf16 twin is 2-12 % slower than 2.5.12 at n 4, L >= 256, and within -3 .. +5 % at n 2 (the scale multiplies; at L128 it is faster because the
FP8 route caches its packed weights and launches no per-call prepare kernel), so the e4m3 gain over the same kernel structure is 1.4-1.5x. The sigmoid form alone does not pay in bf16:
the bf16 twin with the tanh sigmoid was 1.03-1.12x the default at n 4, L >= 256 (`res/sp2.jsonl`; MUFU drops to 20 % but the tensor pipe stays the limit).
With e4m3 GEMMs the tanh sigmoid pays: `SIGT=0` (ex2 + rcp) 0.86-0.94 (n 4) / 0.78-0.82 (n 2) of the default against 0.68-0.80 with tanh
(`res/sp1.jsonl`, before the silu fold of the final build). FP8 kernel
counters (D128 n4 L1024, first build): tensor 43 %, MUFU 29 %, FMA 28 %, DRAM 24 %, issue 47 %: no unit is saturated, the per-warpgroup chain
(expand -> SwiGLU -> squeeze, two warpgroups) is the limit. Ablation of the e4m3 kernel: no SwiGLU math 0.398 ms, full 0.478 ms, HBM floor 0.195 ms.

D 384 / 512 (wide chain; `h8` = h + squeeze e4m3, expand bf16 (`G1=0`); `full` = expand e4m3 too):

| D, n | L | 2.5.12 ms | h8 / default | full / default | control |
|---|---|---|---|---|---|
| 384, 4 | 256 | 0.403 / 0.444 | 0.845 / 0.859 | 0.642 / 0.649 | 0.996 / 1.002 |
| | 512 | 1.520 / 1.711 | 0.863 / 0.886 | 0.662 / 0.651 | 0.996 / 0.997 |
| | 1024 | 5.887 / 6.582 | 0.869 / 0.905 | 0.644 / 0.671 | 0.996 / 1.002 |
| | 2048 | 24.11 / 26.41 | 0.846 / 0.884 | 0.686 / 0.654 | 1.008 / 0.976 |
| 384, 2 | 256 | 0.249 / 0.257 | 0.876 / 0.907 | 0.715 / 0.737 | 0.995 / 1.018 |
| | 512 | 0.895 / 0.981 | 0.897 / 0.903 | 0.719 / 0.722 | 0.997 / 0.982 |
| | 1024 | 3.442 / 3.738 | 0.903 / 0.927 | 0.717 / 0.750 | 0.997 / 0.987 |
| | 2048 | 13.61 / 16.10 | 0.901 / 0.917 | 0.713 / 0.720 | 0.998 / 0.994 |
| 512, 4 | 256 | 0.653 / 0.702 | 0.801 / 0.846 | 0.586 / 0.617 | 0.998 / 1.010 |
| | 512 | 2.552 / 2.879 | 0.799 / 0.788 | 0.575 / 0.559 | 0.997 / 0.999 |
| | 768 | 5.604 / 6.502 | 0.796 / 0.803 | 0.570 / 0.568 | 0.994 / 0.995 |
| 512, 2 | 256 | 0.367 / 0.396 | 0.831 / 0.841 | 0.641 / 0.659 | 0.994 / 0.992 |
| | 512 | 1.388 / 1.531 | 0.843 / 0.859 | 0.635 / 0.650 | 0.999 / 1.006 |
| | 768 | 3.016 / 3.470 | 0.844 / 0.836 | 0.632 / 0.623 | 0.999 / 0.992 |

### Error against fp64

`scripts/f8bench.py err` (`res/err2.jsonl`, `res/errw2.jsonl`): the module in float64 on the PyTorch backend (fp32 masters -> float64), same inputs;
weights N(0, 1/fan_in), LayerNorm weight 1 +- 0.1, bias +- 0.1, x ~ N(0, 1). judge = max|y - y64| / max|y64| (`measure/accuracy.py`; tolerance
max(err_pytorch, 8e-3) = 8e-3 on every row); rms = rms(y - y64) / rms(y64); update = rms(y - y64) / rms(y64 - x). Ranges over L128 / 512 / 1024
(D128) and L256 / 512 (wide). No NaN / inf in any case; no saturation (peak 15-28 against 448), no guard retry or fallback.

| route | judge | rms | update rms |
|---|---|---|---|
| PyTorch bf16 (tolerance reference) | 0.0050-0.0068 | 0.0036-0.0038 | |
| 2.5.12 default (D128 / wide) | 0.0035-0.0042 | 0.0027-0.0028 | 0.0052-0.0053 |
| D128 full e4m3 | 0.041-0.045 | 0.034-0.035 | 0.066 |
| D128 expand only (G1) | 0.033-0.045 | 0.028-0.029 | 0.055 |
| D128 squeeze only (G2) | 0.026-0.037 | 0.0195-0.020 | 0.038 |
| D384 / 512 h8 (h + squeeze) | 0.020-0.029 | 0.0195-0.0197 | 0.038 |
| D384 / 512 full | 0.035-0.042 | 0.034 | 0.066 |

Every FP8 variant is outside the 8e-3 tolerance (3-5x), as expected for e4m3 activations (3 mantissa bits; the error is per-element storage, the
scales only keep it out of the subnormal range). The update error (6.6 %) is above the TriMul planes' (3.8 %): both Transition GEMM inputs are
activations, the TriMul route quantizes one plane pair. End-level effect (structures) was not measured.

### Default route, verification

- Default bytes: sha1 of the default output, separate processes, this branch against the base tree (91a9241fb): 14 / 14 identical (D 64 n 4 / 2,
  D 128 n 4 / 2 incl. 12800 rows and L2048, D 256 n 4 / 2, D 384 n 4 / 2 (2048 rows: the generic path), D 512 n 4, D 768 n 2; `res/sha_base.json`,
  `res/sha_fp8b.json`). No 2.5.12 source changed: the FP8 kernels are new files; the dispatcher gained a `fp8=True` parameter and one branch behind
  `POLICY`.
- Tests: `tests/gpu/test_transition_fp8_h100.py` (fragment permutation, error vs fp64 for D128 4 placements x n x rows and D384 / 512 x n x G1, bf16 twin
  = 2.5.12 bytes, guard sync retry and lazy `overflow_peak` / `rescale_all`, default untouched when off). ruff check / pyright clean on the touched
  Python files (`ruff format` reports one 2.5.12 line in `cuda/__init__.py` that this branch leaves as it is).
- Not in the prebuild plan: the FP8 extensions JIT-build on first use (about 90 s each).

### Not done, and why

- D 64 (`wide_d64_fwd_small_f3`, 282-385 TFLOP/s): x / y traffic is half of the time at n 2 (1.5 TB/s of x + y alone) and the LayerNorm / SwiGLU
  share is the largest of all widths; e4m3 GEMMs bound below 1.3x. Not built (measured bound: section "D 64" below).
- D 256: built afterwards (section "D 256" below).
- Per-channel weight scales, e4m3 x / y storage (the interface is bf16), e5m2: not tried. Training: not ported (forward first, as briefed).
- The 2.5.12 D128 bf16 default with the tanh sigmoid alone: slower (above), so no non-FP8 spin-off.

### D 256 (`transition_fp8_d256_sm90a.cu`, branch `fp8x/transition-d256`, base ffcd465f7, 2026-11-17)

Chai-1 / ESMFold2 / Protenix-v2 pair transition. Scripts and raw rows: `~/miniworld-engine-scratch/fp8x-transition-d256/{scripts,res}`.

Design: the 2.5.12 D256 forward (`wide/kernels/transition_fwd_d256.cu`: a producer warpgroup + two ping-pong consumer warpgroups, a 128-row tile,
the 64 x 256 fp32 output accumulator of each warpgroup held across the 32-unit hidden chunks), not the wide chain: the D 128 placements carry over
because the chunk's [a | b] C fragment holds exactly one 32-wide h block, i.e. one k32 e4m3 A fragment after the D 128 K permutation.

- LayerNorm (the 2.5.12 code) writes e4m3 xn (LayerNorm-bound scale, cannot saturate) into its own 32 KB buffer, so x stays resident for the
  residual: the 2.5.12 kernel's second x load per tile (and its `xn_free` / `x_res` hand-off) is gone.
- per chunk: expand = 8 x `wgmma m64n64k32 .e4m3` SS (xn8, [Wa8_j; Wb8_j] K-major 128B swizzle, 16 KB); h8 = e4m3(silu(a) b sh) in registers
  (tanh silu, max |h sh| to the guard); squeeze = ONE `wgmma m64n256k32 .e4m3` RS over Ws8_j ([256 d][32 hs] K-major, 32B swizzle, 8 KB, K
  permuted on the host). The slot is 24 KB (48 KB in 2.5.12): 4 slots + x 64 KB + xn8 32 KB = 192 KB.
- `G1=0` ("h + squeeze"): the 2.5.12 bf16 expand over the in-place bf16 xn (the 2.5.12 x reload kept), h and the squeeze e4m3 (40 KB slots).
- scales / guard / calibration / packing per parameter version: the D 128 code (`_pack_d256`); `ROW_GATE[256]` = 8192 rows (below).

Module time against the 2.5.12 default (`scripts/bench.py speed`, the protocol above: one process, CUDA graph of one call each, interleaved, the
default captured twice as the same-cubin control; cells `cool / sustained`; `res/sp1.jsonl`). full = expand + squeeze e4m3, h8 = h + squeeze only
(`G1=0`). Lazy guard, no overflow in any case (peak 15-26 of 448):

| n | L (rows L^2) | 2.5.12 ms | full / default | h8 / default | control |
|---|---|---|---|---|---|
| 4 | 256 | 0.175 / 0.190 | 0.733 / 0.741 | 0.915 / 0.902 | 0.999 / 0.993 |
| | 384 | 0.375 / 0.417 | 0.719 / 0.741 | 0.915 / 0.915 | 1.002 / 1.002 |
| | 512 | 0.657 / 0.724 | 0.708 / 0.735 | 0.912 / 0.937 | 0.996 / 0.999 |
| | 768 | 1.416 / 1.698 | 0.701 / 0.716 | 0.917 / 0.896 | 0.997 / 0.987 |
| | 1024 | 2.516 / 2.929 | 0.695 / 0.723 | 0.922 / 0.922 | 0.999 / 0.995 |
| | 1536 | 5.594 / 6.550 | 0.690 / 0.717 | 0.921 / 0.909 | 0.999 / 0.993 |
| | 2048 | 9.958 / 11.94 | 0.691 / 0.712 | 0.918 / 0.908 | 0.997 / 0.984 |
| 2 | 256 | 0.109 / 0.113 | 0.762 / 0.768 | 0.913 / 0.907 | 1.000 / 0.997 |
| | 384 | 0.224 / 0.244 | 0.756 / 0.760 | 0.923 / 0.913 | 1.002 / 1.001 |
| | 512 | 0.385 / 0.428 | 0.747 / 0.749 | 0.923 / 0.909 | 0.999 / 0.994 |
| | 768 | 0.818 / 0.942 | 0.746 / 0.752 | 0.923 / 0.926 | 0.998 / 1.008 |
| | 1024 | 1.452 / 1.630 | 0.750 / 0.751 | 0.929 / 0.924 | 0.999 / 0.999 |
| | 1536 | 3.198 / 3.672 | 0.742 / 0.757 | 0.927 / 0.935 | 0.998 / 1.004 |
| | 2048 | 5.670 / 6.667 | 0.740 / 0.740 | 0.929 / 0.917 | 0.999 / 0.989 |

Weight-ring depth (`res/sp4.jsonl`, L256 / 512 / 1024): 3 slots within +-1 % of 4, 5 slots 1-2 % faster at n 4 (sustained), equal at n 2: the
default stays 4 (shared with D 128).

Small row counts (`res/sp3.jsonl`, `MINROWS=0`; full / default, cool): n 4: 2048 rows 1.91, 4096 1.39, 6144 1.09, 8192 0.70, 16384 0.70,
32768 0.69; n 2: 2048 1.63, 4096 1.24, 6144 0.99, 8192 0.69, 16384 0.71, 32768 0.73. Below 8192 rows the default runs the generic kernels
(`_PREFER_GENERIC_ROWS[256]` = 6144) and the fused FP8 kernel has 48 or fewer 128-row tiles for 132 SMs, so `ROW_GATE[256]` = 8192.

Error against fp64 (`scripts/bench.py err`, `res/err1.jsonl` / `res/err2.jsonl`; rows 128, 16384, 38400, 65536, 262144, n 4 / 2; tolerance
max(err_pytorch, 8e-3) = 8e-3 on every row). No NaN / inf, no guard retry or fallback:

| route | judge | rms | update rms |
|---|---|---|---|
| PyTorch bf16 (tolerance reference) | 0.0044-0.0056 | 0.0036-0.0037 | 0.0070-0.0071 |
| 2.5.12 default | 0.0035-0.0040 | 0.0027-0.0028 | 0.0052 |
| D256 full e4m3 | 0.035-0.040 | 0.034-0.035 | 0.066 |
| D256 h8 (h + squeeze, `G1=0`) | 0.021-0.029 | 0.019-0.020 | 0.038 |

The same as D 128 / wide (4.4-5x the tolerance for full, 2.7-3.7x for h8): the e4m3 storage of the two GEMM inputs, not the kernel.

Verification: default bytes, sha1 in separate processes against the base tree ffcd465f7: 18 / 18 identical (D 64 n 4 / 2; D 128 n 4 / 2 incl.
12800 rows and L2048; D 256 n 4 / 2 at 2048, 8192, 16384, 2^20, 2^22 rows; D 384 n 4 / 2 incl. 2048 rows; D 512 n 4; D 768 n 2 at 512 / 16384;
`res/sha_base.json`, `res/sha_fp8.json`). No 2.5.12 source changed (the kernel and its wgmma header are new files; `fp8_sm90a.py` gained the
D 256 build / pack / launch and the gate entry). Tests: `tests/gpu/test_transition_fp8_d256_h100.py` (17: error vs fp64 for n x G1 x rows,
CUDA-graph replay = eager bytes, sync and lazy guard, row gate, default untouched) and `tests/gpu/test_transition_fp8_h100.py` (37) pass on H100;
`tests/cpu/test_capture_api.py` (13) passes; ruff check / format and pyright clean on the touched Python files. The extension JIT-builds on first
use (about 85 s per (n, G1, slots)).

### Single-representation transition at token counts (AF3 single D 384 n 4; D 768 n 2), `res/sp2.jsonl`, `res/ub1.jsonl`

Question (b): does a token-count-sized single transition (rows = tokens, 256-4096) gain from e4m3 at all? The 65536-row gate kept it on the
2.5.12 route (the generic kernels at these rows). Measured with the gate off (`MINROWS=0`), full / default (cool; sustained within 1 %):

| D 384 n 4 rows | 256 | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 |
|---|---|---|---|---|---|---|---|---|
| 2.5.12 ms | 0.0140 | 0.0143 | 0.0174 | 0.0180 | 0.0214 | 0.0220 | 0.0303 | 0.0343 |
| wide FP8 full / default | 3.44 | 3.34 | 2.79 | 2.72 | 2.30 | 2.29 | 1.70 | 1.51 |
| wide FP8 h8 / default | 4.82 | 4.69 | 3.91 | 3.80 | 3.20 | 3.17 | 2.33 | 2.07 |

The wide chain (128-row tiles, 2 to 32 CTAs, h through HBM) is 1.5-3.4x slower. Upper bound of any e4m3 kernel: cuBLAS(Lt) e4m3 against bf16
at the two GEMM shapes (`scripts/ub2.py`, CUDA graph, ms):

| D, n, rows | 2.5.12 module | expand bf16 / e4m3 | squeeze bf16 / e4m3 | GEMM time e4m3 saves (share of the module) |
|---|---|---|---|---|
| 384, 4, 256 | 0.0146 | 0.0048 / 0.0052 | 0.0058 / 0.0050 | 0.0004 (3 %) |
| 384, 4, 1024 | 0.0183 | 0.0078 / 0.0077 | 0.0071 / 0.0067 | 0.0005 (3 %) |
| 384, 4, 4096 | 0.0345 | 0.0215 / 0.0184 | 0.0105 / 0.0078 | 0.0058 (17 %) |
| 768, 2, 256 | 0.0165 | 0.0060 / 0.0057 | 0.0061 / 0.0051 | 0.0013 (8 %) |
| 768, 2, 512 | 0.0172 | 0.0073 / 0.0064 | 0.0076 / 0.0061 | 0.0024 (14 %) |
| 768, 2, 1024 | 0.0474 | 0.0105 / 0.0092 | 0.0080 / 0.0069 | 0.0024 (5 %) |
| 768, 2, 2048 | 0.0624 | 0.0174 / 0.0151 | 0.0107 / 0.0079 | 0.0051 (8 %) |
| 768, 2, 4096 | 0.0794 | 0.0331 / 0.0236 | 0.0162 / 0.0120 | 0.0137 (17 %) |

At token counts the GEMMs are latency-bound (a few waves of small tiles): e4m3 saves 0-17 % of the module even with perfect fusion, before the
quantization work, and the existing FP8 kernels lose 1.5-3.4x. The gate stays (D 384 / 512: 65536 rows); no token-count FP8 kernel was built.
(Side finding, bf16 not FP8: the 2.5.12 D 768 module at 1024-4096 rows takes 1.6-2.6x the time of its two bf16 cuBLAS GEMMs.)

### D 64: bound from data (`scripts/d64.py`, `res/d64a.jsonl`; not built)

Model: module time = HBM floor (a CUDA-graph `y.copy_(x)` of the same bytes) + the rest, and e4m3 GEMMs shrink only the rest. Calibrated on
D 128, where the FP8 kernel exists: (fp8 - floor) / (default - floor) = 0.58-0.61 at n 4 and 0.34-0.48 at n 2 (rows 2^18 / 2^20 / 2^22).

| D 64 | rows | 2.5.12 ms | floor ms | floor share | module bound with the D 128 factor |
|---|---|---|---|---|---|
| n 4 | 262144 / 1048576 / 4194304 | 0.081 / 0.270 / 1.062 | 0.025 / 0.093 / 0.448 | 32-42 % | 0.73 / 0.74 / 0.76 |
| n 2 | 262144 / 1048576 / 4194304 | 0.051 / 0.185 / 0.703 | 0.025 / 0.092 / 0.446 | 50-63 % | 0.74 / 0.72 / 0.76 |

So an e4m3 D 64 kernel (the D 128 recipe on `wide_d64_fwd_small_f3`: LayerNorm fragments quantized in registers, K = 64 = two k32 fragments,
h -> k32 fragment, m64n64k32 squeeze) could reach about 0.72-0.76 of the default by this model, i.e. the >= 10 % bar is met on paper. The model
assumes the D 128 overlap carries over to 64-row tiles with 3 CTAs per SM; it is an estimate, not a measurement. Not built on this branch
(optional item; next candidate).

## TriAttn: where else FP8 can go (2026-11-17, branch `fp8x/triattn`, base `fp8/3.0.0-on-2512` 91a9241fb)

Question: beyond the shipped e4m3 P / V (FP8 PV route), which operand of the H100 inference route can go to e4m3, what can it gain at most, and what does it cost in
accuracy? No algorithm change. Scripts and raw rows: `~/miniworld-engine-scratch/fp8x-triattn/scripts`, `res/` (prof2, emu1, real1, qk1, sp1, cb1).

### Stage times at 2.5.12 (ms per launch, torch profiler, B1, no mask; module = CUDA-graph replay; "def" = 2.5.12 default, "f8" = FP8 PV hi route with every gate removed)

| config, L | module def / f8 | core def / f8 | front def / f8 | tail or out-proj def / f8 | other |
|---|---|---|---|---|---|
| 128:4:32, 1024 | 2.434 / 2.111 | 1.750 / 1.449 (69 %) | 0.405 / 0.361 (17 %) | tail 0.274 / 0.277 (13 %) | |
| 128:4:32, 2048 | 15.35 / 13.00 | 13.51 / 10.87 (81 %) | 1.595 / 1.414 (11 %) | tail 1.076 / 1.085 (8 %) | |
| 256:8:32, 1024 | 5.434 / 5.069 | 3.582 / 3.122 (61 %) | wide 1.254 / 1.497 (29 %) | cuBLASLt 0.524 / 0.525 (10 %) | |
| 256:8:32, 2048 | 33.62 / 31.44 | 28.45 / 23.13 | wide 5.53 / 6.17 | 2.09 / 2.09 | |
| 384:12:32, 1024 | 9.087 / 8.707 | 5.960 / 4.811 (55 %) | ln_bias 0.78 + cuBLAS q\|k\|v\|g 1.99 (23 %) | 0.79 / 0.79 (9 %) | vt_quant - / 0.54 (6 %) |
| 384:12:32, 2048 | 57.87 / 53.40 | 22.88 / 19.04 per launch (row blocks) | ln 1.3-1.4 + cuBLAS 2.0-2.1 | 1.60 / 1.62 | vt_quant - / 1.13 |
| 64:4:16, 1024 | 2.263 / 1.915 | 1.769 / 1.434 (76 %) | front_v3 0.322 / 0.325 (17 %) | 0.128 / 0.128 (7 %) | |
| 64:4:16, 2048 | 14.27 / 11.86 | 12.29 / 9.87 | 1.248 / 1.275 | 0.532 / 0.533 | |
| 64:4:32, 1024 | 2.434 / 2.148 | 1.796 / 1.490 | 0.458 / 0.457 | 0.172 / 0.172 | |

Shapes, FLOPs and bound (L1024, M = L^2 = 1.05 M rows; bound from bytes / FLOPs over the measured time):
- d128 front (`tri_front_sm90`): LN + [Wq; Wk; Wv] and the bias, K = 128, N = 384 + 16: 107 GFLOP; reads x 256 B / row, writes q, k fp16 512 B, v 256 B (FP8 route: e4m3 V^T 128 B), bias 8 B
  = 1032 (904) B / row: 2.6-2.7 TB/s, HBM-bound (as documented). The e4m3 MMA rate buys nothing here; only fewer bytes do.
- core: QK^T and PV 2 L^3 H 32 each = 275 + 275 GFLOP (4 heads, L1024), not bandwidth-bound. Removal ablation of the F8 core's QK^T wgmma (`TA_ABL=1`, time only,
  random inputs, module level): 128:4:32 -6.0 % (L1024) / -8.3 % (L2048), 256:8:32 -5.0 % (L1024), 64:4:16 0 %; on real activations the QK^T share of the F8 core is
  2.5-7.4 % (fourth-round ablation). With the 2x e4m3 rate the core can gain at most about half of that.
- d128 tail (`tri_tail_sm90<true>`): gate GEMM LN(x) Wg^T + u Wo^T, 69 GFLOP; o 256 + x 256 + out 256 = 768 B / row: 2.9 TB/s, HBM-bound.
- wide front (256:8:32, `tri_front_wide_sm90`): 4 projections + bias, K = 256: 554 GFLOP, 443 TFLOP/s: tensor-bound (GEMM-only ablation 1.07 of 1.25 ms, see
  docs/h100/triangle_attention.md). Its FP8 variant (group-2 CTA writes the V^T plane) is 0.24 ms slower than the plain front at L1024 in this profile.
- 384:12:32 front: `ln_bias_fwd` + cuBLAS fp16 [M, 384] x [384, 1536] = 1.24 TFLOP at 625 TFLOP/s: tensor-bound.
- out-projection (cuBLASLt `lt_mm`, residual in the epilogue; 256 / 384 / 64): 137 / 309 / 34 GFLOP, 3.05-3.07 TB/s: HBM-bound.

### Candidates (upper bound = the stage share that the e4m3 change can remove, at L1024 / L2048 of the FP8 PV route)

| stage | e4m3 operand | share of the stage | upper bound (module) | error cost (real activations, attention output) | done |
|---|---|---|---|---|---|
| core QK^T (all configs) | q, k (m64n128k32 e4m3, fp16 accumulator) | ablation 0-8 % of the module | 0-4 % | rel L2 2.2-3.1 % mean, max-norm up to 22 % (q, k alone, emulated) | yes |
| d128 front stores | q, k written as e4m3 | 256 of 904 B / row (-28 %) of 17 % / 11 % | 4.8 % / 3.1 % | (same operand as the row above) | yes (with the core) |
| d128 tail | o (core epilogue writes e4m3, tail reads it) | 128 of 768 B / row (-17 %) of 13 % / 8 % | 2.2 % / 1.4 % | +2.65 % rel L2 on o (e4m3 floor; v+p+o 3.1-3.3 % vs v+p 1.8-2.2 %) | no: gain below 2.5 %, error +50 % |
| wide front (256:8:32) GEMM | x (per-row scale), W (per-column) | GEMM 1.07 of 1.25 ms, 29 % | ~10 % (L1024) | projection from e4m3 operands 3.8 % rel L2 (stored e4m3: 2.65 %): q / k / bias worse than q, k e4m3 storage | no: q / k / bias need >= fp16; v / g alone do not shorten the 4-CTA lockstep group (one projection per CTA) |
| 384 cuBLAS q\|k\|v\|g | y, W of the v \| g half only | <= 25 % of the GEMM, 23 % / ~7 % | <= 5.7 % / 2 % | v: 3.8 % before its own V^T quantization | no: needs an extra e4m3 copy of y (+384 B / row, ~0.13 ms at L1024) and a split GEMM; net <= 4 %, worse V |
| out-projection (256 / 384 / 64) | u (core epilogue), Wo | 1/6 of the bytes of an HBM-bound GEMM, 7-10 % | <= 1.5 % | +2.65 % on u | no |
| bias projection | Wb x | 8-24 B of >= 900 B / row | ~0 | logits | no |
| bidirectional bias-only | P, V planes for `torch._scaled_mm` | 4 x scaled_mm 131 vs 171 us bf16 (L768, per direction) | 80 us for both directions, against >= 110 us for the V planes (+ transposed plane for the ending contraction) | 3.8 % | no: still no-go; a fused quant would need e4m3 output from the bf16 planes GEMM (cuBLASLt has none) or a hand GEMM (the fused LN + planes GEMM was already slower) |
| training forward / backward | - | - | - | - | no: inference gain is small, the e4m3 backward had no speed gain (Known limits) |

### Implemented: FP8 QK (EXPERIMENTAL, `h100_sm90.FP8_QK`, default off)

- Core `triattn_h16_sm90.cu` `-DTA_F8QK=1` (with `TA_F8PV`): q / k arrive as e4m3 rows (32 bytes a head); the Q tile and the K tiles are 32-byte rows in the 32-byte
  swizzle (the TMA box and wgmma descriptor of the 16-channel K16 path); the Q fragment of one k32 step is one `ldmatrix.x4` of the same bytes as the fp16 k16 layout;
  `S += Q K^T` is one `wgmma m64n128k32.f16.e4m3.e4m3` (fp16 accumulator, so the bias-initialised accumulator and the f16x2 softmax are unchanged). 32-channel heads only.
- Scale: per head a power of two `sq` with q8 = q / sq and k8 = k * sq: the product is 1, so the bias tile and the softmax scale are untouched; `sq` = 2^round(log2(amax_q / amax_k) / 2),
  refused (fp16 score GEMM kept) when amax_q * amax_k > (448 / 2)^2. Real activations: amax_q <= 13.8, amax_k <= 17.1, product <= 186 (limit 50 176).
  Calibrated on the site's first eager call (the V calibration call); saturation sets the shared overflow flag; `rescale_all` turns the q / k e4m3 off for the site
  (a product-1 scale cannot widen).
- Front `tri_front_sm90<3, VTW, true>` (d128 fused route only): the q / k chunks leave as e4m3 rows (16-bit stores of e4m3 pairs into the 128B-swizzled stage, one
  uint8 TMA store per 64 rows). `FP8_QK_FRONT = False`: a torch pass instead (measurement only: +130-180 % module time). Other configs (wide / cuBLAS / d64 fronts) are
  unchanged; their core gain alone was measured with pre-made e4m3 q / k (below).
- Test: `tests/gpu/test_triattn_f8qk_h100.py`.

Module time, 128:4:32, one sample, CUDA graph, interleaved with a same-cubin control (cool / sustained, protocol of "Base 2.5.12"; random weights 0.08, no overflow):

| L | 128 | 256 | 512 | 768 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|---|---|
| 2.5.12 default ms | 0.031 / 0.030 | 0.103 / 0.101 | 0.435 / 0.468 | 1.159 / 1.228 | 2.415 / 2.561 | 7.051 / 8.152 | 15.38 / 17.10 |
| FP8 PV / default | 1.086 / 1.089 | 0.934 / 0.934 | 0.909 / 0.881 | 0.878 / 0.886 | 0.865 / 0.868 | 0.844 / 0.837 | 0.845 / 0.849 |
| FP8 PV + QK / default | 1.059 / 1.060 | 0.930 / 0.931 | 0.848 / 0.822 | 0.819 / 0.837 | 0.804 / 0.822 | 0.794 / 0.801 | 0.795 / 0.812 |
| FP8 PV + QK / FP8 PV | 0.975 / 0.973 | 0.995 / 0.996 | 0.932 / 0.932 | 0.933 / 0.944 | 0.929 / 0.946 | 0.940 / 0.957 | 0.940 / 0.955 |
| same-cubin control | 0.980 / 0.984 | 0.997 / 0.999 | 1.002 / 0.991 | 0.999 / 0.995 | 0.997 / 1.000 | 0.998 / 0.993 | 0.998 / 1.000 |

Per kernel (profiler, FP8 PV -> + QK): L1024 core 1.443 -> 1.366 ms (-5.3 %), front 0.372 -> 0.301 ms (-19 %); L2048 core 11.07 -> 9.90 (-10.6 %), front 1.407 -> 1.164 (-17 %); tail unchanged.
Core alone, pre-made e4m3 q / k (random, cool, control within 0.7 % except 12:2048 2.3 %): 4 heads L1024 / 2048 0.949 / 0.942, 8 heads 0.952 / 0.986, 12 heads 0.950 / 1.003:
without a front that writes e4m3 q / k the wide / cuBLAS sites would gain at most ~3 % of the module at L1024 and nothing at L2048, before paying a quantization pass.

Error (vs fp64):

| case | 2.5.12 default | FP8 PV | FP8 PV + QK |
|---|---|---|---|
| real activations, kernels on the dumps (5I28 x2, L256, 6 layers x 2 nodes), attention output rel L2 mean (max), Protenix v1 | 0.18 % (0.19) | 2.23 % (2.67) | 3.96 % (5.93) |
| same, Boltz-2 | 0.18 % (0.19) | 2.07 % (2.73) | 3.16 % (4.68) |
| same, AF3 | 0.17 % (0.20) | 2.17 % (2.57) | 3.21 % (4.51) |
| same, OpenDDE (12 heads) | 0.17 % (0.19) | 2.40 % (2.57) | 3.12 % (3.97) |
| same, max-norm (max\|o - o64\| / max\|o64\|), worst over all | 0.52 % | 6.2 % | 23.6 % |
| module 128:4:32 L512, update rms / update max | 1.86 % / 2.07 % | 3.31 % / 4.67 % | 4.39 % / 6.78 % |
| module 128:4:32 L1024, update rms / update max | 2.07 % / 2.15 % | 3.09 % / 3.36 % | 4.06 % / 6.61 % |
| judge form max\|y - y64\| / max\|y64\|, L512 / L1024 (err_pytorch = the PyTorch backend, bf16 autocast: 0.32 % / 0.30 %, so tol = 8e-3) | 0.29 % / 0.28 % | 0.66 % / 0.43 % (inside) | 0.95 % / 0.85 % (outside) |

No NaN / inf, no saturation in any run. The emulation (float64, e4m3 rounding of the named operand only, the same dumps) agrees: q / k alone 1.9-3.1 % mean rel L2, q / k + V + P
2.9-3.7 % against 1.8-2.2 % for V + P; q / k feed the logits, so the error concentrates on the rows with a few dominant keys (max-norm).

Verdict: FP8 QK pays 5-7 % of the module on the fused d128 route from L512 (on top of the FP8 PV route's 9-16 %; 0.80-0.82 of 2.5.12 together), but it is the
first placement whose judge-form error crosses 8e-3, and its worst element error is 4x the FP8 PV route's. Not whitelisted; end-level (FoldForge) not checked.
Default route: bit-identical to the base (see the verification line below).

Verification (2026-11-17): default route (no switch) against the base tree 91a9241fb, separate processes, sha1 of every output (`bit.py ta`: the 25 module shapes of
"Verification on the 2.5.12 base" + 6 module-level calls incl. training with every gradient): 31 / 31 identical. GPU tests: `test_triattn_f8qk_h100.py` 6 (new),
`test_triattn_f8pv_h100.py` 31, `test_triattn_h100_tail_gate.py` 15, `..._key_tiles.py` 15, `test_triattn_h16_core_dh16.py` 2, `..._front_lnb.py` 6, `..._front_y16.py` 14,
`..._variants_infer.py` 17, `test_cuda_refuses_not_raises.py` 8 passed. CPU: `test_prebuild_plan`, `test_requirements`, `test_judge`, `test_plot` 1196 passed; ruff / pyright clean on
the touched Python files. The F8QK core and the QK8 front instance are not in the prebuild plan (JIT on first use; experimental).

## Token DiT forward: e4m3 GEMM operands (EXPERIMENTAL, `conditioned_transition.cuda.token_dit_fp8`, branch `fp8x/token-dit`)

Scope: the H100 bf16 inference block of `TokenDiTProtenix` / `TokenDiTMiniWorld` with the default model options (the CondFold route of
`docs/h100/token_dit.md`) and Chai-1's option block (parallel, identity AdaLN; expand / out / squeeze, see "Chai-1's option block" below), width 768. Off by default; with no switch the route is bit-identical to 2.5.12 (below). The block runs
24 x per diffusion step x 200 steps, so on small targets it is a large share of the sampler; the trunk FP8 routes do not reach it.

```
MINIWORLD_TOKEN_DIT_FP8=qkvg,expand        # placements: qkvg, expand, out, squeeze ("1" = qkvg,expand; "all")
MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS=<n>       # one row gate for all (default per placement: qkvg 2560, expand 640, out / squeeze 2560 rows M = A B L)
```

Scales: **dynamic per row** for every activation operand (row max / 448, computed by the pass that makes the row), static per output
channel for the weights (made once per weight pack, `_capture.lookup` like every pack). Nothing can saturate, whatever a step's noise
level, so there is no calibration and no guard: the TriMul / TriAttn static-scale + `take_overflow` contract is not needed and not
duplicated. The cost is one more block reduction in the row pass (below).

### Where the time goes (A = 5 samples, shared conditioning, 16 x 48, pair 128; torch profiler kernel time, eager after warm-up)

| stage | work (M = A L) | L768 us | L1536 us | counters (ncu, L768 / L1536) | e4m3 placement | upper bound |
|---|---|---|---|---|---|---|
| conditioning GEMM `[L, 384] x [384, 6 x 768]` fp32 out | 2.7 / 5.4 GF | 9.8 | 15.3 | (cuBLAS, not profiled) | none: per-token AdaLN / gate tables, small, fp32 by design | - |
| `adaln_in_cf` rows | 5.9 MB in, 5.9 MB out (bf16) | 12.6 | 23.6 | issue / latency bound: SM 60-69 %, DRAM 11 / 26 MB, XU 31-34 % | write xa as e4m3 + row scale (`adaln_in_rows_cf_q8`) | -1.5 us (half the write) |
| q\|k\|v\|g GEMM `[M, 768] x [768, 3072]` | 18.1 / 36.2 GF | 31.0 | 56.8 | cuBLAS 582 / 635 TFLOP/s | xa e4m3 (row) x W e4m3 (channel), `torch._scaled_mm` row-wise, bf16 out + bias | GEMM alone 31.1 -> 24.0 / 58.8 -> 44.8 us (micro-bench, row-wise) |
| attention core `attn_fwd` (gated) | QK + PV 9.1 / 36.2 GF | 46.4 | 145.7 | tensor pipe 24 / 29 %, issue active 50 / 57 %, eligible warps 0.97 / 1.15 per scheduler, XU 36 / 42 %, ALU 37 / 43 %, L2 47 %, DRAM 25 % | P.V on e4m3 (TriAttn-style V^T planes) | small: the core is latency bound, not tensor bound; PV is about half of a 24-29 % tensor pipe -- not built (see below) |
| out GEMM `[M, 768] x [768, 768]` | 4.5 / 9.1 GF | 11.0 | 16.9 | cuBLAS | og e4m3 needs its own row pass (heads of a row are written by 16 CTAs: no per-row scale in the core's epilogue) | measured loss (below) |
| `resgate_adaln_cf` rows | | 14.0 | 26.7 | as adaln_in | write the expand input as e4m3 + row scale | -1.5 us |
| expand + SwiGLU (`gemm_swiglu_sm90`) `[M, 768] x [768, 2 x 1536]` | 18.1 / 36.2 GF | 29.2 | 57.8 | wgmma, `sm__throughput` 64-73 % | e4m3 x e4m3 wgmma k32 (`gemm_swiglu_f8_sm90.cu`, same tiles / ring / epilogue + scales) | measured 20.8 / 42.8 us |
| squeeze GEMM `[M, 1536] x [1536, 768]` | 9.1 / 18.1 GF | 15.9 | 29.7 | cuBLAS | h e4m3 needs a row pass (h's row spans 12 / 24 tiles) or a static scale + guard | measured loss with the row pass |
| `resgate_out_cf` rows | | 13.0 | 23.8 | | none (bf16 output) | - |
| total | | 183.3 | 398.7 | | | |

Not placed: the pair bias is already hoisted (built once per pair / mask / weights, `_capture.lookup_inputs`; not per step). Storing it as
e4m3 was rejected without building: it is a log2-unit logit, and e4m3 spacing at |bias| 2-8 is 0.125-0.5, i.e. a 9-41 % probability
error per key. FP8 P.V in the attention core: not built -- at head dim 48 the core issues on half the cycles with ~1 eligible warp and the
tensor pipe at 24-29 %, so halving the PV tensor time does not shorten the critical path (the exp / max / bias / ex2 work does), and it
would need e4m3 V^T planes written by the q|k|v|g GEMM plus a P re-layout (an algorithm change, out of scope).

### Kernel time per placement (A5, profiler, us; bf16 -> e4m3)

| | L768 | L1536 |
|---|---|---|
| q\|k\|v\|g GEMM | 31.0 -> 25.9 | 56.8 -> 45.3 |
| `adaln_in` rows (+ e4m3 + row max) | 12.6 -> 14.0 | 23.6 -> 25.7 |
| expand + SwiGLU | 29.2 -> 20.8 | 57.8 -> 42.8 |
| `resgate_adaln` rows (+ e4m3 + row max) | 14.0 -> 15.4 | 26.7 -> 28.4 |
| out GEMM (+ og row pass) | 11.0 -> 10.4 + 5.3 | 16.9 -> 13.6 + 9.3 |
| squeeze GEMM (+ h row pass) | 15.9 -> 13.8 + 7.0 | 29.7 -> 19.5 + 15.1 |

The row passes get slower although they write half the bytes: the third block reduction (the row max) costs more than the bytes save
(one block per row, latency bound).

### Module time against the 2.5.12 default (one block call, CUDA graph, same process, interleaved; `cool / sustained`)

Protocol as the TriAttn / TriMul tables above (cool = 2 s idle + one ~10 ms interleaved burst, 3 reps, median; sustained = 2 s idle, 1 s
back-to-back, 3 timed blocks, mean); master weights (fp32 parameters, bf16 activations), shared conditioning, all-ones mask, every
row gate removed. Raw rows: `~/miniworld-engine-scratch/fp8x-tokendit/res/speed2.jsonl` (`speed1.jsonl`: first build of the row passes,
plus out / squeeze alone).

| A:L (M) | 2.5.12 default ms | same-cubin control | qkvg | expand | qkvg + expand | all four |
|---|---|---|---|---|---|---|
| 5:256 (1280) | 0.0742 / 0.0721 | 0.995 / 0.999 | 0.967 / 0.971 | 0.974 / 0.977 | **0.940 / 0.941** | 1.006 / 1.005 |
| 5:384 (1920) | 0.0968 / 0.0940 | 0.998 / 1.001 | 1.008 / 1.012 | 0.962 / 0.961 | 0.970 / 0.971 | 1.025 / 1.028 |
| 5:512 (2560) | 0.1264 / 0.1249 | 0.999 / 1.006 | 0.985 / 0.990 | 0.960 / 0.967 | **0.945 / 0.948** | 0.979 / 0.979 |
| 5:768 (3840) | 0.1816 / 0.1832 | 0.998 / 1.003 | 0.980 / 0.986 | 0.957 / 0.970 | **0.934 / 0.950** | 0.976 / 0.992 |
| 5:1024 (5120) | 0.2433 / 0.2519 | 0.995 / 0.998 | 0.969 / 0.973 | 0.957 / 0.949 | **0.935 / 0.927** | 0.956 / 0.960 |
| 5:1536 (7680) | 0.4004 / 0.4203 | 0.994 / 1.010 | 0.970 / 0.977 | 0.960 / 0.970 | **0.938 / 0.949** | 0.978 / 0.981 |
| 5:2048 (10240) | 0.5686 / 0.6030 | 0.995 / 0.995 | 0.971 / 0.975 | 0.964 / 0.973 | **0.944 / 0.939** | 0.991 / 0.969 |
| 1:768 (768) | 0.0746 / 0.0728 | 0.995 / 0.998 | 1.017 / 1.020 | 0.972 / 0.976 | 0.989 / 0.992 | 1.059 / 1.059 |
| 1:1536 (1536) | 0.1343 / 0.1318 | 0.995 / 0.997 | 1.007 / 1.011 | 0.966 / 0.966 | 0.981 / 0.982 | 1.009 / 1.009 |

out alone and squeeze alone (first run, `speed1.jsonl`): 1.02-1.05 at every shape (the extra row pass outweighs the e4m3 GEMM). The
default row gates follow this table: expand from 640 rows, qkvg from 2560 (it lost at 768 / 1536 / 1920 rows).

### Error against fp64 PyTorch (same module and inputs; random weights, LN weights 1 +- 0.1, biases +- 0.1, 3 masked keys)

`~/miniworld-engine-scratch/fp8x-tokendit/scripts/err.py`, raw rows `res/err1.jsonl`, `res/err2.jsonl`. Update error = rms / max of (y - y64)
relative to rms / max of (y64 - x); judge = max|y - y64| / max|y64| (`measure/accuracy._rel`), tolerance max(err of PyTorch bf16, 8e-3).
Every gate removed; no NaN / inf in any case (no saturation is possible with row scales).

| case | default 2.5.12 | PyTorch bf16 | qkvg | out | expand | squeeze | qkvg + expand | all four |
|---|---|---|---|---|---|---|---|---|
| A5 L384 | 0.66 / 0.65 % (3.6e-3) | 0.94 / 1.07 % (6.0e-3) | 1.25 / 1.16 % (**6.4e-3**) | 0.86 / 0.80 % (4.4e-3) | 5.35 / 5.64 % (3.1e-2) | 3.71 / 4.17 % (2.3e-2) | 5.45 / 5.19 % (2.9e-2) | 6.58 / 7.83 % (4.4e-2) |
| A5 L768 | 0.66 / 0.67 % (4.1e-3) | 0.94 / 1.05 % (6.4e-3) | 1.07 / 1.10 % (**6.7e-3**) | 0.79 / 0.77 % (4.7e-3) | 5.36 / 5.61 % (3.4e-2) | 3.73 / 3.86 % (2.4e-2) | 5.43 / 5.36 % (3.3e-2) | 6.56 / 7.20 % (4.4e-2) |
| A5 L1536 | 0.66 / 0.73 % (3.7e-3) | 0.94 / 1.14 % (5.8e-3) | 0.94 / 1.04 % (**5.3e-3**) | 0.74 / 0.76 % (3.9e-3) | 5.39 / 5.51 % (2.8e-2) | 3.73 / 4.19 % (2.1e-2) | 5.44 / 5.74 % (2.9e-2) | 6.57 / 6.39 % (3.2e-2) |
| A1 L768 | 0.65 / 0.58 % (4.0e-3) | 0.94 / 0.97 % (6.8e-3) | 1.06 / 1.00 % (**7.0e-3**) | 0.79 / 0.66 % (4.6e-3) | 5.38 / 4.57 % (3.2e-2) | 3.73 / 3.36 % (2.3e-2) | 5.44 / 4.57 % (3.2e-2) | 6.57 / 5.85 % (4.1e-2) |
| A5 L768, input x 0.05 | 0.64 / 0.69 % (7.0e-3) | 0.97 / 1.13 % (1.2e-2) | 3.54 / 3.86 % (3.9e-2) | 1.91 / 2.04 % (2.1e-2) | 5.36 / 6.03 % (6.1e-2) | 3.72 / 4.74 % (4.8e-2) | 6.39 / 6.42 % (6.5e-2) | 7.58 / 8.33 % (8.5e-2) |
| A5 L768, input x 20 | 7.71 / 7.65 % (2.5e-3) | 9.69 / 13.7 % (4.5e-3) | 7.76 / 7.87 % (2.6e-3) | 7.72 / 7.73 % (2.5e-3) | 9.37 / 10.7 % (3.5e-3) | 8.54 / 8.94 % (2.9e-3) | 9.41 / 10.7 % (3.5e-3) | 10.1 / 10.7 % (3.5e-3) |

Cells: update rms / update max (judge). Reading: every e4m3 GEMM whose both operands are e4m3 has ~3.7 % relative error (the e4m3
rounding of two operands: the squeeze row is exactly that); the expand puts it on both a and b of the SwiGLU (5.4 %). The transition
carries most of the block's update, so expand / squeeze show almost fully, while the attention placements (qkvg, out) are diluted by the
residual: at input scale 1 qkvg stays inside the judge tolerance (5.3-7.0e-3 < 8e-3), out too; expand and squeeze fail it 3-4x (as the
TriMul FP8 planes do). The dilution depends on the regime: with a small block input (x 0.05: the attention update dominates the
transition's AdaLN input) qkvg rises to 3.5 % / 3.9e-2. Random weights only: the real models' activation statistics along a 200-step
trajectory are not measured here (the per-row scales make the route independent of a step's overall scale, not of its row shape).

### Verification

- Default route (no switch) against the base tree (91a9241fb = `fp8/3.0.0-on-2512`), separate processes, sha1 of the output: 18 / 18
  identical (`TokenDiTProtenix`, `TokenDiTMiniWorld`, `TokenDiTRoseTTAFold3` x A5 L384, A5 L768, A1 L256 x bf16 / fp32; the
  Boltz-2 / Chai-1 calls of the script were refused identically by both trees). With `MINIWORLD_TOKEN_DIT_FP8=qkvg` only the four
  bf16 A5 Protenix / MiniWorld cases change (fp32, A1 L256 below the gate and the RoseTTAFold3 option block keep the default bytes).
- No existing source changed in a way that reaches a compiler: the FP8 kernels are two new files of a new extension
  (`token_dit_fp8_sm90`: `gemm_swiglu_f8_sm90.cu`, `token_dit_fp8_rows.cu`); `token_dit_rows.cu`, `gemm_swiglu_sm90.cu`, the shared
  `tmn_kernels.cuh` are untouched. Not in the prebuild plan (opt-in: JIT, ~90 s on first use).
- Tests: `tests/gpu/test_token_dit_fp8.py` (the e4m3 SwiGLU GEMM against a torch reference on the same e4m3 operands at ht 64 / 128,
  the row quantizer's scales and rounding bound, each placement's error against the bf16 route, switch-off and below-gate calls bit-equal
  to the default).

### Chai-1's option block (branch `fp8c/chai1`, 2026-11-18)

Why Chai-1 got nothing from `MINIWORLD_TOKEN_DIT_FP8=expand` in 3.0.1 (probe: FoldForge a9e1549 + engine 3.0.1, 5I28 n = 4 = 512 tokens, `MINIWORLD_FP8=safe`):
`token_dit_fp8.active(M, D, fp32, plain)` was asked with `plain = False` on all 48 token DiT calls of the probe (M = 2560 = 5 x 512, D = 768, bf16,
conditioning bf16 [5, 1, 512, 384], CondFold on): `plain` is `variant == VARIANT_DEFAULT`, and Chai-1's variant is `(parallel 1, sig 0, eps 0.1, gated 0,
up 0, kq_ln 0, ..)`. The AF3 / Protenix / OpenDDE blocks are plain; Chai-1 (identity AdaLN, parallel attention / transition, no gating query), RoseTTAFold3
(parallel, k-q LayerNorm) and Boltz-2 (up-gate) are not. Width, dtype, row count and conditioning were all fine.

Route now: the identity block (`parallel and not sig and not up and not kq_ln and not qk`) takes `expand`, `out`, `squeeze`; **not `qkvg`**. The
parallel block's transition input is the second AdaLN of the attention input's LayerNorm pass, so the new row kernel `adaln_in_rows_cf_par_q8`
(`token_dit_fp8_rows.cu`, the same extension) writes the attention input as bf16 (bit-identical to `adaln_in_cf_kernel<192, true, false>`'s output,
tested) and the transition input as e4m3 with a dynamic per-row scale; the e4m3 SwiGLU GEMM of the plain route is reused unchanged (same weight pack
key, which holds the variant). The sequential block's `resgate_adaln_cf_q8` is skipped for the parallel schedule. RoseTTAFold3 and Boltz-2 are not
opened by this change (they stay `active() == {}`).

Kernel time of 5I28 n = 4 (profiler, three forwards: 9632 token DiT blocks, 200 steps x 16 blocks per forward + warm-up): `gemm_swiglu_sm90` 198.7 ms -> `gemm_swiglu_f8_sm90` 145.3 ms,
`adaln_in_cf_kernel` 113.8 ms -> `adaln_in_cf_par_q8_kernel` 123.2 ms (the e4m3 image + row max): net -44 ms of 6949 ms profiled kernel time (0.6 %; per warm forward
~15 ms of 1.63 s).

Module error against fp64 PyTorch (`TokenDiTChai1`, random weights, master weights, bf16 activations / conditioning, 3 seeds each; update rms = rms(y - y64) / rms(y64 - x)):

| case | default | PyTorch bf16 | expand | out | squeeze | expand + out + squeeze |
|---|---|---|---|---|---|---|
| A5 L512 | 0.62 % (judge 5.4-6.4e-3) | 0.96 % (8.2-9.2e-3) | 5.0 % (4.7-5.1e-2) | 1.5 % | 3.5 % | 6.2 % |
| A5 L512, input x 0.05 | 0.55 % | 0.74 % | 5.3 % (judge 4.7-6.4e-2) | 0.98 % | 3.6 % | 6.4 % |
| A5 L512, input x 20 | 2.97 % (judge 2.5-2.8e-3) | 4.2 % (4.7-5.0e-3) | 5.8 % (judge 4.8-5.5e-3) | 3.3 % | 4.5 % | 6.8 % |
| A1 L512 | 0.62 % | 0.96 % | 5.0 % (4.3-5.3e-2) | - | - | - |
| A5 L1024 | 0.61 % | 0.96 % | 5.1 % (4.1-5.0e-2) | 1.4 % | 3.5 % | 6.3 % |

Same character as the plain block's expand (5.4 %, above): two e4m3 operands in the SwiGLU, well above the module tolerance (judge 8e-3) at input scale 1 -- the expand does not pass the
module rule, exactly as for AF3, where the end-level FoldBench (27 targets, 0 flagged) is the evidence. Raw rows `~/miniworld-engine-scratch/fp8c-chai1/res/err1.jsonl`.

End level (FoldForge a9e1549 `miniworld`, bf16, mode exact, 4 recycles, 200 steps, 5 samples, diffusion_seed 0, **trunk seeds 0-3**, six FoldBench picks 268-658 tokens;
arms: bf16; `fp8old` = `MINIWORLD_FP8=safe` with the token DiT route off = what 3.0.1 does for Chai-1; `fp8new` = `safe` with this branch). RMSD in A (Kabsch, all atoms present in both, per sample;
"bf16 pair" = largest same-sample RMSD between two bf16 trunk seeds; inside = max same-seed RMSD <= max(1.1 x bf16 pair, 0.5 A)):

| target (tokens) | bf16 pair max | fp8old same-seed max / mean | fp8new same-seed max / mean | max abs d ipTM old / new | inside (new) |
|---|---|---|---|---|---|
| 8dhj (268) | 2.04 | 1.75 / 0.66 | 1.73 / 0.67 | 0.009 / 0.010 | yes |
| 8hxq (358) | 20.85 | 19.94 / 5.59 | 19.90 / 5.71 | 0.029 / 0.028 | yes (weak probe: bf16 trunk seeds already differ by 20.9 A) |
| 8vfj (392) | 0.85 | 0.71 / 0.43 | 0.71 / 0.44 | 0.001 / 0.001 | yes |
| 8avq (400) | 2.68 | 1.82 / 1.00 | 1.83 / 1.02 | 0.004 / 0.005 | yes |
| 9gsq (450) | 24.17 | 9.32 / 3.66 | 9.35 / 3.72 | 0.051 / 0.052 | yes (weak probe) |
| 8ts0 (658) | 18.27 | 13.61 / 6.17 | 13.73 / 6.07 | 0.027 / 0.027 | yes (weak probe) |

6 / 6 inside; the tight targets (8vfj, 8dhj, 8avq: bf16 pair 0.85-2.7 A) show the token DiT route adding nothing measurable to the FP8 `safe` set's deviation that Chai-1 already had
(same-seed mean +0.00 .. +0.02 A). Files (6 targets x 4 trunk seeds x 3 arms = 72 runs) in `~/miniworld-engine-scratch/fp8c-chai1/e2e`, script `e2e_chai1.py`, table `an_e2e.py`.
This is six targets, not the 27 of the AF3 FoldBench: the lDDT / DockQ evaluator was not run for Chai-1.

5I28 (FoldForge a9e1549, warm forward, median of 6, CUDA graph + compile, node02; `run_1118_301_fp8` = the recorded 3.0.1 FP8 run, `bf16` = `run_1118_301_steady`; the
three arms of the new job (`fp8c-chai1/bench`) ran back to back in one job, bf16 control = engine default of this branch):

| n (tokens) | bf16 recorded s | FP8 3.0.1 recorded s | ratio | job: bf16 s | job: fp8old s | job: fp8new s | fp8old / bf16 | fp8new / bf16 | fp8new / fp8old |
|---|---|---|---|---|---|---|---|---|---|
| 1 (128) | 0.3394 | 0.3324 | 0.979 | 0.3181 | 0.3313 | 0.3277 | 1.041 | 1.030 | 0.989 |
| 2 (256) | 0.6303 | 0.6174 | 0.979 | 0.6286 | 0.6171 | 0.6099 | 0.982 | 0.970 | 0.988 |
| 4 (512) | 1.6922 | 1.6297 | 0.963 | 1.6926 | 1.6348 | 1.6283 | 0.966 | 0.962 | 0.996 |
| 8 (1024) | 6.1203 | 5.8481 | 0.956 | 6.1211 | 5.8951 | 5.8757 | 0.963 | 0.960 | 0.997 |
| 12 (1536) | 14.997 | 14.286 | 0.953 | 15.140 | 14.340 | 14.330 | 0.947 | 0.947 | 0.999 |
| 16 (2048) | 29.322 | 27.873 | 0.951 | 29.632 | 27.729 | 27.622 | 0.936 | 0.932 | 0.996 |

fp8new against the recorded bf16: 0.965 / 0.968 / 0.962 / 0.960 / 0.956 / 0.942. The token DiT route gains 0.1-1.2 % of the forward (the whole gain of the expand is 0.6 % of the profiled kernel time);
the n = 1 cell is noise-dominated (bf16 repeats 0.318-0.560 s in the job; the recorded and the job bf16 differ by 6 %). Chai-1 is the weakest FP8 model because its trunk is dominated by
what FP8 does not serve (TriAttn d64: 14 % of the kernel time, no Chai-1 whitelist; the sampler's attention core; elementwise adds 6 %; 7 % host-to-device copies), not because a route is closed.

OuterProductMean (the second gap): not closed, on purpose. Chai-1's OPM is the grouped form (8 groups x 8, product LayerNorm, no count division; `kernels.outer_product_mean.chai`); `opm_fp8` serves groups = 1
(`opm_train.SHAPES`) only, so `OuterProductMean._cuda_route` returns `chai` for all 72 calls of the probe (msa [1, 1024, 512, 64], d_pair 256), and the OPM FP8 counter stays 0 whatever the S gate says
(S = 1024 at L = 512 would pass it). A grouped e4m3 product would be a new kernel family (the product is a batched cuBLAS GEMM over 8 groups with K = S = 1024, then the LayerNorm epilogue); the
whole grouped OPM is 0.7 % of the profiled kernel time (product `nvjet_tst_256x128` 26.6 ms, `chai_epilogue_kernel` 19.2 ms of 6949 ms over three forwards, plus the projection prologue), so an
e4m3 product (at best -40 % of the product's share) is worth ~0.2 % of the forward. Not built (Chai-1's MSA in the 5I28 benchmark is 1024 rows; a deeper MSA would change that share).

Verification: `tests/gpu/test_token_dit_fp8.py` +4 tests (placements `active()` logic; every placement of the identity block against the bf16 route; `qkvg` / switch off = default route bytes; the parallel row kernel:
bf16 output bit-equal to the bf16 pass, e4m3 image and row scale of the transition AdaLN) -- 46 passed with `test_token_dit_cond_fold.py` and `test_token_dit_shared_cond.py` (default-route regressions, 11 min
including the JIT build); `tests/cpu/test_capture_api.py`, `test_token_swiglu_tiles.py` 15 passed; ruff and pyright clean on the touched files. The FP8-off default route is untouched by construction (new kernels are in the FP8
extension only; the Python side adds one `identity` boolean that is computed but unused while `PLACES` is empty). The `dev code-graph` CLI is not in the tree, so the test files were chosen by import reach of `token_dit_fp8` / the token DiT cuda route.

## Atom DiT forward: e4m3 row-kernel GEMMs (`local_dit_h100.FP8_ROWS`, EXPERIMENTAL, opt-in; branch `fp8x/atomdit`, 2026-11-17)

Scope: the H100 `"cuda"` backend of the local atom DiT block (`modules/atom_dit.py`, AF3 32 x 128 windows; `kernels/integrations/local_dit_h100.py`),
bf16 (`master`) inference with a conditioning shared by the samples (the sampler step: `cond` an expanded view over A > 1), the AdaLN classes
(Protenix / OpenDDE, AF3 / OpenFold3 / IntelliFold2) and Chai-1 (`IDENT` / `PARS` / `NOOUT`). Boltz-2 (`UP`), RoseTTAFold3 (`KQLN`), ESMFold2
(`RMS`), per-sample conditioning tables, fp32 and training keep the bf16 / TF32 kernels. Default off; with it off the route is the 2.5.12 one.

| switch | effect |
|---|---|
| `MINIWORLD_ATOMDIT_FP8=1` (or `all`) at import, `local_dit_h100.enable_fp8(True)` | both row kernels on e4m3 wgmma (`pre90_f8`, `post90_f8`) |
| `MINIWORLD_ATOMDIT_FP8=pre` / `enable_fp8("pre")` | only the q / gate / k / v projections (`pre90_f8`): a quarter of the output error of `all` (table below) |
| `MINIWORLD_ATOMDIT_FP8=post` / `enable_fp8("post")` | only the out-projection, SwiGLU and squeeze (`post90_f8`) |
| `sm90_atom.f8.MINB` | override of the register target per kernel (default: by grid size, see `f8.py`) |

### Where FP8 can go: stages of the 2.5.12 inference step (AF3 block, 4096 atoms, A = 5 shared cond, bf16)

Per sampler step the block runs 5 kernels; the conditioning tables (`cond90`), the pair bias (`h_pair_bias`) and the slid-edge tables are made once
per fold (hoisted, `static_inputs()`), so they are not per-step work and FP8 there buys nothing over the 200 steps. Kernel times: torch profiler,
eager, warm (the graph replay of the step is 100-102 us). Counters: `ncu` (one call, same shape; `ncu` durations are inflated, ratios only).

| stage (kernel) | GEMMs per row (M = A N rows) | FLOP / row | HBM bytes / row | step time us (share) | ncu: tensor pipe / DRAM / occupancy / issue | e4m3 operand candidates |
|---|---|---|---|---|---|---|
| `pre90_fwd` | [128] x [128, 512] (q, gate, k, v) | 131 k | read 256 (x; mod / mkv shared: L2), write 1024 (q, sg, k, v); weights 128 KB per 64-row CTA from L2 | 21.4 (21 %) | 15.6 % / 12.7 % / 15 % / 40 % | A = x1, xkv (per row), B = Wq/g/k/v; stored q / k / v |
| `h_attn_fwd` | per head 32 x 128 QK^T (k = 32), 32 x 128 PV (k = 128) | 66 k | read q / k / v ~768 (k / v re-read from L2), write o 256 + lse 16 | 18.5 (18 %) | 13.4 % / 33.8 % / 12 % / 37 % | Q / K (k32 = head dim), P, V (needs V^T) |
| `edge_attn_fwd` (slide) | 5 windows, fp32 | - | ~0 | 16.2 (16 %, fixed: 100 CTAs) | 0.1 % / 1.8 % / 6 % / 17 % | none (latency bound, fp32 by design) |
| `post90_fwd` | [128] x [128, 128] (Wo), [128] x [128, 512] ([a, b]), [256] x [256, 128] (squeeze) | 229 k | read 768 (a, o, sg), write 256; weights 224 KB per CTA from L2 | 44.9 (44 %) | 16.3 % / 12.1 % / 11 % / 34 % | A = gated o, x2, h (per row), B = Wo, Wa / Wb, Wd |
| `tail_attn_fwd` | padding windows | - | ~0 | 2.3 (2 %) | - | none |

No stage is near a roofline: tensor pipe <= 16 %, DRAM <= 46 % (`h_attn_fwd` at 16384 atoms), 1 warpgroup x 2-3 CTAs per SM: the row kernels are
latency bound on the serial chain `cp.async weights -> wgmma (commit + wait) -> epilogue`. The head dimension (32) is exactly one e4m3 k32 step and
K = 128 / 256 of the row GEMMs is 4 / 8 steps, so the FP8 shapes fit; what pays is decided by the ablations.

**Removal ablations (honest upper bounds; module = the captured step, static weights + inputs, same process, interleaved, 5 x 100 replays, median;
scratch copies of the kernels with the ablation defines, outputs garbage, time only; `~/miniworld-engine-scratch/fp8x-atomdit/scripts/abl.py`):**

| ablation (what FP8 would remove) | 1024 | 4096 | 8192 | 16384 atoms (A = 5) |
|---|---|---|---|---|
| step time, us (same-source control) | 50.4 | 102.0 | 166.1 | 296.7 |
| row kernels: half of the wgmma k steps (e4m3 products) | 0.913 | 0.873 | 0.887 | 0.869 |
| row kernels: half of the weight bytes (e4m3 weights in shared memory) | 1.064 | 1.050 | 1.043 | 1.033 |
| both | 0.964 | 0.918 | 0.909 | 0.900 |
| attention: half of the q / k / v bytes (e4m3 storage) | 0.989 | 0.992 | 0.992 | 0.987 |
| attention: half of the mma.sync (e4m3 QK^T / PV) | 0.996 | 0.987 | 0.993 | 0.987 |
| attention: both | 0.993 | 0.998 | 1.003 | 0.985 |

So the only placement with a real upper bound is the e4m3 product of the row kernels (9-13 % of the step); weight bytes are not the limit, and the
window attention (softmax / latency bound with 32-wide heads) has at most 1-1.5 %: e4m3 q / k / v storage and an e4m3 attention core were not built.

### What is built

- `sm90_atom/f8w90.cuh`, `pre90_f8.cu`, `post90_f8.cu`, `sm90_atom/f8.py` (new files; the bf16 / TF32 sources are byte-identical to 2.5.12, so
  no prebuilt artifact re-keys). Same pipeline and math as `pre90_fwd` / `pre90_split` / `post90_fwd`, products on
  `wgmma.m64n{64,128}k32.f32.e4m3.e4m3` with the A operand from registers.
- A operand: the activation tile (x1, xkv, gated o, x2, h_j) scaled per row by a power of two from the row's exact amax (`2^floor(log2(448 / amax))`,
  exponent arithmetic), cast `cvt.rn.satfinite.e4m3x2` once into 16 registers (instead of the 64 fp32 of the tile) and reused by every product
  that reads it. Dynamic and exact: no calibration, no static scale, no overflow guard (a cast cannot saturate), so the noise-level dependence
  of the activations over the 200 steps does not matter (every A operand except h and the gated o is a LayerNorm output; o is a convex
  combination of v rows).
- B operand: the bf16 pack's weights scaled by a power of two per 64-row unit / matrix (q lo / hi, gate, k, v; Wo units; Wa_j / Wb_j halves; Wd),
  e4m3, once per weight version (`f8.pack8`, cached like the bf16 pack). The k32 slot order is the accumulator order of the previous product
  (slot `16 h + 4 t + 2 a + b` holds logical column `16 h + 8 a + 2 t + b`), baked into the weights' K axis, so the accumulator pairs are the A
  fragment with no shuffle. 128-byte swizzle, K-major, as the bf16 tiles (the Wd chunk's 64-byte rows padded to 128).
- Epilogue: `acc * (1 / row scale) * (1 / weight scale)` (powers of two: exact). The squeeze over the 4 hidden chunks (each chunk its own row
  scale) is accumulated in fp32 registers after dequantization; a first version that rescaled the wgmma accumulator between chunks gave wrong
  results (95 % of the update) whenever the x2 operand was kept quantized across the chunks -- not understood (ptxas register handling around
  register-A wgmma suspected); the fp32 sum avoids writing a wgmma accumulator between products.
- Registers: `pre90_f8` 236, `post90_f8` 253 at 1 CTA minimum (2 per SM); capped at 168 (3 per SM, ~130-270 B spill) they win where a third
  CTA per SM makes the grid one wave less: `pre90_f8` 3 up to 800 CTAs, `post90_f8` 3 only between 2 x and 3 x the SM count (`f8._minb`, measured
  26.1 -> 19.5 us / 42.7 -> 36.9 us at 320 CTAs; 79.2 -> 85.8 / 120.6 -> 131.5 us at 1280, so 1 there).

### Module time against the 2.5.12 default (H100 80GB HBM3, 700 W cap)

Protocol (`~/miniworld-engine-scratch/fp8x-atomdit/scripts/speed.py`, raw `res/speed2.jsonl`): registry cases (`master`, eval, A = 5, shared cond = `atom_single_shared_cond`),
the module captured in CUDA graphs of one call under `static_weights()` + `static_inputs()` (the sampler step: tables hoisted) -- the default route,
the default again (same-cubin control), FP8 pre + post, pre only, post only -- in one process; cool = 2 s idle then one interleaved burst
(~10 ms per route), 3 reps, median; sustained = 2 s idle, 1 s of back-to-back interleaved replays, then 3 interleaved blocks (~20 ms per route), mean.
Cells `cool / sustained`, ratio to the default.

| case (class: stream: atoms: A) | 2.5.12 default us | FP8 pre + post | FP8 pre only | FP8 post only | same-cubin control |
|---|---|---|---|---|---|
| AtomDiTOpenFold3: shared: 1024: 5 | 50.6 / 49.0 | 0.915 / 0.916 | 0.942 / 0.940 | 0.974 / 0.973 | 0.999 / 0.999 |
| AtomDiTOpenFold3: shared: 2048: 5 | 65.4 / 63.5 | 0.937 / 0.944 | 0.946 / 0.952 | 0.957 / 0.958 | 0.974 / 0.974 |
| AtomDiTOpenFold3: shared: 4096: 5 | 101.8 / 99.6 | 0.912 / 0.915 | 0.986 / 0.988 | 0.935 / 0.934 | 1.000 / 1.002 |
| AtomDiTOpenFold3: shared: 8192: 5 | 166.9 / 163.6 | 0.970 / 0.974 | 0.987 / 0.993 | 0.968 / 0.979 | 0.991 / 0.997 |
| AtomDiTOpenFold3: shared: 12288: 5 | 234.0 / 230.1 | 0.959 / 0.961 | 0.975 / 0.977 | 0.982 / 0.985 | 1.001 / 1.003 |
| AtomDiTOpenFold3: shared: 16384: 5 | 298.9 / 295.6 | 0.950 / 0.949 | 0.965 / 0.963 | 0.983 / 0.982 | 0.999 / 0.997 |
| AtomDiTProtenix: shared: 4096: 5 | 82.7 / 81.0 | 0.902 / 0.905 | 0.987 / 0.989 | 0.912 / 0.915 | 0.998 / 0.999 |
| AtomDiTProtenix: shared: 16384: 5 | 276.1 / 272.1 | 0.954 / 0.956 | 0.971 / 0.972 | 0.978 / 0.983 | 0.999 / 1.000 |
| AtomDiTChai1: shared: 4096: 5 (slide) | 91.8 / 89.6 | 0.949 / 0.955 | 0.990 / 0.993 | 0.946 / 0.957 | 0.998 / 0.998 |
| AtomDiTOpenFold3: per-sample cond: 4096: 1 (before the shared-cond gate) | 51.6 / 50.1 | 1.087 / 1.091 | 1.116 / 1.119 | 0.996 / 1.006 | 0.984 / 0.984 |
| AtomDiTOpenFold3: per-sample cond: 16384: 1 (before the gate) | 100.5 / 98.2 | 1.019 / 1.022 | 1.016 / 1.022 | 0.990 / 1.001 | 0.996 / 1.000 |

The per-sample rows were measured before the route was gated to a shared conditioning (they are why it is); with the gate they run the default.
Gain: 3-10 % of the block step (pre + post), about half to two thirds of the ablation bound; the rest goes to the quantization (row amax,
casts, dequantizing epilogues) and the register / occupancy trade. The atom DiT is 3-12 % of a FoldForge warm forward (`activations.md`, 5I28
n = 2 / 4 / 8), so the end-to-end upper bound is about 0.3-1.2 %.

### Error against fp64 (registry inputs, `~/miniworld-engine-scratch/fp8x-atomdit/scripts/err.py`, raw `res/err3.jsonl`)

judge = max|y - y64| / max|y64| (the engine metric, `measure/accuracy.py`), tol = max(err_pytorch, 8e-3); rms_rel = ||y - y64|| / ||y64||.

| case | tol | PyTorch master judge | bf16 cuda (default) judge / rms_rel | FP8 pre + post judge / rms_rel / max abs | FP8 pre only judge / rms_rel | FP8 post only judge / rms_rel |
|---|---|---|---|---|---|---|
| AtomDiTOpenFold3: shared: 1024: 5 | 8.0e-3 | 5.79e-3 | 5.79e-3 / 3.85e-3 | 5.73e-2 / 2.87e-2 / 0.364 | 1.25e-2 / 7.86e-3 | 5.24e-2 / 2.78e-2 |
| AtomDiTOpenFold3: shared: 4096: 5 | 8.0e-3 | 7.66e-3 | 6.37e-3 / 3.84e-3 | 4.83e-2 / 2.86e-2 / 0.282 | 2.14e-2 / 7.99e-3 | 4.60e-2 / 2.77e-2 |
| AtomDiTOpenFold3: shared: 16384: 5 | 8.0e-3 | 7.23e-3 | 7.50e-3 / 3.85e-3 | 5.51e-2 / 2.86e-2 / 0.357 | 1.65e-2 / 7.86e-3 | 5.35e-2 / 2.78e-2 |
| AtomDiTOpenFold3: shared: 4096: 5 (slide_qblock) | 8.0e-3 | 7.66e-3 | 6.37e-3 / 3.84e-3 | 4.83e-2 / 2.86e-2 / 0.282 | 2.14e-2 / 7.99e-3 | 4.60e-2 / 2.77e-2 |
| AtomDiTProtenix: shared: 4096: 5 | 8.0e-3 | 7.74e-3 | 6.90e-3 / 3.85e-3 | 4.78e-2 / 2.86e-2 / 0.273 | 1.65e-2 / 8.10e-3 | 5.05e-2 / 2.77e-2 |
| AtomDiTChai1: shared: 4096: 5 (slide) | 9.6e-3 | 9.63e-3 | 9.07e-3 / 6.08e-3 | 9.77e-2 / 5.64e-2 / 1.200 | 9.71e-2 / 3.53e-2 | 6.80e-2 / 4.45e-2 |

Every FP8 output finite; no saturation is possible (dynamic per-row scales). FP8 pre + post is 6-11x the tolerance (rms 2.9 %, 7x the bf16
route); FP8 pre only is 1.6-2.7x (rms 0.8 %, 2x bf16) for the AdaLN models: its error passes through the softmax average and the gates, while
`post90`'s products feed the residual output directly. Chai-1 (identity AdaLN: raw c in the tables) is about twice as large in both.

### Tried, no gain / not built

- e4m3 q / k / v storage and an e4m3 window-attention core: upper bound 0.4-1.5 % of the step (ablations above); not built.
- e4m3 weights alone (bytes): the ablation is slower (1.03-1.06), the row kernels are not limited by the weight stream.
- First `pre90_f8` (fragments re-made per product from the scaled fp32 tile, per-row weight scales read in the epilogue): 1.09-1.32x SLOWER than
  the default (230-255 registers, 2 CTAs per SM, epilogue loads); quantizing the A operand once + scalar weight scales + the grid-size register
  target made it 0.90-0.97.
- Training: not ported (the forward gain is small; the backward kernels are the same register-A wgmma design).
- SWA atom transformer (MiniWorld `SWAAtomTransformer`, `kernels/swa_dit`) and the `edge.cu` slid-window kernels: not covered (the former is not a
  FoldForge inference path, the latter fp32 and latency bound).

Verification: default route bit-identical to 2.5.12 (sha1 of every output and, in train, every gradient: 69 / 69 tensors over 9 cases --
AF3 shared 1024 / 4096, AF3 per-sample 4096, Protenix, Chai-1 slide, Boltz-2, RF3 eval; AF3 A1 and Protenix A48 training -- branch tree vs the base
tree 91a9241fb, separate processes, `~/miniworld-engine-scratch/fp8x-atomdit/scripts/bit2.sh`). Tests: `tests/gpu/test_atom_dit_fp8_rows.py` (error bounds per part, finite,
off = default bitwise, Boltz-2 / per-sample / training fall back bitwise, switch parsing): 11 passed. CPU: `ruff check` / `ruff format` / `pyright` clean on the touched Python files; `test_capture_api.py`, `test_supports_signature.py`, `test_init.py`, `test_measure_shared_cond.py`: 621 passed, 1 failed -- `test_every_module_level_dict_is_registered_or_listed`, the same failure on the base tree (the TriAttn / TriMul FP8 module dicts; nothing of this branch). No registry row, tolerance or `backend_versions` change.
Prebuild: the two FP8 cubins are not in the prebuild plan (first FP8 call JIT-compiles them, ~3 s).

## Known limits

- Boltz-2 9c90 is bistable under any numerical noise of 0.1-2 % on the TriAttn output; V on e4m3 flips it (hi + lo too: 2 of 4 trunk seeds). In the current environment its own bf16 trunk seeds fall into 5 modes (24 seeds, 15.3 A, ipTM 0.40-0.85) and FP8 changes the mode of 9 of 24 same trunk seeds while the mode population stays equal (see the widened end-level section). Hence whitelist only, and Boltz-2 is not on it (open decision).
- Scales are per tensor (TriMul) or per head (TriAttn), powers of two. The calibration caches are keyed by a tensor address (TriMul: to_left's weight; TriAttn: the fused weight pack, an unfused site's V weight); on the 2.5.12 base an entry is dropped when that tensor is freed (before, a later module whose weight landed at the same address read the stale state: 137 % update error for an incoming module after an outgoing one, NaN for a bidirectional one; tests `test_calibration_state_follows_the_module_lifetime`, `test_static_scales_follow_the_module_lifetime`). `weights_changed()` does not invalidate it (inference only today: after an in-place weight update the cache is stale).
- A saturation on an out-of-distribution input is caught only after the forward (lazy guard); a caller that does not read `overflow_peak()` runs unguarded.
- Prebuild (3.0.0): every FP8 kernel the `safe` preset can reach is in the sm90a plan (560 artifacts against 2.5.14's 513; TriAttn F8 cores incl. F8QK, TriMul D64-D384 FP8 K1 / K3, OPM, PWA 8 x 32 + wide, Transition D128 / D256 / wide, token DiT, atom DiT pre / post; see "Release line 3.0.0: verification"). Earlier state: the TriAttn FP8 variants (F8 core hi / hi + lo, the V^T-plane fronts VTW 1 / 2, bf16 and fp16-exact: 6 artifacts) are in the prebuild plan (`kernels/triangle_attention/cuda/__init__.py`), so they ship in the wheel. The TriMul FP8 variants (the FP8-plane K1 tile(s) and the K3 with `K3_CSCALE`, any-L row form and for D256 also the token-flat form; D256 K1 per length of the requirements row) are in the plan too (`kernels/trimul_inproj/cuda/__init__.py`); `_h100_fp8` builds nothing (cuBLASLt e4m3 is a library call). Verified: a prebuilt bundle (`_prebuild --only trimul_inproj cast`) + an EMPTY local JIT cache + an FP8 forward (eager calibration, 3 calls, then a graph capture and replay) at L256 / 300 / 512 / 1024, D128 / D256, both directions: 0 compiled, 5 (D128) / 6 (D256) artifacts resolved from the packaged root, none from the local cache.
- cuBLASLt e4m3 heuristics and `cvt.satfinite` depend on the CUDA 12 toolchain.
- The TriMul and TriAttn backward are not ported (e5m2 no-go; e4m3 backward no speed gain as built).
- End to end gain is small: TriMul FP8 1.5-4.5 % (compile + graph warm forward; measured before the 2026-11-15 `fp8/tm2` fixes and gates, not re-measured).
- TriAttn unfused sites: the static V scale is calibrated on the first eager call of a site (as the fused front's); after `take_overflow()` the caller must `rescale_all()` and redo. A site not reached through `_infer` has no site key and runs the dynamic pass.
- TriAttn FP8 below the S gate (standalone-pass sites: 1024 tokens, 512 for 16-channel heads; front-made sites: see above, from L256 for d_pair 64) runs the fp16 core: short targets see no change at all.

## Verification on the 2.5.12 base (branch `fp8/3.0.0-on-2512`, 2026-11-17)

- Default (no FP8 switch) route against the 2.5.12 tree (origin/main 7c544cf2f), separate processes, sha1 of every output tensor (training: output, input
  gradient, every parameter gradient): **169 / 169 identical**, on the merged tree and again on the final head. Cases: TriAttn the 25 module shapes of the
  verification below (64:4:32 / 64:2:32 / 64:4:16 / 128:4:32 / 256:8:32 / 384:12:32, L256-1536 rotating start / end / flip / masked, plus 128:4:32 L2048)
  and 6 module-level calls (eval both nodes, training with every gradient); bidirectional TriAttn 6 (L128-768, B1 / 2, mask / none, eval + training);
  TriMul one direction D64 / 128 / 256 / 384 / 512, outgoing / incoming, L256 / 300 / 512 / 1024 / 1536 (D384 / 512: L256 / 512 / 1024), B1 / 2: 68 inference
  + 16 training (D64 / 128 / 256 / 512, L256 / 384); bidirectional TriMul D64 / D128 L128-2048 (port K3 range included), D256 L256-1024, mask / none, B2,
  training: 44; OuterProductMean S512 / 1024 / 4096 / 8192 (the S > 4096 route included): 4.
- FP8 routes: error identical to the 2.5.9-based branch (table in "Base 2.5.12"); the speed / gate re-measure is in the same section.
- `tmn_kernels.cuh` carries the FP8 K1 code (under `TMN_K1_F8` / `TMN_K1_F8C`) and the schedule-6 ring generalisation, so every TriMul artifact that
  includes it re-keys (output bytes unchanged, above): the 3.0.0 sm90 wheel needs a rebuild of those (107 artifacts of trimul_inproj / triangle_attention
  in the plan were not in the 2.5.12 cache). The anchor-patch chains of the families that patch the header by anchor strings
  apply: every planned artifact built, `test_prebuild_plan.py` passes.
- Tests. GPU: the code-graph CLI (`dev code-graph --diff --tests`) is not in the tree since 2.5.0 (65f37dc3c), so the files were chosen by import reach of
  the changed modules / sources (32 files; the 3 sm80-only files left out): one 4-GPU work-queue job, 28 / 29 files pass;
  `test_trimul_big_plane.py::test_triton_training_past_2_31_elements[False-14]` (Triton backend, L4200 training) runs out of memory, the same on the 2.5.12
  tree alone (not touched here). The FP8 files again after the last code change: `test_triattn_f8pv_h100.py` 31, `test_trimul_fp8_planes.py` 25,
  `test_trimul_fp8_bidir.py` 16, `test_trimul_fp8_k1.py` 6 passed. CPU: `test_requirements`, `test_prebuild_plan`, `test_judge`, `test_modules_cpu`,
  `test_imports`, `test_supports_signature`: 2926 passed. `ruff check` and `pyright` clean on the touched Python files (`ruff format` reports three
  2.5.12 lines that this branch leaves as they are). `_prebuild --check` sm90a / sm80 / sm120: the same findings as the 2.5.12 tree (sm90a plans 526
  artifacts against 513: the FP8 variants). `judge owed` H100 / A100 / RTX 5090: accuracy and speed 0 (no `version` / `backend_versions` change).

## Verification of the merged branch (`fp8/main`, 2026-11-15)

- Default (non-FP8) route, TriAttn: 25 module shapes (d_pair 64:4:32 / 64:2:32 / 64:4:16 / 128:4:32 / 256:8:32 / 384:12:32 at L256 / 512 / 1024 / 1536 rotating start / end / flip / masked, plus 128:4:32 L2048), output bytes equal to main ec11768fc: 25 / 25 identical, before and after the macro cleanup. TriMul default (D64 / 128 / 256 / 384, both directions, L256 / 300 / 512 / 1024 / 1536): 40 / 40 identical to main after the cleanup.
- FP8 route against `fp8/vt` (c7bf0ee76; before `TA_F8N24`): hi and hi + lo, the same 25 shapes with every S gate removed, output bytes: 25 / 25 and 25 / 25 identical (`TA_F8N24` is bit-identical, as reported by the core work). After the cleanup against the merged state before it: 25 / 25 and 25 / 25 identical.
- TriMul FP8 planes (D128 / D256, both directions, L256 / 300 / 512 / 1024 / 1536) and the experimental FP8-K1 route: output bytes equal to `fp8/3.0.0` (e58b9488c) after the cleanup, 20 / 20 and 20 / 20.

## Verification of `fp8/tm2` (TriMul FP8 glue fixes, gates, prebuild plan; 2026-11-15)

- Output bytes: TriMul FP8 planes (D128 / D256, both directions, L256 / 300 / 512 / 1024 / 1536) 20 / 20 identical to `fp8/main`; TriMul default (D64 / 128 / 256 / 384, both directions, same L) 40 / 40 identical (the header edit is under `TMN_K1_F8`; the default K1 builds re-key through the header text, their SASS is unchanged).
- CPU tests `tests/cpu/test_prebuild_plan.py`, `test_requirements.py`, `test_judge.py`: 1188 passed. GPU tests `tests/gpu/test_trimul_fp8_planes.py`, `test_trimul_fp8_k1.py`: 30 passed (one job, 1 GPU).
- Prebuild: see "Known limits" (0 compiled with a prebuilt bundle and an empty JIT cache).

## End level, AF3 FoldBench, all routes (2026-11-17, this tree ffcd465f7)

AF3 on the 27 FoldBench targets of the paper run (3 additional trunk passes, 200 steps, 5 samples, trunk_seed = diffusion_seed = 0, mode
exact, bf16, FoldBench's evaluator), control bf16 2.5.12 (this tree with every switch off is byte-identical to it end to end). Full tables:
`paper/data/foldbench_runs_fp8all/RESULTS.md`, `paper/data/foldbench_runs_fp8safe/RESULTS.md`; speed: `paper/data/fp8all_speed/RESULTS.md`,
`paper/data/fp8safe_speed/RESULTS.md` (FoldForge benchmark mode, compile + CUDA graph, warm forward, interleaved arms, same node per target).
Flag = mean of 5 worse than the control by more than max(2 x the larger sample sd, floor 0.02 lDDT / 0.05 DockQ, lDDT-PLI).

| arm | routes | flagged targets | lDDT mean5 - bf16 2.5.12 (22 targets) | same-sample RMSD median | mean signed dpLDDT | whole forward / 2.5.12 (128 .. 2008 tokens) |
|---|---|---|---|---|---|---|
| all-FP8 | TriAttn PV + q / k, TriMul, OPM bound, PWA, Transition, token DiT qkvg + expand, atom DiT | 3 (8j62 -0.045, 7uyn -0.023, 8xgu -0.021) | -0.0148 | 0.83 A | +0.120 | 0.974 .. 0.791 |
| token DiT qkvg only (+ every other route) | as all-FP8, `MINIWORLD_TOKEN_DIT_FP8=qkvg` | 2 (8j62 -0.053, 8xgu -0.027) | -0.0086 | 0.70 A | -0.145 | - |
| **fp8safe** | every route except token DiT | **0** | -0.0019 (top-1 +0.0021) | 0.60 A | +0.012 | 0.988 .. 0.806 |
| FP8 earlier | TriAttn PV + TriMul planes (2.5.9 tree) | 0 | - | 0.25 A (vs its 2.5.9 control) | +0.078 | 1.000 .. 0.851 |

References: bf16 2.5.12 vs PyTorch 0.38 A, different samples of bf16 2.5.12 1.71 A.

- The loss is the token DiT **qkvg** placement. Split on the 3 flagged + 2 control targets (every other route on): `td=qkvg` reproduces the
  drop (8j62 -0.056 / -0.053 in two runs), `td=expand` stays at -0.007 .. +0.001 (like token DiT off). This is the reverse of the module-level
  ranking in "Token DiT forward" (qkvg 0.9-1.3 % update error, inside the judge tolerance; expand 5.4 %): random-weight module error did not
  predict the end-level effect. qkvg fires only at M = 5 L >= 2560 rows (the 12 targets >= 392 tokens), where the drop is.
- PWA FP8 never fires on AF3 (8 heads x 8 channels go to `pwa_wide`); the single-representation Transition (D384) never passes its row gate.
- fp8safe keeps the full trunk gain (trunk 0.974 .. 0.736 of 2.5.12, equal to all-FP8 within 0.003) and gives up 1.3-3.9 % of the whole
  forward against all-FP8 (token DiT FP8 was 15-60 % of the diffusion gain; atom DiT FP8 stays).
- Not done: `td=expand` alone over all 27 targets (clean on the 5 split targets; it would recover part of the diffusion gain).
- Two runs with the same switches are not byte-identical when the first target of the process differs: the process-wide static scales
  (TriAttn, TriMul, Transition calib) are taken on the first calls (same-sample RMSD 0.23 A between the two qkvg runs).

### fp8safe2: + PWA wide + token DiT `expand` (2026-11-17, this tree c6a1e10ff = the merges of `fp8x/pwa-wide` and `fp8x/transition-d256`)

Full tables: `paper/data/foldbench_runs_fp8safe2/RESULTS.md`, speed `paper/data/fp8safe2_speed/RESULTS.md` (4 interleaved rounds per target).
Same protocol and flag rule as above; all 27 targets in one process (static scales calibrated on 8gzl, as fp8safe; PWA v scales on 8e4h, the
first target where PWA fires). FoldForge pinned to 399d0b4 (the commit of every earlier run): with it the tree with every switch off is again
byte-identical to 2.5.12 end to end (8gzl / 7sxp, 10 / 10 CIFs); the moving ff_2512 checkout (da381a5, engine LayerNorm everywhere) is not.

| arm | routes | flagged targets | lDDT mean5 - bf16 2.5.12 (22 targets) | same-sample RMSD median | mean signed dpLDDT | whole forward / 2.5.12 (128 .. 2008 tokens) |
|---|---|---|---|---|---|---|
| **fp8safe2** | every route incl. PWA wide (`MINIWORLD_PWA_FP8=1`, AF3 8 x 8 now served), token DiT `expand` only | **0** | **+0.0000** (top-1 -0.0021) | 0.54 A | +0.030 | **0.973 .. 0.795** |
| fp8safe (same jobs) | every route, token DiT off, PWA wide gated off | 0 | -0.0019 | 0.60 A | +0.012 | 0.985 .. 0.804 |

- PWA FP8 now fires on AF3: 21 / 27 targets (bucket >= 256 tokens), 0 overflow / rescale, peak |v / scale| 21 of 448. It costs nothing in
  accuracy and gives nothing measurable in time: AF3's trunk has 4 MSA blocks at S 1024, so the PWA module is ~0.1-1.7 ms of a 0.1-9.5 s trunk
  (fp8safe2 / fp8safe trunk 0.998-1.005).
- Token DiT `expand` (without qkvg) is clean over all 27 (8j62 -0.010, limit -0.029; 8xgu / 7uyn / 8b3w within +-0.006) and gives the
  diffusion gain: fp8safe2 / fp8safe diffusion + heads 0.958-0.987, whole 0.974-0.989 (1-2.6 % faster); its whole-forward ratios are 0.000-0.012 above
  all-FP8's (earlier run: 0.974 .. 0.791) without its flags.
- fp8safe1 (PWA wide, token DiT off) was not run: fp8safe2 has no flag and no drop, so PWA's own effect needed no isolation.
- Transition D256 (`MINIWORLD_TRANSITION_FP8`, merged here) is not reached by AF3 (pair D128, single D384 below its row gate).
- Merge verification: default route bit-identical to 2.5.12 (`wt-2512-base`, separate processes) on 44 cases: the registry rows of every FP8
  module (eval + train) plus PWA wide inference at 64:8 / 128:8 / 128:16 (N 256 / 768 / 777 / 2048 / 400 / 512 / 384), PWA wide training
  (64:8 N384, 128:16 N256: y, dmsa, dpair, parameter grads), the 8 x 32 PWA route, Transition D64 / 128 / 256 / 384 / 512 / 768 incl. the D256
  row-gate sizes 2048 / 8192 / 16384 / 2^20. GPU: `test_pwa_fp8_wide`, `test_pwa_fp8`, `test_pwa_long_inference`, `test_transition_fp8_d256_h100`,
  `test_transition_fp8_h100`: 82 passed. CPU: ruff / format / pyright clean on the merged Python files, `test_capture_api` + `test_init` 274
  passed, 2 skipped (no new module-level dict needed registering).

## Open decisions (owner)

Decided for 3.0.0 (see "Decisions for the 3.0.0 release" at the top): no FP8 tolerance / registry row (FP8 stays outside the judge); Boltz-2 stays
off the TriAttn table; TriAttn FP8 only for models in `fp8.TRIATTN_MODELS`; e4m3 q / k in `safe` for af3; experiment harnesses out of the release tree.
Still open:

- Gate re-measure: two candidates (256:8:32 front gate 768 -> 512, 32-channel standalone-pass gate 1024 -> 768), not changed.
- TriMul FP8 in the shipped prebuild: rel/3.0.0 plans and builds every FP8 variant the `safe` preset reaches on sm90a (no first-call JIT); dropping any is the owner's call.
- The experimental FP8-compute K1 (`MINIWORLD_TRIMUL_FP8_K1`, `TMN_K1_F8C`: 6.9 % update error, failed the Protenix pLDDT criterion) is still in the
  code and the shared header; removing it would simplify `tmn_kernels.cuh`.
- FoldForge: wire `fp8.check()` (and `MINIWORLD_FP8_MODEL` / `enable(model=)`) into its forward loop before offering `MINIWORLD_FP8=safe` there.
