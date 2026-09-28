# Large-D TriMul on Vast H100 (2026-09-27)

Scope: bidirectional BF16, FP32 affine, D=256/384/512 and L=384/768,
complete forward+backward timing. NCU is intentionally unavailable.

[PYTORCH_LATENCY_TABLE.md](PYTORCH_LATENCY_TABLE.md) is the current comparison:
eight-cell paired PyTorch/native full F+B latency, using torch.compile as the main
baseline and eager in a separate table. All 16 full-workload runs passed normal
and changed-input graph/reference checks. Native selection is unchanged.
[LATENCY_TABLE.md](LATENCY_TABLE.md) retains the previous Triton comparison and
the separately labeled output-LN diagnostic.

For the follow-up question about L384 speedup versus D128, see
[LENGTH_SCALING.md](LENGTH_SCALING.md). Paired profiles and a controlled intervention
identify a wide Triton output-LN dispatch penalty at L768. This diagnosis is not
a new native speedup. [SHORT_WORK.md](SHORT_WORK.md) records rejected short-shape
pilots and the current handoff; the qualified split4 selection below is unchanged.

Read ../../docs/operations/vast-h100.md before accessing the host. Use
`scripts/vast-sync.sh run GPU COMMAND...`; this session uses both GPUs in separate
sequential queues. Shared source pushes are blocked while those jobs hold locks.

The production engine wide path is older than the latest qualified standalone
research. Start from D256 `d256_pool_checkpoint`, D384 `wide_checkpoint23`, and
D512 `wide_checkpoint24`, not an early September 23 plan. Source snapshot origin
and SHA256s are in `source-manifest.json`. Local snapshot:
`.bench/trimul-large-d-source/runs`; remote snapshot:
`/workspace/experiments/trimul-large-d/runs`. The imported files retain their
original bytes. No third-party toolkit headers or binaries were transferred.

The snapshot deliberately has no `.engine-release-2.0.0/src` tree; imports use
current installed `/workspace/miniworld-engine/src`. Revalidate against that
runtime and CUDA 12.8 compiler, rather than claiming historical qualification
transfers unchanged. Hard-coded cuBLASLt algorithm assertions must pass or be
revalidated; never silently remove them. Historical complete F+B was approximately
D256 2.621/10.506 ms, D384 4.585/19.096 ms, D512 7.048/28.665 ms.

Results go under `/workspace/vast-results/trimul-large-d/` and are pulled into
`.bench/vast-20260927/results/trimul-large-d/`. Keep original checkpoints frozen.
Candidate changes belong in this experiment directory. No production dispatch
promotion without strict gradients, changed-input graph replay, independent
reference, sanitizers and paired full-workload measurements.

Portability findings:

- Package versions alone did not identify the loaded runtime: Conda's cuBLASLt
  12.8.5 took precedence over the PyTorch wheel's 12.8.4. The frozen algorithm
  assertions rejected it. Experiment wrappers preload the wheel's `libcublasLt`
  and `libcublas`; `/proc/self/maps` and `cublasLtGetVersion()` verify 120804.
  Assertions remain enabled. The shared environment is not globally overridden.
- D512/L384 checkpoint24 reproduced all five strict fixtures, changed-input
  graph replay, and independent PyTorch validation after the runtime correction.
- D256's register-allocation audit requires CUDA_HOME/bin/cuobjdump; install the
  matching CUDA 12.8 binary tools rather than skipping this check.

`validate.sh WIDTH LENGTH` reruns historical strict validation and saves logs.
`pilot.sh WIDTH LENGTH [--tune]` measures frozen full/BWD graphs, exports a CUDA
activity trace, and optionally searches cuBLASLt candidates. The search accepts
only bitwise-identical intermediate outputs on the initial fixture; additional
stress/graph/sanitizer qualification is still required before selecting a change.

To recreate the snapshot on this local server, run
`python3 experiments/trimul_large_d_vast/stage_sources.py`. It checks every source
against the manifest and refuses changed historical inputs. Transfer that staged
`runs/` directory to the isolated remote snapshot path using rsync (no delete),
only while no experiment is reading it. The manifest does not include CUDA
caches/binaries and is not a substitute for the newly compiled artifact hashes.

First Vast pilots: D256/L384 full 2.632 ms and D512/L384 full 7.003 ms.
The first exact-output Lt search did not improve either full workload; rejected.
A subsequent D512-short experiment (`overlap.py`, `check_overlap.py`) overlaps
independent output weight gradients with the remaining input backward and uses
explicit stream fork/join. Its status is experimental until recorded validation.

## Candidate and validation scope

`selected_vast.py:make_plan(leaves, mask, dropscale, dy)` exposes the explicit current
research selection. Add this experiment directory to PYTHONPATH, use a fresh
process per width, and call the returned plan inside `torch.no_grad()` on its
construction stream. This is not registered into the production module/autograd
path. Plans own saved state; do not reuse them across overlapping forwards.

The new change is only **D512/L384 input-weight gradient: 8 -> 4 FP32 splits,
Lt index 0**. Forward, GP, dX and affine formulas are preserved. Partial payload
falls from 64 MiB to 32 MiB (not a claim that total allocated memory is halved).
D256, D384 and D512/L768 retain their qualified historical checkpoints.

The original five-fixture strict validator passed, including changed inputs,
weights/mask/dropout, zero gamma, zero mask, zero dropout, graph replay and an
independent PyTorch oracle. Worst recorded source-weight relative error was
about 4.50e-4 against the original strict reference, below the unchanged 5e-4
limit; dX was identical to that reference. This leaves limited numerical margin,
so the selection must not be generalized to longer lengths or other dtypes.

Three paired repetitions on each GPU showed small full F+B improvements
(~0.3–1.3%); exact samples are in `repeat-input-split-gpu{0,1}.json`. These compare
against the latest standalone checkpoint, not the older production wide path.
Consult RESULTS.md and sanitizer logs for final qualification status.
Reproduce paired timings with `repeat_input_split.py --width 512 --length 384`
through `vast-sync.sh run`, preloading the two wheel cuBLAS libraries as in
`pilot.sh`. Reproduce strict checks with `qualify_input_split.py` and full
sanitizers with `sanitize.sh`. All commands run on Vast, not the local host.

Rejected experiments are retained: exact-output Lt retuning (no full gain),
three output-weight overlap phases (neutral/slower), 14 saved-forward tile
configurations (slower), and CUDA 13.1 recompilation (runtime errors). Failed
compiler candidates used separate content-addressed cubins; fresh CUDA probes
on both GPUs and the CUDA 12.8 D256 strict/graph validator passed afterwards.

Sanitizer runtime note: tool injection can override the application's preload
and a bare `ctypes.CDLL('libcublasLt.so.12')` can resolve to the Conda development
library. `runtime.py` pins **only the experimental Lt wrapper's** loader to the
wheel's absolute 12.8.4 path; frozen algorithm identity assertions remain enabled.
Use the sanitizer's `--preload-library` flag separately for each wheel BLAS
library. The attempted Conda library alignment was not applied (metapackage
constraint/active-session lock); no other session's environment was changed.

A fresh isolated qualification copy is at
`/workspace/experiments/trimul-large-d/qualification-v2/`. It was created without
rewriting the shared remote checkout while another session held the source lock.
Qualification completed under the exclusive GPU1 lock without interrupting other
sessions. All three full-workload sanitizer checks passed with 2026.1.1.0;
CANDIDATE.json records the qualified explicit selection and matching source hashes.

Compute Sanitizer 2025.4.1 reported shared-memory hazards in cuBLAS `nvjet_*`
kernels. Keep those logs; do not suppress them and claim a clean whole workload.
NVIDIA's 2026.1 release notes document fixes for Hopper warpgroup false positives:
https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html
A separate `/workspace/tools/sanitizer132` installation provides version
2026.1.1.0. `sanitize.sh` explicitly uses it. This changes no compiler, Torch,
cuBLAS or shared engine dependency. Qualification requires the full workload to
pass with this tool and records its version; baseline comparisons distinguish
an instrumentation issue from a candidate-specific hazard.

Recreate this diagnostic tool separately if needed:

```sh
/opt/miniforge3/bin/conda create -y -p /workspace/tools/sanitizer132 -c nvidia cuda-sanitizer-api=13.2
```
