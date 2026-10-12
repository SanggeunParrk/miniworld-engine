# TriMul H100 B200-style opt4 — 2026-10-08

D512 명시적 B200-style 실험 경로의 folded output K3를 개선했다. 기본 선택은 `MINIWORLD_TRIMUL_H100_B200STYLE=0`으로 유지한다. D256/D384와 backward 선택은 그대로다.

The baseline below is the previous selected **opt3 B200-style** path, not the project default or a vendor kernel. Both sides use flag1, identical seeded inputs/parameters, BF16 activations and linear weights, FP32 normalization, batch1, mask probability0.1, inference dropout0 and training dropout0.25. The unchanged official runner captures `torch.compile` + manual CUDA Graph complete module steps. Each comparison has ten warmup pairs and ninety alternating AB/BA event pairs. Training includes full forward+backward. Inference profiles establish the actual `mw_b200style_k3reorder` dispatch. Some post-capture training profiles omit the forward K3 name; direct native full F+B profiles (job27845, both outgoing and bidirectional) and actual module serving tests independently verify the selected K3. These profiles are dispatch evidence only, not the reported performance measurements.

## Selected change

K3 computes projection before gate and packs the normalized projection before starting the gate GEMM. Output LN statistics use four `ldmatrix.x4.trans` instructions per 64-channel chunk instead of sixteen scalar shared loads per thread. Four lanes share a token row and reduce the statistics with quad shuffles. Statistics reside in warp-owned shared slots; one warp synchronization replaces the previous cross-warp reduction barriers. Projection, gate, residual and dropout BF16 rounding and input/output folding remain at their existing semantic points. The statistics accumulation order changes, and the independent FP32 oracle validates output and every gradient.

Shortening the statistics, packed projection and input-statistics lifetimes reduced spills; changing register roles from producer40/consumer232 to producer24/consumer240 removed the remaining spills. Both allocations use 64,512 registers per 384-thread CTA. D512 H512/H1024 report REG168, STACK0, zero local instructions, max SASS R237 and dynamic shared226,880B plus static1,024B. This change reduces shared-load and reduction overhead; no nominal HBM reduction or SOL claim is made for the selected implementation.

## Paired complete module measurements

Authoritative selected-dispatch comparison: job27838, frozen `snapshot-final`. Speeds are old/new median, so values greater than one favor the new path.

| Mode | Direction | L | Previous ms | New ms | Speedup |
|---|---|---:|---:|---:|---:|
| inference | outgoing | 384 | 1.349664 | 1.347840 | 1.001353x |
| inference | outgoing | 768 | 5.175104 | 5.128288 | 1.009129x |
| inference | bidirectional | 384 | 2.144928 | 2.129408 | 1.007288x |
| inference | bidirectional | 768 | 8.659040 | 8.609824 | 1.005716x |
| training | outgoing | 384 | 4.966976 | 4.932160 | 1.007059x |
| training | outgoing | 768 | 19.401855 | 19.273439 | 1.006663x |
| training | bidirectional | 384 | 8.181504 | 8.167552 | 1.001708x |
| training | bidirectional | 768 | 32.906368 | 32.880127 | 1.000798x |

Geometric mean across four cells per mode: inference **1.005868x**, training **1.004053x**.

Within-job bootstrap confidence intervals of paired mean time differences are positive for seven cells. D512 L768 bidirectional training is **inconclusive**: its median ratio is1.000798x, while the mean-difference95% interval is[-0.0670,0.2249]ms. Ten thousand paired resamples do not model temporal dependence or between-job variation. Earlier independent pilots of the same algorithm at this shape gave1.009206x inference/1.004121x training (job27832); do not replace the authoritative final result with that more favorable pilot.

Separate paired comparison against the project default: job27840, same shape D512 L768 bidirectional. Do not combine absolute times from separate jobs into a three-way comparison.

| Mode | Project default ms | New explicit path ms | Default/new |
|---|---:|---:|---:|
| inference | 7.801376 | 8.781568 | 0.888381x |
| training | 28.113377 | 34.200832 | 0.822009x |

The project default remains faster, so the default route is retained. No vendor leaderboard, SOL90, incoming-wide timing, batch>1 performance, or FP32-master timing was measured in this round.

## Validation and installation

- Candidate job27831:49 strict tests passed. Final selected frozen job27839:49 tests passed, including independent FP32 output/all-gradient oracle, full L384/768 shapes, mask/dropout/zero cases, changed input/weight/dy graph replay, and actual module dispatch with BF16 and FP32 master weights. Thresholds remain output relative<0.006 and each gradient relative<0.01.
- Sanitizer job27834:memcheck25 tests/errors0; racecheck five native F+B/changed-graph scenarios/hazards0/errors0/warnings0; synccheck25 tests/errors0. No filters or suppressions. Instrumented candidate math is byte-identical to the final selected K3 CUDA source; final driver differs only by Python formatting and D512-only selection.
- Live checkout jobs27842/27843:25 native/edge/changed-graph tests and12 actual-module dispatch/all-gradient tests passed after copying exactly the selected K3 CUDA, driver, and selector. All19 active source files match the frozen selected snapshot bytewise. The full source-family cache digest can differ because the live checkout retains unrelated experimental sources; active-source equality plus live tests establishes this serving path.
- `ruff check` and `git diff --check` pass. No commits or pushes. The integration flag defaults to0; flag1 now uses the new K3 for D512 only.

## Rejected or incomplete experiments retained

| Experiment | Evidence | Complete workload result |
|---|---|---|
| Batched in-place DT+LN epilogue barriers | job27803:49pass; MT64 BN128 NS3 register spills0 | job27804 training0.998884x |
| Same DT+LN, WGMMA wait2 | job27806:49pass | job27807 training0.982147x |
| Cross-tile dual-accumulator DT GEMM + previous LN epilogue | compact variant job27810:49pass; spills0 | job27811 training0.959026x |
| Full resident triangle tile K3, MT64 BN128 NS2 | job27816:49pass; spills0 | job27817 inference0.651158x, training0.881240x |
| Triangle-retaining/residual-streaming L2 hints | job27819:49pass | job27820 inference0.998729x, training0.997945x |
| Matrix-statistics K3 without reordering | CPU jobs27822/27823:40B/48B stack,22/26 local instructions | compile-stage rejection; no GPU timing |
| Projection-first K3, original register roles | jobs27824/27825:56B/48B stack; job27826:49pass | job27827 inference0.997562x, training1.002820x |
| Projection-first with shorter lifetimes,40/232 roles | job27828:8B stack/4 local instructions | job27830 inference1.005285x, training1.002512x |

These experiments do not establish an algorithmic limit for fusion. The resident triangle prototype lowers nominal triangle reloads but changes CTA/tile geometry; the observed slowdown does not isolate a hardware bottleneck. The dual-accumulator overlap prototype also changes live registers and instruction scheduling, so it does not prove that every GEMM/epilogue overlap must lose.

Initial setup jobs27796/27799 failed because of wrapper permission and an omitted prototype source; resource job27800 printed valid resources then failed in its fake CPU occupancy driver, and initial GPU job27801 failed wrapper permissions. Resident job27814 and attempted benchmark27815 failed with a Python kernel-name shadowing error; corrected jobs27816/27817 provide the usable evidence. Extended pilot27833 completed five comparisons, then the official source-stability guard rejected a later capture after formatting a hardlinked candidate Python driver changed its source bytes. The failed run and its partial samples are retained. Job27838 restarted the entire eight-cell comparison after formatting finished on stable sources; its result alone defines the final matrix. Official runner checks were not relaxed.

Evidence: `runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt4_20261008/` contains logs, raw paired samples, confidence intervals, official runner artifacts, scripts, candidate source archives, and source manifests. Scratch snapshots live at `/home/psk6950/miniworld-engine-scratch/trimul-b200style-opt4-20261008/`. Reports from earlier rounds were not modified.
