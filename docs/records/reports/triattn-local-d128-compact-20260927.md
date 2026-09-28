# Local H100 TriangleAttention D128 compact bias partials

The first isolated candidate reduces complete F+B median latency by 1.79–1.95% across L384/L768 and both directions. It is not installed in production dispatch.

## Matched measurements

Local node02 H100; QoS normal_h100; one GPU. Current baseline and candidate run in the same process/GPU, 90 interleaved event samples each, torch.compile fullgraph, manual CUDA Graph, BF16 activations/linear weights, FP32 LayerNorm, B1/D128/H4, dropout 0.25 and 10% token mask. These are fresh local paired results, not ratios against Vast timings.

| L | Direction | Baseline F+B ms | Candidate F+B ms | Reduction |
| --- | --- | ---: | ---: | ---: |
| 384 | starting | 1.232 | 1.210 | 1.79% |
| 384 | ending | 1.254 | 1.231 | 1.85% |
| 768 | starting | 6.590 | 6.471 | 1.80% |
| 768 | ending | 6.736 | 6.604 | 1.95% |

## Change and remaining opportunity

`bias_compact.cu` copies the current grouped dK/dV kernel, retaining its scheduling and barriers. Per-eight-row bias sums remain FP32, but HBM partials use BF16 and the final reduction accumulates in FP32. This halves partial storage: 905,969,664 → 452,984,832 bytes at L768. Combined writes plus reads drop by 905,969,664 bytes per backward. No forward or dK/dV math changes.

The single CUDA activity trace shows L768 starting bias reduction 298 → 181 microseconds; grouped dK/dV itself remains around 2.05–2.09 ms. This explains why the complete gain is modest. The next larger targets are grouped dK/dV (~2.05 ms) and dQ (~1.02 ms); no gain from changing those is claimed here.

## Validation

- Actual candidate dispatch: `grouped_dkdv_compact<8>`, verified in each timed trace.
- PTXAS: 168 registers, no spills; shared memory 175,104 bytes.
- Full-module dropout-zero outputs/all gradients: maximum incremental relative L2 0.4272%, under the 0.5% incremental gate; deterministic changed input/weight/dy/mask CUDA Graph replay agrees exactly with fresh execution.
- Matched dropout RNG (two seeds, both lengths/directions): output bitwise identical; all input/parameter gradients pass 0.5% incremental gate.
- FP64 core stress: 10 mask/magnitude fixtures, including dO magnitude 65536, zero input and all-masked cases; dK/dV bitwise identical to baseline. Ordinary nonzero dBias error rises from about 0.237–0.239% to 0.288–0.291%. One-key mathematical-zero dBias uses an absolute-noise criterion, not a meaningless relative error against zero.
- Full stress JSON reconstructed from retained PASS records because the original sanitizer probe reused its JSON path; output filenames now distinguish full stress and sanitizer probes.
- Sanitizer: core L64 memcheck/racecheck/synccheck passed; core L768 memcheck passed; full compiled module F+B with dropout at both lengths/directions passed memcheck. Native-filtered core L256 racecheck and L768 synccheck passed with zero hazards/errors in job 19737. Filters include both baseline/candidate grouped dK/dV and bias reduction kernels.

## Artifacts and execution

Experiment: `experiments/triattn_local_d128/`. Production source is untouched. `build.json` records CUDA source/header/binary hashes; timing JSON records production source hashes and the script hash. All builds/GPU work use Slurm `--partition=h100 --account=cssb --qos=normal_h100 --gres=gpu:h100:1`.

- 19731: isolated build + L768 paired run, completed in 2m06s.
- 19732: FP64 stress + L384 paired run + core sanitizer checks, completed in 52s.
- 19734: whole-module dropout/memcheck completed successfully (8 matched-seed cases, zero memcheck errors). The subsequent unfiltered L256 racecheck was intentionally cancelled at 3m18s to avoid instrumenting the FP64/vendor oracle; its incomplete log is retained and is not counted as a pass.
- 19737: focused native L256 racecheck and L768 synccheck, completed in 27s, zero hazards/errors.
- Results: `experiments/triattn_local_d128/results/`, including raw timing samples, kernel traces, summary.csv, dropout/stress JSON and sanitizer logs.

No hardware-counter/SOL measurement. BF16 bias partials change rounding: these are validated experimental results, not bitwise-equivalent full gradients.
