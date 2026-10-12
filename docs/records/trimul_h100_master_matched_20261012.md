# TriMul H100 — matched FP32-master training audit, 2026-10-12

The installed **opt8** native configuration beats **cuEquivariance 0.12.0 in all 120 tested training shapes** with BF16 activations and unrounded FP32 master parameters. The 119 resident paired comparisons give **1.146457–3.051949x** complete forward+backward speedup. The largest bidirectional shape, measured with one implementation resident at a time, gives **1.995516x** by ABBA means and **1.973292x** with the conservative comparison.

This is a refreshed deployment-condition baseline, not a kernel optimization gain. It measures opt8 before selection of the subsequent B7 scheduling candidate. It provides no new inference/Anthropic result or SOL claim. Earlier BF16-weight reports retain their explicitly stated parameter policy.

## Runner initialization repair

The current official runner defaults to `param_master=True`. However, its TriMul MiniWorld constructor first converted each layer to BF16, loaded the common FP32 state into it, and then `_place` converted it back to FP32. Vendor layers loaded the original FP32 values directly. The parameter dtypes matched while their values differed.

Job **28192** retained the strict tensor-equality guard and failed the first vendor comparison. No speedup from that unmatched attempt is counted. New constructor/placement regression tests reproduce four failures before the fix: unidirectional and bidirectional modules under FP32 and mixed-precision master policies. The explicit BF16-weight cases already match. Removing the premature MiniWorld-only cast lets `_place` apply the same requested policy after shared-state loading. All **155 module-benchmark contract tests pass** after this repair, including exact parameter-value equality across MiniWorld, cuEquivariance and PyTorch. No numerical threshold or vendor kernel was changed.

The repaired official runner SHA256 is `f0ad9729a094a76739865370bc5ace02a1aff3d8178def949876c1090e5aa26d`; it remains unchanged throughout the successful measurements. The original failing runner SHA256 is `9b724e9e76e36c75cad6b23bc5ce83fcb79503b7117fdfc7ab3d90f0572c2490`. Both versions and the before/after regression logs are preserved.

## Scope and timings

Batch 1, one module layer, D64/128/256/384/512, hidden width D per direction, lengths 128 through 1024 in steps of 128, outgoing/incoming/bidirectional, BF16 activations, FP32 master weights and norm affine parameters, dropout 0.25, mask probability 0.1. Measurements use the official `benchmarks/runners/bench.py`, compilation and manual CUDA Graphs. Native dispatch uses CUDA and cuBLAS; actual graph and compiled-invocation profiles contain no Triton kernels.

For the 119 paired cells, both captured steps remain resident on the same allocated H100 and are measured in 90 alternating CUDA-event pairs. Input and every parameter tensor must be exactly equal; the successful observer also checks mask and dy equality. Dropout probability and regime match, but independent timed replays are not claimed to use the same random draw. Ratios divide sample medians and are not confidence intervals.

| Width | Paired cells | cuEquivariance / opt8 F+B |
|---|---:|---:|
| D64 | 24 | 1.997877–3.051949x |
| D128 | 24 | 1.572048–2.532025x |
| D256 | 24 | 1.146457–2.666520x |
| D384 | 24 | 1.331589–2.595756x |
| D512 | 23 | 1.315570–2.999269x |

The smallest margin is outgoing D256/L128: native **0.392416 ms**, cuEquivariance **0.449888 ms**. Incoming at the same shape is 1.147028x, and bidirectional is 1.210956x.

For D512/L1024 bidirectional, job **28210** runs four separate processes in vendor/native/native/vendor order. Their complete F+B times are **118.476799 / 60.040161 / 58.982273 / 119.034370 ms**. All four processes have matching SHA256 fingerprints for the logical input, ten parameters, dy and mask, and all report compiled manual-graph F+B. The mean ratio is 1.995516x; dividing the faster vendor time by the slower native time gives 1.973292x. This is a separate ABBA result, not another resident paired cell.

## Validation and retained failures

Every successful native cell checks the output and all eleven gradients against the FP32 reference under the runner's paired correctness dropout. Across the 119 paired cells, maximum output relative error is **0.00387201** and maximum gradient relative error is **0.00726650**, below the existing 0.006/0.01 limits. The vendor also stays within these limits: maxima **0.00387151 / 0.00843852**, with no failed vendor accuracy rows. The largest ABBA native case has **0.00357973 / 0.00601462** output/gradient errors; the vendor has **0.00374800 / 0.00626948**. All observed gradients are finite. Correctness tolerances are unchanged.

Job **28198** completed the first 39 paired cells, then ran out of memory when adding the outgoing D512/L1024 vendor graph. The observer had retained the previous shape's native and vendor closures in local variables after clearing its capture dictionary. Its revised version releases those observer-owned references after measurements and profiling. It preserves the first 39 results and resumes only missing cells. Outgoing and incoming D512/L1024 then both fit and complete their paired comparisons. The original OOM, initial observer, initial results and resumed observer are all retained; the failed cell is not counted as a timing result.

This audit changes runner initialization and observer ownership, not the opt8 native kernel files. It does not replace the existing graph-value and sanitizer qualification of opt8 or qualify a different experimental kernel. The selected six-file opt8 manifest matches the frozen native baseline. The explicit native flags remain those documented in [opt8](trimul_b200style_h100_opt8_20261012.md), including `MINIWORLD_TRIMUL_H100_B200STYLE=1`; that flag's default is unchanged.

All imports, tests, compilation, benchmarks and archive checks ran through Slurm on compute nodes, with explicit memory requests, `h100/cssb/normal_h100`, and one H100 per GPU job. No commit or push occurred.

Evidence: `runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/master_matched_20261012/`, **798 files listed in `SHA256.json`**, including paired raw samples and profiles, per-parameter errors, four ABBA audits and process logs, source manifests, both runner versions, constructor regression logs, original state-mismatch and OOM failures, and official paired-runner artifacts. All 798 returned files passed hash verification. `results/master-matched-grid-summary.json`, `results/master-matched-grid.csv` and `results/current-master-abba-summary.json` provide the machine-readable results.
