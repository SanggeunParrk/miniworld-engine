# Training forward audit and correction

H100 node02; BF16; batch1/C128/H128 per direction; dropout25%; pair mask, residual, all backward saves and live weight packing included.
All paths explicit CUDA Graph. Old routes static fullgraph compile. Six paths alternated for 600 event samples each. No RNG, optimizer, CPU dispatch or compilation time.

| L | Old Triton ms | Old H100 ms | Reported adapter ms | Fixed packing ms | Fixed + selected configs ms | Speedup vs old H100 |
|---|---|---|---|---|---|---|
| 384 | 0.537 | 0.461 | 0.458 | 0.444 | 0.438 | 1.051x |
| 768 | 2.132 | 1.800 | 1.761 | 1.745 | 1.737 | 1.036x |

## Findings

- Actual graph traces confirm old H100 uses our CuTe front/f567 and separate input/output LN. It does not contain Anthropic. Current uses mw_saved_front and mw_k3_train.
- The reported current adapter launched six copies, two concatenations and one stack conversion per call. Old static compile merged its packing. Untimed-prepack control removes about24us from the current adapter.
- Corrected adapter rebuilds live weights every call into one contiguous allocation, exposing all layout conversions to one compiled packing operation. It retains the original shapes through views, and passes original Wg directly to K3 instead of transposing it twice.
- L768 had used the L384 K1 schedule. The recorded L768 schedule is restored, but its effect was small and varied across paired runs. No claim of a large gain from this setting.
- All24 feasible configurations in the existing saved-training K3 search space were compiled and checked at both lengths. Five screening leaders plus the starting schedule were confirmed with600 alternating samples. L384 selects (1,128,4,2,232,1); L768 retains (2,64,4,1,232,1). This is exhaustive only within the existing bounded space.
- Both K3 output and BF16 saves must be bit-exact to the starting schedule; FP32 statistics require relative L2<=1e-6. All admitted candidates satisfy these tests.
- Packing correction preserves every output and saved tensor bit-for-bit. Corrected graph was replayed after modifying x, WL and Wg: all outputs/saves match a fresh call exactly. Restoring inputs restores exact forward output.
- Current learned-training derivative is not the upstream inference-only K1/K3 payload. Backward saves/dropout exist here; old H100 also saves its backward intermediates. Saving alone is not a sufficient explanation for the small relative improvement.
- The raw pre-fix trace shows core region savings around16us/56us at L384/L768. Packing explains the erased margin at384; it does not hide a huge forward improvement.
- This correction remeasures training FORWARD. Earlier directly measured full forward+backward results remain historical pre-correction results. Do not estimate a new full total by subtracting standalone forward times.
- Only the experimental adapter/benchmark changes here; no production default dispatch change.

## Artifacts

- audit_training_forward.py / forward-audit-L*.json: actual old/new kernel traces and initial timing.
- training_forward_adapter.py: corrected live packing and original-gate-weight adapter.
- tune_training_k3_audit.py / training-k3-audit-L*.json: all24 configurations and held-out finalist samples.
- audit_corrected_forward.py / forward-corrected-L*.json:600 samples/path, source hashes, changed-input replay checks and corrected kernel traces.
