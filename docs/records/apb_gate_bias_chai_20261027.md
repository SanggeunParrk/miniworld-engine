# AttentionPairBias gate bias and Chai-1's width on H100 (2026-10-27)

Branch `feat/foldforge-gaps` (base `integ/ladder-all2` b93a655e). The last open engine item of FoldForge's `docs/engine-kernel-gaps.md` (3b): Chai-1's Pairformer single attention
(`single_attention_gate_bias = 1.0`, 16 heads x 24, `d_single` 384, `d_pair` 256) ran the composed PyTorch statements on H100. Now it runs the native APB path for inference.

H100 SXM, torch 2.13 / cu129, CUDA graph, `benchmarks/runners/ladder.py run --group '^(apb_)' --repeats 3` (median of 3; ms; ratios CUDA / portable and CUDA / compiled PyTorch):

| group | mode | L | PyTorch best | portable | CUDA | CUDA / portable | CUDA / PyTorch | status |
|---|---|---|---|---|---|---|---|---|
| apb_chai (16x24 pair256, gate bias 1.0) | inference | 384 | 0.1538 | 0.1160 | 0.0600 | 0.517 | 0.390 | OK |
| apb_chai | inference | 768 | 0.4237 | 0.2629 | 0.1488 | 0.566 | 0.351 | OK |
| apb_gb (16x24 pair128, gate bias 1.0) | inference | 384 | 0.1068 | 0.0837 | 0.0422 | 0.504 | 0.395 | OK |

The rows of the other `apb_*` groups of the same run (no gate bias, unchanged code path) are all OK (11 rows, no violation). Before this change the two new groups' CUDA rung was the PyTorch statements
Accuracy and route: `tests/gpu/sm90/test_apb_gate_bias_h100_gpu.py` (44 passed, 5 min on one H100).
