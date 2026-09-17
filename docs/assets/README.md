# Graphical abstract

[miniworld-engine-graphical-abstract.png](miniworld-engine-graphical-abstract.png) illustrates
AF3-style operations, representative GPU kernel fusion, and the declared-workload tuning,
validation and cache-reuse workflow. The molecular ribbon provides application context;
MiniWorld Engine supplies kernels and building blocks, not a full structure-prediction model.

The evidence strip reports A6000 DiT results from the
[2026-09-11 production audit](../records/a6000-production-audit.md). It compares ordinary
token DiT (384 tokens) and atom DiT (4096 atoms), depth 1, BF16, against compiled PyTorch.
Inference uses A5 with CUDA Graph; training uses A48 without graphs and measures forward
plus backward. These results do not qualify other GPUs or measure a complete model.

Generated with the built-in image generation tool. The exact
[generation prompt](miniworld-engine-graphical-abstract.prompt.txt) is preserved alongside
the image. Illustrations are schematic, not measured tensor or hardware layouts.
