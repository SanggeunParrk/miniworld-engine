# Boltz-2 / RoseTTAFold3 / Chai-1 variants on H100: second pass (2026-10-22)

Same harness as [model_variants_h100_20261022.md](model_variants_h100_20261022.md) (A = 5, bf16, CUDA graph, ms; "compiled" =
`torch.compile` of the torch statements). Changes: the variant stacks' weight pack is cached outside the CUDA-graph replays (it was
re-made in every replay, more than half of a token step), 128-wide rows warp-per-row, the atom edge windows' bias made once per pair,
RoseTTAFold3's OPM biases in the wide H100 prologue.

| stack | size | before | after | compiled | after vs compiled |
|---|---|---|---|---|---|
| Boltz-2 token (24) | L 384 | 5.92 | 2.78 | 7.90 | 2.8x |
| Boltz-2 token (24) | L 768 | 9.22 | 5.91 | 21.4 | 3.6x |
| RoseTTAFold3 token (24) | L 384 | 5.77 | 2.60 | 7.75 | 3.0x |
| RoseTTAFold3 token (24) | L 768 | 8.64 | 5.44 | 21.2 | 3.9x |
| Chai-1 token (16) | L 384 | 3.22 | 1.74 | 5.22 | 3.0x |
| Chai-1 token (16) | L 768 | 5.26 | 3.74 | 15.2 | 4.1x |
| Boltz-2 atom (3) | 8192 atoms | 1.64 | 1.22 | 1.92 | 1.6x |
| RoseTTAFold3 atom (3) | 8192 atoms | 1.63 | 1.19 | 1.92 | 1.6x |
| Chai-1 atom (3) | 8192 atoms | 1.46 | 1.13 | 1.75 | 1.6x |
| RoseTTAFold3 OPM | S 1024, N 384 | 1.02 | 0.63 | 0.92 | 1.5x |
| RoseTTAFold3 OPM | S 4096, N 384 | 2.87 | 1.99 | 2.43 | 1.2x |
| RoseTTAFold3 OPM | S 1024, N 768 | 3.43 | 2.33 | 3.31 | 1.4x |

The pair bias (token and atom) is still made per call when the pair tensor changes between calls, as FoldForge's per-step pair
conditioning does; Chai-1's grouped OPM is unchanged (1.1-1.2x).
