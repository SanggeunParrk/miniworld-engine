# H100 local atom DiT — whole matrix after the small-row work (2026-10-15)

Worktree `h100/local-dit-opt` (uncommitted). H100 80 GB, harness `bench.py target=local_dit level=module`, CUDA graph, ms. `new` = this tree;
`old Triton` = the same tree with `local_dit_h100_mma=False` (the previous best H100 path: Triton plus the CUDA pieces); Anthropic = their
windowed atom attention + atom row kernels composed with torch GEMMs (inference only); PyTorch eager / `torch.compile` = the reference
module (training: eager under `cudagraph=manual`; compiled bf16 training fails the harness's run-to-run check, 3e-3 > 1e-4, so its bf16
numbers come from `cmp_train.py`, below).

Cells are (augmentations A, atoms N, precision). Every cell: `new` is the fastest column.

| mode | cell | new | Anthropic / PyTorch eager (compiled) | PyTorch compiled (inference) | old Triton |
|---|---|---|---|---|---|
| inference | A1 N1024 32 | 0.037856 | 0.097536 | 0.118848 | 0.348944 |
| inference | A1 N1024 bf16-mixed | 0.03568 | 0.09296 | 0.122624 | 0.307936 |
| inference | A1 N128 32 | 0.034912 | 0.09008 | 0.095104 | 0.1536 |
| inference | A1 N128 bf16-mixed | 0.034656 | 0.084896 | 0.102784 | 0.159136 |
| inference | A1 N4096 32 | 0.053792 | 0.139744 | 0.211392 | 1.01162 |
| inference | A1 N4096 bf16-mixed | 0.046752 | 0.126336 | 0.181792 | 0.093376 |
| inference | A1 N8192 32 | 0.075616 | 0.212384 | 0.350864 | 1.96139 |
| inference | A1 N8192 bf16-mixed | 0.058176 | 0.173024 | 0.26512 | 1.56138 |
| inference | A48 N1024 32 | 0.288832 | 1.01408 | 1.05498 | 1.84323 |
| inference | A48 N1024 bf16-mixed | 0.181968 | 0.758176 | 0.73808 | 1.77866 |
| inference | A48 N2048 32 | 0.538576 | 2.00176 | 1.98733 | 3.59402 |
| inference | A48 N2048 bf16-mixed | 0.345616 | 1.56445 | 1.48365 | 3.49235 |
| inference | A48 N4096 32 | 0.96288 | 3.87466 | 3.84954 | 7.02198 |
| inference | A48 N4096 bf16-mixed | 0.616816 | 3.0368 | 2.80392 | 1.25654 |
| inference | A5 N1024 32 | 0.061248 | 0.163216 | 0.187488 | 0.437824 |
| inference | A5 N1024 bf16-mixed | 0.048864 | 0.135616 | 0.164928 | 0.402368 |
| inference | A5 N2048 32 | 0.089792 | 0.254176 | 0.314688 | 0.815008 |
| inference | A5 N2048 bf16-mixed | 0.058496 | 0.193312 | 0.233248 | 0.107696 |
| inference | A5 N256 32 | 0.04944 | 0.099616 | 0.109824 | 0.201632 |
| inference | A5 N256 bf16-mixed | 0.043232 | 0.093888 | 0.11168 | 0.197504 |
| inference | A5 N4096 32 | 0.153088 | 0.465344 | 0.533952 | 1.56728 |
| inference | A5 N4096 bf16-mixed | 0.097568 | 0.32144 | 0.379088 | 0.165376 |
| inference | A5 N8192 32 | 0.257248 | 0.867904 | 1.02234 | 3.04685 |
| inference | A5 N8192 bf16-mixed | 0.165376 | 0.60456 | 0.673184 | 2.61013 |
| training | A1 N1024 32 | 0.29584 | 0.912928 (eager) / 0.380896 (compiled) | - | 0.836848 |
| training | A1 N1024 bf16-mixed | 0.251568 | 0.967856 (eager) / n/a (compiled) | - | 0.697728 |
| training | A1 N128 32 | 0.244352 | 0.520672 (eager) / 0.27744 (compiled) | - | 0.48624 |
| training | A1 N128 bf16-mixed | 0.225088 | 0.63792 (eager) / n/a (compiled) | - | 0.413216 |
| training | A1 N4096 32 | 0.452096 | 2.23818 (eager) / 0.679792 (compiled) | - | 2.08818 |
| training | A1 N4096 bf16-mixed | 0.37152 | 2.12582 (eager) / n/a (compiled) | - | 0.478384 |
| training | A48 N1024 32 | 1.78342 | 7.41146 (eager) / 3.38722 (compiled) | - | 5.7199 |
| training | A48 N1024 bf16-mixed | 1.14722 | 7.13962 (eager) / n/a (compiled) | - | 4.6719 |
| training | A48 N2048 32 | 3.19629 | 13.3456 (eager) / 6.3472 (compiled) | - | 11.0729 |
| training | A48 N2048 bf16-mixed | 1.92266 | 13.0445 (eager) / n/a (compiled) | - | 9.14045 |
| training | A48 N4096 32 | 5.80522 | 26.1892 (eager) / 12.4396 (compiled) | - | 21.7684 |
| training | A48 N4096 bf16-mixed | 3.5016 | 25.2565 (eager) / n/a (compiled) | - | 7.75638 |
| training | A48 N8192 32 | 11.0817 | 50.3333 (eager) / 24.3405 (compiled) | - | 42.2642 |
| training | A48 N8192 bf16-mixed | 6.61664 | 49.2224 (eager) / n/a (compiled) | - | 15.2518 |
| training | A5 N2048 32 | 0.614752 | 2.33837 (eager) / 0.973968 (compiled) | - | 2.03309 |
| training | A5 N2048 bf16-mixed | 0.441024 | 2.30442 (eager) / n/a (compiled) | - | 1.69666 |
| training | A5 N4096 32 | 1.03098 | 4.41026 (eager) / 1.69821 (compiled) | - | 3.82576 |
| training | A5 N4096 bf16-mixed | 0.65792 | 4.12573 (eager) / n/a (compiled) | - | 1.14214 |
| training | A5 N512 32 | 0.375392 | 1.09432 (eager) / 0.450528 (compiled) | - | 0.99928 |
| training | A5 N512 bf16-mixed | 0.321856 | 1.20854 (eager) / n/a (compiled) | - | 0.887232 |

Training under one graph step (`cmp_train.py`, fwd + bwd, same measurement for the three columns), bf16 autocast / fp32:

| cell | ours bf16 | eager | compiled | ours fp32 | eager | compiled |
|---|---|---|---|---|---|---|
| A1 N128 | 0.169 | 0.665 | 0.264 | 0.214 | 0.534 | 0.284 |
| A1 N1024 | 0.242 | 1.045 | 0.355 | 0.316 | 0.951 | 0.451 |
| A1 N4096 | 0.382 | 2.200 | 0.669 | 0.470 | 2.298 | 0.769 |
| A5 N512 | 0.325 | 1.277 | 0.456 | 0.393 | 1.143 | 0.546 |
| A5 N2048 | 0.454 | 2.384 | 0.975 | 0.608 | 2.364 | 1.200 |
| A5 N4096 | 0.708 | 4.230 | 1.509 | 1.024 | 4.471 | 1.928 |
| A48 N1024 | 1.226 | 7.237 | 2.785 | 1.787 | 7.510 | 3.855 |
| A48 N2048 | 2.067 | 13.116 | 5.438 | 3.216 | 13.450 | 7.231 |
| A48 N4096 | 3.806 | 25.316 | 10.346 | 5.809 | 26.261 | 14.162 |
| A48 N8192 | 7.289 | 49.282 | 20.054 | 11.134 | 50.414 | 27.436 |

## What changed on the small cells

The first matrix of the day lost the fp32 training cells with few rows to `torch.compile` (A1 N128 0.538 vs 0.277 ms). A graph replay
profile (`graph_gaps.py`) showed 184 kernels, 230 us of them the weight packing of every step (about 150 casts / copies / concatenations: 69
elementwise copies, 19 cats, 29 + 29 integer ops of the TF32 rounding), and the row kernels at 20-30 us each because a tile of 64 rows is one
CTA that walks every weight unit.

| change | A1 N128 fp32 train |
|---|---|
| before | 0.538 ms |
| `pack90.cu`: all weight forms in one launch (bit-exact against the torch expression) | 0.311 |
| cond / pre split over CTAs by weight unit, tr_bwd by hidden chunk | 0.258 |
| post_bwd and cond_bwd split-K with a last-CTA fixed-order finish | 0.242 |

Large cells use the unsplit kernels (the split build is a separate compile, `-DSPLITK=1`: sharing the code cost post_bwd 436 -> 740 us at
A48). The split picks about 264 CTAs per call (`_split_for`).

## Wide matrix (2026-10-15, after the attention chunk and finalize changes)

`cmp_wide.py`: ours vs PyTorch eager vs `torch.compile` PyTorch, fwd (inference) or fwd + bwd (training), one CUDA graph each, 336 cells =
A in {1, 2, 3, 5, 12, 24, 48} x N in {64, 100, 300, 1000, 2048, 4096} (N not a multiple of 32 included) x {fp32, bf16} x {per-sample
conditioning, shared (expanded)} x {inference, training}. Ours is the fastest column in every cell. Speed-up over the faster of eager /
compiled:

| mode / precision / conditioning | cells | smallest (cell) | mean |
|---|---|---|---|
| infer bf16 per-sample cond | 42 | 1.83x (A2 N64) | 4.19x |
| infer bf16 shared cond | 42 | 2.01x (A1 N64) | 8.59x |
| infer fp32 per-sample cond | 42 | 1.41x (A24 N64) | 2.14x |
| infer fp32 shared cond | 42 | 1.69x (A5 N64) | 2.69x |
| train bf16 per-sample cond | 42 | 2.40x (A24 N64) | 4.26x |
| train bf16 shared cond | 42 | 2.16x (A24 N64) | 4.22x |
| train fp32 per-sample cond | 42 | 1.09x (A5 N64) | 1.76x |
| train fp32 shared cond | 42 | 1.27x (A1 N300) | 3.05x |

The first pass of this matrix had five fp32 training cells with N = 64 lost to `torch.compile` (A12 N64 0.461 vs 0.385 ms). The window
attention kernels ran one CTA per (key block, head) with the 16-sample chunk, i.e. 4 CTAs at N = 64: `_chunk(A, tiles)` now shrinks the
chunk until the grid has about one CTA per SM (A12 N64: attention 147 -> 26 us), and `finalize90` runs 16 x 27 blocks instead of 4 x 27
(24 -> ~6 us at small sizes).
