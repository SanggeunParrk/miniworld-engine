# Where a cache build spends its time

A record of one measurement, not a current reference. The three defects it names are fixed; the
numbers are the "before".

Taken 2026-08-26 from the logs of the A6000 rebuild that ran on 2026-08-25 (job 1655289, gpu03:
8x RTX A6000, 96 cores, `--gpus 8 --compile-jobs 15`, timed out at 12 h with 292 of 859 units
done). 283 of those units left both a claim file and a finished log, so their wall time is
`claim mtime -> log mtime` and the rest is read out of the log.

## The split

| | of 92.2 h of unit wall |
|---|---:|
| compile, in the precompile pool | 50.2 h (54%) |
| compile, serially in the parent | 16.7 h (18%) |
| bench, on the card | ~18.8 h (20%) |
| launch enqueue and setup | ~6.5 h (7%) |

Unit wall sums over eight slots; the job's own wall was about an eighth of it.

The bench figure is derived, not read: `do_bench(warmup=25, rep=100)` costs its budget by
construction -- it picks `n_repeat` from a measured estimate to fill 100 ms -- so timing 542,146
configs costs 542,146 x ~125 ms. The build now measures it directly (`[bench]` in the per-unit
log) because deriving it is how it was got wrong: the only bench-shaped number the build printed
was `[first-launch] fn.run`, and **a round's first compile happens inside `fn.run`**, so that
line was carrying the whole precompile pool. Read as bench it says the card is 1.2% of a build.

## Three defects

### 1. A second autotune key compiled 14,996 configs on one core

A kernel is tuned once per autotune key. Each key reaches triton's compiler with its own
signature and specialisation, so the two keys of one kernel do **not** share a compiled binary.
The settled-set that answers "did the pool already compile this?" was keyed on (kernel, config)
with no key component. So for the second key: the precompile round was skipped as "already
compiled", the settled-set then told the serial compile guard those configs were warm cache hits,
and the parent really compiled every one of them, one at a time, on one core, while fifteen pool
workers sat idle.

Visible in the logs as the cost of a "warm cache hit":

| unit | forkless calls | seconds | ms each | rounds skipped |
|---|---:|---:|---:|---:|
| `adaln_gemm_gate-bf16-L256` | 29,976 | 32,492 | 1084 | 1 |
| `adaln_gemm_gate-bf16-L512` | 29,992 | 17,800 | 594 | 1 |
| `adaln_gemm_gate-bf16-L1024` | 30,054 | 6,333 | 211 | 1 |
| `cond_transition_bwd_gemm-fp32-L256` | 15,513 | 81 | 5.2 | 0 |
| `cond_transition_bwd_swiglu_dx_packed` | 15,485 | 63 | 4.1 | 0 |

A warm hit is 3.5 ms. 1084 ms is a compile. 15.8 of the 16.7 serial CPU-hours are in units that
skipped a round; 9.0 h of them are in the first row alone, on one core.

Fixed by putting triton's own autotune key in the settled-set key.

### 2. Half the pool waited on a tail

Aggregate worker occupancy over 461 rounds was 50% -- 429 CPU-hours of workers idle inside a
round that had not finished. The chunks a round is dealt into were contiguous slices of the
config list, and the configs that run the full 60 s compile budget before being SIGKILLed are the
deep-pipeline, big-tile end of the grid, which the grid lists together. One worker drew a slice
holding about thirteen of them and fifteen waited ~800 s.

| round | wall | ideal (child time / workers) | idle | configs | killed |
|---|---:|---:|---:|---:|---:|
| `trimul_gemm_gate-bf16-L128` | 1962 s | 1059 s | 903 s | 2592 | 120 |
| `trimul_bwd_gate_recompute-bf16` | 1976 s | 1097 s | 879 s | 2592 | 120 |
| `cond_transition_fwd_b2b_saveact` | 2170 s | 1349 s | 821 s | 7776 | 328 |

Fixed by ordering a round most-expensive-first and dealing the chunks round-robin, so each chunk
gets one expensive config instead of one chunk getting thirty-two. Each round now prints its
occupancy.

### 3. Compiling and measuring never overlapped

One unit per card, and a unit alternates between compiling (a pool of processes, no card) and
measuring (one card, one core). So for a fifth of every unit its fifteen compile workers sat
idle, and for the other four fifths the card did nothing.

`--units-per-gpu N` puts N units on each card. What they must not do is measure at the same time:
two kernels sharing the SMs both read slower by an amount that drifts over a round, and a build
whose readings drift picks a different config. Units sharing a card take that card's lock for a
whole tuning round -- once per round, not once per config; 542,146 lock operations would cost
more than the contention.

## What this does not say

* Whether compile cost per config is uniform. It is not measured, and the configs the smem
  predictor skips are the big-tile end, which is the expensive end.
* Anything about a card other than sm86. The split between compile and bench depends on both the
  host's core count and the card's speed.
