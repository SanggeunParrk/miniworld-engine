## Final node02 timings

Milliseconds; mean of two independent capture medians. Speedup is Triton / H100.

| L | D | Mode | Triton ms | Legacy ms | H100 residual ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| 384 | 128 | inference | 0.2753 | 0.1604 | 0.1605 | 1.716x |
| 384 | 128 | training | 1.0492 | 1.0366 | 0.9961 | 1.053x |
| 768 | 128 | inference | 1.0362 | 0.5681 | 0.5670 | 1.828x |
| 768 | 128 | training | 3.9755 | 3.9279 | 3.7653 | 1.056x |
| 384 | 512 | inference | 2.3097 | 1.8150 | 1.8276 | 1.264x |
| 384 | 512 | training | 7.3882 | 7.3261 | 7.1850 | 1.028x |

Validation: **16 Hopper GPU tests**, **20 CPU checks**, and **6 compute-sanitizer tests with 0 memory errors**. All final benchmark rows passed the unchanged graph replay checks.
