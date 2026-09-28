# Matched module timings

Milliseconds; median of two captures in reverse arm order. BF16 activations/weights, FP32 LN affine; n4 B1, nonzero squeeze, static torch.compile (one graph observed), manual CUDA Graph. Training is forward+backward, no optimizer. General Transition has no dropout.

| D | L | Mode | split | full-output b2b | segmented b2b |
|---|---|---|---:|---:|---:|
| 128 | 384 | inference | 0.2752 | 0.1792 | — |
| 128 | 384 | training | 1.0479 | 0.9619 | — |
| 256 | 384 | inference | 0.7166 | 0.8103 | — |
| 256 | 384 | training | 2.4332 | 2.4642 | — |
| 128 | 768 | inference | 1.0351 | 0.6573 | — |
| 128 | 768 | training | 3.9796 | 3.6249 | — |
| 256 | 768 | inference | 2.9199 | 3.0979 | — |
| 256 | 768 | training | 9.6206 | 9.6003 | — |
| 384 | 384 | inference | 1.4194 | 1.6936 | 1.5791 |
| 384 | 384 | training | 4.7139 | 4.9379 | 4.7986 |
| 512 | 384 | inference | 2.3276 | 2.7596 | 3.0972 |
| 512 | 384 | training | 7.6009 | 7.8297 | 8.1110 |
| 384 | 768 | inference | 5.7888 | 6.4270 | 5.9395 |
| 384 | 768 | training | 18.9477 | 19.6665 | 19.1300 |
| 512 | 768 | inference | 9.1911 | 10.5803 | 11.9604 |
| 512 | 768 | training | 29.3983 | 31.4365 | 32.5344 |
