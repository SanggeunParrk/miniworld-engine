# Default Triton module performance

Milliseconds, median of two captures in reverse arm order. Static compile + manual CUDA Graph, B1 pair L384/768, n4, BF16 and FP32 affine, nonzero squeeze. Training = forward+backward, no optimizer.

| D | L | Mode | Previous split | New segmented default | Speedup |
|---|---|---|---:|---:|---:|
| 128 | 384 | inference | 0.2763 | 0.1654 | 1.670x |
| 128 | 384 | training | 1.0522 | 0.9429 | 1.116x |
| 256 | 384 | inference | 0.7268 | 0.5662 | 1.284x |
| 256 | 384 | training | 2.4350 | 2.2666 | 1.074x |
| 128 | 768 | inference | 1.0346 | 0.5937 | 1.743x |
| 128 | 768 | training | 3.9811 | 3.5283 | 1.128x |
| 256 | 768 | inference | 2.8957 | 2.1512 | 1.346x |
| 256 | 768 | training | 9.6258 | 8.9613 | 1.074x |

## Full-output b2b and both segmented variants

| D | L | Mode | split | full-output b2b | segmented LN separate | segmented LN fused |
|---|---|---|---:|---:|---:|---:|
| 128 | 384 | inference | 0.2746 | 0.1788 | 0.1646 | 0.1831 |
| 128 | 384 | training | 1.0459 | 0.9580 | 0.9345 | 0.9653 |
| 256 | 384 | inference | 0.7120 | 0.8077 | 0.5657 | 0.6337 |
| 256 | 384 | training | 2.4241 | 2.4178 | 2.2668 | 2.3377 |
| 128 | 768 | inference | 1.0331 | 0.6422 | 0.5991 | 0.6810 |
| 128 | 768 | training | 3.9769 | 3.6106 | 3.5382 | 3.6453 |
| 256 | 768 | inference | 2.8813 | 3.1076 | 2.1690 | 2.3912 |
| 256 | 768 | training | 9.6019 | 9.5455 | 8.9901 | 9.3687 |
