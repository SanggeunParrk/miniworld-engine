## Default auto: node02 module timing

| L | D | Mode | Triton ms | New auto ms | Speedup |
|---|---|---|---:|---:|---:|
| 384 | 128 | inference | 0.2756 | 0.1610 | 1.712x |
| 384 | 128 | training | 1.0468 | 0.9402 | 1.113x |
| 768 | 128 | inference | 1.0367 | 0.5712 | 1.815x |
| 768 | 128 | training | 3.9754 | 3.5245 | 1.128x |
| 384 | 256 | inference | 0.7258 | 0.5270 | 1.377x |
| 384 | 256 | training | 2.4374 | 2.2630 | 1.077x |

## Matched explicit variants (before cache publication; same winning configs)

| L | D | Mode | Triton | Recompute b2b | Saved b2b | CuTe |
|---|---|---|---:|---:|---:|---:|
| 384 | 128 | inference | 0.2757 | 0.1599 | — | 0.3088 |
| 384 | 128 | training | 1.0502 | 0.9985 | 0.9360 | 1.1980 |
| 768 | 128 | inference | 1.0364 | 0.5712 | — | 1.1701 |
| 768 | 128 | training | 3.9772 | 3.7591 | 3.5277 | 4.6041 |
| 384 | 256 | inference | 0.7115 | 0.5250 | — | 0.6517 |
| 384 | 256 | training | 2.4135 | 2.3984 | 2.2450 | 2.5328 |
