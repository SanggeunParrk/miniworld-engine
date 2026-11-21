
### h100  (repeats: r1, r2, r3; median of the repeats' medians, ms; ratio = backend / cuda)

| mode | d | L | pytorch | cuda | fast_trimul | gpumode | geonn |
|---|---|---|---|---|---|---|---|
| eval | 128 | 512 | 3.180 | 0.249 | 2.724 (x10.95) | 1.920 (x7.71) | 2.168 (x8.71) |
| eval | 128 | 1024 | 19.493 | 0.971 | 17.013 (x17.53) | 10.977 (x11.31) | 14.815 (x15.26) |
| eval | 128 | 2048 | 80.234 | 5.847 | 71.826 (x12.29) | 53.479 (x9.15) | 62.140 (x10.63) |
| train | 128 | 512 | 10.756 | 1.123 | 12.777 (x11.38) | - | - |
| train | 128 | 1024 | 80.988 | 4.617 | 91.505 (x19.82) | - | - |
| train | 128 | 2048 | 336.726 | 23.780 | 384.211 (x16.16) | - | - |

- geonn (train): 3 cell(s) not measured: unsupported: training: the library has no backward kernel (its autograd path is its

- gpumode (train): 3 cell(s) not measured: unsupported: training: forward-only kernels (no backward)

### a100  (repeats: r1, r2, r3; median of the repeats' medians, ms; ratio = backend / cuda)

| mode | d | L | pytorch | cuda | fast_trimul | gpumode | geonn |
|---|---|---|---|---|---|---|---|
| eval | 128 | 512 | 6.744 | 0.589 | 5.631 (x9.56) | 5.379 (x9.13) | 5.097 (x8.66) |
| eval | 128 | 1024 | 33.845 | 3.107 | 28.502 (x9.17) | 24.493 (x7.88) | 25.776 (x8.30) |
| eval | 128 | 2048 | 141.410 | 17.556 | 121.056 (x6.90) | 124.110 (x7.07) | 108.134 (x6.16) |
| train | 128 | 512 | 25.012 | 2.670 | 28.475 (x10.66) | - | - |
| train | 128 | 1024 | 139.631 | 12.607 | 158.185 (x12.55) | - | - |
| train | 128 | 2048 | 572.946 | 66.151 | 653.018 (x9.87) | - | - |

- geonn (train): 3 cell(s) not measured: unsupported: training: the library has no backward kernel (its autograd path is its

- gpumode (train): 3 cell(s) not measured: unsupported: training: forward-only kernels (no backward)

### rtx5090  (repeats: r1, r2, r3; median of the repeats' medians, ms; ratio = backend / cuda)

| mode | d | L | pytorch | cuda | fast_trimul | gpumode | geonn |
|---|---|---|---|---|---|---|---|
| eval | 128 | 512 | 3.711 | 0.513 | 3.001 (x5.85) | 2.725 (x5.31) | 2.727 (x5.32) |
| eval | 128 | 1024 | 25.401 | 2.660 | 18.478 (x6.95) | 13.103 (x4.93) | 19.512 (x7.34) |
| eval | 128 | 2048 | 113.148 | 15.439 | 85.741 (x5.55) | 66.127 (x4.28) | 95.894 (x6.21) |
| train | 128 | 512 | 12.241 | 2.660 | 13.248 (x4.98) | - | - |
| train | 128 | 1024 | 104.408 | 13.030 | 112.962 (x8.67) | - | - |
| train | 128 | 2048 | - | - | - | - | - |

- cuda (train): 1 cell(s) not measured: skipped: OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB. G

- fast_trimul (train): 1 cell(s) not measured: skipped: OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB. G

- geonn (train): 13 cell(s) not measured: unsupported: training: the library has no backward kernel (its autograd path is its

- gpumode (train): 13 cell(s) not measured: unsupported: training: forward-only kernels (no backward)

- pytorch (train): 1 cell(s) not measured: skipped: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.12 GiB. GPU 

- pytorch (train): 1 cell(s) not measured: skipped: OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB. G
