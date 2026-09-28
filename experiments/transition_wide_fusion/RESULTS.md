# Local H100 wide Transition results

One allocated H100 SXM, `normal_h100` QoS; BF16 pair tensors `[1,L,L,D]`, expansion 4.
Actual compiled modules, fresh full F+B, input and all five parameter gradients, 100 alternating CUDA Graph event samples.
Candidate is explicit only. D128/D256 use unchanged native kernels. Timings are milliseconds.
D128 was added in a separate local job with the same harness/protocol; wide rows retain the preceding qualified measurements.

| L | D | PyTorch compiled | Native before | Experiment | vs native | vs PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 128 | 1.194 | 0.528 | 0.530 | 0.998x | 2.254x |
| 384 | 256 | 2.358 | 1.830 | 1.831 | 0.999x | 1.288x |
| 384 | 384 | 4.208 | 3.771 | 3.652 | 1.033x | 1.152x |
| 384 | 512 | 6.452 | 6.136 | 5.920 | 1.036x | 1.090x |
| 768 | 128 | 4.506 | 2.031 | 2.032 | 1.000x | 2.218x |
| 768 | 256 | 9.075 | 7.120 | 7.132 | 0.998x | 1.273x |
| 768 | 384 | 16.307 | 14.630 | 14.229 | 1.028x | 1.146x |
| 768 | 512 | 25.294 | 23.973 | 23.344 | 1.027x | 1.084x |

All eight shapes passed full-gradient, compiled graph and changed-state replay checks.
Both changed widths passed full-L768 module memcheck and isolated changed-kernel racecheck/synccheck.
See [VALIDATION.json](VALIDATION.json) for exact gates, job IDs and source hashes, and [README.md](README.md) for rejected fusion designs.
D384/D512 retain cuBLAS GEMMs; the gain comes from the persistent LN backward + residual epilogue. This does not solve full gate/weight-gradient fusion.
Raw records and unsuccessful attempts remain under `.bench/transition-wide-local/`.
