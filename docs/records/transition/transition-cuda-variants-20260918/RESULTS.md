# Measured results

node02 H100; BF16 activations/weights, FP32 LN affine; residual ON; expansion 4. Static compile + manual CUDA Graph. Milliseconds, median of two harness repeats. No optimizer. L384 seed-sweep configs are reused at L768.

## inference module

| D | L | Triton split | Triton streamed | CUDA streamed | Triton full | CUDA full |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 384 | 0.2755 | 0.2889 | 0.2108 | 0.1659 | 0.2167 |
| 128 | 768 | 1.0324 | 1.0536 | 0.7734 | 0.5939 | 0.7838 |
| 256 | 384 | 0.7078 | 0.7435 | 0.6667 | 0.5651 | 0.6465 |
| 256 | 768 | 2.8965 | 2.8114 | 2.6040 | 2.1154 | 2.4305 |
| 384 | 384 | 1.4128 | 1.5822 | 1.8576 | 4.3679 | 1.6431 |
| 384 | 768 | 5.7812 | 5.9575 | 7.0844 | 16.4760 | 6.3429 |
| 512 | 384 | 2.3445 | 3.1202 | 3.4022 | 7.2727 | 3.3561 |
| 512 | 768 | 9.2645 | 12.0253 | 13.0988 | 27.9370 | 12.8118 |

## training module

| D | L | Triton split | Triton streamed | CUDA streamed | Triton full | CUDA full |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 384 | 1.0471 | 1.0629 | 1.0799 | 0.9359 | 1.0326 |
| 128 | 768 | 3.9805 | 3.9906 | 4.0584 | 3.5238 | 3.8379 |
| 256 | 384 | 2.4144 | 2.4311 | 2.5985 | 2.2516 | 2.4343 |
| 256 | 768 | 9.5647 | 9.5930 | 10.1680 | 8.9021 | 9.5993 |
| 384 | 384 | 4.7629 | 4.8935 | 5.5708 | 7.5800 | 5.4241 |
| 384 | 768 | 19.2152 | 19.5724 | 22.1335 | 29.6532 | 21.4924 |
| 512 | 384 | 7.7213 | 8.1914 | 9.2868 | 12.2336 | 9.7384 |
| 512 | 768 | 29.2018 | 32.5667 | 36.4323 | 48.8622 | 39.5957 |

## Direction timings and peak allocated memory

Separate fullgraph `autograd.grad` fixture with `donated_buffer=False` for retained-graph backward; backward is directly measured, not total minus inference. Peak extra allocation measures an eager invocation of the compiled step, including backward temporaries and returned gradients; it does not include graph private pools and excludes preexisting fixture allocations and reserved allocator memory.

| D | L | Arm | Training fwd | Bwd | Full step | Inference | Peak extra MiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 128 | 384 | triton_split | 0.2659 | 0.7758 | 1.0353 | 0.2659 | 649.1 |
| 128 | 384 | triton:streamed_k | 0.2791 | 0.7709 | 1.0480 | 0.2824 | 649.1 |
| 128 | 384 | cuda:streamed_k | 0.2037 | 0.8735 | 1.0754 | 0.2014 | 649.1 |
| 128 | 384 | triton:full_k | 0.1561 | 0.7739 | 0.9262 | 0.1538 | 649.1 |
| 128 | 384 | cuda:full_k | 0.2088 | 0.8218 | 1.0265 | 0.2065 | 649.1 |
| 128 | 768 | triton_split | 1.0280 | 2.9431 | 3.9641 | 1.0247 | 2596.5 |
| 128 | 768 | triton:streamed_k | 1.0444 | 2.9558 | 3.9815 | 1.0330 | 2596.5 |
| 128 | 768 | cuda:streamed_k | 0.7718 | 3.3078 | 4.0521 | 0.7690 | 2596.5 |
| 128 | 768 | triton:full_k | 0.5847 | 2.9925 | 3.5425 | 0.5856 | 2596.5 |
| 128 | 768 | cuda:full_k | 0.7712 | 3.0923 | 3.8318 | 0.7711 | 2596.5 |
| 256 | 384 | triton_split | 0.7269 | 1.7373 | 2.4548 | 0.7145 | 1297.1 |
| 256 | 384 | triton:streamed_k | 0.7316 | 1.7300 | 2.4966 | 0.7330 | 1297.1 |
| 256 | 384 | cuda:streamed_k | 0.6539 | 2.0684 | 2.6824 | 0.6646 | 1297.1 |
| 256 | 384 | triton:full_k | 0.5540 | 1.7145 | 2.3166 | 0.5541 | 1297.1 |
| 256 | 384 | cuda:full_k | 0.6362 | 1.8687 | 2.5089 | 0.6368 | 1297.1 |
| 256 | 768 | triton_split | 2.9257 | 6.7907 | 9.8263 | 2.9304 | 5188.5 |
| 256 | 768 | triton:streamed_k | 2.8071 | 6.7587 | 9.9297 | 2.8254 | 5188.5 |
| 256 | 768 | cuda:streamed_k | 2.6126 | 7.7363 | 10.5147 | 2.6222 | 5188.5 |
| 256 | 768 | triton:full_k | 2.1612 | 6.7720 | 9.2026 | 2.1631 | 5188.5 |
| 256 | 768 | cuda:full_k | 2.4206 | 7.2176 | 9.8266 | 2.4223 | 5188.5 |
| 384 | 384 | triton_split | 1.4464 | 3.3630 | 5.0647 | 1.4459 | 1945.1 |
| 384 | 384 | triton:streamed_k | 1.5618 | 3.3982 | 5.0177 | 1.5641 | 1945.1 |
| 384 | 384 | cuda:streamed_k | 1.8353 | 3.7798 | 5.6219 | 1.8335 | 1945.1 |
| 384 | 384 | triton:full_k | 4.2948 | 3.4067 | 7.5545 | 4.3032 | 1945.1 |
| 384 | 384 | cuda:full_k | 1.6351 | 3.8461 | 5.4744 | 1.6249 | 1945.1 |
| 384 | 768 | triton_split | 5.7834 | 13.5290 | 19.5663 | 5.8169 | 7780.5 |
| 384 | 768 | triton:streamed_k | 6.0426 | 13.5474 | 19.9997 | 6.0445 | 7780.5 |
| 384 | 768 | cuda:streamed_k | 7.3764 | 14.9388 | 22.2919 | 7.3853 | 7780.5 |
| 384 | 768 | triton:full_k | 16.7162 | 13.4933 | 30.0021 | 16.7162 | 7780.5 |
| 384 | 768 | cuda:full_k | 6.3229 | 15.2517 | 21.5470 | 6.3303 | 7780.5 |
| 512 | 384 | triton_split | 2.3435 | 5.3697 | 7.7278 | 2.3424 | 2593.1 |
| 512 | 384 | triton:streamed_k | 3.1032 | 5.3651 | 8.2402 | 3.1077 | 2593.1 |
| 512 | 384 | cuda:streamed_k | 3.3679 | 5.9249 | 9.2111 | 3.3673 | 2593.1 |
| 512 | 384 | triton:full_k | 7.2666 | 5.3418 | 12.1717 | 7.2652 | 2593.1 |
| 512 | 384 | cuda:full_k | 3.3308 | 6.3642 | 9.7008 | 3.3308 | 2593.1 |
| 512 | 768 | triton_split | 9.4457 | 20.9491 | 30.5318 | 9.4143 | 10372.5 |
| 512 | 768 | triton:streamed_k | 12.0920 | 20.9564 | 32.6893 | 12.0941 | 10372.5 |
| 512 | 768 | cuda:streamed_k | 13.2264 | 23.4503 | 36.7630 | 13.2350 | 10372.5 |
| 512 | 768 | triton:full_k | 28.3902 | 20.9876 | 49.2358 | 28.4505 | 10372.5 |
| 512 | 768 | cuda:full_k | 12.8957 | 25.2943 | 38.7084 | 12.8288 | 10372.5 |

## Isolated common backward operations

Independent graph timings; do not add these to infer module latency.

| D | L | dh | dWs | dWab | dxn | weight cat |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 384 | 0.0743 | 0.0680 | 0.1286 | 0.1169 | 0.0018 |
| 128 | 768 | 0.2799 | 0.2622 | 0.4769 | 0.4547 | 0.0019 |
| 256 | 384 | 0.1458 | 0.1297 | 0.2304 | 0.2705 | 0.0024 |
| 256 | 768 | 0.6264 | 0.4900 | 0.9006 | 0.9933 | 0.0024 |
| 384 | 384 | 0.3028 | 0.2443 | 0.5206 | 0.5379 | 0.0028 |
| 384 | 768 | 1.3065 | 0.9890 | 2.0210 | 2.2316 | 0.0029 |
| 512 | 384 | 0.4650 | 0.4510 | 0.8547 | 0.9034 | 0.0034 |
| 512 | 768 | 2.0139 | 1.7212 | 3.7159 | 3.9531 | 0.0036 |

## Selected native machine instructions

Static SASS instruction-site counts for the selected direction, not executed instruction counts or traffic bytes. `UTMALDG` and `HGMMA` confirm explicit TMA/WGMMA. `LDL`/`STL` show remaining local-memory traffic in some wide configurations.

| Variant | D | Direction | UTMALDG | HGMMA | LDL | STL |
|---|---:|---|---:|---:|---:|---:|
| full_k | 128 | forward | 12 | 24 | 0 | 0 |
| full_k | 128 | backward | 11 | 16 | 0 | 0 |
| full_k | 256 | forward | 22 | 40 | 45 | 35 |
| full_k | 256 | backward | 21 | 32 | 0 | 0 |
| full_k | 384 | forward | 24 | 60 | 16 | 8 |
| full_k | 384 | backward | 19 | 48 | 0 | 0 |
| full_k | 512 | forward | 48 | 72 | 56 | 44 |
| full_k | 512 | backward | 25 | 64 | 29 | 29 |
| streamed_k | 128 | forward | 13 | 32 | 0 | 0 |
| streamed_k | 128 | backward | 13 | 16 | 0 | 0 |
| streamed_k | 256 | forward | 13 | 24 | 30 | 20 |
| streamed_k | 256 | backward | 10 | 8 | 0 | 0 |
| streamed_k | 384 | forward | 12 | 32 | 40 | 60 |
| streamed_k | 384 | backward | 10 | 8 | 0 | 0 |
| streamed_k | 512 | forward | 20 | 32 | 122 | 106 |
| streamed_k | 512 | backward | 10 | 8 | 0 | 0 |

## Validation

18 GPU checks passed, including both complete schedules, output and all six gradients, tail rows, residual identity, static compile/CUDA Graph and explicit module wiring. Unfiltered Compute Sanitizer: 10 numerical/module checks passed, 0 errors. Related import/dispatch checks: 47 passed, 1 skipped. Full logs and source hashes accompany this record.

