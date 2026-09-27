# Whole Transition module results

node02 H100; B1 pair; one layer; expansion4; static compile + manual CUDA Graph; nonzero squeeze weights; BF16 activations/projection parameters and FP32 LN affine. Medians of two captures per arm, with reversed arm order on the second pass. Same `bench_module_transition` fixture; untouched PyTorch reference. Training includes forward/backward and the four cuBLAS GEMMs; excludes optimizer. No dropout parameter exists in this Transition module.

Old CUDA: archived native binary/source `9502ec6a3ef5`. Both old and new use the same norm settings. Old/new best is the faster of the two CUDA schedules, independently selected from the measurements; all individual arms are retained below. Triton best means full-K at D128/256 and split at D384/512. Speedup = baseline time / new time.

## inference

| D | L | Old best CUDA ms | New best CUDA ms | New schedule | Best Triton ms | Old/new speedup | Triton/new speedup |
|---:|---:|---:|---:|---|---:|---:|---:|
| 128 | 384 | 0.2116 | 0.2030 | streamed_k | 0.1651 | 1.042x | 0.813x |
| 128 | 768 | 0.7762 | 0.7454 | streamed_k | 0.5929 | 1.041x | 0.795x |
| 256 | 384 | 0.6467 | 0.5341 | full_k | 0.5639 | 1.211x | 1.056x |
| 256 | 768 | 2.4511 | 2.0395 | full_k | 2.1475 | 1.202x | 1.053x |
| 384 | 384 | 1.6565 | 1.5376 | full_k | 1.4074 | 1.077x | 0.915x |
| 384 | 768 | 6.4299 | 6.1182 | full_k | 5.7948 | 1.051x | 0.947x |
| 512 | 384 | 3.3455 | 2.8190 | streamed_k | 2.3541 | 1.187x | 0.835x |
| 512 | 768 | 12.7749 | 10.9001 | streamed_k | 9.2507 | 1.172x | 0.849x |

## training

| D | L | Old best CUDA ms | New best CUDA ms | New schedule | Best Triton ms | Old/new speedup | Triton/new speedup |
|---:|---:|---:|---:|---|---:|---:|---:|
| 128 | 384 | 1.0360 | 0.9552 | full_k | 0.9382 | 1.085x | 0.982x |
| 128 | 768 | 3.8542 | 3.5391 | full_k | 3.5185 | 1.089x | 0.994x |
| 256 | 384 | 2.4492 | 2.1956 | full_k | 2.2715 | 1.116x | 1.035x |
| 256 | 768 | 9.6429 | 8.4997 | full_k | 8.8927 | 1.134x | 1.046x |
| 384 | 384 | 5.4726 | 4.9734 | full_k | 4.7506 | 1.100x | 0.955x |
| 384 | 768 | 21.5879 | 20.0409 | full_k | 19.2744 | 1.077x | 0.962x |
| 512 | 384 | 9.4058 | 8.1341 | streamed_k | 7.5051 | 1.156x | 0.923x |
| 512 | 768 | 36.7132 | 32.4004 | streamed_k | 29.9167 | 1.133x | 0.923x |

## Both schedules

| D | L | Mode | CUDA schedule | Old ms | New ms | Speedup |
|---:|---:|---|---|---:|---:|---:|
| 128 | 384 | inference | full_k | 0.2151 | 0.2086 | 1.031x |
| 128 | 384 | inference | streamed_k | 0.2116 | 0.2030 | 1.042x |
| 128 | 384 | training | full_k | 1.0360 | 0.9552 | 1.085x |
| 128 | 384 | training | streamed_k | 1.0811 | 0.9561 | 1.131x |
| 128 | 768 | inference | full_k | 0.7824 | 0.7541 | 1.038x |
| 128 | 768 | inference | streamed_k | 0.7762 | 0.7454 | 1.041x |
| 128 | 768 | training | full_k | 3.8542 | 3.5391 | 1.089x |
| 128 | 768 | training | streamed_k | 4.0611 | 3.5949 | 1.130x |
| 256 | 384 | inference | full_k | 0.6467 | 0.5341 | 1.211x |
| 256 | 384 | inference | streamed_k | 0.6628 | 0.6414 | 1.033x |
| 256 | 384 | training | full_k | 2.4492 | 2.1956 | 1.116x |
| 256 | 384 | training | streamed_k | 2.6354 | 2.3110 | 1.140x |
| 256 | 768 | inference | full_k | 2.4511 | 2.0395 | 1.202x |
| 256 | 768 | inference | streamed_k | 2.5921 | 2.5500 | 1.017x |
| 256 | 768 | training | full_k | 9.6429 | 8.4997 | 1.134x |
| 256 | 768 | training | streamed_k | 10.2416 | 9.0850 | 1.127x |
| 384 | 384 | inference | full_k | 1.6565 | 1.5376 | 1.077x |
| 384 | 384 | inference | streamed_k | 1.8526 | 1.7643 | 1.050x |
| 384 | 384 | training | full_k | 5.4726 | 4.9734 | 1.100x |
| 384 | 384 | training | streamed_k | 5.5923 | 5.1065 | 1.095x |
| 384 | 768 | inference | full_k | 6.4299 | 6.1182 | 1.051x |
| 384 | 768 | inference | streamed_k | 7.0970 | 6.7360 | 1.054x |
| 384 | 768 | training | full_k | 21.5879 | 20.0409 | 1.077x |
| 384 | 768 | training | streamed_k | 22.1177 | 20.2747 | 1.091x |
| 512 | 384 | inference | full_k | 3.3455 | 3.2520 | 1.029x |
| 512 | 384 | inference | streamed_k | 3.4062 | 2.8190 | 1.208x |
| 512 | 384 | training | full_k | 9.7108 | 8.8638 | 1.096x |
| 512 | 384 | training | streamed_k | 9.4058 | 8.1341 | 1.156x |
| 512 | 768 | inference | full_k | 12.7749 | 12.4038 | 1.030x |
| 512 | 768 | inference | streamed_k | 13.0659 | 10.9001 | 1.199x |
| 512 | 768 | training | full_k | 38.6404 | 35.3091 | 1.094x |
| 512 | 768 | training | streamed_k | 36.7132 | 32.4004 | 1.133x |
