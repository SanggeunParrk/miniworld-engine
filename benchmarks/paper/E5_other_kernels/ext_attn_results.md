# E5 ext_attn results (AF3 setup, L 512 / 1024 / 2048, 20 timed iterations, median of 3 repeats; ratio = backend ms / engine cuda ms, > 1 = slower than the engine)

## h100 speed

| case | pytorch ms | cuda ms | gnops | of3tri | megafold | bionemo_ir | turbopf | ligo_pwa | megafold_tr |
|---|---|---|---|---|---|---|---|---|---|
| AttentionPairBias 384,128,16 L1024  float32 eval | 2.407 (4.39x) | 0.548 | 2.558 (4.66x) | 2.558 (4.66x) | 2.407 (4.39x) | - | - | - | - |
| AttentionPairBias 384,128,16 L1024  float32 train | 4.763 (2.24x) | 2.125 | 9.553 (4.5x) | 9.593 (4.52x) | 4.927 (2.32x) | - | - | - | - |
| AttentionPairBias 384,128,16 L1024  master eval | 2.586 (16.92x) | 0.153 | 2.536 (16.6x) | 2.533 (16.57x) | 2.558 (16.73x) | - | - | - | - |
| AttentionPairBias 384,128,16 L1024  master train | 5.196 (10.79x) | 0.482 | 5.131 (10.65x) | 5.128 (10.65x) | 5.256 (10.91x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  float32 eval | 9.474 (4.86x) | 1.95 | 10.358 (5.31x) | 10.36 (5.31x) | 9.29 (4.76x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  float32 train | 18.286 (2.07x) | 8.85 | 35.453 (4.01x) | 35.423 (4x) | 18.345 (2.07x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  master eval | 10.454 (21.22x) | 0.493 | 10.207 (20.72x) | 10.204 (20.71x) | 10.384 (21.08x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  master train | 20.406 (12.31x) | 1.658 | 19.919 (12.02x) | 19.911 (12.01x) | 20.622 (12.44x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  float32 eval | 0.604 (3.11x) | 0.194 | 0.654 (3.37x) | 0.653 (3.37x) | 0.615 (3.17x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  float32 train | 1.369 (1.93x) | 0.711 | 2.618 (3.68x) | 2.63 (3.7x) | 1.381 (1.94x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  master eval | 0.709 (10.77x) | 0.066 | 0.704 (10.69x) | 0.701 (10.65x) | 0.701 (10.65x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  master train | 1.487 (8.14x) | 0.183 | 1.474 (8.07x) | 1.473 (8.06x) | 1.484 (8.12x) | - | - | - | - |
| AttentionPairBias 768,128,16 L1024  master eval | 2.611 | - | 2.552 | 2.548 | 2.565 | - | - | - | - |
| AttentionPairBias 768,128,16 L1024  master train | 5.232 | - | 5.2 | 5.196 | 5.22 | - | - | - | - |
| AttentionPairBias 768,128,16 L2048  master eval | 10.471 | - | 10.229 | 10.228 | 10.329 | - | - | - | - |
| AttentionPairBias 768,128,16 L2048  master train | 20.51 | - | 20.051 | 20.04 | 20.342 | - | - | - | - |
| AttentionPairBias 768,128,16 L512  master eval | 0.721 | - | 0.716 | 0.714 | 0.712 | - | - | - | - |
| AttentionPairBias 768,128,16 L512  master train | 1.52 | - | 1.525 | 1.522 | 1.507 | - | - | - | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L1024 D1024 master eval | 5.565 (9.72x) | 0.572 | - | - | - | 6.918 (12.08x) | - | 100.726 (175.94x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L1024 D1024 master train | 12.205 (5.16x) | 2.364 | - | - | - | - | - | 124.29 (52.58x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L2048 D1024 master eval | 16.442 (9.9x) | 1.661 | - | - | - | 19.015 (11.45x) | - | 1052.027 (633.27x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L2048 D1024 master train | 35.203 | - | - | - | - | - | - | 1114.781 | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L512 D1024 master eval | 2.267 (9.07x) | 0.25 | - | - | - | 2.873 (11.5x) | - | 30.859 (123.48x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L512 D1024 master train | 4.946 (4.56x) | 1.086 | - | - | - | - | - | 39.424 (36.31x) | - |
| OuterProductMean 64,128,32,false,0.001 L1024 D1024 master eval | 9.565 (2.43x) | 3.943 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L1024 D1024 master train | 23.045 (1.78x) | 12.91 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L2048 D1024 master eval | 33.907 (2.15x) | 15.763 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L2048 D1024 master train | 84.385 (1.64x) | 51.444 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L512 D1024 master eval | 3.009 (3.13x) | 0.961 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L512 D1024 master train | 7.155 (2.19x) | 3.266 | - | - | - | - | - | - | - |
| Transition 128,4 L1024  master eval | 6.975 (10.52x) | 0.663 | - | - | - | - | - | - | 3.75 (5.66x) |
| Transition 128,4 L1024  master train | 19.211 (5.46x) | 3.521 | - | - | - | - | - | - | 12.183 (3.46x) |
| Transition 128,4 L2048  master eval | 27.752 (10.07x) | 2.757 | - | - | - | - | - | - | 14.82 (5.38x) |
| Transition 128,4 L2048  master train | 76.279 (5.07x) | 15.035 | - | - | - | - | - | - | 48.342 (3.22x) |
| Transition 128,4 L512  master eval | 1.773 (9.34x) | 0.19 | - | - | - | - | - | - | 0.966 (5.09x) |
| Transition 128,4 L512  master train | 4.936 (5.42x) | 0.912 | - | - | - | - | - | - | 3.177 (3.49x) |
| Transition 64,4 L1024 D384 master eval | 1.637 (14.75x) | 0.111 | - | - | - | - | - | - | 0.856 (7.71x) |
| Transition 64,4 L1024 D384 master train | 4.169 (7.72x) | 0.54 | - | - | - | - | - | - | 2.013 (3.73x) |
| Transition 64,4 L2048 D384 master eval | 3.217 (14.01x) | 0.23 | - | - | - | - | - | - | 1.684 (7.33x) |
| Transition 64,4 L2048 D384 master train | 8.161 (7.86x) | 1.038 | - | - | - | - | - | - | 3.928 (3.79x) |
| Transition 64,4 L512 D384 master eval | 0.849 (14.49x) | 0.059 | - | - | - | - | - | - | 0.445 (7.6x) |
| Transition 64,4 L512 D384 master train | 2.164 (7.66x) | 0.282 | - | - | - | - | - | - | 1.049 (3.72x) |
| TriangleAttention 128,4,true,0.25 L1024  master eval | 56.109 (23.68x) | 2.37 | 9.281 (3.92x) | 9.189 (3.88x) | 14.346 (6.05x) | - | 9.649 (4.07x) | - | - |
| TriangleAttention 128,4,true,0.25 L1024  master train | - | 11.886 | 34.986 (2.94x) | 34.923 (2.94x) | 46.741 (3.93x) | - | 35.892 (3.02x) | - | - |
| TriangleAttention 128,4,true,0.25 L2048  master eval | - | 16.549 | 57.934 (3.5x) | 58.033 (3.51x) | 92.468 (5.59x) | - | 55.749 (3.37x) | - | - |
| TriangleAttention 128,4,true,0.25 L2048  master train | - | 83.221 | 227.331 (2.73x) | 227.269 (2.73x) | 372.698 (4.48x) | - | 228.904 (2.75x) | - | - |
| TriangleAttention 128,4,true,0.25 L512  master eval | 8.004 (18.41x) | 0.435 | 1.762 (4.05x) | 1.75 (4.03x) | 2.624 (6.04x) | - | 2.029 (4.67x) | - | - |
| TriangleAttention 128,4,true,0.25 L512  master train | 19.477 (9.56x) | 2.036 | 7.757 (3.81x) | 7.77 (3.82x) | 8.026 (3.94x) | - | 6.659 (3.27x) | - | - |

## h100 accuracy vs float64 (L 512; err = rms-type error of the judge, pytorch = the reference's own error, light vs max(pytorch, fixed))

```
TriangleAttention(..) L512 S1 master eval                               cuda         measured     0.0033  0.0037  GREEN   
TriangleAttention(..) L512 S1 master eval                               gnops        measured     0.0037  0.0037  GREEN   
TriangleAttention(..) L512 S1 master eval                               of3tri       measured     0.0037  0.0037  GREEN   
TriangleAttention(..) L512 S1 master eval                               megafold     measured     0.0037  0.0037  GREEN   
TriangleAttention(..) L512 S1 master eval                               bionemo_ir   unsupported  -       -       -       native library bionemo_ir.libs._cutedsl_kernels cannot be loaded on this host (glibc 2.28 < 2.34 required): it
TriangleAttention(..) L512 S1 master eval                               turbopf      measured     0.0037  0.0037  GREEN   
TriangleAttention(..) L512 S1 master train                              cuda         measured     0.0079  0.0086  GREEN   
TriangleAttention(..) L512 S1 master train                              gnops        measured     0.0098  0.0088  YELLOW  
TriangleAttention(..) L512 S1 master train                              of3tri       measured     0.0098  0.0088  YELLOW  
TriangleAttention(..) L512 S1 master train                              megafold     measured     0.0098  0.0088  YELLOW  
TriangleAttention(..) L512 S1 master train                              bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
TriangleAttention(..) L512 S1 master train                              turbopf      measured     0.0098  0.0088  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  cuda         measured     0.0002  0.0001  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  gnops        measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  of3tri       measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  megafold     measured     0.0002  0.0001  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  cuda         measured     0.0029  0.0029  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  gnops        measured     0.0029  0.0029  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  of3tri       measured     0.0029  0.0029  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  megafold     measured     0.0029  0.0029  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  cuda         measured     0.0016  0.0005  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  gnops        measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  of3tri       measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  megafold     measured     0.002   0.0006  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  cuda         measured     0.0095  0.0087  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  gnops        measured     0.0107  0.0087  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  of3tri       measured     0.0107  0.0087  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  megafold     measured     0.0116  0.0087  YELLOW  
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  cuda         measured     0.0031  0.0031  GREEN   
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  bionemo_ir   unsupported  -       -       -       native library bionemo_ir.libs._cutedsl_kernels cannot be loaded on this host (glibc 2.28 < 2.34 required): it
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  cuda         measured     0.0066  0.0061  GREEN   
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  cuda         measured     0.0031  0.0032  GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  bionemo_ir   measured     0.0007  0.0032  GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  ligo_pwa     measured     0.0031  0.0032  GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  cuda         measured     0.0158  0.0123  YELLOW  
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  ligo_pwa     measured     0.0158  0.0123  YELLOW  
Transition(d_hidden=128;n=4) L512 S1 master eval                        cuda         measured     0.0038  0.0069  GREEN   
Transition(d_hidden=128;n=4) L512 S1 master eval                        bionemo_ir   unsupported  -       -       -       width 128, hidden 512: no shipped silu_gate (no bias / no mask / residual) variant on this SM
Transition(d_hidden=128;n=4) L512 S1 master eval                        megafold_tr  measured     0.0081  0.0069  YELLOW  
Transition(d_hidden=128;n=4) L512 S1 master train                       cuda         measured     0.0075  0.0079  GREEN   
Transition(d_hidden=128;n=4) L512 S1 master train                       bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
Transition(d_hidden=128;n=4) L512 S1 master train                       megafold_tr  measured     0.0109  0.0079  YELLOW  
Transition(d_hidden=64;n=4) L512 D384 S1 master eval                    cuda         measured     0.0045  0.0061  GREEN   
Transition(d_hidden=64;n=4) L512 D384 S1 master eval                    bionemo_ir   unsupported  -       -       -       width 64, hidden 256: no shipped silu_gate (no bias / no mask / residual) variant on this SM
Transition(d_hidden=64;n=4) L512 D384 S1 master eval                    megafold_tr  measured     0.0081  0.0061  YELLOW  
Transition(d_hidden=64;n=4) L512 D384 S1 master train                   cuda         measured     0.0062  0.0085  GREEN   
Transition(d_hidden=64;n=4) L512 D384 S1 master train                   bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
Transition(d_hidden=64;n=4) L512 D384 S1 master train                   megafold_tr  measured     0.0104  0.0085  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  cuda         unsupported  -       -       -       16 heads x d_single 768: the kernels are built for (heads, d_single) in ((8, 384), (12, 384), (16, 384), (24, 
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  gnops        measured     0.0033  0.0033  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  of3tri       measured     0.0033  0.0033  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  megafold     measured     0.0033  0.0033  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  cuda         unsupported  -       -       -       16 heads x d_single 768: the kernels are built for (heads, d_single) in ((8, 384), (12, 384), (16, 384), (24, 
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  gnops        measured     0.0091  0.0085  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  of3tri       measured     0.0091  0.0085  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  megafold     measured     0.0092  0.0087  YELLOW  
```

## a100 speed

| case | pytorch ms | cuda ms | gnops | of3tri | megafold | bionemo_ir | turbopf | ligo_pwa | megafold_tr |
|---|---|---|---|---|---|---|---|---|---|
| AttentionPairBias 384,128,16 L1024  float32 eval | 3.973 (6.22x) | 0.639 | 4.381 (6.86x) | 4.38 (6.85x) | 4.056 (6.35x) | - | - | - | - |
| AttentionPairBias 384,128,16 L1024  float32 train | 8.156 (2.82x) | 2.894 | 26.036 (9x) | 26.062 (9.01x) | 8.215 (2.84x) | - | - | - | - |
| AttentionPairBias 384,128,16 L1024  master eval | 4.413 (15.18x) | 0.291 | 4.366 (15.01x) | 4.36 (14.99x) | 4.44 (15.27x) | - | - | - | - |
| AttentionPairBias 384,128,16 L1024  master train | 8.832 (9.4x) | 0.94 | 8.831 (9.39x) | 8.821 (9.38x) | 9.038 (9.61x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  float32 eval | 15.534 (6.78x) | 2.293 | 16.67 (7.27x) | 16.672 (7.27x) | 15.596 (6.8x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  float32 train | 31.449 (2.85x) | 11.016 | 98.795 (8.97x) | 98.825 (8.97x) | 31.559 (2.86x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  master eval | 17.511 (17.5x) | 1 | 17.117 (17.11x) | 17.104 (17.1x) | 17.321 (17.31x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  master train | 34.68 (10.3x) | 3.368 | 34.336 (10.19x) | 34.316 (10.19x) | 35.128 (10.43x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  float32 eval | 0.95 (4.24x) | 0.224 | 1.111 (4.95x) | 1.108 (4.94x) | 0.989 (4.41x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  float32 train | 2.165 (2.42x) | 0.896 | 7.819 (8.73x) | 7.82 (8.73x) | 2.234 (2.49x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  master eval | 1.141 (11.25x) | 0.101 | 1.115 (11x) | 1.113 (10.98x) | 1.126 (11.11x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  master train | 2.396 (7.31x) | 0.328 | 2.4 (7.32x) | 2.398 (7.32x) | 2.433 (7.42x) | - | - | - | - |
| AttentionPairBias 768,128,16 L1024  master eval | 4.445 (9.6x) | 0.463 | 4.366 (9.43x) | 4.355 (9.41x) | 4.397 (9.5x) | - | - | - | - |
| AttentionPairBias 768,128,16 L1024  master train | 8.947 (4.04x) | 2.212 | 11.807 (5.34x) | 11.794 (5.33x) | 9.029 (4.08x) | - | - | - | - |
| AttentionPairBias 768,128,16 L2048  master eval | 17.578 (11.02x) | 1.595 | 17.195 (10.78x) | 17.185 (10.77x) | 17.27 (10.83x) | - | - | - | - |
| AttentionPairBias 768,128,16 L2048  master train | 34.878 (4.4x) | 7.931 | 42.967 (5.42x) | 42.918 (5.41x) | 34.863 (4.4x) | - | - | - | - |
| AttentionPairBias 768,128,16 L512  master eval | 1.163 (6.76x) | 0.172 | 1.144 (6.65x) | 1.141 (6.63x) | 1.153 (6.7x) | - | - | - | - |
| AttentionPairBias 768,128,16 L512  master train | 2.463 (3.18x) | 0.774 | 3.503 (4.53x) | 3.506 (4.53x) | 2.478 (3.2x) | - | - | - | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L1024 D1024 master eval | 9.756 (7.69x) | 1.269 | - | - | - | 11.926 (9.4x) | - | 262.367 (206.79x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L1024 D1024 master train | 22.003 (4.43x) | 4.966 | - | - | - | - | - | 308.385 (62.09x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L2048 D1024 master eval | 29.051 (6.83x) | 4.255 | - | - | - | 33.388 (7.85x) | - | 2429.728 (571.07x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L2048 D1024 master train | 64.367 | - | - | - | - | - | - | 2578.564 | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L512 D1024 master eval | 3.865 (7.98x) | 0.484 | - | - | - | 4.957 (10.23x) | - | 65.829 (135.91x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L512 D1024 master train | 8.822 (5.08x) | 1.736 | - | - | - | - | - | 82.754 (47.68x) | - |
| OuterProductMean 64,128,32,false,0.001 L1024 D1024 master eval | 21.121 (1.7x) | 12.448 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L1024 D1024 master train | 55.326 (1.52x) | 36.336 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L2048 D1024 master eval | 78.788 (1.61x) | 48.804 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L2048 D1024 master train | 206.114 (1.44x) | 143.144 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L512 D1024 master eval | 6.284 (2.08x) | 3.021 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L512 D1024 master train | 15.852 (1.72x) | 9.23 | - | - | - | - | - | - | - |
| Transition 128,4 L1024  master eval | 13.62 (7.3x) | 1.865 | - | - | - | - | - | - | 8.121 (4.35x) |
| Transition 128,4 L1024  master train | 36.623 (3.69x) | 9.919 | - | - | - | - | - | - | 23.204 (2.34x) |
| Transition 128,4 L2048  master eval | 54.592 (6.91x) | 7.898 | - | - | - | - | - | - | 32.623 (4.13x) |
| Transition 128,4 L2048  master train | 146.321 (3.66x) | 39.998 | - | - | - | - | - | - | 93.373 (2.33x) |
| Transition 128,4 L512  master eval | 3.414 (7.52x) | 0.454 | - | - | - | - | - | - | 1.96 (4.32x) |
| Transition 128,4 L512  master train | 9.504 (3.82x) | 2.485 | - | - | - | - | - | - | 5.912 (2.38x) |
| Transition 64,4 L1024 D384 master eval | 2.979 (13.26x) | 0.225 | - | - | - | - | - | - | 1.499 (6.67x) |
| Transition 64,4 L1024 D384 master train | 7.727 (6.39x) | 1.208 | - | - | - | - | - | - | 3.879 (3.21x) |
| Transition 64,4 L2048 D384 master eval | 5.884 (13.62x) | 0.432 | - | - | - | - | - | - | 3.019 (6.99x) |
| Transition 64,4 L2048 D384 master train | 15.175 (6.33x) | 2.399 | - | - | - | - | - | - | 7.735 (3.22x) |
| Transition 64,4 L512 D384 master eval | 1.544 (12.57x) | 0.123 | - | - | - | - | - | - | 0.765 (6.23x) |
| Transition 64,4 L512 D384 master train | 3.998 (6.06x) | 0.659 | - | - | - | - | - | - | 1.907 (2.89x) |
| TriangleAttention 128,4,true,0.25 L1024  master eval | 106.863 (17.56x) | 6.087 | 17.373 (2.85x) | 17.332 (2.85x) | 25.212 (4.14x) | - | - | - | - |
| TriangleAttention 128,4,true,0.25 L1024  master train | - | 31.272 | 408.223 (13.05x) | 408.956 (13.08x) | 98.394 (3.15x) | - | - | - | - |
| TriangleAttention 128,4,true,0.25 L2048  master eval | - | 39.241 | 121.552 (3.1x) | 121.528 (3.1x) | 167.816 (4.28x) | - | - | - | - |
| TriangleAttention 128,4,true,0.25 L2048  master train | - | 216.111 | 2788.807 (12.9x) | 2788.99 (12.91x) | 788.105 (3.65x) | - | - | - | - |
| TriangleAttention 128,4,true,0.25 L512  master eval | 14.128 (14.26x) | 0.991 | 3.124 (3.15x) | 3.132 (3.16x) | 4.502 (4.54x) | - | - | - | - |
| TriangleAttention 128,4,true,0.25 L512  master train | 35.022 (6.82x) | 5.135 | 73.456 (14.31x) | 73.492 (14.31x) | 15.984 (3.11x) | - | - | - | - |

## a100 accuracy vs float64 (L 512; err = rms-type error of the judge, pytorch = the reference's own error, light vs max(pytorch, fixed))

```
TriangleAttention(..) L512 S1 master eval                               cuda         measured     0.0037  0.0034  GREEN   
TriangleAttention(..) L512 S1 master eval                               gnops        measured     0.0034  0.0034  GREEN   
TriangleAttention(..) L512 S1 master eval                               of3tri       measured     0.0034  0.0034  GREEN   
TriangleAttention(..) L512 S1 master eval                               megafold     measured     0.0034  0.0034  GREEN   
TriangleAttention(..) L512 S1 master eval                               bionemo_ir   unsupported  -       -       -       native library bionemo_ir.libs._cutedsl_kernels cannot be loaded on this host (cuModuleLoadData(015e13181a6001
TriangleAttention(..) L512 S1 master eval                               turbopf      unsupported  -       -       -       turbopf: ImportError: Hopper (sm_90) only; this device is capability (8, 0) (see third_party/ README, fetch.sh
TriangleAttention(..) L512 S1 master train                              cuda         measured     0.0091  0.0121  GREEN   
TriangleAttention(..) L512 S1 master train                              gnops        measured     0.0068  0.0075  GREEN   
TriangleAttention(..) L512 S1 master train                              of3tri       measured     0.0068  0.0075  GREEN   
TriangleAttention(..) L512 S1 master train                              megafold     measured     0.0107  0.0106  YELLOW  
TriangleAttention(..) L512 S1 master train                              bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
TriangleAttention(..) L512 S1 master train                              turbopf      unsupported  -       -       -       turbopf: ImportError: Hopper (sm_90) only; this device is capability (8, 0) (see third_party/ README, fetch.sh
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  cuda         measured     0.0001  0.0001  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  gnops        measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  of3tri       measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  megafold     measured     0.0001  0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  cuda         measured     0.0023  0.0023  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  gnops        measured     0.0023  0.0023  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  of3tri       measured     0.0023  0.0023  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  megafold     measured     0.0023  0.0023  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  cuda         measured     0.0007  0.0005  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  gnops        measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  of3tri       measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  megafold     measured     0.0021  0.0007  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  cuda         measured     0.0112  0.008   YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  gnops        measured     0.0088  0.0088  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  of3tri       measured     0.0088  0.0088  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  megafold     measured     0.01    0.0087  YELLOW  
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  cuda         measured     0.0031  0.0031  GREEN   
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  bionemo_ir   unsupported  -       -       -       native library bionemo_ir.libs._cutedsl_kernels cannot be loaded on this host (cuModuleLoadData(015e13181a6001
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  cuda         measured     0.0063  0.0061  GREEN   
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  cuda         measured     0.0032  0.0032  GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  bionemo_ir   measured     0.0007  0.0032  GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  ligo_pwa     measured     0.0031  0.0032  GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  cuda         measured     0.0153  0.0153  GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  ligo_pwa     measured     0.0153  0.0153  GREEN   
Transition(d_hidden=128;n=4) L512 S1 master eval                        cuda         measured     0.0069  0.0069  GREEN   
Transition(d_hidden=128;n=4) L512 S1 master eval                        bionemo_ir   unsupported  -       -       -       width 128, hidden 512: no shipped silu_gate (no bias / no mask / residual) variant on this SM
Transition(d_hidden=128;n=4) L512 S1 master eval                        megafold_tr  measured     0.0081  0.0069  YELLOW  
Transition(d_hidden=128;n=4) L512 S1 master train                       cuda         measured     0.0069  0.0069  GREEN   
Transition(d_hidden=128;n=4) L512 S1 master train                       bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
Transition(d_hidden=128;n=4) L512 S1 master train                       megafold_tr  measured     0.0109  0.0079  YELLOW  
Transition(d_hidden=64;n=4) L512 D384 S1 master eval                    cuda         measured     0.0041  0.0061  GREEN   
Transition(d_hidden=64;n=4) L512 D384 S1 master eval                    bionemo_ir   unsupported  -       -       -       width 64, hidden 256: no shipped silu_gate (no bias / no mask / residual) variant on this SM
Transition(d_hidden=64;n=4) L512 D384 S1 master eval                    megafold_tr  measured     0.0081  0.0061  YELLOW  
Transition(d_hidden=64;n=4) L512 D384 S1 master train                   cuda         measured     0.0045  0.0066  GREEN   
Transition(d_hidden=64;n=4) L512 D384 S1 master train                   bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
Transition(d_hidden=64;n=4) L512 D384 S1 master train                   megafold_tr  measured     0.0098  0.0085  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  cuda         measured     0.0031  0.0032  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  gnops        measured     0.0031  0.0032  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  of3tri       measured     0.0031  0.0032  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  megafold     measured     0.0031  0.0032  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  cuda         measured     0.0085  0.0082  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  gnops        measured     0.0086  0.0084  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  of3tri       measured     0.0086  0.0084  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  megafold     measured     0.0082  0.0071  YELLOW  
```

## rtx5090 speed

| case | pytorch ms | cuda ms | gnops | of3tri | megafold | bionemo_ir | turbopf | ligo_pwa | megafold_tr |
|---|---|---|---|---|---|---|---|---|---|
| AttentionPairBias 384,128,16 L1024  float32 eval | 1.887 (3.97x) | 0.475 | 2.216 (4.66x) | 2.213 (4.66x) | 2.104 (4.43x) | - | - | - | - |
| AttentionPairBias 384,128,16 L1024  float32 train | 5.142 (2.9x) | 1.775 | 22.161 (12.49x) | 21.836 (12.3x) | 20.946 (11.8x) | - | - | - | - |
| AttentionPairBias 384,128,16 L1024  master eval | 2.594 (10.23x) | 0.253 | 2.549 (10.06x) | 2.54 (10.02x) | 2.536 (10.01x) | - | - | - | - |
| AttentionPairBias 384,128,16 L1024  master train | 5.945 (7.39x) | 0.805 | 5.881 (7.31x) | 5.88 (7.31x) | 5.923 (7.36x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  float32 eval | 8.367 (4.73x) | 1.768 | 8.894 (5.03x) | 8.744 (4.94x) | 8.494 (4.8x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  float32 train | 20.596 (3.18x) | 6.477 | 87.829 (13.56x) | 87.741 (13.55x) | 84.136 (12.99x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  master eval | 10.855 (11.5x) | 0.944 | 10.725 (11.36x) | 10.735 (11.37x) | 10.676 (11.31x) | - | - | - | - |
| AttentionPairBias 384,128,16 L2048  master train | 24.474 (8.01x) | 3.057 | 24.302 (7.95x) | 24.29 (7.95x) | 24.465 (8x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  float32 eval | 0.506 (3.34x) | 0.152 | 0.645 (4.26x) | 0.643 (4.24x) | 0.61 (4.03x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  float32 train | 1.328 (2.48x) | 0.535 | 6.1 (11.41x) | 6.048 (11.31x) | 5.607 (10.49x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  master eval | 0.655 (8.21x) | 0.08 | 0.644 (8.06x) | 0.642 (8.03x) | 0.645 (8.07x) | - | - | - | - |
| AttentionPairBias 384,128,16 L512  master train | 1.565 (6.88x) | 0.227 | 1.579 (6.95x) | 1.58 (6.95x) | 1.564 (6.88x) | - | - | - | - |
| AttentionPairBias 768,128,16 L1024  master eval | 2.64 (9.14x) | 0.289 | 2.58 (8.94x) | 2.579 (8.93x) | 2.606 (9.03x) | - | - | - | - |
| AttentionPairBias 768,128,16 L1024  master train | 6.257 (5.12x) | 1.222 | 6.255 (5.12x) | 6.215 (5.09x) | 6.076 (4.97x) | - | - | - | - |
| AttentionPairBias 768,128,16 L2048  master eval | 11.013 (11.1x) | 0.992 | 11.169 (11.26x) | 11.3 (11.39x) | 11.329 (11.42x) | - | - | - | - |
| AttentionPairBias 768,128,16 L2048  master train | 24.933 (5.9x) | 4.229 | 25.273 (5.98x) | 25.271 (5.98x) | 24.796 (5.86x) | - | - | - | - |
| AttentionPairBias 768,128,16 L512  master eval | 0.655 (6.4x) | 0.102 | 0.69 (6.74x) | 0.688 (6.72x) | 0.69 (6.74x) | - | - | - | - |
| AttentionPairBias 768,128,16 L512  master train | 1.705 (3.14x) | 0.544 | 1.752 (3.22x) | 1.742 (3.2x) | 1.672 (3.08x) | - | - | - | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L1024 D1024 master eval | 6.268 (4.84x) | 1.294 | - | - | - | 8.108 (6.27x) | - | 140.569 (108.65x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L1024 D1024 master train | 16.013 (3.93x) | 4.071 | - | - | - | - | - | 168.357 (41.35x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L2048 D1024 master eval | 18.947 (4.67x) | 4.054 | - | - | - | 22.688 (5.6x) | - | 495.024 (122.11x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L2048 D1024 master train | 47.858 | - | - | - | - | - | - | 579.095 | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L512 D1024 master eval | 2.406 (5.2x) | 0.463 | - | - | - | 3.387 (7.32x) | - | 41.858 (90.44x) | - |
| MSAPairWeightedAveraging 64,128,8,8,0.25 L512 D1024 master train | 6.204 (4.19x) | 1.479 | - | - | - | - | - | 50.487 (34.13x) | - |
| OuterProductMean 64,128,32,false,0.001 L1024 D1024 master eval | 16.849 (1.46x) | 11.538 | - | - | - | 14.048 (1.22x) | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L1024 D1024 master train | 46.809 (1.28x) | 36.443 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L2048 D1024 master eval | 62.457 (1.36x) | 45.86 | - | - | - | 52.071 (1.14x) | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L2048 D1024 master train | - | 142.216 | - | - | - | - | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L512 D1024 master eval | 4.807 (1.62x) | 2.969 | - | - | - | 4.396 (1.48x) | - | - | - |
| OuterProductMean 64,128,32,false,0.001 L512 D1024 master train | 13.331 (1.39x) | 9.563 | - | - | - | - | - | - | - |
| Transition 128,4 L1024  master eval | 12.317 (7.77x) | 1.586 | - | - | - | - | - | - | 6.45 (4.07x) |
| Transition 128,4 L1024  master train | 34.929 (4.64x) | 7.521 | - | - | - | - | - | - | 17.647 (2.35x) |
| Transition 128,4 L2048  master eval | 47.52 (8.07x) | 5.886 | - | - | - | - | - | - | 25.377 (4.31x) |
| Transition 128,4 L2048  master train | - | 30.2 | - | - | - | - | - | - | 70.964 (2.35x) |
| Transition 128,4 L512  master eval | 2.846 (6.27x) | 0.454 | - | - | - | - | - | - | 1.531 (3.37x) |
| Transition 128,4 L512  master train | 8.413 (4.41x) | 1.906 | - | - | - | - | - | - | 4.391 (2.3x) |
| Transition 64,4 L1024 D384 master eval | 2.221 (13.9x) | 0.16 | - | - | - | - | - | - | 1.112 (6.96x) |
| Transition 64,4 L1024 D384 master train | 6.472 (6.71x) | 0.965 | - | - | - | - | - | - | 2.87 (2.97x) |
| Transition 64,4 L2048 D384 master eval | 4.588 (14.74x) | 0.311 | - | - | - | - | - | - | 2.355 (7.56x) |
| Transition 64,4 L2048 D384 master train | 13.128 (7.09x) | 1.852 | - | - | - | - | - | - | 5.851 (3.16x) |
| Transition 64,4 L512 D384 master eval | 1.023 (11.62x) | 0.088 | - | - | - | - | - | - | 0.539 (6.12x) |
| Transition 64,4 L512 D384 master train | 3.1 (6.45x) | 0.481 | - | - | - | - | - | - | 1.425 (2.97x) |
| TriangleAttention 128,4,true,0.25 L1024  master eval | - | 6.303 | 9.908 (1.57x) | 9.924 (1.57x) | 14.98 (2.38x) | 10.121 (1.61x) | - | - | - |
| TriangleAttention 128,4,true,0.25 L1024  master train | - | 25.431 | 49.876 (1.96x) | 49.898 (1.96x) | 47.327 (1.86x) | - | - | - | - |
| TriangleAttention 128,4,true,0.25 L2048  master eval | - | 38.166 | 56.199 (1.47x) | 56.158 (1.47x) | 89.395 (2.34x) | 53.701 (1.41x) | - | - | - |
| TriangleAttention 128,4,true,0.25 L2048  master train | - | - | 298.342 | 298.515 | 378.128 | - | - | - | - |
| TriangleAttention 128,4,true,0.25 L512  master eval | 12.071 (10.05x) | 1.201 | 1.998 (1.66x) | 1.996 (1.66x) | 2.827 (2.35x) | 2.246 (1.87x) | - | - | - |
| TriangleAttention 128,4,true,0.25 L512  master train | 31.494 (7.27x) | 4.332 | 10.788 (2.49x) | 10.642 (2.46x) | 8.15 (1.88x) | - | - | - | - |

## rtx5090 accuracy vs float64 (L 512; err = rms-type error of the judge, pytorch = the reference's own error, light vs max(pytorch, fixed))

```
TriangleAttention(..) L512 S1 master eval                               cuda         measured     0.0032  0.0034  GREEN   
TriangleAttention(..) L512 S1 master eval                               gnops        measured     0.0034  0.0034  GREEN   
TriangleAttention(..) L512 S1 master eval                               of3tri       measured     0.0034  0.0034  GREEN   
TriangleAttention(..) L512 S1 master eval                               megafold     measured     0.0034  0.0034  GREEN   
TriangleAttention(..) L512 S1 master eval                               bionemo_ir   measured     0.0034  0.0034  GREEN   
TriangleAttention(..) L512 S1 master eval                               turbopf      unsupported  -       -       -       turbopf: ImportError: Hopper (sm_90) only; this device is capability (12, 0) (see third_party/ README, fetch.s
TriangleAttention(..) L512 S1 master train                              cuda         measured     0.0083  0.009   GREEN   
TriangleAttention(..) L512 S1 master train                              gnops        measured     0.0088  0.0099  GREEN   
TriangleAttention(..) L512 S1 master train                              of3tri       measured     0.0088  0.0099  GREEN   
TriangleAttention(..) L512 S1 master train                              megafold     measured     0.0094  0.0099  GREEN   
TriangleAttention(..) L512 S1 master train                              bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
TriangleAttention(..) L512 S1 master train                              turbopf      unsupported  -       -       -       turbopf: ImportError: Hopper (sm_90) only; this device is capability (12, 0) (see third_party/ README, fetch.s
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  cuda         measured     0.0002  0.0001  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  gnops        measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  of3tri       measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 e  megafold     measured     0.0001  0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  cuda         measured     0.0031  0.0029  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  gnops        measured     0.0031  0.0029  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  of3tri       measured     0.0031  0.0029  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master ev  megafold     measured     0.0031  0.0029  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  cuda         measured     0.001   0.0007  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  gnops        measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  of3tri       measured     0       0.0001  GREEN   
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 float32 t  megafold     measured     0.0023  0.0007  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  cuda         measured     0.0085  0.0076  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  gnops        measured     0.0085  0.0082  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  of3tri       measured     0.0085  0.0082  YELLOW  
AttentionPairBias(d_single=384;d_pair=128;n_head=16) L512 S1 master tr  megafold     measured     0.0104  0.0082  YELLOW  
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  cuda         measured     0.0031  0.0031  GREEN   
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  bionemo_ir   measured     0.0031  0.0031  GREEN   
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  cuda         measured     0.0052  0.0054  GREEN   
OuterProductMean(d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj  bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  cuda         measured     0.003   0.003   GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  bionemo_ir   measured     0.0007  0.003   GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  ligo_pwa     measured     0.003   0.003   GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  cuda         measured     0.0075  0.008   GREEN   
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
MSAPairWeightedAveraging(d_msa=64;d_pair=128;n_head=8;d_hidden=8;p_dro  ligo_pwa     measured     0.0112  0.008   YELLOW  
Transition(d_hidden=128;n=4) L512 S1 master eval                        cuda         measured     0.0049  0.0069  GREEN   
Transition(d_hidden=128;n=4) L512 S1 master eval                        bionemo_ir   unsupported  -       -       -       width 128, hidden 512: no shipped silu_gate (no bias / no mask / residual) variant on this SM
Transition(d_hidden=128;n=4) L512 S1 master eval                        megafold_tr  measured     0.0081  0.0069  YELLOW  
Transition(d_hidden=128;n=4) L512 S1 master train                       cuda         measured     0.0049  0.0048  GREEN   
Transition(d_hidden=128;n=4) L512 S1 master train                       bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
Transition(d_hidden=128;n=4) L512 S1 master train                       megafold_tr  measured     0.0109  0.0079  YELLOW  
Transition(d_hidden=64;n=4) L512 D384 S1 master eval                    cuda         measured     0.0045  0.0061  GREEN   
Transition(d_hidden=64;n=4) L512 D384 S1 master eval                    bionemo_ir   unsupported  -       -       -       width 64, hidden 256: no shipped silu_gate (no bias / no mask / residual) variant on this SM
Transition(d_hidden=64;n=4) L512 D384 S1 master eval                    megafold_tr  measured     0.0081  0.0061  YELLOW  
Transition(d_hidden=64;n=4) L512 D384 S1 master train                   cuda         measured     0.0053  0.0067  GREEN   
Transition(d_hidden=64;n=4) L512 D384 S1 master train                   bionemo_ir   unsupported  -       -       -       BioIR kernels are inference only (no backward)
Transition(d_hidden=64;n=4) L512 D384 S1 master train                   megafold_tr  measured     0.0104  0.0085  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  cuda         measured     0.0028  0.0028  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  gnops        measured     0.0028  0.0028  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  of3tri       measured     0.0028  0.0028  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master ev  megafold     measured     0.0028  0.0028  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  cuda         measured     0.0079  0.0082  GREEN   
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  gnops        measured     0.0109  0.0099  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  of3tri       measured     0.0109  0.0099  YELLOW  
AttentionPairBias(d_single=768;d_pair=128;n_head=16) L512 S1 master tr  megafold     measured     0.0111  0.0099  YELLOW  
```

