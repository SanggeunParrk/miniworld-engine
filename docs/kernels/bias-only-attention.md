# bias-only attention

This document records the MiniWorld bias-only TriangleAttention benchmark path.
It covers `TriangleAttention(use_self_attention=False)`: LayerNorm/projection,
bias-only attention, and output gating. The `softmax -> bmm` portion is kept as
the unfused attention operation; the optimization target is the surrounding
kernel structure. Full triangular self-attention is tracked separately under the
`triangle_attention` benchmark target.

## Scope

- Module target: `bias_only_attention`
- Benchmark config: `benchmarks/modules/bias_only_attention/configs/bench.yaml`
- Runner: `benchmarks/runners/bench.py`
- Batch entrypoint: `submits/run_bench.sbatch` with `BENCH_TARGET=bias_only_attention`
- Artifacts: `benchmarks/modules/bias_only_attention/artifacts/`

Implementations in the final benchmark:

- `pytorch`: compiled module reference
- `cuequivariance`: cuEquivariance module path
- `old_triton`: vendored Team-GM bias-only Triton attention kernel
- `miniworld`: MiniWorld LayerNorm/projection/gate dispatch path

The benchmark uses `compile=true`, bf16 mixed precision, `mask_prob=0.0`,
`n_layers=1`, `L in {384, 512, 640, 768, 896, 1024}`, and
`d_pair in {128, 256, 512}` at fixed `L=384` for d sweeps.

## CUDA graph regime

The primary comparison is `compile=true + cudagraph=manual`. This is the fair
steady-state regime for this composed module because the MiniWorld path has
multiple launches around the unfused `softmax -> bmm` core. Without CUDA graph
capture, launch overhead dominates and hides kernel-side improvements.

`compile=true + cudagraph=disabled` is still run as a diagnostic. It currently
shows MiniWorld slower than compiled PyTorch/cuEquivariance for several points,
which is expected for the launch-bound regime and is not the target result.

## d_pair 128/256 training fix

The previous full run failed in training for `d_pair=128/256` with:

```text
TypeError: dynamic_func() missing 2 required positional arguments: 'N' and 'HAS_ROWSCALE'
```

The issue was not bias-only tiling. `layer_norm_bwd_dx_fused` had gained a
`Rowscale` argument and `HAS_ROWSCALE` meta parameter, while the
`layernorm_linear` and compile-native LayerNorm callers were still using the old
signature. The fixed callers pass a placeholder rowscale tensor and
`HAS_ROWSCALE=False`.

Validated targeted repros:

- training, `compile=true`, `cudagraph=manual`, `L=384`, MiniWorld only:
  `d_pair=128` at `1.16346 ms`, `d_pair=256` at `2.25346 ms`
- training, `compile=true`, `cudagraph=disabled`, `L=384`, MiniWorld only:
  `d_pair=128` at `1.83210 ms`, `d_pair=256` at `2.34046 ms`

## H100 benchmark: job 10265

Run:

```bash
sbatch --export=ALL,BENCH_TARGET=bias_only_attention,MINIWORLD_BIASONLY_AUTOTUNE=off \
  --output=benchmarks/modules/bias_only_attention/artifacts/bias_only_%j.out \
  submits/run_bench.sbatch
```

Result log:

```text
benchmarks/modules/bias_only_attention/artifacts/bias_only_10265.out
MODULE BENCH DONE target=bias_only_attention
```

No `failed`, `Traceback`, or `RuntimeError` rows were present in the completed
run. CSVs and SVGs were written under:

```text
benchmarks/modules/bias_only_attention/artifacts/NVIDIA H100 80GB HBM3/
```

### Manual CUDA graph results

Training d sweep, fixed `L=384`:

| d_pair | PyTorch | cuEquivariance | old_triton | MiniWorld | speedup vs best baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2.067 ms | 2.067 ms | 3.125 ms | 1.157 ms | 1.79x |
| 256 | 3.660 ms | 3.656 ms | 4.750 ms | 2.246 ms | 1.63x |
| 512 | 6.951 ms | 6.959 ms | 8.043 ms | 4.325 ms | 1.61x |

Inference d sweep, fixed `L=384`:

| d_pair | PyTorch | cuEquivariance | old_triton | MiniWorld | speedup vs best baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.801 ms | 0.798 ms | 0.926 ms | 0.396 ms | 2.02x |
| 256 | 1.362 ms | 1.363 ms | 1.491 ms | 0.846 ms | 1.61x |
| 512 | 2.502 ms | 2.501 ms | 2.605 ms | 1.497 ms | 1.67x |

Training L sweep, fixed `d_pair=128`:

| L | best baseline | MiniWorld | speedup |
| ---: | ---: | ---: | ---: |
| 384 | 2.063 ms | 1.160 ms | 1.78x |
| 512 | 3.519 ms | 1.940 ms | 1.81x |
| 640 | 5.443 ms | 3.021 ms | 1.80x |
| 768 | 7.755 ms | 4.245 ms | 1.83x |
| 896 | 10.700 ms | 5.962 ms | 1.80x |
| 1024 | 13.817 ms | 7.631 ms | 1.81x |

Inference L sweep, fixed `d_pair=128`:

| L | best baseline | MiniWorld | speedup |
| ---: | ---: | ---: | ---: |
| 384 | 0.798 ms | 0.396 ms | 2.02x |
| 512 | 1.383 ms | 0.671 ms | 2.06x |
| 640 | 2.142 ms | 1.054 ms | 2.03x |
| 768 | 3.075 ms | 1.532 ms | 2.01x |
| 896 | 4.228 ms | 2.120 ms | 1.99x |
| 1024 | 5.514 ms | 2.793 ms | 1.97x |

## Compile-only diagnostic

With `compile=true` and `cudagraph=disabled`, MiniWorld completes but is slower
than the best compiled PyTorch/cuEquivariance baseline:

- inference d sweep at `L=384`: `0.37x` to `0.71x`
- training d sweep at `L=384`: `0.52x` to `0.75x`
- training L sweep at `d_pair=128`: `0.52x` to `0.80x`

This diagnostic confirms the old `d_pair=128/256` failure is fixed, but it is
not the performance regime used for the final module comparison.

## 시도했고 진 것: strided-gather GEMM (삭제됨)

`kernels/bias_only_attention/triton/fused.py` 에 있던 `_bias_only_gemm` 은 2026-08-20 에 삭제했다.
코드는 지웠지만 결론은 남긴다 — **다시 시도하지 말 것.**

연산:

    out[b,h,i,j,d] = sum_k softmax_k(bias[b,h,j,k]) * value[b,h,i,k,d]

아이디어는 "torch 의 permute 왕복을 피한다"였다. `k` 가 축약축이자 softmax 축이고
`softmax(bias)` 는 `i` 에 무관하므로, `A = softmax(bias)` 를 한 번만 구하고 `(b,h)` 당 GEMM
하나로 접는다. torch 는 이미 그 GEMM 을 하지만 입력에서 value-permute, 출력에서 output-permute
를 낸다. 그래서 value 를 `V'[k,(i,d)]` 로 **strided 하게 읽고** 출력도 strided 로 써서 왕복을
둘 다 없애려 했다.

**결과: torch.einsum 보다 ~8배 느리다** (정확도는 문제없음, cosine 1.0). 이유는
`value[i,k,d]` 의 i-stride 가 `L*D` 라서, per-k 로드가 심하게 non-coalesced 가 되기 때문이다.
torch 의 permute -> contiguous -> cuBLAS 경로가 압도적으로 이긴다. op 레벨 승자는 그냥
`torch.einsum` 이고, 실제 이득은 모듈 레벨(LN + `.contiguous()` + gate)에 있다.

덧붙여 이 커널은 축 설계 자체가 잘못돼 있었다: 누산기가
`acc = tl.zeros([BLOCK_M1_J, BLOCK_M1_I * BLOCK_K])` 로, N extent 가 **튜닝 축 두 개의 곱**이다.
all-64 조합이 64x4096 fp32 누산기가 되어 컴파일만 45분씩 걸렸고, 프로덕션에서 쓰이지도 않으면서
모든 검증 실행의 벽시계를 이 하나가 좌우했다. dispatch 는 이 경로를 참조한 적이 없다
(`modules/triangle_attention/module.py` -> `bias_only_attention.dispatch` -> `main.py` / `gate_out.py`).
유일한 호출자는 벤치 브랜치 하나였고 그것도 함께 지웠다.
