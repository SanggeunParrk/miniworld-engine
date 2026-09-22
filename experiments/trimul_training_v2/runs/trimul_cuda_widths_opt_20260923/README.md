# TriMul D별 CUDA 최적화 · 2026-09-23

D64/128/256/384/512 × L384/768을 연결·검증했다. D128 L384는 기존 선택을 유지했다. 나머지는 이전 CUDA 포트보다 빨라졌다. **D128 외 네 폭은 여전히 Triton보다 느리며, 성능 개발 완료나 SoL90 달성을 주장하지 않는다.**

## 왜 D128 L768의 배속이 떨어졌나

L384는 단일 B7, L768은 분리 B7이었다. 이전 L768 trace에서 뒤쪽 두 커널은 825 + 1207 = 2032us를 차지했다. 단일 후보는 dWL 상대 오차0.051913%가 기존 기준0.050000%를 초과해 선택하지 못했다.

단일 커널 내부에서 dW의 긴 FP32 누적을 두 구간으로 나누고 마지막에 합치도록 수정했다. 기존 x_n 저장·재계산 정책과 미분 수식은 유지했다. dWL 오차는0.013900%로 낮아졌다. 입력·weight 변경, gamma=0, graph/eager, strict LN 기준을 통과해 L768도 단일 B7로 연결했다. B1과 forward는 기존 경로다.

## 다른 D는 왜 느렸나 / 이번 변경

앞선 것은 D128의 최적화 수준을 확장한 구현이 아니라, 폭을 지원하도록 만든 기능 포트였다. 큰 전역 작업 공간과 반복적인 입력/가중치 로드, GEMM 사이의 동기화가 남아 있었다. D가 커지면 GEMM 연산량도 D²에 비례해 커진다.

- 입력 projection/gate 재계산에 Anthropic K1의 register operand + TMA producer + WGMMA consumer를 차용했다. D64/256은 입력 x_n 타일을 레지스터에 유지한다. D384/512는 shared memory에 유지하고 WGMMA SS가 직접 읽어 register spill을 없앤다. 가중치는 순서대로 공급한다.
- dLeft/dRight를 TMA로 읽고 dp/dg를 TMA로 저장한다. 이 작업 공간은 channel-major로 바꿔 dX/dW GEMM이 같은 값을 소비한다.
- NCU에서 register-operand D512의 local load 요청202,205,424 sectors, ptxas spill load436bytes를 발견했다. 무작정 register 한도를 풀면 CTA 상주 수가 줄어 더 느렸다. 큰 폭의 x_n을 shared operand로 바꿔 같은 두 CTA/SM에서 ptxas spill 0을 달성했다.
- 한 weight block 전체가 끝날 때까지 기다리는 대신, 필요한 K 조각의 WGMMA가 끝나면 그 shared slot을 즉시 반환한다. 가중치 버퍼를 줄여 큰 D에서도 B7 CTA 두 개가 SM에 상주할 수 있게 했다.
- **B7은 하나의 CUDA kernel launch다.** 성능을 진단한 분리 producer 후보는 최종 연결에 넣지 않았다. dX와 dW를 별도 kernel로 분리하지 않았다. 다만 단일 launch 내부의 global dp/dg·partial dW 작업 공간은 남아 있다.
- forward/B1은 이전의 검증된 CUDA cubin을 재사용한다. B7의 자원 요구량 때문에 forward/B1까지 occupancy가 줄지 않게 grid를 별도로 계산한다.
- m64n128, 더 큰 warp-group 타일, projection accumulator 압축 등도 시험했으나 전체가 느려져 선택하지 않았다.

## 같은 GPU에서 이전 CUDA와 교대 측정

이전은 바로 앞 D별 CUDA 포트다. Anthropic 또는 cuEquivariance 대비 배속이 아니다. D128 L384의 차이는 잡음 수준이다.

| D | L | 이전 CUDA ms | 현재 CUDA ms | 배속 | 시간 감소 |
|---:|---:|---:|---:|---:|---:|
| 64 | 384 | 1.628 | 1.339 | 1.22x | 17.7% |
| 64 | 768 | 6.257 | 4.961 | 1.26x | 20.7% |
| 128 | 384 | 0.991 | 0.990 | 1.00x | 0.1% |
| 128 | 768 | 4.795 | 3.935 | 1.22x | 17.9% |
| 256 | 384 | 6.672 | 5.428 | 1.23x | 18.6% |
| 256 | 768 | 27.218 | 22.054 | 1.23x | 19.0% |
| 384 | 384 | 11.762 | 9.596 | 1.23x | 18.4% |
| 384 | 768 | 48.510 | 39.323 | 1.23x | 18.9% |
| 512 | 384 | 16.631 | 14.395 | 1.16x | 13.4% |
| 512 | 768 | 69.617 | 60.242 | 1.16x | 13.5% |

## 별도의 같은 실행 Triton/CUDA 교대 측정

위 표와 별도 실행이다. 각 행의 두 값은 같은 프로세스에서 측정했다. 1x 미만이면 CUDA가 느리다. Triton 기준은 기존 벤치의 정적 compile 경로이며 cache miss 시 heuristic24 설정을 사용하는 경우가 있어, 모든 config를 재튜닝한 최상 성능이라고 주장하지 않는다.

| D | L | Triton ms | CUDA ms | Triton 대비 배속 |
|---:|---:|---:|---:|---:|
| 64 | 384 | 0.810 | 1.354 | 0.60x |
| 64 | 768 | 3.050 | 4.982 | 0.61x |
| 128 | 384 | 1.651 | 0.988 | 1.67x |
| 128 | 768 | 6.528 | 4.031 | 1.62x |
| 256 | 384 | 3.451 | 5.434 | 0.64x |
| 256 | 768 | 17.492 | 22.116 | 0.79x |
| 384 | 384 | 6.184 | 9.697 | 0.64x |
| 384 | 768 | 26.967 | 40.336 | 0.67x |
| 512 | 384 | 9.290 | 14.590 | 0.64x |
| 512 | 768 | 41.252 | 61.238 | 0.67x |

## 실제 배선 / HBM 흐름

```
forward: x -> Anthropic-derived K1 -> left/right + saved x_n
         left/right -> cuBLAS x2 -> tri -> existing CUDA output -> y
backward: tri, x_n, dy -> existing CUDA B1 -> dTri, dGate, output weight/LN grads
          dTri, left/right -> cuBLAS x4 -> dLeft/dRight
          x, x_n, dLeft/dRight, dGate, dy -> one CUDA B7 -> dx, input weight/LN grads
B7 internal: TMA x_n / weights / dLeft / dRight
             -> projection/gate recompute -> dp/dg global workspace
             -> dX and split-K dW -> LN derivative + residual + partial reductions
```

출처: Anthropic 추론 커널·TMA/WGMMA 구현을 계승한 학습 확장이다. 원본 파일은 수정하지 않았고 파생 파일과 SHA를 보관했다. 입력 LN 출력 x_n은 저장하며 출력 LN activation 저장 정책은 사용하지 않는다. D512 forward 입력 LN은 기존처럼 별도 경로다.

## 선택한 큰 폭 B7 설정

| D | producer token tile | weight slot 수 | slot당 K64 chunk | CTA/SM | dW split 수 |
|---:|---:|---:|---:|---:|---:|
|64|64|8|1|2|128|
|256|64|1|4|2|32|
|384|64|1|6|2|32|
|512|64|1|4|2|32|

D128은 기존 특화 ring 기반 구현을 사용한다. 폭별 커널 config와 선택 근거는 `gp-tune-*`, `fused-tune-*`, `tune-*` JSON 및 job 로그에 보관했다. 모든 가능한 config를 탐색했다는 뜻은 아니다.

## NCU 진단 · D512 L384 B7

register-operand → shared-operand 변경 후 NCU kernel 시간은7.302 →6.131ms, tensor pipe 활성률은26.8% →31.7%였다. local load/store 요청은202,205,424 /58,952,892 sectors →0/0으로 감소했다. 총 DRAM 읽기+쓰기는 약8.04GB →8.00GB로 거의 같다. 즉 개선은 큰 HBM 트래픽 감소가 아니라 spill 처리 비용을 없앤 효과다. 이 활성률을 SoL32%라고 부르지 않는다.

## 검증과 측정 범위

- public PyTorch autograd 진입점 10shape, 모든11개gradient, 두 번 forward 후 backward의 저장값 독립성, in-place 변경 탐지 통과.
- 각 shape에서 PyTorch/Triton BF16 비교: output 상대L2<0.005, 각 gradient<0.01. 이는 교차 구현 기준이며 D128 같은 수식의 엄격 LN5e-6 기준을 완화한 것이 아니다.
- 입력/weight 변경 후 CUDA graph 검증. 새 네 폭은 graph/eager 일반 output/gradient bit-exact, atomic LN gradient 상대L2≤5e-6.
- 네 새 폭 L384와 수정된 D128 L768에서 compute-sanitizer memcheck/racecheck 모두0errors/0hazards.
- 측정은 CUDA graph replay, dropout25% 고정 mask/scale, residual 포함. RNG 생성·optimizer·CPU dispatch·compile·plan 생성은 제외했다.
- 개발 진입점만 변경했다. production 자동 dispatch나 원격 push는 하지 않았다. 선택 B7의 NCU D256/D512 L384 자료를 `ncu-*`에 기록했다. 파이프 활성률은 roofline/SoL90 달성률과 다르다. Cache와 clock을 강제 고정하지 않은 진단 측정이다.

다음 병목은 넓은 D의 B1과 B7 GEMM·전역 작업 공간 전달이다. 특히 D512 개선폭은 작다. 단일 launch 자체만으로 이 비용이 사라지지는 않는다.

[HTML](index.html) · [수치](summary.json) · [재현·SHA](manifest.json)
