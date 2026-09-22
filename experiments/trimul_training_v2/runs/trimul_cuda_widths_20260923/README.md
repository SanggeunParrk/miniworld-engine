> 이 문서는 직전 측정 기록입니다. [최신 폭별 최적화·검증](../trimul_cuda_widths_opt_20260923/README.md)을 참고하세요.

# 양방향 TriMul · D64–512 CUDA backward 연결

**다섯 폭 모두 현재 개발 학습 진입점에서 CUDA backward로 연결했다.** 이전 표처럼 D128만 CUDA로 측정하고 나머지는 기존 Triton을 현재 결과로 표시하지 않는다. 다만 **D64/256/384/512 CUDA 포트는 아직 Triton보다 느리다.** 연결·정확성 검증과 성능 최적화 완료는 별개다.

## 학습 전체: 같은 실행의 기존 Triton 대비

B1, H100, BF16, hidden은 방향별 D / 합친 폭2D. dropout25%, pair mask, residual, 모든11개gradient, 실시간 weight packing 포함. static compile + 수동 CUDA graph. 각 행의 기존/현재는 같은 GPU 프로세스 교대 측정. optimizer·dropout RNG·compile·초기 plan 생성·CPU dispatch 제외. 1×보다 작으면 CUDA가 느리다.

| D | L | 기존 Triton ms | 현재 CUDA ms | 기존 대비 배속 |
|---:|---:|---:|---:|---:|
| 64 | 384 | 0.809 | 1.635 | 0.50× |
| 64 | 768 | 3.067 | 6.257 | 0.49× |
| 128 | 384 | 1.654 | 0.990 | 1.67× |
| 128 | 768 | 6.479 | 4.794 | 1.35× |
| 256 | 384 | 3.440 | 6.695 | 0.51× |
| 256 | 768 | 17.082 | 27.606 | 0.62× |
| 384 | 384 | 6.147 | 11.730 | 0.52× |
| 384 | 768 | 26.686 | 48.864 | 0.55× |
| 512 | 384 | 9.342 | 16.810 | 0.56× |
| 512 | 768 | 40.495 | 71.551 | 0.57× |

## 학습 forward

| D | L | 기존 Triton ms | 현재 CUDA ms | 배속 |
|---:|---:|---:|---:|---:|
| 64 | 384 | 0.256 | 0.270 | 0.95× |
| 64 | 768 | 0.959 | 1.003 | 0.96× |
| 128 | 384 | 0.558 | 0.293 | 1.90× |
| 128 | 768 | 2.175 | 1.161 | 1.87× |
| 256 | 384 | 1.281 | 1.151 | 1.11× |
| 256 | 768 | 5.304 | 4.674 | 1.13× |
| 384 | 384 | 2.272 | 2.020 | 1.12× |
| 384 | 768 | 9.396 | 8.186 | 1.15× |
| 512 | 384 | 3.595 | 3.261 | 1.10× |
| 512 | 768 | 14.796 | 13.473 | 1.10× |

## 추론 forward · 앞선 측정 유지

추론 전용 설정이다. 이번에 다시 측정한 학습 forward와 다른 경로이며 학습의 dropout은25%, 추론은0이다.

| D | L | 기존 Triton ms | 선택 추론 경로 ms | 배속 |
|---:|---:|---:|---:|---:|
| 64 | 384 | 0.205 | 0.131 | 1.56× |
| 64 | 768 | 0.812 | 0.530 | 1.53× |
| 128 | 384 | 0.419 | 0.276 | 1.52× |
| 128 | 768 | 1.684 | 1.172 | 1.44× |
| 256 | 384 | 1.120 | 0.971 | 1.15× |
| 256 | 768 | 4.661 | 3.986 | 1.17× |
| 384 | 384 | 2.231 | 1.817 | 1.23× |
| 384 | 768 | 9.118 | 7.446 | 1.22× |
| 512 | 384 | 3.770 | 3.283 | 1.15× |
| 512 | 768 | 15.332 | 13.673 | 1.12× |

## 실제 연결

- `runs/trimul_training_current.py:Training(a)`가 D로 분기한다. D128은 기존에 검증한 cache-policy B1과 B7, 나머지 네 폭은 새 CUDA 포트다.
- 같은 파일의 `bidirectional_trimul_cuda(...)`는 PyTorch autograd 진입점이다. 모든 D에서 11개 gradient를 반환한다. 폭 미지원 시 예외를 내며 Triton backward로 조용히 대체하지 않는다.
- D128 L384는 수정된 단일 B7, L768은 검증된 분리형 CUDA B7을 유지한다. L768 단일 후보의 dWL 오차0.0519%가 기준0.0500%를 초과한 기록은 이전 실험에 남겨 두었다.
- D64/256/384/512의 backward는 `width_b1` 1회 → cuBLAS contraction backward4회 → `width_b7` 1회. profiler의 실제 커널 이름으로 확인했다.
- 새로운 폭별 커널은 Anthropic의 TMA/WGMMA primitives와 기존 재계산 수식을 차용했다. D128의 저수준 스트리밍 배선을 그대로 복제한 것은 아니다. 큰 폭에서는 shared memory 한도를 지키기 위해 일시적인 전역 작업 공간을 사용한다.
- 입력 projection/gate는 앞서 튜닝한 Anthropic 파생 K1을 사용한다. 입력 x_n은 저장하며 backward가 재사용한다. projection/gate는 backward에서 재계산한다. 두 방향 contraction은 cuBLAS다.

## 넓은 폭의 HBM 흐름

1. Forward: x → K1 → left/right + x_n → cuBLAS → tri → 출력 CUDA → y. D512 입력 LN은 별도 경로다. 출력 CUDA는 내부 phase 사이에서 임시 normalized tri 버퍼를 쓴다.
2. B1 입력: tri, x_n, dy, dropout scale, output weights/affine. LN_out·projection·gate 재계산 → dp/dg → dNorm/dW → LN 미분. 출력: dTri, dGate, dWproj, dWgate, dgamma_out, dbeta_out.
3. cuBLAS: dTri + left/right → dLeft/dRight.
4. B7 입력: x, x_n, dLeft/dRight, dGate, dy, input weights/affine, mask. input projection/gate 재계산 → dp/dg → dx_n와 dW → LN 미분+residual. 출력: dx, 네 input dW, dgamma_in, dbeta_in.

**현재 느린 이유:** 한 CUDA launch 안에 묶었지만 normalized tri, dp/dg, dx_n 등의 중간값을 전역 workspace에 써서 phase 간 전달한다. D128 스트리밍 구현처럼 메모리 왕복을 충분히 줄이지 못했다. 이 상태를 D128과 동등하게 튜닝됐다고 주장하지 않는다. 다음 개선은 전역 작업 공간을 작은 ring 또는 shared memory 타일로 줄이는 것이다.

## 타일·자원

GEMM은 m64n64k16 WGMMA를 사용하며 K는64단위로 이동한다. D64는 1 warp-group/N64, 나머지는2 warp-group/N128이 A 타일을 공유한다. TMA는2stage로 계산과 겹친다. dW split-K는 D64에서128, 나머지에서32. cooperative grid는 드라이버의 실제 occupancy 한도 내에서 선택한다. 이 값들은 현재 검증한 설정이며 넓은 autotune 공간 탐색을 완료한 상태는 아니다.

## 검증

- 실제 public autograd 경로: D64/128/256/384/512 × L384/768, **10개 shape 통과**.
- 출력 relativeL2≤0.005, 각 gradient≤0.01: BF16 교차 구현 기준. 이 수치는 기존 D128 동일 수식 비교의 엄격한5e-6 LN 검증 기준을 대체하지 않는다.
- Forward를 두 번 수행한 후 각 backward의 저장값 독립성, forward 후 in-place 변경 탐지 통과.
- CUDA graph에서 입력·가중치 변경이 반영된다. 새 폭별 경로 graph/eager는 일반 출력·gradient bit-exact, atomic LN parameter gradient relativeL2≤5e-6를 별도 확인했다.
- 새 네 폭 L384에서 compute-sanitizer memcheck와 racecheck 모두0errors/0hazards. L768도 수치·graph·autograd 검증을 수행했다.
- 실제 TMA/WGMMA cubin, source 및 측정 JSON SHA를 [manifest.json](manifest.json)에 기록한다. NCU SoL을 새로 측정했다는 주장은 하지 않는다.

## 재현·범위

`final.sbatch`: 새 네 폭 두 길이의 같은 실행 Triton/CUDA 비교. `autograd.sbatch`: 현재 개발 진입점10shape. `sanitize.sbatch`: memcheck/racecheck. 공통환경은 `runs/anthropic_adoption_20260919/env.sh`.

이 진입점은 **개발용**이며 production 엔진의 자동 dispatch 변경이나 push는 하지 않았다. 한 프로세스에 H100 한 장이 보이는 환경, B1, L384/768, BF16을 지원한다. [HTML](index.html) · [수치 JSON](summary.json).
