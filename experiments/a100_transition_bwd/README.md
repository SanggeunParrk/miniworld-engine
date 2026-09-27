# A100 Transition backward (CUDA, sm_80) — 개발 기록

pair Transition(D = 128, n = 4, bf16)의 backward를 A100용 CUDA 커널로 만든다. forward는 [../a100_transition_fwd](../a100_transition_fwd/README.md)
(추론·학습 공용, bwd를 위해 아무것도 저장하지 않는다)다.

- **fixture / 측정**: fwd와 같다(`TA.fixture`, CUDA graph replay, 7 × 50의 중앙값). dy는 `randn`이다.
  정확도는 같은 weight의 fp32 autograd와 비교하며, 여섯 gradient 모두 rel-RMS로 본다.
- **엔진 기준**(`baseline.py`, `records/engine-baseline.json`, gpu05): 현재 A100 dispatch(Triton residual 경로, bwd 5 launch)다.
  bwd ≈ fwd+bwd − fwd.

## SoL

bwd의 모든 곱을 한 번씩 하면 `16·M·D·H` FLOP이다(a, b, dh 재계산 포함).
240 TFLOP/s(이 카드의 cuBLAS 천장) 기준 **L384 644 µs, L768 2577 µs**다.

## 현재 (gpu05)

| L | 이 구현 | % SoL | 엔진 bwd | 배율 | 학습 step (fwd+bwd) 이 구현 / 엔진 |
|---:|---:|---:|---:|---:|---:|
| 384 | **1247.8 µs** | 51.6% | 1565 µs | **1.25×** | 1536 / 2119 µs (**1.38×**) |
| 768 | **4858.8 µs** | 53.0% | 6138 µs | **1.26×** | 5946 / 8301 µs (**1.40×**) |

정확도(rel-RMS vs fp32, L384)는 여섯 개 모두 엔진보다 좋다. 출력은 bit 단위로 재현된다.

| | dx | dγ | dβ | dWa | dWb | dWs |
|---|---:|---:|---:|---:|---:|---:|
| 이 구현 | 2.33e-3 | 2.29e-3 | 2.55e-3 | 3.08e-3 | 2.94e-3 | 2.94e-3 |
| 엔진 | 3.09e-3 | 3.04e-3 | 3.44e-3 | 4.23e-3 | 4.11e-3 | 3.75e-3 |

## 설계

세 역할이 `16·M·D·H`를 나눠 하고, 모든 곱은 한 번씩만 한다.

| 커널 | 한 일 | FLOP | L384 |
|---|---|---:|---:|
| **P** `tr_bwd_p_kernel` | x, dy → LN 재계산(fwd와 bit 동일) → a, b, dh → dA, dB, h(bf16)를 **fragment-native 블록**으로 기록, 행 통계 | 6·MDH | 477 µs |
| **X** `tr_bwd_x_kernel` | d_xn = [dA\|dB]·[Wa; Wb](f32 누산). A fragment는 블록에서 LDG.128로 바로 읽고, B는 ring + ldmatrix.trans → LN bwd + residual → dx, dγ/dβ 부분합 | 4·MDH | 341 µs |
| **W** `tr_bwd_w_kernel` | hidden 128-slice × 행 replica. dWa, dWb, dWsᵀ(f32, 누산기 192/thread). A = 블록에 movmatrix.trans, B = xn / dy 타일에 ldmatrix.trans. x 타일은 smem에서 정규화 | 6·MDH | 373 µs |

- **fragment-native 블록**: (16행 × 16 hidden) 하나는 512 B = 32 lane × lane의 m16k16 A fragment word다.
  X는 이것을 A fragment로 그대로 쓰고, W는 `movmatrix.trans`로 전치된 A fragment를 얻는다. 둘 다 shared memory를 거치지 않는다.
  dA | dB | h 블록은 `[R][K][3][32]`로 붙여 두어 포인터 하나로 접근한다.
- P와 X는 fwd의 골격을 쓴다. 8 warp × 32행, 32-hidden 청크 weight ring(슬롯별 full/empty mbarrier), half tile 우선, fwd의 열 순열.
- dγ/dβ: warp별 smem 슬롯에 고정 순서로 누적한다(atomic 없음 → bit 재현).

## 단계별 기록 (L384, µs)

| 단계 | P | X | W | 합 | 내용 |
|---|---:|---:|---:|---:|---|
| v1 | 622 | 345 | 542 | 1550 | 첫 버전(정확도 첫 실행부터 통과) |
| P: A fragment quad 충돌 제거 | 484 | | | | xn STG.128 / dy LDG.128 묶음이 fragment quad와 충돌해 HMMA마다 MOV 4개(명령의 38%). xn 저장을 없애고(W가 x를 정규화) dy는 32-bit 로드 |
| W v2 | | | 683 → 380 | | 8 warp × 192 누산기, stage에 A 블록 + 통계. granule별 포인터 SEL(22%)을 컴파일 타임 슬롯으로 |
| 블록 인터리브, ring 카운터 | 483 | 341 | 375 | 1254 | spill된 ring 상태를 `LDL`로 읽는 것이 STG 뒤에서 대기 |
| dγ/dβ 결정적 | 477 | 341 | 373 | **1248** | |

기각:

| 시도 | 결과 | 이유 |
|---|---|---|
| PX(P와 X를 한 커널에, warp 0–3 P / 4–7 X, smem 인계) | 856 대 824 µs | SMSP당 P warp 하나 → 에필로그를 가릴 warp가 없다, spill 176 B |
| X의 A fragment 2 step 선행 | 341 → 415 µs | 레지스터 +16 → spill |
| 한 persistent 커널 + L2 ring | (미구현) | 타일 단위로는 동시에 떠 있는 중간값이 약 33 MB로 L2 초과. 청크 단위 결합이 필요 |

## H100 방식: launch 1회, 재계산 (`csrc/tr_bwd_fused_sm80.cuh`)

H100 커널(`../transition_fused/src/transition_bwd.cu`)의 배치를 그대로 옮겼다. 두 역할은 통신하지 않고 둘 다 a, b, dh를 계산해서
**22·M·D·H**가 된다.

- **prologue:** LN → xn, 통계. 그 뒤 grid barrier.
- **DX CTA:** a, b, dh를 계산하고 SwiGLU bwd → d_xn(f16 누산) → LN bwd → dx, dγ/dβ.
- **DW CTA:** hidden 64-slice × 행 replica. a, b, dh를 재계산해 h, dA, dB → dWa, dWb, dWsᵀ.

역할별 단독 커널로 최적화한 과정 (L384):

| 역할 | 단계 | 시간 | 처리량 |
|---|---|---:|---:|
| DW (12·MDH) | 첫 버전 (slice CTA가 x를 smem에서 정규화) | 1442 µs | 80 TFLOP/s |
| | xn 입력(정규화 pass 제거) | 917 µs | 126 |
| | phase 2를 8 warp로 재분할(B 공유 순서), A를 fragment 순서로 ldmatrix | 726 µs | 160 |
| | padding 레이아웃 / phase-1 fragment 선행 로드 | 761 / 728 µs | (기각) |
| DX (10·MDH) | 첫 버전 (f16 d_xn) | 648 µs | 149 |
| | step을 8-unit 절반 둘로 (누산기 48 → 24, MOV·spill 감소) | 619 µs | 156 |
| | 에필로그 대수 정리 (2 dA, 0.5 Wa를 f16 WX에) | 618 µs | 156 |

한 커널로 합쳤을 때 (DX CTA 수 스윕, gpu05):

| DX CTA | 44 | 52 | 60 |
|---|---:|---:|---:|
| L384 | 1541 µs | **1522 µs** | 1688 µs |
| L768 | **5891 µs** | 6030 µs | 6709 µs |

정확도는 여섯 gradient 모두 3-커널 경로와 같고(d_xn f16 누산이어도 dx 2.34e-3), 결과는 bit 재현된다.
**3-커널(1238 / 4855 µs)보다 23% / 21% 느리다.**

- **연산량:** 재계산 때문에 FLOP이 1.375배다.
- **DW:** A100에서는 dW 누산기 때문에 xn/dy fragment를 레지스터에 둘 수 없다. 그래서 phase 1에서 A를 smem에서 계속 다시 읽는다.
  warp 하나가 hidden 16개만 계산하므로 재사용이 적다.
- **DX:** xn + dy + d_xn이 레지스터를 채워서 spill이 생긴다.
- **분배:** 역할을 SM 단위로 나누니, 타일 단위 tail과 역할 간 불균형이 더해진다.

## 남은 시간은 어디에 있나

- **DRAM**: 중간값(dA, dB, h)이 행당 3 KB로 쓰이고 3 KB(W) + 2 KB(X)로 읽힌다. 합계 약 10 KB/행 = L384에서 1.47 GB, 1.6 TB/s로 약 920 µs다.
  세 커널 모두 레지스터 한도(255)에 있어서 latency를 가리기 어렵다.
- 다음 레버: W(와 X)가 P 직후 같은 타일을 읽게 해서 L2에서 받기. 청크 단위 결합 persistent 커널, 또는 SM을 나눈 동시 커널 + 전역 flag다.

## 재현

```bash
cd experiments/a100_transition_bwd
sbatch run.sbatch baseline.py --length 384 768 --out records/engine-baseline.json   # 엔진 기준
sbatch run.sbatch bench_bwd.py --length 384 768                                      # 정확도 + 시간
sbatch run.sbatch time_parts.py 384 768                                              # 커널별 시간
sbatch prof.sbatch tr_bwd_p 384 <tag>                                                # ncu stall + SASS
```
