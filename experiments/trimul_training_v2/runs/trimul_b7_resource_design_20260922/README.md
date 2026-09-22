# B7–B12: 재사용과 레지스터·동기화·버퍼를 함께 설계하기

후속 구현·실측 완료: [K128 파이프라인 결과](../trimul_b7_weight_batch128_20260922/index.html).
두 compute WG 공유는 구현했지만 spill/직렬화 비용으로 미채택했다. 기존 CTA 배치에 K128 전송 단계를 적용한 구성을 선택했다.
아래 내용은 실험 전 설계 기록이다.

2026-09-22. L384, 양방향, B7–B12 단일 CUDA 호출. **설계 검토이며 새 성능 실측이 아니다.**
현재 선택은 `../trimul_b7_l2_reuse_20260922/selected.json`의 2-CTA 배치다.
기존 선택의 361.616 µs는 해당 기록의 교차 측정값이며 이번 설계의 성능이 아니다.

## 판단

우선순위는 **이미 BF16으로 반올림된 값의 FP32 레지스터 수명을 줄인 뒤, 독립적인 두 계산 warpgroup이 가중치 버퍼를 공유**하도록 하는 것이다.
warpgroup은 함께 WGMMA를 실행하는 128개 스레드 묶음이다.
한 warpgroup에 두 행의 누산기를 맡기는 이전 실패안과 구분한다.
이 구조도 아직 이득을 보장하지 않는다. 특히 같은 launch의 source CTA 자원까지 맞아야 한다.

## 현재 소스에서 확인한 비용

`../trimul_b7_ring_cluster_schedule_20260922/joint.cu` 기준:

- source 16 CTA / 논리 그룹: 각 CTA는 hidden 32채널의 dp/dg를 만들고 dW를 FP32로 장기 누적한다.
- dX 10 CTA / 그룹: 각 CTA는 한 64행 타일의 dX를 계산한다. source와 전역 ring으로 연결된다.
- 그룹 10개, 총 260 CTA, 256 threads/CTA, dynamic shared 112 KiB.
- 컴파일 기록: static shared 1024 B, static register allocation 128/thread, stack/spill 0.
- 실행 중 register budget은 producer 64 / compute 192로 재분배된다. static allocation과 혼동하면 안 된다.
- `consumer_compute`: dX FP32 `acc[64]`를 가진 상태에서 output-gate `gate[64]`를 계산한다.
- gate 합산 직후 `acc[j] = round_bf16(...)`이지만 이후에도 64개의 FP32 레지스터 표현으로 LN 미분까지 유지한다.
- `source_compute`: dW[64], x_n fragment[32], 두 GP 결과 a0/a1[32씩] 등 수명이 겹친다. 이는 소스 수준의 배열 크기이며 실제 동시 live register 수는 컴파일 결과로 확인해야 한다.

### 논리적 전송량: HBM 실측값이 아님

L384는 M=147456행, 64행 타일 2304개다. BF16 기준:

| 전송 | 현재 논리 payload | 두 dX warpgroup 공유 후보 |
|---|---:|---:|
| source의 x_n 읽기 | 576 MiB | 유지 |
| dp/dg ring 쓰기+읽기 | 288+288 MiB | 유지 |
| dX 주 가중치 읽기 | 576 MiB | 약 288 MiB + 끝 타일 오버헤드 |
| output-gate 가중치 읽기 | 72 MiB | 공유하면 약 36 MiB + 끝 타일 오버헤드 |
| dW partial 쓰기+최종 읽기 | 5+5 MiB | 유지 |

ring 용량은 15 MiB이지만 전체 실행에서 전달되는 양은 위와 다르다.
전송량은 캐시 적중·요청 분할·재시도 이전의 소스 수준 계산이다. 절감량을 HBM 절감이나 latency 절감률로 치환하지 않는다.

## 1. 먼저 값의 수명부터 줄인다

### dX: 이미 반올림한 누산값을 BF16 두 개씩 포장

현재 수식의 반올림 지점을 그대로 둔다:

```python
dxn = round_bf16(dx_main_fp32 + round_bf16(dx_gate_fp32))
# 이 지점 이후만: FP32 64개 대신 BF16x2 32개로 보관
# LN 미분에서 필요한 순간 정확히 FP32로 펼쳐 사용
dx = layernorm_backward(dxn, raw_x, gamma) + residual
```

새 BF16 반올림을 GEMM 누산 중간에 넣으면 안 된다. LN reduction의 합산 순서도 유지한다.
gate 결과는 N64 또는 N32 조각으로 처리하고 사용한 dX 구간을 즉시 포장하는 후보를 비교한다.
N32는 gate WGMMA 명령 수가 증가하므로 레지스터 감소만 보고 채택하지 않는다.
LN에는 포장된 dX 32개와 raw-x fragment 32개를 유지하고, FP32 정규화 값 전체를 장기 보관하지 않는다.

### source: GP 결과의 기존 BF16 변환을 앞당긴다

`pair_glu`는 이미 gate/projection preactivation을 BF16으로 변환한 뒤 sigmoid와 곱셈을 한다.
첫 GP가 완료된 뒤 그 결과를 BF16x2 16개로 포장하고 두 번째 GP를 실행하는 후보를 만든다.
기존 반올림 위치의 수학적 의미는 유지하면서 첫 결과의 FP32 배열 수명을 줄일 수 있다.
변환을 중복하지 않고, WGMMA 완료와 operand 수명을 침범하지 않아야 한다.
이는 register-pressure 가설이며 실제 ptxas/SASS live range와 시간으로 검증한다.

## 2. 두 계산 warpgroup이 한 가중치 타일을 공유

```text
source CTA: 기존 dp/dg + 장기 dW 누적 → 기존 global ring
                                                   │
dX CTA, 384 threads                                │
  producer WG:  W tile 한 번 TMA load ─────────┐     │
                행 A/B derivative load       │     │
  compute WG A: 행 A dp/dg × 공유 W → acc A 64 FP32 │
  compute WG B: 행 B dp/dg × 공유 W → acc B 64 FP32 │
                각자 gate 합산 → BF16 포장 → LN 미분 → dX
```

각 compute WG는 한 행 타일만 맡는다. 두 번째 dX 누산기를 shared에 저장했다 다시 읽는 경로는 만들지 않는다.
기존 논리 dX consumer 10개를 물리 CTA 5개 × 2 WG로 매핑하면 그룹당 source 16 + dX 5 = 21 CTA다.
총 210 CTA, compute WG 총수는 기존 260개와 같다. 더 많은 계산 그룹으로 밀어붙이는 안이 아니다.

### Shared memory 수명 배치

| 단계 | 살아 있는 버퍼의 계획 | 합계 |
|---|---|---:|
| 주 dX GEMM | 행 A ring 16 KiB×2 + 행 B ring 16 KiB×2 + 공용 W 16 KiB×2 | 96 KiB |
| gate/LN | 행별 raw-x 16 KiB + gate-gradient 16 KiB, 공용 Wgate 32 KiB | 96 KiB |
| 출력 | 읽기가 완료된 행별 raw-x 공간을 dX 출력으로 재사용; LN partial scratch 별도 소량 | 112 KiB 안에서 설계 |

두 단계는 같은 공간을 재사용한다. gate-gradient는 GEMM 전체에 걸쳐 미리 유지하지 않는다.
raw-x를 덮어쓰기 전에 해당 행의 LN 입력 읽기가 모두 끝나야 한다.
residual은 벡터 global load 후보를 먼저 검토한다. coalescing과 명령 증가가 손해면 버퍼 배치를 다시 선택해야 한다.
producer가 epilogue 영역을 덮기 전에 **두 WG의 마지막 WGMMA 완료**를 확인한다.

### 동기화

- 공용 W 슬롯 하나에 full/empty 프로토콜 하나. full은 두 compute WG가 기다리고 empty는 두 WG 모두 WGMMA 완료 후 알린다.
- derivative 슬롯과 완료는 행별로 독립 관리한다. 한 행의 ring을 다른 행 때문에 불필요하게 붙잡지 않는다.
- 단계마다 전체 CTA/cluster barrier를 추가하지 않는다. 공유 W의 재사용 지점에서 두 소비자 완료 확인은 필수다.
- 공유 때문에 느린 WG를 기다리는 비용은 남는다. 두 행의 동일 타일 크기와 꼬리 타일 처리가 중요하다.
- grid-wide dW/LN 최종 합산 barrier는 그대로 필요하다.

## 3. 가장 중요한 실행 가능성 조건: 같은 launch의 source도 맞춰야 한다

384-thread 커널은 source CTA에도 적용된다. source에 비활성 WG가 하나 생긴다고 그 레지스터 비용이 자동으로 0이라고 가정하면 안 된다.
H100 SM은 64K 32-bit registers, 228 KiB shared를 가진다. 목표는 기존처럼 두 CTA/SM을 유지하는 것이다.
아래는 보수적인 **설계 예산**이지 컴파일러가 달성했다는 결과가 아니다:

| 역할 | WG별 register/thread 예산 | CTA 합계 |
|---|---|---:|
| dX | producer 32 + compute 104 + compute 104 | 30720 registers |
| source | producer 24 + compute 192 + inactive 24 | 30720 registers |

384-thread CTA에 80/thread allocation을 가정한 예산이다. 112-register compute WG 둘을 무작정 넣으면 예산을 초과한다.
할당 단위·정적 shared·컴파일러 제약을 포함한 실제 occupancy API 확인이 필수다.
source producer를 64→24로 줄이는 데 따른 성능 손해도 별도 대조군으로 확인해야 한다.
source의 자원 조건을 만족하지 못하면 두-WG 안 전체를 보류한다. dX 부분만 빨라진 결과로 채택하지 않는다.
현재 hardware-cluster 설정은 그룹당 26 CTA를 가정한다. 새 21 CTA 매핑에 기존 divisibility 조건/launch를 그대로 복사하면 안 된다.

## 왜 이전 실패안과 다른가 / 무엇이 여전히 위험한가

- 이전 dxpair: 한 WG에 두 누산기 → 한 누산기를 shared에 보관. 새 안: 행마다 별도 WG와 레지스터 소유권.
- 이전 batch2/3 atomic: dW 기여 전송량 감소 대신 224 KiB shared와 누산값 이동. 새 안: 기존 dW 장기 누적 유지.
- 이전 multicast: CTA 사이 준비·소비 동기화. 새 안의 W 공유는 같은 CTA 안에서 수행한다.
- 그래도 세 번째 WG의 register 비용, source producer 축소, 얕아진 행별 derivative pipeline, 추가 gate 명령, 두 소비자 간 불균형 비용이 남는다.
- 전역 ring 왕복은 제거되지 않는다. 이를 제거했다거나 모든 비용이 줄었다고 설명하면 안 된다.

## 검증 순서 / 채택 조건

1. 기존 256-thread 커널에서 BF16 포장만 각각 대조한다. 수식·7개 출력 허용 오차는 유지한다.
2. 384-thread source 대조군으로 producer 축소·비활성 WG 비용을 먼저 확인한다.
3. 두-WG dX: ptxas spill/stack, WGMMA serialization, 실제 occupancy를 확인한다. 2 CTA/SM 목표 실패 시 원인 수정 전 장시간 튜닝하지 않는다.
4. 선택 커널과 같은 프로세스·입력·교차 graph timing. L384만 측정한다.
5. 정확도/반복 replay/poison/flag 재사용 검사, memcheck/racecheck, 동일 cubin NCU.
6. NCU에서는 전체 L2 sectors, HBM bytes, local spills, 유효 tensor 처리와 critical-path 대기를 함께 본다. L2 busy% 하나로 SoL90을 판정하지 않는다.

252 µs와 SoL90 달성 여부는 새 실측 전에는 미정이다. 이번 작업은 소스·기존 실패 기록 대조와 설계/비용 계산까지다.

## 근거

- [선택 및 NCU 기록](../trimul_b7_l2_reuse_20260922/README.md)
- [이전 두 행/atomic/DSM 실패 분석](../trimul_b7_accum_analysis_20260922/README.md)
- [이전 atomic 전송량 분석](../trimul_b7_atomic_audit_20260922/README.md)
- [선택 CUDA 소스](../trimul_b7_ring_cluster_schedule_20260922/joint.cu)
- [NVIDIA Hopper 자원·occupancy 안내](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [NVIDIA warpgroup register 재분배 규칙](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html)

Anthropic 유래 TMA/WGMMA primitives를 계승하는 학습 확장이라는 기존 설명과 라이선스를 유지한다.
