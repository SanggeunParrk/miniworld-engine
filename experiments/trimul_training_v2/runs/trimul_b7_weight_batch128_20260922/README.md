# B7–B12: 가중치 K128 전송 파이프라인

L384 양방향 B7–B12 단일 CUDA 커널. 이번 실험의 선택은 기존 2-CTA 배치에서 가중치 전송 단위를 K64→K128로 바꾼 구성이다.
Anthropic 유래 TMA/WGMMA primitives에 기반한 학습 확장이라는 기존 기조와 Apache-2.0 설명을 유지한다.

## 최종 같은 실행 비교

| 구현 | µs |
| --- | ---: |
| 이전 선택: 2-CTA, K64 weight stages | 369.376 |
| 새 선택: K128, ring12, producer32 | 356.592 |
| K128, ring12, producer24 | 360.256 |
| K128, ring10, producer32 | 358.976 |
| 분리형 | 414.688 |

이전 선택 대비 3.46%, 분리형 대비 14.01% 시간 단축. 5×160 교차 graph 측정의 통합 중앙값이다.
5개 라운드 모두 이전 선택보다 빨랐다. 과거 다른 GPU/클록 조건의 절대 µs와 섞어 비교하지 않는다.
252 µs·SoL90은 미달. B1–B4, 전체 모듈, L768의 새 측정은 아니다.

## 선택 배선

| 항목 | 이전 | 새 선택 |
| --- | --- | --- |
| derivative shared slots | 16 KiB × 4 = 64 KiB | 16 KiB × 2 = 32 KiB |
| weight shared slots | 16 KiB × 2 = 32 KiB | 32 KiB × 2 = 64 KiB |
| 별도 dGate 공간 | 16 KiB | 16 KiB |
| dynamic shared 합계 | 112 KiB | 112 KiB |
| weight ready/empty 단계 / 행 타일 | 16 | 8 |
| weight TMA load 명령 / 행 타일 | 16 | 16 |
| WGMMA commit / 주 dX 행 타일 | 16 | 8 |
| 주 dX WGMMA wait / 행 타일 | 8 | 8 |
| producer/compute register budget | 64/192 | 32/224 |

WGMMA N128, 누산 순서, 반올림, 수식, global ring, dW 장기 누적, 260 CTA/256 threads, hardware cluster2를 유지한다.
가중치의 논리 payload는 유지하며 두 load를 한 ready 단계로 묶어 다음 K128 묶음을 미리 읽는다.
raw-x/residual/output-gate 전송은 마지막 main GEMM 소비 완료 후 alias 공간에 넣는다.
기존 LN 입력 선행 전송과의 차이까지 포함한 실제 측정이다. 변경 각각의 단독 효과를 분리한 결과는 아니다.
occupancy API는 cluster2 132개까지 반환했고 260 CTA로 실행했다. 선택 cubin은 stack/spill 0, WGMMA serialization 경고 없음.

## 다른 구조 실험

총 40개 측정 항목(대조군·재측정 포함)을 기록했다. 각 항목은 3~5개의 mutation/eager/graph/poison/flag 검사를 통과했다.

- BF16 포장, GP 결과 수명 단축, N64 gate, 레지스터 배분 변경만으로는 안정적인 이득을 얻지 못했다.
- 384-thread 제어군: N128 WGMMA는 80-register target에서 컴파일 실패. N64로 바꿔 실행한 제어군도 약 5% 느렸다.
- 두 compute WG의 weight 공유를 실제 구현했다. 첫 버전 1031 µs vs 같은 실행의 기존 340 µs. local spill 요청이 크게 증가했다.
- gate N32 및 LN fragment 수명을 줄인 버전은 581 µs vs 같은 실행의 기존 342 µs. 개선됐지만 선택 기준보다 느렸다.
- 추가 registers/1-CTA residency, K128 배치, WG 역할 template 특수화도 비교했고 더 빠르지 않았다.
- 두-WG 후보의 파이프라인·spill·명령 증가를 확인한 뒤 기존 배치에 K128만 적용해 위 이득을 얻었다.

## NCU

선택 cubin 별도 profile: SM 38.04%, memory 59.50%, L2 80.09%, HBM 39.18%.
local load/store sectors: 0/0.
별도 측정의 hardware throughput 지표이며, 알고리즘의 달성 가능한 최소 시간 대비 효율을 측정한 값은 아니다.
profile latency와 위 CUDA-event latency를 직접 섞어 비교하지 않는다. `results.json`에 이전/두-WG/lowreg/선택 profile과 단위를 보관했다.

## 검증 / 제한

선택 cubin의 5개 변형 입력·7개 출력 검사, 반복 graph, scratch poison, counter/flag 초기화 검사 통과.
원래 relative-L2 한도를 유지했다. 같은 cubin의 memcheck/racecheck 모두 0 errors.
추가로 producer24 후보도 sanitizer를 통과했으며 `*-p24.*`로 기록을 보관했다.
production dispatch와 upstream selector는 변경하지 않았다. commit/push/온라인 게시 없음.

재현: `selected.json`의 환경으로 `Plan(..., clusters=10, mode=52)` 실행.
`run.slurm --configs 12:24,12:32,10:32 --cases 5 --rounds 5 --iterations 160`이 최종 비교다.
`prepare.py`는 후보 생성 기록이다. 직접 재현에는 현재 `sweep.py`와 `plan.py`를 사용한다.


## 후속 전체 연결 검사 · job15526

최신 B1과 결합한 L384 전체 fwd+bwd는1009.808µs (동일 실행의 이전 개선 B7 조합1065.824µs).
단, 변형 입력에서 입력 LN gamma/beta gradient가 기존 상대L2 한도5e-6를 초과했다 (최대9.535e-6).
개별 커널 입력 검사·sanitizer 통과와 전체 연결 정확도 통과는 구분한다. 최신 전체 값은 진단용이며 기본 경로에 승격하지 않았다.
[전체 결과와 실패 수치](../trimul_full_latest_20260922/index.html).
