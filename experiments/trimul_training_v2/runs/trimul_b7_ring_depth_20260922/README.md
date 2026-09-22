# B7–B12: dW 합산 배치 + ring depth 12

H100 / L384 / C128 / projection width256 / 단일 cooperative CUDA launch.
Anthropic 유래 Apache-2.0 TMA/WGMMA primitives를 사용한 학습 확장이다.

## 동일 실행 비교

| 구현 | 시간 (µs) |
| --- | ---: |
| 재개 출발점 통합형 | 415.376 |
| 현재 분리형 | 393.344 |
| 직전 선택 통합형 | 369.296 |
| dW 합산 배치 개선 | 360.672 |
| 새 선택: 합산 개선 + ring 12타일 | 348.112 |

직전 통합형보다 5.74%, 현재 분리형보다 11.50%, 재개 출발점보다 16.19% 시간 단축.
5개 변형 입력을 검증한 뒤 같은 입력·프로세스에서 순서를 교대로 바꾸며 5×160회 측정했다.
B7–B12의 초기화와 최종 reduction을 포함한다. 전체 학습 / L768 결과가 아니다.
252 µs 목표는 미달이며 SoL90은 입증하지 않았다.

## 변경

1. source의 dW partial을 [2,128,32] 순서로 shared에서 재배치한 뒤 연속 global store한다.
2. 최종 reducer도 partial의 연속 주소를 담당하도록 바꿨다. 기존 값과 합산 순서는 유지한다.
3. 그룹당 global ring을 8 → 12타일로 늘렸다. dX CTA 10개에 전달할 데이터를 source가 더 앞서 생성할 수 있다.
4. dX의 shared ring은 기존과 동일하게 16 KiB × 4이다. global ring 크기 변경과 구분해야 한다.

source 160 CTA / dX 100 CTA / producer 64 registers / compute 192 registers.
global ring은 10 → 15 MiB. 총 논리적 dp/dg 크기는 그대로이며 실제 HBM 트래픽은 별도다.

## 시도와 선택

- source dW/다음 projection rolling: 421.62 µs vs 같은 실행 기준 366.78 µs, 제외.
- dX WGMMA rolling: 404.08 µs vs 369.86 µs, 제외. 결합 후보도 더 느렸다.
- 첫 dp/dg 타일을 먼저 전달: 381.79–387.87 µs vs 368.05 µs, 제외.
- dW partial/reducer 배치 세 방식: 모두 정확도 통과, 결합 방식을 선택.
- ring depth 4/8/10/12/14/16/20 비교: 12 선택. 4는 심한 대기, 과도한 크기는 이득 감소.
- 새 depth에서 CTA/레지스터 배분 재조정: G10/U10/P64 유지.

서로 다른 실행의 절대 시간을 합쳐 속도 향상을 계산하지 않는다. 최종 비교는 confirmation.json이다.

## 검증

5개 입력/가중치/마스크 변형에서 7개 출력 모두 기존 허용오차 통과.
선택 후보는 직전 통합형과 5건 모두 bit-exact. eager/graph, scratch poison, counter/flag 재사용 통과.
선택한 동일 cubin의 memcheck/racecheck 오류 0건. NCU local spill traffic 0.

NCU 별도 실행에서 새 후보 HBM read 303.24 MB / write 144.61 MB,
이전 후보 read 303.76 MB / write 56.80 MB다. ring 확대는 실제 쓰기를 늘리는 비용이 있다.
pipeline 대기 감소의 이득이 이 비용을 넘었다. HBM 왕복을 줄여 얻은 성능이라고 해석하면 안 된다.
Tensor active 37.42%는 SoL 수치가 아니다. 프로파일 latency 340.19 µs는 graph 교차 측정과 별도다.

## 재현

선택 설정: selected.json. 소스/cubin SHA-256: confirmation.json 및 selected.json.
최종 측정: `sbatch confirm.slurm --depths 12 --cases 5` (저장소 루트의 전체 경로로 실행).
검증 환경: B7_CONSUMERS=10, B7_PRODUCER_REGS=64, B7_RING_DEPTH=12, MODE=52.
production dispatch 및 전체 모델 배선은 변경하지 않았다.
