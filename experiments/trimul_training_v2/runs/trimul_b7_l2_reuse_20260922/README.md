# B7–B12: L2 재사용 구조 실험 및 CTA cluster 배치

최종 선택은 multicast 없이 2-CTA hardware cluster로 배치하고 불필요한 초기 cluster sync를 제거한 후보다.
수식, 저장값, ring 크기, WGMMA 연산 순서, 단일 cooperative kernel 범위를 유지한다.

## 최종 같은 실행 비교 (5×160회)

| 구현 | µs |
| --- | ---: |
| 이전 선택 | 365.792 |
| 1-CTA cluster 속성 | 370.944 |
| 새 2-CTA 배치 | 361.616 |
| 분리형 | 413.184 |

이전 선택 대비 1.14%, 분리형 대비 12.48% 시간 단축.
5개 측정 라운드 모두 이전 선택보다 빨랐다. 크기가 작은 이득이며 과거 348.11 µs 기록과 절대 시간을 섞어 비교하지 않는다.
252 µs 목표는 미달. 전체 학습 또는 L768 결과가 아니다.

## 구조 실험

총 41개 설정/대조군을 비교했다. 각 3~5개 입력 변형에서 정확도·eager/graph·scratch poison·flag 재사용 검사를 통과했다.
다양한 구조 실험의 커널들은 보관하되 선택하지 않았다.

- TMA multicast: 2/4 CTA가 x_n을 공유. 준비 barrier와 cluster 동시 실행 한도 비용으로 전체 시간은 증가했다.
- dX 재사용: 한 CTA가 두 행 타일을 계산하며 주 경로 가중치 load를 공유. 두 번째 누산 결과는 shared에만 보관한다.
- source 재사용: 한 CTA가 두 채널 타일을 계산하며 x_n을 공유. 두 dW 누산기를 유지해 레지스터 압박이 증가했다.
- 위 구조들에서 CTA 비율, ring 크기, 레지스터 분배, GEMM pipeline, SS WGMMA를 비교했다.
- 기존 배선에 SS WGMMA만 적용한 후보도 더 느렸다.
- 4-CTA cluster 최초 설정은 occupancy API의 248 CTA 한도보다 커 실행 전에 제외했다. 240/224 CTA로 수정해 검증했다.

## NCU 해석

별도 프로파일에서 기존 L2 sectors 약 75.92M → dX 재사용 69.63M, source 재사용 75.81M.
dX의 특정 가중치 load는 절반이 되었지만 커널 전체 L2 트래픽 감소는 약 8.3%였다.
동시에 dX 재사용의 HBM write는 144.61 → 334.46 MB로 증가했다. source 재사용은 L2 총량이 거의 줄지 않았다.
따라서 반복 읽기를 줄이는 수식상의 계산만으로 전체 성능 이득을 주장하지 않는다.

새 선택의 NCU: SM 37.45%, memory 58.14%, L2 80.00%, HBM 40.15%.
L2 처리율 상승은 더 빠른 시간만의 효과가 아니다. L2 sectors도 약 80.86M로 늘었다.
SoL90은 미달이며 이 수치가 알고리즘의 최소 실행 시간 대비 달성률은 아니다.
프로파일 latency 339.296 µs는 위 graph 교차 측정과 별도이며 서로 비교하지 않는다.

## 검증·재현

새 선택은 5개 변형 입력의 7개 출력 검사 통과. 동일 cubin의 memcheck/racecheck 오류 0건.
선택 cubin의 ptxas spill 0 bytes. 소스·cubin SHA-256 및 환경: selected.json.
NCU·실험 원본 경로: results.json. production dispatch 변경 없음.
Anthropic 유래 Apache-2.0 TMA/WGMMA primitives에 기반한 학습 확장이다.
