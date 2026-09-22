# 최신 B1+B7 양방향 TriMul 전체 재측정 · 2026-09-22

> **후속 해결:** 이 문서는 수정 전 측정 이력이다. 이후 gate 미분 곱셈 순서를 복원하여 L384 엄격한 LN gradient 검증을 통과했다. [수정·검증 결과](../trimul_ln_gradient_20260922/README.md). 원래 실패 기록과 당시 시간은 보존한다.

**L384 전체 fwd+bwd: 최신 조합 1009.808 µs. 같은 실행의 이전 개선 B7 조합 1065.824 µs 대비 5.26% 단축, 1.0555배.**

**최신 조합의 수치는 정확도 검증이 남은 후보의 진단값이다.** 입력 LN gamma/beta gradient가 변형 입력에서 기존 상대L2 한도5e-6를 초과했다. 허용치를 완화하거나 기본 경로에 적용하지 않았다.

## 조건과 시간

job15526 · node01 H10080GB · L384/C128/양방향 H256 · BF16 · dropout25% · pair mask·residual.
같은 프로세스/GPU의 3자 교차 CUDA graph 비교. 전체6×300회, 구간별5×300회.
full은 매 replay마다 fresh forward activation을 생성하는 전체 그래프를 직접 측정했다. 별도 구간 시간을 더한 값이 아니다.
전체6개 라운드에서 최신 후보가 이전 개선 B7 조합보다 빨랐다.

| 구간 (µs) | 이전 개선 B7 조합 | 구형 B7 검사 코드 | 최신 B1+단일 B7 후보 |
|---|---:|---:|---:|
| 전체 fwd+bwd 직접 측정 | 1065.824 | 1222.288 | 1009.808 |
| forward | 294.400 | 294.592 | 294.384 |
| backward | 767.296 | 923.712 | 712.624 |
| B1–B4 단독 | 188.448 | 180.064 | 180.448 |
| B7–B12 단독 | 406.016 | 570.592 | 349.792 |

포함: 양방향, B1–B12, cuBLAS contraction, live weight packing, 11개 gradient.
제외: optimizer, dropout RNG 생성, compilation, CPU dispatch. 전체 MiniWorld 학습 step이 아닌 TriMul 모듈 시간이다.

## 왜 이전1,048µs / 직전1,185µs와 다른가

- 과거1,047.808µs 및1,075.888µs 조합의 B1/B7 cubin을 SHA-256으로 대조해 동일함을 확인했다. 이번 실행에서는1,065.824µs다.
- 직전1,185µs 검사는 오래된 split_xn_pc1 B7을 사용했다. 이 검사 조합은 이번 실행에서1,222.288µs다.
- 최신 후보는 선택된 cache-policy B1과 K128/ring12/producer32/cluster2 단일 B7이다. B7 cubin SHA는 selected.json과 일치한다.
- 이번 full 구간의 3자 혼합 workload 중앙 SM clock은1875MHz, 메모리는2619MHz, 전력691.20W다. 과거1,048µs 측정은1920MHz,1,076µs 측정은1845MHz였다. 개별 arm의 클록으로 해석하지 않는다.
- 과거와 이번 절대 시간을 직접 나눠 개선율을 계산하지 않았다. 5.26%는 이번 동일 실행 비교다.

## 정확도

일반 입력 및2개 weights/input/mask/dropout 변형 검사. 모든 경로 graph/eager bit-exact이며 모든 출력은 finite다.
forward, 출력측 gradient는 정확히 일치했다. 최신 후보의 dX/가중치 gradient는 기존 한도내지만 입력 LN 파라미터 gradient는 아래 한도를 넘었다.

| case | gradient | 상대L2 | 기존 한도 |
|---|---|---:|---:|
| 1 | dgamma_in | 8.37042444e-06 | 5e-06 |
| 1 | dbeta_in | 6.89123954e-06 | 5e-06 |
| 2 | dgamma_in | 9.53485596e-06 | 5e-06 |

이전 개선 B7 조합 및 구형 B7 검사 코드는 모든3case에서 기존 검증을 통과했다. 최신 후보는 case1/2가 실패했으며 성능 진단만 진행했다.
이는 새 전체 연결 검사에서 드러난 한계다. 개별 B7의 과거 sanitizer/입력 검사 통과를 전체 정확도 통과로 대체하지 않았다.
job15522는 첫 실패에서 중단했다. 실패 기록을 보존하고 job15526에서 한도를 유지한 채 모든case와 진단 시간을 수집했다.

## 실제 실행과 적용 상태

Profiler에서 이전 개선 조합은 front_b7b12_dw + front_b7b12_dx, 최신 후보는 b7_joint 한 회를 확인했다.
둘 다 infer_k1/save_k3, b1_fused 한 회 및 cuBLAS contraction을 포함한다. 결과JSON에 전체kernel 목록·cubin hash·원본samples·telemetry를 보존했다.
측정용 policy.py에 최신 조합을 구성했으며 생산 dispatch와 runs/trimul_training_current.py는 변경하지 않았다.
최신 조합의 정확도 문제를 해결하기 전까지 검증된 기본 경로로 승격하지 않는다. 새 SoL 주장은 없다.

[원본결과](results.json) · [요약](summary.json) · [과거 cubin 대조](historical-cubin-verification.json) · [벤치](bench.py)
