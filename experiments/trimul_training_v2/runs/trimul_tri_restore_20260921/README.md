# 현재 TriMul 학습 개발 정책: tri + 저장 통계

2026-09-21 · 사용자 지시에 따라 출력 LayerNorm activation 저장 정책을 폐기하고 BF16 tri 입력으로 복구했다.

## 저장 정책과 진입점

- 현재 진입점: `runs/trimul_training_current.py:Training`.
- 구현: `tri_policy.py:Training` → 기존 검증된 `trimul_ln_policy_v4_20260921/final.py:Replacement(...,-1)`.
- 입력 affine BF16 `x_n` 저장은 유지한다.
- 출력 쪽 큰 activation은 contraction이 이미 만든 BF16 `tri`만 보존한다. 복사하지 않는다.
- K3는 행별 FP32 평균 `mu_out`와 역표준편차 `rstd_out`만 추가 저장한다.
- 출력 pre-affine `xhat` 및 affine `z`는 저장하지 않는다. 출력 projection/gate도 저장하지 않는다.
- B1은 tri와 통계를 TMA로 읽고 shared memory에서 `h=(float(tri)-mu)*rstd`, `z=BF16(h*gamma+beta)`를 재구성한다. 평균·분산 reduction은 다시 하지 않는다.
- B4는 재구성 h와 저장 rstd로 LN 미분을 계산한다. B7 및 cuBLAS 연결은 유지한다.
- 정책 선택은 `runs/trimul_training_selection.json`에 명시했다. Production dispatch는 별개이며 기존 B7 정확도 문제로 승격하지 않았다.

## 재측정

node02 H100 GPU 2개에서 길이별 독립 실행. 양방향 C128/H256 BF16, mask, dropout25%, residual 포함. 구간별 600회 교대 CUDA graph 중앙값. Live packing, Wp 전치, cuBLAS, 전체11개 gradient 포함. Optimizer/RNG 생성/CPU dispatch/compile 제외. 구간 중앙값 합과 전체 실측은 다를 수 있다.

| L | 경로 | FWD ms | B1–B4 ms | BWD ms | 전체 ms |
|---|---|---|---|---|---|
| 384 | tri + LN 통계 재계산 | 0.2972 | 0.2593 | 0.9700 | 1.2589 |
| 384 | 폐기: FP32 정규화 저장 | 0.3453 | 0.2978 | 1.0037 | 1.3464 |
| 384 | 현재: tri + 저장 통계 | 0.2975 | 0.2424 | 0.9502 | 1.2403 |
| 768 | tri + LN 통계 재계산 | 1.1638 | 0.8774 | 3.7566 | 4.9436 |
| 768 | 폐기: FP32 정규화 저장 | 1.3866 | 0.9781 | 3.8548 | 5.2769 |
| 768 | 현재: tri + 저장 통계 | 1.1539 | 0.8143 | 3.6910 | 4.8708 |

- L384: 폐기한 FP32 저장 경로 대비 B1 지연 18.60%, BWD 5.33%, 전체 7.88% 감소. 기존 통계 재계산 기준 대비 전체 1.48% 감소.
- L768: 폐기한 FP32 저장 경로 대비 B1 지연 16.75%, BWD 4.25%, 전체 7.70% 감소. 기존 통계 재계산 기준 대비 전체 1.47% 감소.

## 검증

- L384·768: 일반 입력, 가중치/입력/mask/dropout/dy 변경, gamma_out=0을 포함한 출력·11개 gradient가 기존 tri 기준과 bit-exact.
- CUDA graph replay와 eager bit-exact.
- 보존 tensor가 원본 BF16 tri와 같은 주소이며, 출력 LN activation allocation이 없고 통계만 FP32 `[2M]`임을 검사했다.
- 현재 B1 cubin은 기존 stats 경로의 cubin과 동일하다. 해당 소스 SHA-256도 검증했다. 기존 memcheck/racecheck 성공 기록을 재사용하며, 이번 새 배선의 수치 검사는 별도로 다시 실행했다.
- `current-entry-check.json`은 새 공통 진입점을 직접 실행한 별도 검사다.
- 기존 B7 L768 dWL 독립 기준 상대L2 0.055569% 문제(한도0.05%)는 남아 있다. 위 bit-exact는 기존 구현과의 일치이며 독립 기준 문제 해결을 뜻하지 않는다.

## 보존 메모리

출력 쪽 tri+통계: L384 76.68 MB, L768 306.71 MB. 폐기한 FP32 xhat+rstd의151.58/606.34 MB보다 작다. 입력 x_n 및 left/right는 동일하게 보존한다. MB는 십진 버퍼 크기이며 실측 DRAM 트래픽이 아니다.
