# TriMul 입력 LN gradient 엄격 검증 해결 · 2026-09-22

**L384 수정 커널은 기존 한도 5×10⁻⁶을 그대로 유지한 전체 검증을 통과했다.**

## 원인과 수정

단일 B7의 gate 미분에서 FP32 곱셈 결합 순서를 바꾼 것이 원인이었다.

```python
# 기존 계약: da는 mask 적용 후 gradient, p/g는 재계산 projection/sigmoid gate
# dg를 BF16으로 저장하기 전 곱셈 순서를 유지해야 한다.
dp_fp32 = da * g
# 수정 전
# dg = bf16((dp_fp32 * p) * (1 - g))
# 수정 후
dg = bf16(((da * p) * g) * (1 - g))
dp = bf16(dp_fp32)
```

두 식은 실수 산술에서 같지만 FP32 중간 반올림이 달라진다. 이것이 다음 BF16 반올림 경계에서 차이를 만들고 dX_n GEMM과 LN 파라미터 gradient에 전달됐다. 원래 곱셈 순서를 복원했다. 허용 오차, 저장 정책, 단일 B7의 융합 구성, 타일/CTA/ring 설정은 변경하지 않았다.

## 원인 분리 증거

진단 커널은 dX_n과 affine 전 정규화값 xhat을 내보냈다. 정규화값은 전 원소 일치했다. 수정 전 dX_n 불일치는 3개 case에서 각각 150/140/145개였고, 수정 후 모두 **0개**였다.
기존·새 구현 모두 자신의 dX_n/xhat을 FP64로 곱하고 합산한 참조에 대해 LN gradient 오차가 약2–3×10⁻⁷이었다. 따라서 문제는 LN reduction 자체가 아니라 앞단 gate 미분에서 유입된 값 차이였다.
FP64 검사는 이 중간값의 독립 합산 참조이며 전체 네트워크 FP64 autograd 비교라고 주장하지 않는다.

## 기존 실패 3case 재검증

| case | 수정 전 dgamma | 수정 후 dgamma | 수정 전 dbeta | 수정 후 dbeta |
|---|---:|---:|---:|---:|
| 0 | 4.079e-06 | 3.550e-07 | 4.712e-06 | 2.707e-07 |
| 1 | 8.370e-06 | 3.186e-07 | 6.891e-06 | 2.815e-07 |
| 2 | 9.535e-06 | 3.469e-07 | 4.227e-06 | 2.690e-07 |

입력 LN gradient 최대 상대L2: **3.550e-07**, 기존 한도5e-6. forward와 입력 dX는 기존 계약 참조와 bit-exact. 가중치 gradient도 기존5e-4 한도를 통과했다. graph/eager 전 출력 bit-exact.

## 추가 검증

- 12개 입력 조건 × TriMul 단독/새 CUDA Transition 연결 블록 = 24개 조합.
- random 입력·가중치·upstream gradient, 작은 입력, 0인 LN scale 채널, 비정규 affine 값, mask/dropout 변경, 전체 mask0, 전체 dropout scale0 포함.
- forward0, dX2e-5, LN gradient5e-6, 나머지 gradient5e-4: 기존 한도 유지. 모든 조건 통과, graph/eager bit-exact.
- MiniPairformer 연결 LN gradient 최대 상대L2 **3.458e-07**.
- 동일 수정 cubin sanitizer: **memcheck: 통과; racecheck: 통과**.
- 현재 개발 진입점: **검증본과 전체 출력 bit-exact, 동일 cubin 확인**.

## 성능

같은 실행 job15612에서 수정 전/후를 교차 측정했다.

| 구간 | 수정 전 | 수정 후 |
|---|---:|---:|
| B7 단독 | 350.752µs | 351.936µs |
| TriMul fwd+bwd | 1006.800µs | 1009.168µs |

전체 시간 변화 **+0.24%**. 성능 개선을 주장하지 않으며, 정확도를 고치면서 기존 성능을 거의 유지했다.
별도 job15616에서 MiniPairformer 블록 학습 전체는 수정 전 1.576ms, 수정 후 1.580ms였다. 서로 다른 job의 절대 시간을 섞어 비율을 계산하지 않는다.

## 적용 범위

`runs/trimul_training_current.py:Training`의 **L384 개발 경로**에 수정된 단일 B7을 연결했다. 다른 길이는 기존 개발 경로를 유지한다. **L768의 새 단일 B7 및 엔진 production auto-dispatch 승격은 이 작업 범위가 아니다.**
이전에 "LN gradient 검증 미완료"라고 표시했던 L384 제한은 해결됐다. 과거 벤치 숫자는 수정 전 커널의 역사 기록으로 보존하며, 당시 실패 결과를 지우거나 통과로 바꾸지 않는다.

## 재현·근거

- [수정 커널](fixed/joint.cu), [현재 선택 policy](policy.py), [선택·SHA-256](selected.json)
- [원인 분리](diagnose.json), [기존 실패 case 전체 검증과 시간표본](validation.json)
- [12case·MiniPairformer 연결](stress_block.json)
- [현재 개발 진입점](../trimul_training_current.py)
- `sbatch runs/trimul_ln_gradient_20260922/validate.sbatch`
- `sbatch runs/trimul_ln_gradient_20260922/stress_block.sbatch`
- `sbatch runs/trimul_ln_gradient_20260922/sanitize.sbatch`
- `sbatch runs/trimul_ln_gradient_20260922/verify_entry.sbatch`
