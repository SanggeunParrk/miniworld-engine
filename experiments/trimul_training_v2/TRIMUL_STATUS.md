# TriMul 개발 현황 · 2026-09-23

**L384 입력 LN gradient 엄격 검증 문제를 해결했다.** 기존 상대L2 한도5e-6를 유지했고, 실패3case의 최대 오차는9.535e-6 → **3.550e-7**로 감소했다.

- 원인: gate 미분 `((da*g)*p)*(1-g)`가 원래 `((da*p)*g)*(1-g)`와 FP32 중간 반올림 순서가 달랐다. 기존 순서를 복원했다.
- 검증: 기존 실패3case, 추가12조건 × TriMul/전체 블록, graph/eager bit-exact, 같은 cubin memcheck/racecheck 통과.
- 적용: `runs/trimul_training_current.py`의 L384 개발 경로. 엔진 production auto-dispatch는 변경하지 않았다.
- 성능: job15612 같은 실행 TriMul 전체1006.800 →1009.168µs(+0.24%). MiniPairformer는 job15616에서1575.856 →1579.664µs.
- 남은 범위: 큰 폭의 추가 최적화, production 승격, SoL90/252µs 목표. 이번 정확도 수정으로 완료됐다고 주장하지 않는다.

[수정·검증 상세](runs/trimul_ln_gradient_20260922/index.html) · [선택·SHA-256](runs/trimul_ln_gradient_20260922/selected.json) · [수정 전 마무리 기록](docs/trimul-fusion/closeout-20260922/README.md)

## MiniPairformer 1블록 · 새 CUDA Transition 포함

L384/C128 H100, job15581: 최신 추론 **0.406ms**, 학습 forward **0.431ms**, 학습 fwd+bwd **1.582ms**. 동일 실행의 이전 H100 TriMul + 기존 Triton Transition은 각각 0.578/0.626/2.470ms. 수정 전 측정 기록이며, 이후 L384 LN gradient 엄격 검증을 통과했다.

[비교표·배선·검증](runs/minipairformer_block_cuda_20260922/index.html)

## MiniPairformer 1블록 · PyTorch / cuEquivariance

job15592, 같은 실행 최신 추론 0.386ms / 학습 forward 0.412ms / 학습 fwd+bwd 1.520ms. 학습은 PyTorch compile 대비 3.43배, cuEq primitives + PyTorch Transition 대비 2.28배, cuEq + 동일 CUDA Transition 대비 1.81배. 수정 전 측정 기록이며, 이후 L384 엄격한 LN gradient 검증을 통과했다.

[조건·전체 표·검증](runs/minipairformer_block_baselines_20260922/index.html)

## Transition backward 추가 개선 · 2026-09-22

| 범위 | 이전 µs | 변경 µs | 시간 감소 |
|---|---:|---:|---:|
| Transition backward · L384 | 443.840 | 429.312 | 3.27% |
| Transition backward · L768 | 1714.896 | 1703.360 | 0.67% |
| MiniPairformer fwd+bwd · L384 | 1565.920 | 1548.688 | 1.10% |

기존 hand-CUDA 대비. 엄격 검증·memcheck/racecheck 통과, Transition 개발 작업 트리에 반영. [구조·결과](runs/transition_bwd_upgrade_20260922/index.html).

## TriMul D별 최적화 · 2026-09-23

[최신 표·배선·검증](runs/trimul_cuda_widths_opt_20260923/index.html). D128 L768은 단일 B7로 전환했다. D64/256/384/512도 이전 CUDA 대비 개선했지만 Triton보다 느리다. 10shape autograd와 수정 커널 sanitizer 검증 통과.
