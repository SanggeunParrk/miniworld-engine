# FP32 정규화 값 저장 / raw tri 없는 B1–B4

2026-09-21 · 현재 학습 개발 방향. Anthropic 파생 CUDA / H100 node02 / 양방향 C128 / dropout 25%, mask, residual / L384·768.

## 실제 저장·계산 정책

- Forward K3에서 입력 affine `x_n`(BF16), 출력 pre-affine `xhat`(FP32), 출력 `rstd`(FP32)를 저장한다.
- B1은 `tri`도 평균도 받지 않는다. 평균·분산 reduction 및 `(tri-mean)*rstd` 재계산이 없다.
- `xn_out = xhat * gamma_out + beta_out`만 복원한다. Projection/gate 및 그 미분은 기존 융합 구조로 계산한다.
- LN 미분은 `q=dNorm*gamma`, `dTri=rstd*(q-mean(q)-xhat*mean(q*xhat))`를 사용한다.
- Input LN 값은 계속 저장한다. B7은 기존 split_xn_pc1 구현을 유지한다.
- 학습 어댑터: `training.py:Training`. Production dispatch를 바꾸지는 않았다.

## 구현 변경

1. Gate 가중치를 shared memory에 유지해 타일별 재로드를 없앴다.
2. FP32 xhat를 CUDA fragment 순서의 native 타일 배치로 저장한다. 물리 shape `[M/64,16,4,2,32,4]`, 논리 shape `[M,256]`.
3. Forward는 정렬된 128-bit vector store, backward는 16 KiB ×4의 H100 TMA bulk copy로 읽는다. 읽은 값은 shared memory의 vector load로 곧바로 소비한다. 별도 layout 변환 커널은 없다.
4. Forward 저장을 affine 계산의 의존 순서에 배치해 K3의 register spill을 제거했다. 재계산이나 저정밀 압축으로 바꾸지 않았다.
5. dTri는 LN 미분이 끝난 뒤 비게 된 shared memory를 재사용해 TMA로 출력한다.
6. 길이별 K3 후보64개 중 컴파일/파이프라인 조건을 통과한44개와 B1 20개를 검증·교대 측정했다. K3 `(BI,BJ,slots,acc,regs,serial)=(1,128,4,1,1,1)`, B1 132 CTA·unroll8 선택. 실제 탐색 결과는 `tune-L*.json`.

## 전체 모듈 측정

각 구간별 600회 CUDA graph 교대 측정 중앙값. weight packing, backward Wp 전치, 양방향 cuBLAS, B7, 11개 gradient를 포함한다. Optimizer/RNG 생성/컴파일/CPU dispatch는 제외한다. 구간별 중앙값을 더한 값과 전체 실측은 다를 수 있다.

| L | 경로 | FWD ms | B1–B4 ms | BWD ms | 전체 ms |
|---|---|---|---|---|---|
| 384 | raw tri 재계산 기준 | 0.2953 | 0.2575 | 0.9720 | 1.2648 |
| 384 | 이전 FP32 x̂ | 0.3443 | 0.2977 | 1.0108 | 1.3534 |
| 384 | 현재 native FP32 x̂ | 0.3435 | 0.2936 | 1.0055 | 1.3459 |
| 768 | raw tri 재계산 기준 | 1.1596 | 0.8760 | 3.7681 | 4.9581 |
| 768 | 이전 FP32 x̂ | 1.3788 | 1.0061 | 3.8942 | 5.3205 |
| 768 | 현재 native FP32 x̂ | 1.3842 | 0.9776 | 3.8765 | 5.2976 |

비교 대상 `prior_xhat`는 이전 정규화 FP32 저장 경로이며, `baseline`은 raw tri에서 출력 LN을 재계산하는 shared B1 경로다. Anthropic 원본이나 cuEq 대비 수치가 아니다. raw tri 기준은 비교용이며 현재 선택 경로로 되돌리지 않는다.

## 검증과 한계

- L384·768 출력 및 11개 gradient: raw tri 기준과 bit-exact. 입력/가중치/mask/dropout/dy 변경 및 gamma_out=0 검사 포함.
- Forward 이후 raw tri를 NaN으로 덮어써도 모든 backward 결과 bit-exact. backward 보존 tensor에 tri가 없음을 확인했다.
- CUDA graph replay와 eager 결과 bit-exact.
- FP32 xhat+rstd 저장 payload: L384 151.58 MB, L768 606.34 MB. BF16 tri를 보존할 때보다 커지므로 LN 재계산 제거가 자동으로 전체 속도 향상을 의미하지 않는다.
- 기존 B7의 L768 dWL 독립 기준 상대 L2 0.055569% 문제(한도0.05%)는 이 작업과 별개로 남아 있다. 기존/신규가 같은 값이므로 production 승격은 보류한다.

Sanitizer/NCU 실행 상태: `verification-L*.json`.

L384·768 전체 memcheck 0 errors. L384 B1/K3 racecheck 0 hazards. NCU 두 길이 모두 정상 완료.

## B1 NCU 실측

| L | 경로 | 시간 | DRAM read | DRAM write | DRAM peak % | Tensor peak % |
|---|---|---|---|---|---|---|
| 384 | prior_xhat | 299.584000 us | 320.437248 Mbyte | 141.634560 Mbyte | 46.010735 % | 13.452837 % |
| 384 | xhat_fp32 | 288.160000 us | 320.969472 Mbyte | 142.375680 Mbyte | 47.965712 % | 13.441785 % |
| 768 | prior_xhat | 1.030848 ms | 1.228378 Gbyte | 481.541632 Mbyte | 49.482291 % | 15.377927 % |
| 768 | xhat_fp32 | 999.744000 us | 1.230742 Gbyte | 482.683392 Mbyte | 51.125724 % | 15.738783 % |

NCU 시간은 replay/profiling 비용을 포함한 별도 실행이므로 위 CUDA event 벤치 시간과 섞지 않는다. peak 비율만으로 전체 알고리즘 SoL90 달성을 주장하지 않는다.
