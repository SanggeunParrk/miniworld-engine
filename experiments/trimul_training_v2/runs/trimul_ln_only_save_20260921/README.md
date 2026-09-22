# Forward: LN activation만 저장할 때의 비용 · 2026-09-21

## 결과

입력 LN 출력만 저장하면 전체 forward는 **L384 +3.61%, L768 +2.69%**. 출력 LN 출력만 저장하면 **+6.70%, +4.27%**. 둘 다 저장하면 **+10.77%, +7.96%**.

**저장하는 것은 affine 이후 BF16 `x_n`(C128), `xn_out`(C256)뿐이다. 평균·역표준편차, projection, gate, input preactivation은 저장하지 않는다.** 기존처럼 left/right/tri는 유지한다. 따라서 '저장 없음'은 LN 추가 저장 없음이라는 뜻이다.

H100 node02 GPU0/1, BF16 batch1/C128, bidirectional shared output LN256, L384/768, mask/dropout25%/residual. 기존 Anthropic 파생 no-save forward를 기준으로 같은 실행에서 측정했다. 각 variant마다 600회 교대 CUDA graph timing. Live weight packing 포함, RNG/optimizer/CPU dispatch/컴파일 제외. Forward만 비교했으며 backward 이득은 아직 측정하지 않았다.

| L | 추가 저장 | 전체 forward ms | 증가 μs | 증가율 |
|---|---|---|---|---|
| 384 | 추가 LN 저장 없음 | 0.284704 | +0.000 | +0.00% |
| 384 | 입력 x_n만 | 0.294976 | +10.272 | +3.61% |
| 384 | 출력 xn_out만 | 0.303792 | +19.088 | +6.70% |
| 384 | 입력·출력 둘 다 | 0.315376 | +30.672 | +10.77% |
| 768 | 추가 LN 저장 없음 | 1.172720 | +0.000 | +0.00% |
| 768 | 입력 x_n만 | 1.204224 | +31.504 | +2.69% |
| 768 | 출력 xn_out만 | 1.222784 | +50.064 | +4.27% |
| 768 | 입력·출력 둘 다 | 1.266080 | +93.360 | +7.96% |

## 통제한 구현

K1, cuBLAS contractions, K3의 LN/GEMM/반올림/epilogue는 유지한다. K3가 원래 계산하는 두 LN 결과를 필요한 경우에만 global buffer에 추가로 기록한다. 입력 LN 저장도 K3에서 수행하여 K1을 변경하거나 K3의 입력 LN 재계산을 없애지 않았다. 따라서 LN 분리형과 융합형을 다시 비교한 결과가 아니다.

새 activation 복원용 커널은 없다. K3 내부에서 기존 output staging shared memory를 재사용한 TMA store와 register→global vector store를 비교했다. **모든 저장 조합, 두 L에서 TMA가 더 빨랐다.** K3 config는 `(BI=2,BJ=64,slots=4,NACC=1,regs24/240,LN_serial=1)`로 공통 고정했다. 저장 스케줄 두 개를 비교했으며 전체 tile/config 공간 재튜닝 결과가 아니다.

수정된 커널의 저장OFF와 완전 미수정 기존 forward도 비교했다: L384 284.704 vs284.768 μs, L7681172.720 vs1174.032 μs. 약0.02%/0.11% 차이로, 저장OFF 대조군이 기존 경로와 같은 성능임을 확인했다.

## K3 단독: 같은 입출력 버퍼에서의 저장 방법 비교

| L | 저장 | TMA μs | vector STG μs |
|---|---|---|---|
| 384 | 입력 x_n만 | 94.368 | 110.720 |
| 384 | 출력 xn_out만 | 103.008 | 151.392 |
| 384 | 입력·출력 둘 다 | 117.120 | 183.744 |
| 768 | 입력 x_n만 | 313.728 | 363.840 |
| 768 | 출력 xn_out만 | 346.400 | 518.128 |
| 768 | 입력·출력 둘 다 | 405.152 | 665.520 |


K3 단독과 전체 forward는 캐시 상태와 타이밍 범위가 달라 증가량이 정확히 같지는 않다. 최종 답변에는 전체 forward 실측을 사용한다.

## 추가 유지 메모리

| L | 저장 | 추가 tensor 크기 |
|---|---|---|
| 384 | 추가 LN 저장 없음 | 0.000 MB |
| 384 | 입력 x_n만 | 37.749 MB |
| 384 | 출력 xn_out만 | 75.497 MB |
| 384 | 입력·출력 둘 다 | 113.246 MB |
| 768 | 추가 LN 저장 없음 | 0.000 MB |
| 768 | 입력 x_n만 | 150.995 MB |
| 768 | 출력 xn_out만 | 301.990 MB |
| 768 | 입력·출력 둘 다 | 452.985 MB |


논리적인 tensor 크기 합계다. peak GPU memory 실측은 아니다.

## 검증과 한계

- 모든7개 K3 후보에서 최종 y가 미수정 forward와 bit-exact.
- 저장된 LN 출력은 독립 FP32 수식→BF16 기준 상대L2 약0.000009–0.000011. 검사 한도0.0005. Input/output LN 각각 검사했다.
- x, LN gamma, dropout scale, mask를 바꾼 CUDA graph replay에서도 y는 미수정 경로와 bit-exact. 저장된 activation도 변경된 입력에 대한 독립 기준과 일치한다.
- L384 K3 racecheck: 오류0/hazard0. L768 K3 memcheck: 오류0. 필터는 `kns=infer_k3`; 전체 backward sanitizer가 아니다.
- 저장OFF K3는 기존 커널과 동일하게 ptxas16B spill stores/32B spill loads가 있다. 저장ON도 설정에 따라 다르다. 입력 LN만 TMA 저장하는 경로는52B spill stores/68B loads, 입력·출력 둘 다 TMA 저장하는 경로는0이다. 저장 비용뿐 아니라 컴파일러 스케줄 변화가 포함된 실제 실행시간 비교이며, 전 후보 spill0이라고 주장하지 않는다.
- 이 실험은 BF16 LN activation만 저장한다. Mean/rstd까지 저장하는 경우와 backward 연결/성능은 별도 평가가 필요하다. Production dispatch는 변경하지 않았다.

## 재현

node02에 할당된 H100에서 `MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build` 설정 후 `bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_ln_only_save_20260921/bench.py --length 384` 실행. L768도 동일하다. `--check-only`는 정확도만 검사한다.

Anthropic Apache-2.0 native v5 TMA/WGMMA·LN 구현을 계승한 저장 정책 실험. 기존 `no_save_k3.cu`에서의 변경은 derive.py와 derivation.json에 기록했다.
