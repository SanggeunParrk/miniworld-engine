# Transition backward 부분합 최적화 · 2026-09-22

기준은 **직전 hand-CUDA Transition**이다. Anthropic/Triton과의 새 비교가 아니다. Anthropic v5의 TMA/WGMMA primitives를 계승한 학습 구현의 부분합 처리를 개선했다.

## 결과

H100 SXM, BF16, D128/H512. 같은 프로세스에서 순서를 번갈아 CUDA graph 실행, 5×250 표본의 중앙값. 전체 블록은 정확도를 수정한 동일 TriMul + Transition, L384, dropout25%, 두 residual과 모든 파라미터 gradient 포함. RNG 생성·optimizer·CPU dispatch·compile 제외.

| 범위 | 이전 µs | 변경 µs | 시간 감소 |
|---|---:|---:|---:|
| Transition backward · L384 | 443.840 | 429.312 | 3.27% |
| Transition backward · L768 | 1714.896 | 1703.360 | 0.67% |
| MiniPairformer fwd+bwd · L384 | 1565.920 | 1548.688 | 1.10% |

Standalone job15646; block job15648. 표의 행들은 서로 다른 실행의 측정이며 단독 시간 합으로 전체 시간을 산출하지 않는다.

## 채택

- main backward 본체는 그대로 유지: 64 dW CTA + 68 dX CTA, 255 registers/thread, spill 0.
- LN gradient: 한 CTA에서 채널당 544개 부분합을 순차 합산하던 작업을 4 CTA × 8개 부분합 그룹으로 병렬화. 고정된 합산 순서, atomic 없음.
- dWs: 16×16 tile을 shared memory에서 전치하여 출력의 인접 주소에 저장. dW의 합산 순서와 BF16 반올림 지점 유지.
- reducer launch grid: 769 → 772 CTA. forward, 저장 정책, backward 본체의 fusion과 입력·출력은 동일.
- 적용: `/home/psk6950/miniworld-engine-tbwd/src/miniworld_engine/kernels/transition/cuda/transition_fused_bwd_sm90a_kernel.cu`. 이 Transition 개발 작업 트리의 D128/H512 경로에 반영. 중앙 main 레포로 병합/push한 것은 아니다.

## 검증

- L384/768 각 12조건, 총24조건: 실제 forward saves에서 계산. 작은 입력, gamma=0 채널, x=0, dy=0, 가중치 변경 포함.
- dX와 세 dW 모두 기존 native와 bit-exact. LN 최대 상대L2 **6.046e-07**, 기존 제한5e-6 유지.
- 전체 블록8조건 통과. graph/eager는 모든 gradient에서 bit-exact.
- 두 길이 memcheck0errors, racecheck0hazards (job15651). 필터는 main backward와 reducer를 모두 포함.
- PyTorch/cuEq와의 새로운 비교는 이번 범위가 아니다. 기존 CUDA 연산 계약을 유지하는 변경을 검증했다.

## NCU

NCU는 cache-control none / clock-control none, kernel replay14passes. latency 벤치는 별도 수행.

| 커널 | 이전 µs | 변경 µs |
|---|---:|---:|
| 본체 |390.624|392.288|
| 부분합 축약 |17.920|5.568|

본체 Tensor Core elapsed utilization 약56%. 이는 전체 알고리즘 SoL 비율이 아니다. HBM read/write 합 약165MB, 해당 replay 약391µs. 본체의 PC sampling은 barrier16.5%, wait20.5%, warpgroup-arrive5.8%; 이 비율은 표본 비율이며 시간 절감량이 아니다. 255register와 1CTA/SM, eligible warps/scheduler 약0.60인 상태여서 본체의 스케줄·동기화 개선이 다음 과제다. SoL90 미달 여부를 정량 확정할 calibrated bound는 이번에 산출하지 않았다.

## 제외한 후보

- per-warp global 부분합 float2 벡터화: 뚜렷한 이득 없음.
- CTA 내 부분합 누적 후 한 번 저장: L384 이득은 작고 L768에서는 추가 barrier 비용 때문에 선택안보다 느림.
- DW_REPL7/9: L384 약485/465µs, 선택안426µs보다 느림. L768에서도 느림.

## 재현과 증거

`prepare.py`, `next_candidates.py`, `tune.sbatch`는 직접 cubin 후보 비교. `integrate.py parallel_transpose 8`로 선택 native snapshot 생성 후 `validate.sbatch`, `block.sbatch`, `sanitize.sbatch`, `verify_install.sbatch` 실행. 기본 env는 `runs/anthropic_adoption_20260919/env.sh`.

- [정확도·standalone시간](validation.json), [전체블록](block.json), [7개 후보×2길이](tune.json)
- [NCU이전](ncu-baseline.csv), [NCU선택안](ncu-parallel_transpose.csv), [sanitizer](sanitizer.json)
- [선택소스](selected/transition_fused_bwd_sm90a_kernel.cu), [설치SHA](installed.json)
- [설치경로검증](installed_validation.json), [구조·결과HTML](index.html)
