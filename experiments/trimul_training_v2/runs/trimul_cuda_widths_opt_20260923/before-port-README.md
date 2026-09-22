# 양방향 TriMul 폭별 실험 · 2026-09-22

D64/256/384/512 폭별 실험에 D128을 같은 조건으로 재측정해 추가했다. D128 추론은 기존 Anthropic K1/K3 설정을 유지했으며 재튜닝하지 않았다. **D는 입력 pair 폭이며, outgoing/incoming hidden도 각각 D다. 합친 hidden과 출력 LN 폭은 2D.** B1, L384/768, BF16, mask 및 residual 포함. 학습 dropout25%, 추론 dropout0. 모두 실제 GPU 실행 결과다.

## 추론: 폭별 CUDA 후보

기준은 같은 실행의 기존 Triton/cuBLAS 경로. 타이밍은 static compile + 수동 CUDA graph, 가중치 packing 포함. optimizer, dropout RNG, compile 및 CPU dispatch 제외. 각 행은 같은 GPU 프로세스의 교대 측정이다. 폭/길이 사이에는 잡과 클럭이 다르다.

| D | L | Triton ms | 선택 후보 ms | 배속 |
|---:|---:|---:|---:|---:|
| 64 | 384 | 0.205 | 0.131 | 1.56× |
| 64 | 768 | 0.812 | 0.530 | 1.53× |
| 128 | 384 | 0.419 | 0.276 | 1.52× |
| 128 | 768 | 1.684 | 1.172 | 1.44× |
| 256 | 384 | 1.120 | 0.971 | 1.15× |
| 256 | 768 | 4.661 | 3.986 | 1.17× |
| 384 | 384 | 2.231 | 1.817 | 1.23× |
| 384 | 768 | 9.118 | 7.446 | 1.22× |
| 512 | 384 | 3.770 | 3.283 | 1.15× |
| 512 | 768 | 15.332 | 13.673 | 1.12× |

| D | 입력 LN / K1 | 출력 | 선택 근거 |
|---|---|---|---|
|64|Anthropic 파생 fused K1, 사용하지 않는 x_n 저장 제거|Anthropic K3|두 길이 약1.5배, spill0|
|128|기존 Anthropic K1 설정, x_n 저장 없음|Anthropic K3|기존 설정 그대로 재측정; 재튜닝 없음|
|256|Anthropic 파생 fused K1, x_n 저장|기존 Triton LN/projection/gate|최소 K3 shared memory246.25KiB > H100227KiB, spill0|
|384|Anthropic 파생 fused K1, x_n 저장|기존 Triton LN/projection/gate|LN 분리는 전체0.3–0.5% 차이뿐. 단순한 융합형 유지; K1 spill store/load8B|
|512|별도 Triton LN (torch.compile) → Anthropic 파생 K1|기존 Triton LN/projection/gate|1CTA/SM 허용. LN 분리로 K1 spill store744/load976B →0; 전체 약2–3% 추가 단축|

Native K1은 Anthropic v5 body를 계승한다. upstream source는 그대로 두고 복사본의 MINB(최소 CTA/SM)만 컨피그로 바꿨다. D512는 기존 64-token tile의 최소2CTA/SM 제약으로 shared memory 예산을 만족할 수 없었으며, 1CTA/SM을 허용한 후보를 실행·검증했다. D512 LN 분리에는 이미 존재하는 LNM=0 경로를 사용한다.

튜닝축: BI/BJ, ring slots2/4/6/8, SKCH1/2/4/6/8 중 D를 나누는 값, 최소 CTA1/2. 자원 한도를 먼저 검사한 뒤 모든 남은 조합을 실행했다. K3(D64)는 slots4/6/8, accumulator1/2. [선택 config·cubin SHA](selection.json).

**D256 이상은 전체 CUDA 단일 경로가 아닌 혼합 경로다.** 출력 K3는 두 방향을 합친 H=2D 기준 최소 shared memory가 D256/384/512에서246.25/361.25/476.25KiB여서, 기존 구조의 compile-time 제약에 걸린다. 새 K3를 완성했다고 주장하지 않는다.

## 학습: 최신 연결 결과와 기존 대비 배속

D128은 최신 cache-policy B1–B4를 연결했다. L384 B7–B12는 LN gradient 수정 단일 커널, L768은 기존 분리형 CUDA 커널이다. L384/768 각각 기존 Triton과 같은 GPU 프로세스에서 교대로 측정했다. dropout25%, residual, 매 실행 forward와 저장, 실시간 가중치 packing, 11개 gradient 포함. profiler에서 두 길이의 `b1_fused`와 L384의 단일 `b7_joint` 실행을 확인했다. 두 길이 모두 이전 CUDA 대비 엄격한 수치 검증, Triton 교차 구현 검증, 입력·가중치 변경 후 graph 검증을 통과했다.

| D | L | 기존 Triton 전체 ms | 현재 전체 ms | 기존 대비 배속 | 실제 학습 경로 |
|---:|---:|---:|---:|---:|---|
| 64 | 384 | 0.819 | 0.819 | 1.00× | 기존 Triton · 최신 CUDA 미포팅 |
| 64 | 768 | 3.067 | 3.067 | 1.00× | 기존 Triton · 최신 CUDA 미포팅 |
| 128 | 384 | 1.654 | 0.990 | 1.67× | 최신 CUDA B1 + 단일 B7 |
| 128 | 768 | 6.479 | 4.794 | 1.35× | 최신 CUDA B1 + 분리 CUDA B7 |
| 256 | 384 | 3.465 | 3.465 | 1.00× | 기존 Triton · 최신 CUDA 미포팅 |
| 256 | 768 | 17.401 | 17.401 | 1.00× | 기존 Triton · 최신 CUDA 미포팅 |
| 384 | 384 | 6.188 | 6.188 | 1.00× | 기존 Triton · 최신 CUDA 미포팅 |
| 384 | 768 | 26.515 | 26.515 | 1.00× | 기존 Triton · 최신 CUDA 미포팅 |
| 512 | 384 | 9.334 | 9.334 | 1.00× | 기존 Triton · 최신 CUDA 미포팅 |
| 512 | 768 | 40.277 | 40.277 | 1.00× | 기존 Triton · 최신 CUDA 미포팅 |

L768 새 단일 B7은 dWL relative L2 0.0519%로 엄격한 기준 0.0500%를 초과해 선택하지 않았다. 허용 오차를 완화하지 않았으며 [실패 증거](latest-D128-L768-single-rejected.json)를 남겼다.

다른 D의 1.00×는 최신 CUDA가 같은 속도라는 뜻이 아니라, **아직 포팅하지 않아 기존 Triton을 유지**한다는 뜻이다. D128 기존 시간은 아래의 과거 기준선 수치 대신 최신 CUDA와 같은 실행에서 재측정한 값을 썼다.

## 참고: 기존 경로만의 기준 측정

아래 Triton은 **기존 폭 일반화 경로**다. D128 행 역시 동일한 기존 Triton 기준선이며, 최신 CUDA 결과는 위의 별도 학습 비교표에 있다. 다른 폭으로 해당 최신 커널을 포팅한 결과도 아니다. 또한 위 추론 후보는 training autograd를 제공하지 않는다.

| D | L | Triton train fwd ms | Triton fwd+bwd ms | PyTorch compile fwd+bwd ms | cuEq fwd+bwd ms |
|---:|---:|---:|---:|---:|---:|
| 64 | 384 | 0.257 | 0.819 | 2.038 | 1.091 |
| 64 | 768 | 0.961 | 3.067 | 17.214 | 4.137 |
| 128 | 384 | 0.533 | 1.619 | 3.987 | 2.225 |
| 128 | 768 | 2.075 | 6.326 | 35.694 | 8.648 |
| 256 | 384 | 1.290 | 3.465 | 7.893 | 5.517 |
| 256 | 768 | 5.202 | 17.401 | 73.134 | 21.632 |
| 384 | 384 | 2.271 | 6.188 | 12.564 | 10.213 |
| 384 | 768 | 9.132 | 26.515 | 113.671 | 40.497 |
| 512 | 384 | 3.602 | 9.334 | 17.100 | 16.401 |
| 512 | 768 | 14.499 | 40.277 | 153.679 | 65.660 |

cuEq는 동일한 양방향 수식의 primitives 조합이다. 공개 단방향 TMU 두 번 호출과 다르다. PyTorch도 동일 수식의 reference를 static compile한 결과이며 일반적인 모든 PyTorch 구현의 최선 성능을 뜻하지 않는다. 이전 D128의 다른 실행 수치와 배속을 계산하지 않는다.

Triton의 신규 shape cache miss는 24개 후보를 탐색했다. 전체 config 공간을 완전히 튜닝한 상한 성능은 아니다. 원본 로그에 해당 cache miss 기록을 남겼다.

## D512·L768 주소 오류 수정

기존 `_input_dual_bwd_kernel`의 `rk`가 int32여서 `k*fs1`이 행의 int64 주소에 더해지기 **전에** overflow했다. KP4096, M589824에서 최대 열 offset은2,415,329,280 elements로2^31-1을 넘는다. `rk`를 int64로 바꿔 곱셈부터64비트로 계산하도록 수정했다.

- 원본 memcheck: out-of-bounds read 재현, 종료99. 수정본:0errors.
- 수정 후 D512/L768 전체 추론·학습 forward·모든 gradient 검증 통과.
- `miniworld-engine-k1k3`, `miniworld-engine-tbwd`의 해당 커널에 반영. [반영 파일·SHA](installed-offset-fix.json).
- 실험용 engine snapshot은 덮어쓰지 않았다. 벤치에서는 동일한 수정 커널을 명시적으로 연결했다. [원본/수정 검사](offset-sanitizer.json).

## 검증·적용 범위

모든 기존 경로는 forward relativeL2≤.005, gradient≤.01의 BF16 교차 구현 기준을 통과. 입력 변경 후 graph/eager는 일반 출력 bit-exact, 기존 atomic LN gradient는5e-6 기준을 유지했다. 새 추론 후보도 전체 출력≤.005, 변경된 입력/가중치가 graph에 반영되는지 확인했다. 기존 D64/256/384/512 native 선택 후보는 두 길이에서 mask=0, memcheck/racecheck를 모두 통과했다. 추가 D128은 전체 forward 및 graph 입력 변경 검증을 수행했으며 이번 추가 측정에서 sanitizer는 재실행하지 않았다. 해당 sanitizer JSON과 로그에 결과를 기록했다.

실험용 [selected.py](selected.py)의 `Inference`에 D64/256/384/512의 폭/길이별 선택을 모았다. D128 표 행은 별도 `native_d128.py`로 측정했다. `torch.no_grad()`에서 사용하며 input/weight 내용 변경을 지원한다. graph replay 중 tensor storage 교체는 지원하지 않는다. **엔진 auto-dispatch에 추론 후보를 승격하거나 push하지 않았다.** 엔진에 적용한 변경은 주소 overflow 수정이다.

다음 우선순위: D512의 streamed-K K1 및 출력 K3 새 설계, 넓은 폭의 training backward 설계. D64도 최신 B1/B7을 새 폭에 맞게 포팅해야 한다. 지금의 학습 성능표만으로 D128 통합 알고리즘의 폭별 성능을 판단할 수 없다.

## 증거와 재현

`latest.sbatch`: D128 최신 CUDA와 기존 Triton 학습 비교. `d128.sbatch`: D128 기준선 두 길이 및 기존 CUDA 추론 재측정. `bench.sbatch`: 나머지 4폭×2길이, PyTorch/Triton/cuEq. `native.sbatch`, `native-nosave.sbatch`, `native-separate.sbatch`: Native 튜닝. `compare-ln.sbatch`: D384/512 같은 실행 LN 융합/분리 비교. `offset.sbatch`: 주소 오류 재현/수정. `native-sanitize*.sbatch`, `check-selected.sbatch`: 메모리·race·최종 adapter 검증. H100 node01, 공통 env `runs/anthropic_adoption_20260919/env.sh`.

[요약 JSON](summary.json) · [구조/전체 표 HTML](index.html) · [최종 선택](selection.json). NCU 기반 roofline/SoL은 이번 폭별 실험에서 측정하지 않았고, ptxas resource 정보 및 torch profiler kernel breakdown을 기록했다.
