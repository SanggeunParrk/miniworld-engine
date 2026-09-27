# A100 Transition forward (CUDA, sm_80) — 개발 기록

목표: pair Transition 추론 forward(`out = x + Ws·(silu(Wa·LN(x)) ⊙ (Wb·LN(x)))`, D = 128, H = 512, bf16)를 A100용 CUDA 커널 하나로 만들고
SoL 90%까지 올린다. 비교 대상은 Anthropic 공개 커널 최선 row(`pf`, Triton, [baseline](../a100_anthropic_baseline/README.md))다.

- **카드**: A100 80GB PCIe, 전력 한도 300 W. 모든 버전이 전력 한도에서 돌고, SM 클럭은 1215–1305 MHz 사이에서 움직인다(`clock_probe.py`).
- **fixture**: baseline과 같다. `Transition(128, n=4)`에 학습된 것처럼 보이는 weight를 넣고(모듈 기본값은 squeeze weight가 0이라
  GEMM2가 0을 곱하게 된다. 전력이 덜 들고 클럭이 올라가 시간이 좋게 나온다), bf16 입력 `randn`, fp32 모듈과 비교한다.
  `transition_a100.fixture()`를 bench / probe / trace / ncu가 모두 쓴다.
- **측정**: CUDA graph replay, 7 × 50의 회당 중앙값(`bench.py`). 노드마다 ±1–2% 차이가 나서 공식 수치는 SoL 피크를 잰 gpu08에서 쟀다.

## SoL 정의

`max(essential bytes / BW, FLOP / tensor)`. 천장은 이 카드에서 잰 값이다(`../a100_trimul_fwd/records/peaks-gpu08.json`): cuBLAS large-GEMM
**240 TFLOP/s**(전력 한도, 1140 MHz), streaming copy 1.60 TB/s. gpu05는 238.5 TFLOP/s로 거의 같다(`records/peaks-gpu05.json`).

| | bytes / row | FLOP / row | L384 floor | L768 floor | SoL90 |
|---|---|---|---:|---:|---:|
| Transition fwd | 512 (x in + out) | 6·D·H = 393 216 | 241.6 µs (tensor) | 966.4 µs (tensor) | ≤ 268.4 / ≤ 1073.7 µs |

## 현재 (v15, `records/v15-gpu08.json`, gpu08)

| L | 이 커널 | % of SoL | Anthropic `pf` (같은 프로세스) | 배율 | rel-RMS vs fp32 |
|---:|---:|---:|---:|---:|---:|
| 384 | **288.1 µs** | 83.9% | 392.2 µs (61.6%) | **1.36×** | 2.62e-3 (pf 3.27e-3, bf16 모듈 3.35e-3) |
| 768 | **1086.9 µs** | 88.9% | 1513.8 µs (63.8%) | **1.39×** | 2.62e-3 (pf 3.26e-3, bf16 모듈 3.34e-3) |

출력은 run 간 bit 단위로 재현된다. **SoL90은 아직 도달하지 못했다**: L768은 1.2%, L384는 7% 모자란다.

## 설계

한 warp가 32행 × 전체 hidden을 맡아서 `[M][512]` 활성값이 warp 밖으로 나가지 않는다.

- **레지스터**: LN(x)는 GEMM1 A fragment로 레지스터에 상주한다(bf16, 64개). 출력 누산기 32 × 128도 512 hidden 내내 레지스터에 상주한다.
  **GEMM2는 f16**으로 돈다(h, Ws f16, f16 누산). sm_80에서는 f32 누산과 tensor 속도가 같고 레지스터는 절반(64개)이다.
  누산기는 **x에서 시작**하므로 residual이 공짜다.
- **step (16 hidden)**: GEMM1(a | b, 64 MMA) → C fragment에서 SwiGLU → 그대로 GEMM2 A fragment(m16n8 C = m16k16 A) → GEMM2(32 MMA).
  `Wa`는 host에서 0.5배(정확)해 두고 `silu(a)·b = a'b(1 + tanh a')`, hidden 원소당 MUFU 하나다.
- **host 순열**: 쿼드의 스레드 q가 x와 out 모두에서 16 B 벡터 i의 열 `32i + 8q … +7`을 갖도록 GEMM1 k 순서와 GEMM2 n 순서를 바꾼다.
  LN 입력이 셔플 없이 A fragment가 되고, 누산기 타일 J가 x word J가 되며, 쿼드의 store가 연속 64 B(full sector)가 된다.
- **weight 공급**: SM당 8-warp CTA 하나(256행 타일). 32-hidden 청크(24 KB)를 6-slot cp.async ring으로 L2에서 흘린다. CTA barrier는 없다.
  청크 u에서 각 warp가 청크 u+2의 1/8을 청크 u−4의 슬롯에 복사한다(8 warp가 모두 반납한 뒤, 느린 warp가 잡고 있으면 청크 끝으로 미룬다).
  그리고 FULL mbarrier를 기다리고 EMPTY에 arrive한다. warp 간 격차가 2청크까지 허용되어, 같은 SM sub-partition의 두 warp가 위상이
  어긋난다(한쪽 에필로그가 다른 쪽 MMA 밑으로 들어간다).
- **x**: staging 없이 LN이 L2에서 읽는다(이전 item 시작 때 `prefetch.global.L2`). 그 64 KB를 ring이 쓴다.
- **스케줄**: CTA별 작업 목록을 smem에 둔다. 타일 수의 나머지는 half tile(warp당 16행)로 **먼저** 돌려서, 어떤 라운드도 대부분 노는
  full 라운드가 되지 않고 시작 때 DRAM burst도 절반이 된다.

## 단계별 기록 (bench, 실제 weight, µs)

| 버전 | 변경 | L384 | L768 | 노드 |
|---|---|---:|---:|---|
| v1 | 4-warp CTA 2개/SM, 2-slot ring + 청크당 CTA barrier, fp32 GEMM2, residual은 L2에서 재읽기 | 331.6 | 1275.8 | gpu08 |
| v2 | 8-warp CTA 1개/SM, 3-slot, warp 0–3만 리필 | 349.5 | 1352.1 | gpu08 |
| v3 | 4-slot, 8 warp가 1/8씩 리필(단일 리필 warp 제거), half tile tail | 338.0 | 1292.4 | gpu08 |
| v4 | GEMM2 f16 누산(레지스터 −64, spill 160 B → 12 B), B fragment 1 k-step 선행 | 323.4 | 1212.2 | gpu05 |
| v5 | 누산기를 x로 초기화(residual 재읽기·레지스터·마지막 청크 peel 제거) | 307.2 | 1151.1 | gpu05 |
| v7 | half tile을 먼저 | 300.9 | 1156.5 | gpu05 |
| v9 | store가 full sector가 되는 열 순열, smem 작업 목록(turn 시 `LDL` 제거) | 289.7 | 1121.7 | gpu05 |
| v11 | 리필 지연, 즉시 오프셋 cp.async (gpu08 공식) | 288.2 | 1120.2 | gpu08 |
| v12 | x staging 제거 → 6-slot ring(격차 1 → 2청크) | 284.1 | 1095.0 | gpu01 |
| **v15** | v12 정리 (gpu08 공식) | **288.1** | **1086.9** | gpu08 |

기각한 것 (같은 노드에서 기준 대비):

| 시도 | 결과 | 이유 |
|---|---|---|
| mma를 non-volatile로 | 변화 없음 | SASS 스케줄은 ptxas가 따로 한다 |
| 4-warp CTA 2개/SM + f16 (v4 기준) | L768 +2%, L384 +6% | L2 weight 트래픽 2배 |
| 다음 타일 LN 통계를 청크 루프에 분산 (v6) | +3% | 청크 루프 스케줄이 망가진다 |
| GEMM2(t−1)를 SwiGLU(t)와 겹침 (v8) | +4% | spill 60 B, ptxas 스케줄 악화 |
| `tanh.approx.f16x2` (MUFU 절반) | 0% | MUFU 개수가 아니라 의존 사슬의 모양이 문제다 |
| GEMM2 B 선행 2/3 step, AHEAD 1, sleep 0/100, NST 3 | 0 … +13% | |
| 64-hidden 청크(48 KB × 3) | +0.6% | |
| 리필을 step 0 GEMM1 뒤로 | +0.7% | spill 112 B |
| 다음 타일 x를 마지막 청크에서 레지스터로 (v13) | 0% | spill 364 B |
| CTA별 청크 순서 회전(L2 hot spot 가설) | +1.5% | spill 124 B, 이득 없음 |
| 상주 weight 청크 2 / 3개 (ring 4 / 3) | −0.5% / +1.5% | 트래픽 −12.5%는 잡음 수준 |

## 남은 시간은 어디에 있나 (L768, `trace.py` clock64 + ablation)

- 타일 67 K 클럭 중 **청크 90%**(이상치 3072 대 3700 클럭, 83%), **LN 6.7%**(L2 latency 포함), store 1.2%, turn 1.2%.
- **weight 스트리밍 = 전력**: 복사만 뺀 ablation(`ABL_NOCOPY`)이 6.5% 빠르다. 대부분이 **클럭**(1245 → 1305 MHz)이다.
  SM당 256행이 한 번의 weight pass를 공유하므로, L2 → smem weight 트래픽이 행당 1.5 KB로 필수 트래픽(0.5 KB)의 3배다.
  행 수는 레지스터로 막혀 있다(스레드당 255개를 모두 쓴다).
- **SwiGLU 에필로그**: 실제 weight 기준 `ABL_NOEPI`가 v10에서 −3.8%, v15에서 0%다. 지금은 병목이 아니다.
- **L384 tail**: 5.33 라운드 → half tile로 5.5 라운드, 약 3%다. 카드 전체에 고르게 나눠도 SMSP당 2 warp인 것이 같아서 줄지 않는다.
- HMMA micro-benchmark: warp 하나에 독립 누산기가 4개 이상이면 pipe를 채운다(291–300 TF). 청크의 비효율은 warp마다 HMMA를 못 내는
  구간(청크 시작 동기화·발행 약 360 클럭, 에필로그)이 두 warp에서 겹칠 때 생긴다.

## 재현

```bash
cd experiments/a100_transition_fwd
sbatch --nodelist=gpu08 run.sbatch --length 384 768 --anth pf --out records/vNN-gpu08.json   # 정확도 + 시간 + Anthropic
sbatch multi.sbatch "384 768" "" "-DTR_NST=4 -DTR_AHEAD=2"                                  # 빌드 변형 A/B (ptxas 레지스터·spill 포함)
sbatch trace.sbatch                                                                          # clock64 구간 분해 (TR_TRACE)
sbatch prof.sbatch <tag> [flags]                                                             # clock/전력 + ncu stall + SASS
sbatch abl.sbatch 768 "" "-DABL_NOCOPY"                                                      # 시간만 보는 ablation (출력은 틀림)
```

`csrc/archive/`에 각 버전 커널을 남겨 두었다. `sm80_common.cuh`는 `../a100_trimul_fwd/csrc`의 것을 include한다.
