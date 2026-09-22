# Transition backward, 하나의 융합 CUDA 커널 (D=128, H=512, bf16) — 2026-09-21, node02 H100

> **마무리 2026-09-21.** 이 디렉터리는 개발 기록이고, 완성본은 miniworld-engine에 있습니다.
> 패키지: `experiments/transition_fused/` (브랜치 `research/transition-backward-fused`, 워크트리 `~/miniworld-engine-tbwd`, 로컬 전용).
> 배선: `src/miniworld_engine/kernels/transition/cuda/` + `modules.Transition._residual_forward`, 기본 ON.
> 실측(모듈 fwd+bwd, H100): L384 1074 -> 562 us, L768 4022 -> 2092 us. Triton 1.9x, torch.compile 2.2x, eager PyTorch 3.2x.

목표: 학습 backward(현재 엔진 = cuBLAS dh → Triton gate bwd → cuBLAS dWs/dWab/dxn → CUDA LN bwd, 5 launch)를 **커널 하나**로 융합.
참조 설계: B1–B4 TriMul backward(`runs/anthropic_b1b4_pipeline_20260919/dual_ln_prefetch.cu`)의 dual-role + Anthropic native v5 primitives
(`.../pkg/v5/csrc/tmn_ptx.cuh`, `tmn_kernels.cuh`: TMA, wgmma, ldmatrix/stmatrix, mbarrier, swz128, quad_sum).

## 연산 계약 (엔진의 has_xn + fuse_residual 경로와 동일)

```
dh = bf16(dy Ws)                    a = xn Wa^T, b = xn Wb^T (fp32)      sig = 1/(1+2^(-a log2 e)), silu = a sig
h  = bf16(silu b)                   dA = bf16((dh b)(sig + silu(1-sig))) dB = bf16(dh silu)
dWs = dy^T h, dWa = dA^T xn, dWb = dB^T xn   (fp32 누적 → bf16)          d_xn = bf16(dA Wa + dB Wb)
LN bwd(저장된 rstd, c1 = mean rstd): xhat = (x-mean) rstd, wdy = gamma d_xn,
dx = bf16(bf16((wdy - xhat ca - cb) rstd) + dy),  dgamma += d_xn xhat, dbeta += d_xn  (fp32)
```
bf16 반올림 지점과 곱셈 결합 순서는 엔진과 동일하게 유지했다. sigmoid는 `ex2.approx + div.full`(Triton `tl.sigmoid`가 내리는 형태).

## 결과 (`src/tbwd_w12.cu`, `-DDW_REPL=8`)

| | L384 (M=147K) | L768 (M=590K) |
|---|---:|---:|
| 엔진 backward(`_fused_bwd`, cuBLAS+Triton 5 launch) | 779 µs | 2950 µs |
| 1차 설계 v7 (클러스터 reduce-scatter, 1 launch) | 1017 µs | 3920 µs |
| **2차 설계 w12 (2역할 분할, 1 launch + 작은 reduce)** | **411 µs** | **1528 µs** |
| 엔진 대비 | **1.90×** | **1.93×** |
| 2역할 설계의 tensor 하한(22·M·D·H) | 224 µs | 897 µs |
| 절대 tensor 하한(16·M·D·H) | 163 µs | 652 µs |

latency는 CUDA graph replay median(같은 세션; spread ≤ 19 µs). 132 CTA × 256 thread(DW 64 / DX 68), smem 231 KB, 255 레지스터, spill 0.
**6개 출력 전부 비트 재현 가능**하다(w2까지는 dgamma/dbeta만 atomicAdd라 재실행 시 달라졌다).

수치: 6개 출력 모두 fp32 autograd 참조 대비 엔진과 같은 rel_rms(dx 3.067e-3, dWa 3.879e-3 …). 엔진 출력과의 차이는
dx 8.4e-5(원소 0.035%만 상이), dW* 1.3–2.0e-4(0.7%) — 누적 순서 차이뿐이다. dgamma/dbeta만 atomicAdd라 재실행 시 비트 동일하지 않다.

## 1차 설계 구조 (v1–v7, 기각)

16(→15) 클러스터 × 8 CTA, CTA당 256 thread = 2 warpgroup, 1 CTA/SM(smem 206 KB).
CTA rank r은 hidden 슬라이스 [64r, 64r+64)을 소유하고 그 Ws/Wa/Wb 48 KB를 상주시킨다(가중치 스트리밍 없음 = 재계산 0, FLOP 하한 그대로).
클러스터는 64행 타일을 순회하며 타일마다: 한 warpgroup(타일마다 교대)이 dh/a/b → gate → h,dA,dB(stmatrix) → d_xn 부분합
(m64n128, K=64) → **DSMEM reduce-scatter**(CTA r이 타일의 8행을 소유) → LN/residual/dx 저장; 두 warpgroup 모두 자기 몫의
dW(WG0: dWa_s + dWs^T 좌반, WG1: dWb_s + 우반, 96 fp32 누적 레지스터)를 in-flight로 누적한다. 입력은 TMA 더블버퍼.

## 측정으로 확인한 비용 (L384, 진단 변형은 결과가 틀린 타이밍 전용)

| 단계 | µs | 비고 |
|---|---:|---|
| v1 (첫 동작 버전, 클러스터 16개) | 2855 | `cuOccupancyMaxActiveClusters` = 15 → 16번째 클러스터가 2차 웨이브 |
| v2 = v1 − `fence.acq_rel.cluster` | 2787 | 이 fence가 `MEMBAR.ALL.GPU` + **`CCTL.IVALL`(L1 전체 무효화)** 로 내려가 stall 샘플의 15% 차지. `mbarrier.arrive.release.cluster`가 이미 release를 제공 |
| v2, 클러스터 15개 | **1564** | SM active/elapsed 0.48 → 0.95 |
| v4 = v2 + recv 행 stride 512→528 B | 1568 | 기각(뱅크 충돌이 원인이 아니었음) |
| v5 = v4 + d_xn을 dW GEMM보다 먼저 발행(`wgmma_wait<1>`) | 1560 | 미미 |
| **v6 = v5, per-peer mbarrier 핸드셰이크 → 하드웨어 클러스터 배리어** | **1017** | 타일당 `mbarrier.arrive.release.cluster` 16회(한 thread에서 직렬, 각각 DSMEM store drain) = **761 µs**. `barrier.cluster.arrive.release/wait.acquire` 2회로 대체 |
| v7 = v6 + dh/a/b 세 체인을 연속 발행 후 순서대로 소비 | 1017 | 변화 없음(GEMM 지연이 병목이 아님) |
| d6/e2: reduce-scatter를 로컬로만 (원격 DSMEM 제거) | −213 | 원격 DSMEM 비용 |
| e1: 타일당 클러스터 배리어 2개 제거 | −359 | 남은 배리어·스큐 비용 |
| d7/d9: d_xn·scatter·LN 제거, 배리어도 제거 | 396 | dh/a/b + gate + stmatrix + dW GEMM + CTA 배리어만 (tensor 하한 101 µs) |
| d8: sigmoid 제거 | −51 | MUFU는 병목 아님 |
| d10: gate 산술 제거(stmatrix 유지) | −60 | gate 산술도 병목 아님 |

clock64 프로브(v2, DX warpgroup, 타일당 25.9K cycles): issue+retire 4594 · dh 812 · a/b 1408 · gate+stmatrix 1668 ·
dW 발행 888 · d_xn 970 · **recv_free 대기 3822 · DSMEM scatter 8028** · LN+store 1117. 반대편 warpgroup은 타일당 3.5K cycles만 일하고 나머지는 대기.

## 1차 설계 결론: 이 구조로는 엔진을 못 이긴다

v7의 1017 µs 중 **572 µs(56%)가 d_xn의 CTA 간 축약**(클러스터 배리어 359 + 원격 DSMEM 213)이고, 코어 445 µs는 하한 163 µs의 2.7배다.
클러스터로 hidden을 쪼개면 재계산이 0이 되는 대신 타일마다 `[64행][128 d] fp32 부분합 8개`(256 KB/타일/클러스터, 전체 604 MB)를
DSMEM으로 옮기고 클러스터 전체를 2번 동기화해야 한다. 타일을 키우면 landing buffer가 smem을 넘고(64행에 이미 33 KB), 배리어를 1개로
줄이려면 부분합을 bf16으로 낮춰야 한다(반올림 8회 → 수치 계약 변경).

## 2차 설계 = 채택 (B1–B4와 같은 2역할 분할, 클러스터 없음)

- **DW 역할 CTA** (8 슬라이스 × R 복제): 128행 타일, 두 warpgroup이 각자 64행의 dh/a/b/gate를 하고(둘 다 바쁨), 동기화 후
  WG0 = dWa_s + dWs^T 좌반, WG1 = dWb_s + 우반을 128행에 대해 누적. smem = 가중치 48 + 입력 64×2 + h/dA/dB 48 = 224 KB.
- **DX 역할 CTA**: 128행 타일, hidden chunk 8개를 TMA ring(슬롯 48 KB × 2)으로 흘리며 chunk마다 dh/a/b → gate → dA/dB를
  **레지스터의 wgmma A-fragment 그대로**(smem 왕복·stmatrix 없음) → `d_xn += dA·Wa_j + dB·Wb_j` (RS m64n128)로 레지스터에 누적,
  마지막에 LN/residual/dx. CTA 간 통신 0.
- 재계산은 dh/a/b 3개뿐 → 총 22·M·D·H FLOP(하한의 1.375배) = **224 µs 하한**.

### 2역할 설계의 최적화 경과 (L384, R=8)

| 단계 | µs | 내용 / 판단 |
|---|---:|---|
| w1 | 466 | 첫 동작 버전(Triton `div.full` sigmoid) |
| **w2** | **438** | sigmoid를 kit의 `rcp.approx(1 + ex2.approx(-a log2 e))`로. 출력은 w1과 동일 |
| w3 | 426 | `tanh.approx` sigmoid. **미채택**: 엔진 대비 dx가 5.4e-5 → 2.9e-4로 계약이 바뀜 |
| w4 | 527 | DW stage-2 GEMM을 다음 타일로 흘려보내기. **기각**(96개 dW 누적기가 stage-1 위로 살아남아 스케줄 악화) |
| **w5** | **420** | Wa와 Wb를 `[128 n][128 d]` 한 덩어리로 패킹 → `a|b`가 m64n64 두 체인이 아니라 m64n128 한 체인(chunk당 wgmma 32→24개). 출력 동일 |
| **w6** | **419** | 같은 패킹을 d_xn에도: `[dA|dB]`를 m64k128 A-fragment로 보고 K=128 한 체인 |
| w7/w10 | 392–396 | xn 단일버퍼 + x를 global에서 + warp별 smem dgamma. **기각**: dx가 재실행 시 달라짐(간헐, 3회 중 2회). `bar.sync`를 mbarrier로 바꿔도 남아 원인 미해결 |
| w8/w9 | — | smem을 232448까지 쓰려다 illegal instruction. **동적 smem 실효 상한은 231424**(232448 opt-in에서 1 KB reserved가 빠짐) |
| **w12 (최종)** | **411** | dgamma/dbeta 부분합을 shared atomic 대신 **(CTA, warp)별 전용 global 행**에 누적 → atomic 0개, 6개 출력 전부 비트 재현 |
| w13a | 414 | DX 에필로그에서 두 행의 1-pass를 교차(로드 중첩). 변화 없음 — 이때 벽은 DW |
| w13b | 493 | DW의 stage1↔stage2 CTA 배리어를 warp그룹별 단방향 mbarrier로 바꿔 자기 행 몫을 먼저 발행. **크게 기각**: stage 2가 K=64 두 조각으로 쪼개져 체인이 짧아지고 중간 mbarrier 스핀이 붙음 |
| w13 | 486 | 위 둘 동시 적용, 기각 |
| w14 | 501 | DW의 stage-2 GEMM을 다음 타일로 흘리기(w4의 재시도. w4는 in-flight 누적기에 fence를 거는 하자가 있었지만, 고쳐도 결과는 같음). **기각** |

**어느 쪽 하나만 고쳐서는 총합이 안 줄어든다.** w12 기준 스윕(R=7 480 / R=8 415 / R=9 467)에서 역산하면 타일당 DW 2.91 µs,
DX 23.4 µs이고, R=8에서 DW 419 µs vs DX 398 µs라 **DW가 벽**이다. DX를 아무리 줄여도 419에서 멈추고(w12→w13a가 그 증거),
DW를 줄이면 398에서 멈춘다. 20%를 얻으려면 두 역할을 함께 고쳐야 하는데, 시도한 두 변경(w13a/w13b)은 각각 무효·역효과였다.

역할 비율 스윕(w1): R=6 594 · R=7 509 · **R=8 466** · R=9 550 · R=10 625 · R=11 735 µs. R ≥ 8에서는 DX가 타일당 25.8 µs로 일정한
병목이고 R ≤ 7에서는 DW가 2.89 µs로 병목이라 **R=8(DW 64 / DX 68)에서 균형**이다(1152 = 8 × 144이라 DW 타일 수도 정확히 나뉜다).
w12에서도 R=7 480 / R=8 415 / R=9 467로 R=8이 최적.

### DX 역할 비용 분해 (R=11, DX가 지배적인 비율. 기준 653 µs, 타이밍 전용 변형)

| 항목 | µs | 비고 |
|---|---:|---|
| LayerNorm 에필로그 전체 | 192 | 이 중 dgamma/dbeta 106(**shared atomic 77** + warp 셔플 축약 35), dx 전역 store 9 |
| d_xn RS 체인 | 157 | tensor 하한에 가까움(실질 연산) |
| gate 산술 | 108 | 원소당 sigmoid 2 MUFU + 10여 연산, chunk·thread당 32원소 |
| `a|b` 체인 | 82 | |
| dh 체인 | 68 | |
| 가중치 스트리밍 | ~0 | 링이 항상 chunk 0만 읽게 바꿔도 동일 → L2 대역 병목 아님 |

남은 여지: DX 역할이 타일당 약 23 µs(tensor 20.5K cycles, 45%), DW가 2.7 µs(3.1K, 64%). DX의 chunk 루프는
`dh/a|b → wait → gate → RS`가 직렬이고, gate를 다음 chunk의 GEMM과 겹치려면 a|b 누적기 한 벌(64 레지스터)이 더 필요해 255 한도에
막힌다. 링을 3슬롯으로 늘리거나 warp별 smem 축약 버퍼(8 KB)를 두려면 입력 더블버퍼를 포기해야 하는데, 그 방향(w7~w10)에서
미해결 레이스가 나왔다.

### 수치

6개 출력 모두 fp32 autograd 참조 대비 엔진과 같은 rel_rms(dx 3.067e-3, dgamma 3.684e-3, dbeta 3.070e-3, dWa 3.878e-3, dWb 3.774e-3,
dWs 3.414e-3 — 엔진 자신의 값과 1e-5 이내로 일치). 엔진 출력과 직접 비교하면 dx 5.4e-5(원소 0.019%만 상이), dgamma/dbeta 4e-5,
dW* 2.4–2.9e-4(원소 1%)로 누적 순서 차이뿐이다. dgamma/dbeta만 atomicAdd라 재실행 시 비트 동일하지 않고 나머지 4개는 동일하다.

### 개발 중 잡은 버그 (다음 커널에서도 나올 것들)

- 링 슬롯 해제를 타일 마지막에 한 번 더 하면(다음 타일 j=0의 해제와 중복) 슬롯 1만 타일당 5번 arming/해제되어 위상이 어긋나고 **데드락**. 해제는 chunk당 정확히 한 번.
- `__grid_constant__ Par p` 하나로 받는 시그니처는 인자를 개별로 패킹하는 런처와 맞지 않는다(misaligned address). CUtensorMap은 개별 `__grid_constant__` 파라미터로 받고 구조체에는 주소만 담는다.
- 마지막 chunk에서 xn 버퍼를 x로 덮어쓰기 전에 두 warpgroup을 join해야 한다(다른 WG가 아직 a/b GEMM으로 읽는 중).
- dgamma/dbeta를 커널 수명 내내 레지스터에 두면(64개) 652 B spill. 타일마다 quad 밖으로 shuffle 축약한 뒤 **(CTA, warp) 전용 global 행**에
  평범한 read-modify-write로 누적하는 것이 shared atomic보다 싸고 결과도 결정적이다.
- sm_90의 동적 shared memory 실효 상한은 **231424 B**다(opt-in 232448에서 block당 reserved 1 KB가 빠짐). 그 위로 요청하면 런치는 통과하고
  범위 밖 접근이 illegal instruction으로 터진다.
- `bar.sync`는 스레드만 정렬하고 **비동기 프록시(TMA 쓰기 vs wgmma 오퍼랜드 읽기)는 정렬하지 않는다**. 다만 이번 w7 레이스는 mbarrier로
  바꿔도 남아 원인이 다른 곳에 있다(미해결).
- in-flight wgmma 그룹이 물고 있는 누적기에 `wgmma.fence`/operand fence를 다시 걸면 **행**이 걸린다. fence는 그룹 발행 구간을 감싸는 용도다.
- 하네스가 커널 시그니처와 인자 개수가 어긋나면 런치가 `CUDA_ERROR_INVALID_VALUE`로 죽거나, 인자가 밀려 들어가 커널이 무한 대기한다.
  w13의 "데드락"은 실제로는 `dgbw` 인자를 안 넘긴 벤치 쪽 버그였다.

## NCU (w12, R=8, L384, `--set full`, `logs/w12-L384.ncu-rep`)

launch 401.5 µs, SM active/elapsed 0.966, **tensor pipe active 58.1%**, issue active 41.8%, warps active 12.5%(1 CTA/SM × 8 warp),
LSU wavefront 56%, shared 뱅크 충돌 662K이지만 **excessive wavefront는 0**(피할 수 있는 충돌 없음).
issue-active당 stall: wait 0.95 · barrier 0.77 · dispatch 0.46 · not_selected 0.39 · gmma 0.27 · long_scoreboard 0.21 · math_pipe 0.16.
SASS 라인별로는 `WARPGROUP.DEPBAR`(wgmma 대기) 13%, `HGMMA` 자체 9.5%가 최상위이고 **단일 라인 최대가 4.5%인 평평한 분포**다.
고칠 만한 단일 핫스팟이 없다는 뜻이고, 실제로 이후 구조 변경 5건(w4·w7~w10·w13a·w13b·w14)이 모두 무효이거나 역효과였다.

이 설계는 22·M·D·H FLOP을 실행해야 하므로 tensor 58%가 곧 401 µs다. 330 µs에 가려면 70%가 필요하고, 그 격차는 한 곳이 아니라
wgmma 의존 대기 전반에 퍼져 있다.

## 파일

`src/tbwd_w12.cu`(**최종**, `OUT=<name> build.sh tbwd_w12 -DDW_REPL=8`로 빌드), `src/tbwd_w{1..11}.cu`(경과 및 기각안),
`src/tbwd_w{f,g,h,k}*.cu`(역할 격리·단계별 타이밍 진단), `src/tbwd_v{1..7}.cu`(1차 설계, v7 = 그 계열 최선), `src/tbwd_d*.cu`·`tbwd_e*.cu`(타이밍 전용 진단), `src/tbwd_v2p.cu`(clock64 프로브),
`build.sh`(nvcc sm_90a + Anthropic v5 csrc), `bench_bwd.py`(엔진 `_fused_bwd`·fp32 autograd 참조 대비 검증 + CUDA graph 타이밍 + `--prof`),
`ncu_full.sh`, `drv.py`(cuda.bindings 런처: TMA 맵, 클러스터 런치), `occ.py`, `logs/v1-L384.ncu-rep`.
