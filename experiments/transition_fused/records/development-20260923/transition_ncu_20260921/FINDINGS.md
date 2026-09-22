# Transition CUDA 커널 현황 + NCU 프로파일 (2026-09-21, node02 H100, 클럭 고정 없음 1.73–1.80 GHz)

대상: parity checkout(`runs/trimul_sm90_parity_20260917/engine`, 브랜치 perf/trimul-sm90-parity)의 Transition 커널.
- **H100 auto(학습이 실제로 쓰는 경로, D128)**: stats(Triton) → hand-CUDA b2b fwd(saved-xn, `transition_b2b_kernel.cu`) → backward = cuBLAS dh → Triton gate bwd(`_transition_expand_gatebwd_kernel`, savedxn stacked) → cuBLAS dWs/dWab/dxn → CUDA LN bwd.
- **새 CUDA 명시 variant**(`transition_variants_kernel.cu`, streamed_k/full_k, `Transition(implementation="cuda", ...)`): fwd(expand+SwiGLU+squeeze+residual 융합, h HBM 미저장) + gate bwd(h, dA|dB TMA store) + LN/residual bwd. 자동 승격 안 됨.
하네스 `prof_transition.py`(모델 shape M=L², 튠된 config), `ncu_run.sh`, 원본 `logs/*.ncu-rep`, 요약 `ncu_summary.py`, 핫스팟 `ncu_hotspots.py`.

## 같은 세션 커널 시간 (do_bench_cudagraph, ms)

| L | CUDA fwd streamed / full | Triton fwd(full-K b2b) | b2b fwd(auto, stats 포함) | CUDA gate bwd streamed / full | Triton gate bwd |
|---:|---|---:|---:|---|---:|
| 384 | 0.150 / **0.147** | **0.137** | 0.156 | 0.294 / **0.282** | 0.316 |
| 768 | — / 0.600 | **0.537** | 0.607 | — / **1.067** | 1.243 |

하한(D128 L384): fwd 연산 24·M·D² = 58 GFLOP → 1.73 GHz tensor peak 935 TFLOPS에서 **62 µs**; 메모리(x·res·out 113 MB) 38 µs. gate bwd 메모리 647 MB(xn 38 + dh 151 읽기, h 151 + dAB 302 쓰기) → 2.85 TB/s 패턴 하한 **227 µs**.
→ fwd는 세 구현 모두 tensor peak의 **43–48%**, CUDA gate bwd는 패턴 하한의 **80%**(L768 84%).

## NCU (D128 L384, `--set full`)

| 커널 | dur µs | CTA×thr | regs | smem KB | warps/SM | tensor% | XU(MUFU)% | issue% | eligible/cyc | lts% | DRAM rd/wr MB | 상위 stall |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| CUDA fwd full_k | 150.8 | 2304×128 | 175 | 98 | 8 | 42.8 | 28.5 | 28.1 | 0.34 | 55 | 38/38 | wait 1.55, barrier 1.37, long_sb 1.29, gmma 0.69 |
| CUDA fwd streamed_k | 148.6 | 2304×128 | 255 | 115 | 8 | 43.5 | 29.0 | 25.8 | 0.30 | 61 | 38/38 | barrier 1.76, wait 1.66, long_sb 1.44 |
| Triton fwd full-K | 134.4 | 1152×256 | 128 | 106 | 16 | 48.5 | 32.3 | 50.4 | 1.06 | 35 | 38/38 | barrier 1.59, not_selected 1.10 |
| b2b fwd (auto) | 141.3 | 1152×256 | 255 | 230 | 8 | 46.2 | 30.8 | 36.4 | 0.48 | 32 | 39/74(xn 저장) | wait 1.35, barrier 0.91 |
| CUDA gate bwd full_k | 283.6 | 1152×256 | 103 | 98 | 16 | 15.2 | 15.2 | 27.8 | 0.40 | 63 | 194/453 | **long_sb 6.84**, barrier 2.39, wait 1.52 |
| CUDA gate bwd streamed_k | 295.3 | 2304×128 | 67 | 66 | 12 | 14.1 | 14.1 | 41.0 | 0.55 | 63 | 205/453 | long_sb 1.73, wait 1.17, barrier 1.11, gmma 1.06 |
| Triton gate bwd | 318.4 | 18432×128 | 163 | 37 | 12 | 13.1 | 14.3 | 39.8 | 0.59 | 61 | 189/451 | long_sb 2.65 |

local memory(spill) 트래픽: 모든 CUDA 커널 0.

핫스팟(SASS stall 샘플):
- CUDA fwd full_k: `WARPGROUP.DEPBAR`(WGMMA 완료 대기) 18%, HGMMA 발행 10%, mbarrier try_wait 스핀(`@!P BRA`) 11%, 마지막 에필로그의 residual `HADD2` 8%. 명령 구성 FMUL 19%, **MUFU 12.6%**, HGMMA 4.7% — SwiGLU 에필로그(ex2+rcp, 원소당 MUFU 2회 = 151M MUFU → 132 SM × 16/clk에서 **≈41 µs, 커널의 28%**)가 MMA와 겹치지 않고 직렬로 실행된다.
- CUDA gate bwd full_k: 스핀 루프 `@!P0 BRA` **38%** + `SYNCS.ARRIVE`(mbarrier) 13% = 파이프라인 동기화가 stall의 절반. hidden tile(BN=32, 16개)마다 TMA store 3개 → `wait_group.read 0` → `__syncthreads` 2회로 store 완료를 즉시 기다린다(단일 스테이지).

구조 진단(fwd): CTA당 1 warpgroup(128 thr)이 TMA 대기 → 2×WGMMA → `wait<0>` → 에필로그 → squeeze WGMMA → `wait<0>`를 직렬 수행. producer/consumer 분리 없음, k-step 간 MMA 파이프라이닝 없음(항상 `warpgroup_wait<0>`), SM당 8 warp. 또 BM=64 CTA 2304개가 각자 가중치 384 KB(Wa·Wb·Ws)를 L2에서 다시 스트리밍 → 885 MB L2 트래픽(lts 55–61%); Triton은 BM=128로 절반(35%).

## 큰 D (fwd, L384)

| D | CUDA | Triton | tensor% C/T | lts% C/T | CTA/SM | 비고 |
|---:|---:|---:|---|---|---|---|
| 256 | **436 µs** (full_k) | 507 | 57.5 / 50.1 | 36 / 34 | 1 (241 regs, 229 KB) | CUDA 우세 |
| 384 | 1.376 ms (full_k) | 1.389 | 40.2 / 39.9 | 44 / 74 | 1 | 동률 |
| 512 | 2.633 ms (streamed) | 2.702 | 37.1 / 37.3 | 66 / 32 | 1 (255 regs) | 연산 하한 1.0 ms의 2.6배 |

D512: CTA(BM=64) 2304개 × 가중치 6 MB = **14 GB L2 트래픽** → 가중치 재스트리밍이 커널 시간을 결정(lts 66%). 큰 D의 근본 문제는 "행 타일당 전체 가중치를 다시 읽는" 타일링이다.

## 학습 step 구조 (D128 L384, 기록 breakdown: cuda:full_k 총 1.027 ms)

fwd 0.209 | bwd 0.822 = dh cuBLAS 0.074 + gate 0.28–0.32 + dWs 0.068 + dWab 0.129 + dxn 0.117 + LN bwd·기타 ≈0.1.
backward HBM 트래픽 ≈ **1.66 GB**(dh 151 쓰기·읽기, h 151 쓰기·읽기, dAB 302 쓰기·2회 읽기 …) vs 필수 입출력 ≈ 120 MB(xn, dy, dxn). 3 TB/s에서 데이터 이동만 ≈550 µs → backward의 2/3가 중간 텐서 왕복.

## 결론: 어디가 덜 최적화됐나

1. **fwd 커널 구조** — tensor 43%. warp specialization(producer warp + consumer WG 2개 ping-pong: 한 WG가 MMA 하는 동안 다른 WG가 SwiGLU 에필로그)과 `wait<1>` MMA 파이프라이닝으로 에필로그(28%)와 동기화 대기(≈30%)를 MMA 아래로 숨기면 70%+ 가능 → **≈90 µs(−40%)**. 보조: sigmoid를 `tanh.approx` 1 MUFU로(MUFU 절반), BM=128(2 WG가 가중치 타일 공유 → L2 트래픽 절반).
2. **큰 D fwd** — 가중치 L2 재스트리밍. persistent CTA가 행 타일 여러 개를 상주시켜 가중치 타일 1회 적재로 R개 행 타일을 처리(트래픽 1/R), 또는 2-CTA cluster TMA multicast. D512 하한 대비 2.6배 여지.
3. **gate bwd** — 이미 메모리 하한의 80%. store 더블버퍼(다음 타일 MMA와 store 완료 겹침)·BN 64로 남은 20%의 일부.
4. **가장 큰 상금: backward 구조** — B1–B4 TriMul backward와 같은 dual-role 융합(DX 역할: dh=dy·Ws 재계산·gate 미분·dxn=dA·Wa+dB·Wb를 커널 안에서; DW 역할: dWs/dWab를 L2 창에서 누적, h/dAB를 HBM에 쓰지 않음). 연산 145 GFLOP(≈155 µs peak, 60%에서 ≈250 µs) + 필수 트래픽 120 MB → backward 0.82 → **≈0.3 ms**, 학습 step 1.03 → ≈0.5 ms 전망. 실제 학습 경로(auto)는 b2b fwd + Triton gate bwd + cuBLAS 4회이므로 이 융합이 그대로 학습 시간에 반영된다.

---

# (정정) Anthropic Transition 커널 분석 — 2026-09-21

위 1부는 우리 엔진의 CUDA variant 분석이다. 여기가 Anthropic upstream(`third_party/anthropic/upstream/common/opt_core/opt_core/kernels/transition`, rev f4f62fa) 분석이다.

## Anthropic Transition provider의 구성

| row | 구현 | 셀 | 비고 |
|---|---|---|---|
| v2, v1, pf, lnl, af3_fused | **Triton** 단일 커널(LN + a|b GEMM + SiLU·b + W_o + residual) | c 64–512 | C128에서 우리 Triton과 동률(0.158 vs 0.158 ms) |
| **esm_t16** | **CUDA C++/CuTe persistent warp-specialized 커널**(`esm/ef2_t16/ef2_transition_cute.cuh`, 사전 빌드 cubin, driver-load): 132 CTA×384 thr = producer WG(가중치 ring 5×32 KB TMA, x half-tile TMA) + consumer WG 2개(64행씩, 240 regs); LN을 레지스터에서 계산해 smem에 in-place; hidden 64-chunk마다 a,b = x̂·W1ⱼᵀ, x̂·W2ⱼᵀ (wgmma SS m64n64k16, 두 commit group), h = bf16(silu(a)·b)를 wgmma A-fragment 레이아웃으로 레지스터에 직접 구성, acc += h·W3ⱼᵀ (wgmma **RS** m64n256k16, in-flight 유지); 에필로그 out = bf16(x + acc), residual은 L2 재읽기, 32-bit 직접 store | **c=256, hidden=1024 전용** | 이전 이식 측정 C256 L384 0.397 ms (엔진 Triton 0.557, 1.40×) |
| **flash_sm90a** | **CUDA C++/CUTLASS 커널**(`flash_sm90a/csrc/flash_transition_sm90.cu`, 사전 빌드 .so): 128행 타일/CTA(비영속, 1152 CTA), **2-CTA cluster TMA multicast** 가중치 ring([Wa;Wb] 64×256 ×3, Wo 256×32 ×4), producer WG + consumer WG 2개 **FA3식 ping-pong**(named barrier로 wgmma 발행 교대), hidden 32-chunk, GEMM2 RS, smem staging 16 B coalesced 에필로그; **수치 EXACT**(stock bf16 autocast 체인과 비트 동일, SiLU 65536 입력 전수 검사) | c=256, hidden=1024 전용 | 이식 측정 0.832 ms — 커널 388 µs + 외부 fp32 LayerNorm·캐스트 445 µs (LN=0 variant 배선) |

**우리 모델 shape(D128)에는 Anthropic CUDA 커널이 없다.** C128은 Triton v2가 최선이고 우리 것과 동률이다.

## NCU (C256 L384, M=147456, `--set full`, 클럭 고정 없음 ≈1.75 GHz). 하한: 연산 24·M·D² = 232 GFLOP → **245 µs**(946 TFLOPS), 메모리 226 MB → 75 µs

| 커널 | dur µs | grid×thr | regs | smem | tensor active% | SM active/elapsed | XU% | issue% | eligible | lts% | 상위 stall (per issue-active) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **esm_t16** `ef2_transition` | **392.9** | 132×384 (persistent) | 168 | 230.5 KB | 66.7 | 0.936 | 22.6 | 28.0 | 0.36 | 50 | long_scoreboard 2.84, barrier 1.88, wait 1.40, mio 0.77, lg_throttle 0.51 |
| **flash_sm90a** `kernel<Cfg<32,3,4>,0,1>` | **387.7** | 1152×384 (cluster 2) | 168 | 230.4 KB | 66.8 | 0.955 | 22.3 | 43.3 | 0.67 | 32 | barrier 1.67, dispatch 0.90, wait 0.83, long_sb 0.60, sleeping 0.42 |
| 엔진 Triton `_kernel` (+LN 53 µs) | 510.5 | 1152×256 | 210 | 213 KB | 49.0 | 0.957 | 16.3 | 30.8 | 0.43 | 37 | barrier 2.02 |
| esm_t16 L768 | 1456 | 132×384 | 168 | | 67.9 | 0.973 | 23.0 | 28.6 | 0.37 | 58 | 동일 패턴 |
| (참고) v2 Triton C128 L384 | 161.8 | 2304×128 | 243 | 115 KB | 39.3 | 0.96 | 26.6 | 37.2 | 0.52 | | long_sb 1.28 |

두 Anthropic CUDA 커널 모두 **tensor peak의 62%**(tensor active 67% × SM active 93–95%). 우리 Triton(49%)보다 낫지만 상한은 아니다.

핫스팟(SASS stall 샘플 비중):
- **esm_t16**: `WARPGROUP.DEPBAR gsb0,1`(chain a 대기) 13.6% + `DEPBAR 0`(chain b) 7.4% = **21%**; mbarrier try_wait 스핀(`@!P0 BRA`, 가중치 slot w_full 대기) 11.7+5.4+2.5+2.2+… ≈ **23%**; HGMMA 발행 3%; 명령 구성 FMUL 15%, MUFU 9.4%, HGMMA 5.2%.
  - 가중치 ring: 128행 타일마다 W1ⱼ,W2ⱼ,W3ⱼ 16 chunk × 96 KB = 1.5 MB → 1152 타일 × 1.5 MB = **1.77 GB L2→SM**(lts 50–58%); 5 slot × 32 KB ring이 consumer를 자주 멈춘다. multicast 없음(Anthropic 주석: cluster multicast 변형은 "느리거나 같았다").
  - consumer WG 2개가 lockstep: silu(a)의 MUFU(XU 22.6%)는 chain b와만 겹치고 ping-pong이 없어 두 WG의 MUFU 구간에 tensor core가 논다.
  - 타일마다 LN 프롤로그(consumer가 직접 2-pass 통계·warp shuffle 160회/thread)가 MMA와 겹치지 않고, 에필로그의 residual 재읽기(ldg 64회/thread를 마지막 GEMM2 직전에 몰아 발행)가 long_scoreboard 1위·lg_throttle의 원인.
- **flash_sm90a**: `UCGABAR_WAIT`(cluster barrier 대기) **18.8%**, WARPSYNC 5.3%, ERRBAR(named barrier) 3.1%, mbarrier 스핀 4%, HGMMA 3%, DEPBAR 2%. 비영속 2-CTA cluster라 128행 타일(CTA)마다 cluster_sync 2회 + ring 콜드스타트(3+4 slot 채우기)를 반복한다(8.7 wave). BH=32 chunk 32회 × ping-pong named barrier(barrier 1.67, dispatch 0.90). L2 트래픽은 multicast로 절반(lts 32%).
- 공통: GEMM1이 SS 모드 n64(esm_t16은 a,b 두 chain이 같은 A를 각각 읽음)로 smem operand 트래픽이 tensor 풀속도에서 이론 한계(128 B/clk)에 닿는다(l1tex 71%). 1 CTA/SM, 12 warp.

## 결론

- Anthropic Transition **CUDA 커널은 c=256 전용이고 tensor peak의 62%**다. 남은 38%는 (1) 가중치 ring 대기(L2 1.77 GB, 5 slot), (2) 두 consumer WG의 lockstep으로 MUFU 에필로그가 MMA 뒤에 숨지 않음(esm_t16) / 비영속 cluster sync·콜드스타트(flash), (3) 타일당 LN 프롤로그와 residual 재읽기 지연(esm_t16), (4) SS n64 GEMM1의 smem operand 대역.
- 설계 자체(persistent + producer/consumer + 레지스터 h → RS GEMM2 + TMA ring)는 우리 CUDA variant(1 WG 직렬, tensor 43%)보다 낫고 계승할 가치가 있다. 계승 시 개선안: esm_t16 골격 + flash의 ping-pong, 가중치 ring 확대/2-CTA multicast(또는 D128에서는 W 상주), LN을 producer 쪽/별도 warp로 분리, residual을 TMA로 선적재, GEMM1 a|b 패킹(A 1회 읽기, n128), sigmoid 1-MUFU(tanh.approx) 또는 exact 유지 시 MUFU를 chain b 뒤로 완전히 겹치기 → 75–80% 목표(≈310 µs, C256).
- **D128(우리 모델)**: Anthropic CUDA 커널이 없으므로 위 골격을 c=128/hidden=512 셀로 새로 구현해야 한다(연산 하한 62 µs, 현재 최선 Triton 134 µs). 학습 backward는 Anthropic에 없다(esm_kd3 = frozen-weight dX만).
