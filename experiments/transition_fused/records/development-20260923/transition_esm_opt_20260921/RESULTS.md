# Anthropic esm_t16 Transition 커널 최적화 (c=256, hidden=1024, bf16) — 2026-09-21, node02 H100

> **마무리 2026-09-21.** Anthropic esm_t16 forward를 C256에서 SoL 77.6 % (L384) / 82.9 % (L768)까지 올린 기록입니다
> (최종 `src/v21b_resslab.cuh`). **MiniWorld에는 적용되지 않습니다**: MiniWorld의 Transition 폭은 d_pair 128,
> d_single 384, d_msa 64, d_single_token 768이고 C256은 어디에도 없습니다. MiniWorld 폭(D=128)용으로 새로 만든 커널과
> 그 배선은 `runs/transition_bwd_fused_20260921/RESULTS.md`를 보세요.

베이스: `opt_core/kernels/transition/esm/ef2_t16/ef2_transition_cute.cuh`(rev f4f62fa, 사전 빌드 cubin과 비트 동일한 NVRTC 재빌드 `v0_orig` 확인).
flash_sm90a 대신 esm_t16을 고른 이유: persistent 구조, raw PTX + CuTe atom만 쓰는 단순 코드, fast-tier 수치(에필로그 자유도). 하네스 `bench_esm.py`(fp32 참조·shipped cubin 비트 비교·CUDA graph 시간),
`ncu_ab.sh`(NCU per-launch **elapsed cycles** — 이 세션의 클럭 노이즈(graph 벽시계 ±50 µs)를 피하는 기준 지표), `build_nvrtc.py`(Anthropic 로더와 같은 NVRTC 옵션 + -lineinfo).
SoL 분모 = 24·M·D² FLOP / (132 SM × 4096 dense-bf16 FLOP/cycle) = **428.6K cycles**(L384), 1.714M(L768).

## 결과

| | L384 latency | L384 cycles | SoL | L768 latency | L768 cycles | SoL |
|---|---:|---:|---:|---:|---:|---:|
| shipped (Anthropic cubin) | 394 µs | 690K | 62.5% | 1458 µs | 2.581M | 66.5% |
| v14 (1차 최종, 2026-09-21 오전) | 342 µs | 585K | 73.3% | 1288 µs | 2.209M | 77.7% |
| **v21b (최종)** | **325 µs** | **553K** | **77.6%** | **1220 µs** | **2.070M** | **82.9%** |
| tensor 하한 | 245 µs | 428.6K | 100% | 980 µs | 1.714M | 100% |
| shipped 대비 | −17.5% | | | −16.3% | | |

latency = NCU per-launch `gpu__time_duration`(같은 세션, `--clock-control none`, 변형 라운드로빈 median; L384 ±1 µs, L768 ±4 µs). graph 벽시계(node02, ±50 µs 노이즈)는 참고만: shipped 386, v21b 319–323 µs.

수치: fp32 참조 대비 rel_rms 2.315e-3 → 2.315e-3(불변, L768 2.312e-3 동일). shipped 출력 대비 bf16 원소 0.98%가 1 ulp 차이 — 전부 tanh sigmoid(v2)에서 나온 것이며 그 외 변경(v4~v21b)은 v2 출력과 비트 동일(하네스가 인접 변형 간 differing elements = 0 확인). CUDA graph replay 후 출력 동일.
NCU(v14, L384): tensor pipe active 77.4%(shipped 66.4%), SM active/elapsed 0.952, issue 34.5%, lts 50%, XU 13.3%(shipped 22.6%).

## 단계별 (같은 세션 NCU cycles, L384)

| 변형 | 내용 | cycles | SoL | 판단 |
|---|---|---:|---:|---|
| v0_orig | NVRTC 재빌드(shipped와 비트 동일) | 686–692K | 62.5% | 기준 |
| v1/v1b | flash식 [G2(j−1)+G1(j)] 배치 + ping-pong 배리어 | 802K / 519K | | **기각**: G2가 다음 chunk 가중치 대기 뒤로 밀려 ring 대기를 숨기던 in-flight G2가 사라짐 |
| **v2_tanh** | sigmoid = 0.5·tanh.approx(a/2)+0.5 (MUFU 2→1) | 659K | 65.1% | 채택 (tolerance class, 0.98% 원소 1 ulp) |
| v3_pp2 | 원래 순서 + 발행 지점 ping-pong | 421 µs vs 423 | | 기각 (이득 없음) |
| **v4_epi** | 에필로그: residual을 x half-tile로 TMA 재적재 → smem in-place add → TMA store (기존: 32-bit ldg 64개 + stg 64개/thread) | 639K | 67.1% | 채택 (에필로그 11.7K→6.0K cycles/tile) |
| v5_shfl | quad-shuffle 16 B 정렬 LDG/STG 에필로그 | 729K | | 기각 (select/shuffle 오버헤드) |
| **v6_ring** | 스테이징을 마지막 chunk의 W1/W2 ring slot으로 옮겨 x 버퍼 즉시 해제(다음 x 대기 3.0K→0.1K) | 621K | 69.1% | 채택 |
| **v7_lds** | LN·에필로그의 generic 포인터 → 명시적 ld/st.shared | 614K | 69.8% | 채택 |
| v7_ln1 | LN 통계 원패스 | 613K | | 기각 (동일; 수치만 바뀜) |
| v7_diagW3 | 진단: W3 적재 생략(ring 트래픽 −1/3) | 605K | | L2 대역은 주 병목 아님 |
| v8_epi2 | residual 32개 LDS 일괄 발행 후 연산 | 611K | 70.2% | 채택 (미미) |
| v9/v12/v13 | consumer WG 간 시간 오프셋(배리어) | 664K / 639K / 629K | | 기각: 공유 ring이 lockstep으로 되돌림 |
| v10_asym | WG1의 chain 순서 뒤집기(b→a) | 630K | | 기각: WG1의 SiLU가 자기 chain b와 겹치지 않음 |
| **v11_rings** | ring 분리: W1/W2 4-slot(각각 더블버퍼) + W3 1-slot(별도 producer warp, chain a 발행 직후 조기 해제) | 590K | 72.7% | 채택 (W2_{j+1} 적재가 이전 W3 해제를 기다리던 병목 제거) |
| **v14_defer** | TMA store read-wait·slot 해제를 다음 타일 LN 뒤로 미룸 | 585K | 73.3% | 채택 |
| v15_split | h pack/G2 발행을 두 반으로 나눠 교차 | 590K | | 기각 |
| v16_prodln | LN을 producer WG의 유휴 warp 2개로 이동(레지스터 분할 40/232, `ln_full` mbarrier) — consumer 에필로그와 겹치기 위해 | 1022K | 42% | **기각**: warp 1개가 64행 LN에 29K cycles(행당 ~460, 40 regs로 ILP 부족·spill), consumer 232 regs에서 루프에 spill(lmem 64 B)로 루프 54K→76K. 48/232 분할은 setmaxnreg.inc가 해제된 레지스터(15360 < 16384)를 못 받아 데드락 → 40/232 필요 |
| **v17_pp** | ring A 3파트(W1/W2 교대, 슬롯 0–2) + ring B 2슬롯(W3 더블버퍼, 슬롯 3–4) + FA3식 G1 발행 ping-pong(named barrier 3/4: WG c는 3+c에서 sync 후 발행, 발행 후 상대 배리어에 arrive) | 574K | 74.7% | 채택. 단일 W3 슬롯이 lockstep 어트랙터였음(W3_{j+1} 적재가 두 WG의 G2_j를 모두 기다림). raw 프로브: WG1−WG0 오프셋 ≈960 cycles로 타일 내 안정, chunk 주기 3188(프로브 포함) |
| v17b_pp | 배리어 hand-over를 chain a 발행 직후로 | 588K | | 기각 |
| v18a_ldsm | 에필로그 ldmatrix.x4/stmatrix.x4(accumulator fragment = 8×8 b16 fragment; 명령당 512 B) | 568K | 75.5% | 채택 |
| **v18_ldsm** | + TMA store read-wait·슬롯 해제를 다음 타일 LN 통계 뒤로 지연 | 557K | 76.9% | 채택 |
| v19_order | 타일 경계 TMA 순서: residual → 다음 x → 다음 타일 W1_0/W3_0 (`res_go` mbarrier) | 566K | | 기각: residual 대기 2.4K→0.7K이지만 x 대기 +450, 루프 +430 |
| v19b_xgate | x만 residual 뒤로 | 561K vs 560K | | 기각(변화 없음) |
| v20_stats | 다음 타일 LN 통계를 chunk 루프 안에서 ldg(행당 warp 1개/chunk)로 선계산, 1 KB smem 표 | 584K | 73.5% | 기각: 출력 비트 동일하지만 루프가 chunk당 ~300 느려짐(WG 직렬 창이 한계) |
| v21_lnilp | LN 통계 셔플을 단계별 16개 독립 발행(ILP) | 559K vs 560K | | 기각(변화 없음 → 셔플 지연이 아니라 처리량/경합) |
| **v21b_resslab** | residual 재적재를 16 KB 슬랩별 mbarrier 4개로 나눠 add 루프가 슬랩 0부터 시작 | 553–555K | 77.2–77.6% | **채택(최종)** |
| v23_epiorder | G2 retire → 다음 타일 LN 통계(residual 대기 숨김) → add+store → LN apply(store 읽기 숨김) | 572K | 75.0% | 기각: x_{i+1} 대기 1.4K 노출 + 8행 분할·spill(lmem 72)로 통계/apply 느려짐 |

## 프로브(clock64)로 본 최종 타일 구조 (v21b, L384, WG-tile당 cycles, 총 59.2K; tensor 하한 49.2K)

x 대기 92 · **LN 3987**(통계 2574 + apply 1413, apply에 지연된 store read-wait/해제 포함) · **chunk loop 50.6K**(chunk당 3165 vs tensor 3072 = 97%) · G2 대기+해제 696 · residual 슬랩0 대기 658 · add loop(+슬랩 대기) 2076 · store 발행 233.
raw 타임스탬프(v17r): 두 WG의 G1 발행 오프셋 ≈960 cycles(chunk 1~15 일정) — ping-pong이 성립. 선행 WG는 chunk당 W2 대기 ≈470(3파트 ring)이지만 상대 WG의 tensor 작업 아래 숨음.
NCU(v14 --set full 재분석): lts 50%, smem 데이터 파이프 67%(LSU ld/st는 3%; 나머지 = wgmma 오퍼랜드 읽기 + TMA 쓰기), HGMMA stall = mio 43%/wait 24%. L2·smem 모두 포화 아님 → 루프 공백은 lockstep pack→G2 창이었고 v17이 이를 제거.
남은 비용: (a) LN + 에필로그 ≈ 7.7K/tile(13%) — 이 동안 tensor는 상대 WG 1개 분량만 채워짐, (b) 시작·꼬리(1152 tile/132 CTA = 8.73 → 3% 양자화 + 시작 ≈ 5%), (c) 루프 3%.

### 참고: v14 구조 (WG-tile당 62.4K)

x 대기 88 · **LN 3391**(apply 2.6K + 통계 1.0K) · store retire 202 · **chunk loop 53.9K**(chunk당 3.37K vs tensor 3.07K) · G2 대기 796 · residual 대기 819 · **add loop 2657** · store 발행 432.
chunk 내부(v11p): head 131 · W1W2 대기 238 · G1 발행(backpressure = tensor 가동) 1991 · wait<1> 14 · SiLU 463(chain b 아래 숨음) · wait<0> 115 · release+pack 318 · W3 대기 150 · G2 발행 206.
남은 tensor 공백: (a) 두 consumer WG가 lockstep이라 pack→G2 발행 창(≈450/chunk ≈ 7K/tile, 11%)에서 tensor가 놀음, (b) LN + 에필로그 ≈ 7.5K/tile(12%), (c) 시작·꼬리(1152 tile/132 CTA = 8.73 → 3% 양자화 + 시작 ≈ 5%).

LN SASS(v14): 16행/thread에 1394 명령(FFMA 400, FADD 272, SHFL 160, FMUL 160, unpack 158, F2FP 64) → 이슈만 1.4K cycles, 실측 3.4K는 shuffle 체인 지연. 원패스 통계(v7_ln1)로도 안 줄어 명령 수보다 지연이 지배. bf16x2 packed 연산은 수치(단일 fp32 반올림) 계약을 바꿔 제외.

## 확인되지 않은 가설 / 다음 후보

- LN 통계 2.6K가 셔플 배치(v21)로도 안 줄어 "상대 WG의 wgmma 오퍼랜드 읽기가 smem 파이프를 점유해 LN의 LDS/SHFL이 굶는다"는 가설. 진단 변형 `v24d_syncln`(두 WG가 LN을 동시에 시작, named barrier 5)과 프로브 `v24dp_probe`는 빌드까지만 하고 측정하지 않았다. 맞다면 LN 페이즈를 두 WG 동기화로 재배치해 ~2% 가능.
- 2-CTA cluster TMA multicast(가중치 L2→SM 트래픽 절반; 타일 경계의 TMA 큐(x 64 KB + residual 64 KB + 다음 가중치 128 KB) 완화) 예상 2–3%.
- LN 축약을 smem 전치(x half-tile을 스크래치로)로 바꿔 MIO 명령 2배 감축(버터플라이 순서 재현 시 비트 동일) 예상 1–2%.

## 90%에 못 간 이유 (구조)

smem 227 KB = x 타일 64 KB + ring 160 KB로 꽉 차 있어 (1) x 더블버퍼가 없어 LN/에필로그를 다음 타일의 MMA와 겹칠 수 없고, (2) 두 WG를 시간적으로 벌려 서로의 공백을 채우려면 ring 깊이(또는 WG별 ring → L2 트래픽 2배 ≈ 12 TB/s, 불가)가 필요하고, (3) L384의 3% 양자화는 64-row wgmma 단위로는 못 줄임. 시도한 오프셋·ping-pong·비대칭 스케줄은 모두 lockstep 어트랙터로 되돌아갔다(측정).
90%에 가려면 다른 설계가 필요: BH=32 chunk + [Wa;Wb] 패킹(n64 chain, A 1회 읽기) + 16 KB 슬롯 6~8개 + x 더블버퍼(128 KB)로 LN을 chunk 루프 사이에 끼워 넣거나, 2-CTA cluster로 가중치를 multicast하며 CTA당 행 타일 2개를 교대 처리하는 구조. 둘 다 ring lookahead가 1~1.5K cycles 수준으로 줄어 TMA 지연에 민감하므로 이득이 보장되지 않는다.

## 파일

`src/v*.cuh`(모든 변형, `v21b_resslab.cuh` = 최종; v14_defer = 1차 최종), `analyze_raw.py`(v17r raw 타임스탬프 분석), `logs/prof-v17-*.pt`(프로브 덤프), `logs/ab-r16~r22*.ncu-rep`(이번 라운드 A/B), `src/*.json`(variant spec: tm_out/prof/cluster), `build/*.cubin`, `build_nvrtc.py`, `bench_esm.py`, `ncu_ab.sh`, `drv.py`, `logs/ab-r*.ncu-rep`(cycles A/B), `logs/ncu-v{1,4,6,11,14}.ncu-rep`(--set full), `logs/final-L{384,768}.json`.
