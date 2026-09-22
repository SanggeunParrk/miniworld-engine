# B1–B4 H100 CUDA · SoL 상한 측정과 구조 실험 보고 (2026-09-20, Claude)

작업 위치: `runs/anthropic_b1b4_pipeline_20260919` (P). 인계: `CLAUDE_HANDOFF_B1B4.md`. 이전 선택안 감사: `SOL_AUDIT_20260920.md`.
**결론: 선택안 `dual_ln_prefetch`를 유지한다. 새 후보는 채택하지 않았다. production dispatch는 변경하지 않았다.**
선택안은 이 접근 패턴의 실측 메모리 하한 대비 L384 84.7% (종료 단계 보정 88.5%), L768 95.0% (보정 96.0%)다.
L768은 SoL 90%에 도달했고, L384는 도달하지 못했으며 잔여 격차의 귀속을 아래에 적었다.

측정 조건: node02 H100 80GB, CUDA 12.9 sm_90a, BF16 C128/H256, dropout 0.25, 같은 프로세스 paired explicit CUDA graph,
20 warmup + 200 samples (기준 재현은 3 블록 pooled 600). 단위 µs, median / p90. 클록 고정 없음.

## 1. 병목 (source / SASS / NCU / 트래픽 모델)

- 유효 트래픽 (2056 read + 768 write) × L² = 416.4 / 1665.7 MB. warm NCU DRAM 읽기 306 + 쓰기 123 MB (L384) → 모델과 일치.
  두 역할이 dy/gate를 각각 읽지만 tile 진행이 lockstep이라 두 번째 읽기는 L2 hit다 (이 전제는 §3의 traffic_floor로 확인).
- 선택안은 바이트가 아니라 SM 시간이 병목이다: 59.3M warp-instruction, SM당 IPC 1.5, warp 8개/SM.
  stall 샘플: barrier 29%, wait 15%, short-scoreboard 9%, long-scoreboard 7%. 명령 수 상위: 마스크 디코드(66행 5.7M),
  LN 에필로그 셔플 리덕션(132행 3.6M), tmn_ptx bf16 pack/unpack.
- 역할별 body (인계 프로브): DW 2.77 µs/64행 (WGMMA 0.83 µs 포함), DX 6.33 µs/64행. 두 역할은 159–161 µs로 균형.
- DX ragged tail: 2304 = 92×25 + 4 → DX CTA 4개가 26 tile → ≈ 6 µs가 grid barrier에서 노출된다 (L768은 16 CTA가 101 tile, ≈1%).

## 2. 후보별 구현 차이 · 예상 · 실측 · 판단

모든 후보는 ptxas spill 0 (진단 프로브 제외), L384/count132/PART2/dropout .25 strict case + 입력 변경 graph replay 2회에서
dg bit-exact 및 6출력 허용오차 통과. 채택 후보가 없어 24-case / sanitizer / saved-update replay는 재실행하지 않았다.

| 후보 | 배선 차이 | 예상 | L384 | L768 | 판단 · 근거 |
|---|---|---|---|---|---|
| dual_ds_tma | ds 행을 TMA로 smem 적재(DW slot당 16KiB×2, DX 단일 버퍼+mbarrier), 마스크 디코드 제거 | −5~8% | 175.0/181.2 | 705.6/755.8 | 기각. 196KiB 텐서를 132 SM이 같은 주소로 반복 읽어 L2 hot-spot. 레지스터 마스크 캐시가 옳음 |
| dual_ws3 / ws3b | 384 threads, WG0 producer(B1 128thr, empty 대기 후 재적재) + 2 consumer WG, setmaxnreg 40/232·56/224 | DX −15% | 245.6/251.6 (ws3b) | 937.2 | 기각. producer가 empty→TMA→full을 직렬로 대기해 DW tile ≈4.2 µs |
| dual_ws3e | producer = warp0 TMA issuer + warp1-3 B1(96thr); consumer는 empty arrive만 | DX −15% | 227.3/230.9; 비율 1:1 203.1; 5:6 182.9 | 836.0 | 기각. 프로브: DX 5.62 µs/tile(−11%) 그러나 DW 3.50(+26%). B1은 latency-bound라 3 warp로 부족 |
| dual_ws3f (+비율 44/88, 48/84, 52/80) | DX만 WS(56/224), DW는 기존 fused를 240 regs로 + 마스크 워드 smem 캐시, idle WG 24 regs | −13% | 207.6 / 189.8 / 180.3 / 179.5 | 797.9 / 741.7 | 기각. 프로브(48/84): DX 5.20, DW 3.08 µs/tile. NCU warm 174.6 vs 선택안 172.5 (동일), DRAM 75/75%, 읽기 +3%. 프로브 내부 종료 166.6 vs 이벤트 178.0 → 384-thread WS 커널의 launch/exit ≈10 µs |
| dual_dwsplit | DW dp/dg를 공유 32KiB 버퍼에 저장, B1 직후 dy/gate/proj 48KiB를 t+2로 선적재(fullA), xn/norm은 MMA 후(fullB); in-flight 96→144KiB | DW −15% | 171.3/177.0 | 644.8/697.6 | 기각. DW tile 불변 → "SM당 in-flight 한계" 가설 반증 |
| dual_dw32 | DW 32행 tile × 4 slots(48KiB), WGMMA K=32, 32행 TMA map 5개 추가 | DW −15% | 202.9/206.8 | 729.7/739.5 | 기각. tile당 barrier/WGMMA 고정비 2배 |
| dual_ln_prefetch_r240 (대조) | 선택안을 `__maxnreg__(240)`으로만 재컴파일 | 0 | 172.9/175.3 | 639.8/709.2 | L768에서 240 캡만으로 +4% → 3-WG 설계의 DW는 L768에서 구조적으로 불리 |

같은 실행의 선택안 median: 170.4–172.5 (L384), 609.3–616.2 (L768). 원자료 `claude-ab-*.json`, 로그 `claude_logs/`.

## 3. 연산 제거 메모리 하한 (진단 커널, 출력 무의미)

| 커널 | 구성 | L384 | L768 | NCU warm |
|---|---|---|---|---|
| dual_traffic_floor | 선택안 스케줄 그대로(역할·2 slot·TMA·dg/dtri 쓰기), B1/WGMMA/LN 제거 | 184.8/187.8 | 703.2/711.4 | 178.8 µs, DRAM 81.9%, 읽기 367MB(+61MB) |
| dual_stream_floor | 132 CTA 동일 역할, 32행×3 slots(64KiB), 같은 9 스트림 바이트, 연산 0 | **146.1/146.9** | **581.8/584.2** | 148.3 / 580.9 µs, DRAM 83.9% / 86.2%, 바이트 302+115MB / 1.22GB+456MB |

- traffic_floor가 선택안보다 느린 이유: 역할 pacing이 깨져 dy/gate 두 번째 읽기가 L2에서 빠짐(+61MB). 선택안의 연산은 이미 메모리 스케줄 뒤에 숨겨져 있고, 역할 lockstep이 L2 재사용의 전제임을 보여준다.
- stream_floor는 동일 바이트로 이 패턴에서 실측된 가장 빠른 시간이다 (2.85 TB/s). 스트리밍 probe(2.99–3.06)와 peak(3.35) 사이에 있다.

## 4. 같은 프로세스 기준/선택안 시간 (`measure_claude_start.py`, 3블록 pooled 600)

| 범위 | L | Triton/cuBLAS | 이전 balanced | 선택안 ln_prefetch | 배속 |
|---|---|---|---|---|---|
| B1–B4 | 384 | 301.024 / 302.944 | 175.520 / 178.432 | **172.384 / 174.944** | 1.746× |
| B1–B4 | 768 | 1184.112 / 1187.936 | 623.376 / 634.816 | **612.608 / 623.264** | 1.933× |
| 전체 backward | 384 | 1089.088 / 1091.136 | 969.200 / 972.512 | **963.440 / 966.528** | 1.130× |
| 전체 backward | 768 | 4334.944 / 4361.728 | 3798.640 / 3844.224 | **3790.400 / 3833.248** | 1.144× |

전체 backward 11 gradient 상대 L2 최대 1.63e-4 (≤5e-4). 인계 문서 수치(172.640 / 611.232)와 0.2% 이내로 재현됐다.
새 후보는 전체 backward를 측정하지 않았다 (채택하지 않았으므로).

## 5. SoL 판단 (분모를 섞지 않음)

| 분모 | L384 | L768 |
|---|---|---|
| ① 3.35 TB/s 이론 peak (124.3 / 497.2 µs) | 72.1% (NCU DRAM 74.1%) | 81.2% (NCU 82.0%) |
| ② 스트리밍 probe 2.99 / 3.06 TB/s (139.2 / 544.1 µs) | 80.7% | 88.8% |
| ③ 패턴 충실 하한 dual_stream_floor (146.1 / 581.8 µs) | **84.7%** | **95.0%** |
| ④ ③ + 선택안 종료 단계 6.5 µs (dump 2.8, grid 0.6, reduce 2.8, reset 0.3) | 88.5% | 96.0% |

- L768: ③/④ 기준 90%를 넘는다. 남은 4–5%는 종료 단계와 ragged tail이다.
- L384: 90%에 못 미친다. 잔여 ≈20 µs의 귀속: DX ragged tail ≈6 µs, 종료 단계 6.5 µs, 역할 balance/barrier ≈8 µs.
  DX 연산을 tile당 11–14% 줄인 세 변형이 총시간을 줄이지 못했고(프로브·NCU로 확인), 남은 여지는 연산이 아니라
  메모리 스케줄(역할 lockstep, 2-slot 깊이, 384-thread launch 오버헤드)에 있다. 이를 바꾸는 시도(dwsplit, dw32)는 모두 손해였다.
- 한계: ③은 3-deep 32행 파이프라인의 실측치이며 이론적 최소가 아니다. 클록 고정 없음. 모델 전체 학습 step 배속은 측정하지 않았다.

## 6. 파일 · harness 변경 · 재현

- 새 소스(P): `dual_ds_tma.cu/.py/.launch.json`, `dual_ws3.cu`, `dual_ws3b.cu`, `dual_ws3c.cu`, `dual_ws3d.cu`, `dual_ws3e.cu`(+`_r11`,`_r56`,`_probe`),
  `dual_ws3f.cu`(+`_r12`,`_r43_89`,`_r48`,`_r50`,`_r52`,`_probe`,`_probe48`), `dual_dwsplit.cu`, `dual_dw32.cu`, `dual_ln_prefetch_r240.cu`,
  `dual_traffic_floor.cu`, `dual_stream_floor.cu`. 원본과 선택안은 변경하지 않았다.
- 스크립트: `measure_claude_start.py`(기준 재현), `measure_floor.py`(정확성 검사 없는 진단 타이밍), `probe_roles.py`, `probe_end.py`.
- `dual_experiment.py` 확장(추가적·opt-in, 기존 소스 동작 불변): launch.json 키 `ds_tensor_map`, `dw_row32_maps`, `stream32_maps`(tensor map 추가 append),
  진단 전용 `B1B4_ALLOW_SPILL=1`(`*_probe` 소스에만 spill 허용). 원본 백업 `claude_logs/dual_experiment.py.orig`.
- 결과: `claude-start-paired-results.json`, `claude-ab-*.json`, `claude-floor.json`, `claude-stream-floor.json`, `claude-gap.json`,
  `claude_logs/*.log`, `claude_logs/ncu-warm-*.ncu-rep` (ws3f_r48, ln_prefetch, traffic_floor, stream_floor).
- 시각화: `b1b4-sol-experiments.svg` (실험 배선·결과 지도), HTML `site-visuals/dist/trimul.html` 섹션 `#b1b4-sol` (로컬 커밋, 미게시).

```bash
cd /home/psk6950/MiniWorld; P=runs/anthropic_b1b4_pipeline_20260919; ENV=runs/anthropic_adoption_20260919/env.sh   # node02 GPU shell
bash $ENV python -u -B $P/measure_claude_start.py
bash $ENV python -u -B $P/check_experiment.py --source dual_ws3f_r48 --lengths 384 --counts 132 --parts 2 --dropouts .25
bash $ENV python -u -B $P/measure_experiment.py --sources dual_ln_prefetch dual_ws3f_r48 --parts 2 --output claude-ab-dual_ws3f_r48.json
bash $ENV python -u -B $P/measure_floor.py --sources dual_ln_prefetch dual_stream_floor --output claude-stream-floor.json
B1B4_ALLOW_SPILL=1 bash $ENV python -u -B $P/probe_roles.py --source dual_ws3e_probe
```

GPU 사용: 별도 2-GPU 할당(job 13330, node02, 96GB)만 사용. 학습 job 13228과 다른 에이전트 job 13329는 건드리지 않았다.
