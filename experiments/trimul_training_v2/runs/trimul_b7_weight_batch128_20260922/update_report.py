from pathlib import Path
import csv,hashlib,json,re,html
R=Path(__file__).resolve().parent;B=R.parent
bench=R/'configs-12-24_12-32_10-32.json'
r=json.loads(bench.read_text());name='v12p32';t=r['times']
assert len(r['checks'])==5 and len(r['rounds'])==5
assert all(x[name]['valid'] for x in r['checks'].values())
for p,h in r['source_sha256'][name].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
paths=re.findall(r'CUBIN (\S+/trimul_b7_weight_batch128_20260922/build/\S+\.cubin)',(R/'job-15190.log').read_text())
assert len(paths)==3
cubin=Path(paths[1]);log=cubin.with_suffix('.ptxas.log').read_text()
assert '0 bytes spill stores, 0 bytes spill loads' in log and '0 bytes stack frame' in log
assert 'Potential Performance Loss' not in log
san=json.loads((R/'sanitizer-L384.json').read_text())
assert len(san)==2 and all(x['returncode']==0 for x in san)
for f in ('memcheck-L384.log','racecheck-L384.log','ncu-L384-mode52-p32-r12.log'):
 assert str(cubin) in (R/f).read_text(),f

def metrics(folder,filename):
 p=B/folder/filename;rows=list(csv.DictReader(p.open()));u,v=rows[0],rows[-1]
 keys=['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed','gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed','lts__throughput.avg.pct_of_peak_sustained_elapsed','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','lts__t_sectors.sum','dram__bytes_read.sum','dram__bytes_write.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum','smsp__inst_executed.sum']
 return dict(path=str(p),values={k:float(v[k].replace(',','')) for k in keys if v.get(k)},units={k:u.get(k) for k in keys})
profiles={
 'previous':metrics('trimul_b7_ring_cluster_schedule_20260922','ncu-L384-mode52-p64-r12.csv'),
 'two_wg':metrics('trimul_b7_two_wg_20260922','ncu-L384-mode52-p24-r12.csv'),
 'two_wg_lowreg':metrics('trimul_b7_two_wg_lowreg_20260922','ncu-L384-mode52-p24-r12.csv'),
 'selected':metrics(R.name,'ncu-L384-mode52-p32-r12.csv'),
}
folders=['trimul_b7_register_pack_20260922','trimul_b7_gate_half_20260922','trimul_b7_384_n64_20260922','trimul_b7_256_n64_20260922','trimul_b7_two_wg_20260922','trimul_b7_two_wg_uniform_20260922','trimul_b7_two_wg_lowreg_20260922','trimul_b7_two_wg_register_budget_20260922','trimul_b7_two_wg_batch128_20260922','trimul_b7_two_wg_static_20260922',R.name]
experiments=[]
for folder in folders:
 for p in sorted((B/folder).glob('*.json')):
  if p.name in ('results.json','selected.json'):continue
  d=json.loads(p.read_text())
  if not isinstance(d,dict) or 'times' not in d or 'ring_baseline' not in d['times']:continue
  for n,value in d['times'].items():
   if n in ('split','ring_baseline'):continue
   assert all(c[n]['valid'] for c in d['checks'].values())
   experiments.append(dict(folder=folder,record=str(p),variant=n,us=value,baseline_us=d['times']['ring_baseline'],time_reduction_pct=100*(1-value/d['times']['ring_baseline']),cases=len(d['checks'])))
gain=100*(1-t[name]/t['ring_baseline']);splitgain=100*(1-t[name]/t['split'])
rounds=[dict(previous=q['ring_baseline']['median_us'],selected=q[name]['median_us'],time_reduction_pct=100*(1-q[name]['median_us']/q['ring_baseline']['median_us'])) for q in r['rounds']]
assert all(q['selected']<q['previous'] for q in rounds)
selected=dict(status='experimental_validated_small_gain',scope='L384 bidirectional B7-B12 only',plan=str(R/'plan.py'),kwargs=dict(clusters=10,mode=52),environment=dict(B7_CONSUMERS='10',B7_PRODUCER_REGS='32',B7_RING_DEPTH='12',B7_HW_CLUSTER='2',B7_MULTICAST='0'),source_sha256=r['source_sha256'][name],cubin=dict(path=str(cubin),sha256=hashlib.sha256(cubin.read_bytes()).hexdigest()),times_us=t,time_reduction_pct=gain,rounds=rounds,benchmark=str(bench),sanitizer=str(R/'sanitizer-L384.json'),profile=profiles['selected']['path'],target_us=252,target_met=False,sol90_verified=False,production_dispatch_changed=False)
(R/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
(R/'results.json').write_text(json.dumps(dict(selected=selected,experiments=experiments,profiles=profiles,compile_rejection='384-thread 2-CTA/SM N128 control: ptxas needed >=90 registers but target was 80; not benchmarked'),indent=2)+'\n')
m=profiles['selected']['values']
readme=f'''# B7–B12: 가중치 K128 전송 파이프라인

L384 양방향 B7–B12 단일 CUDA 커널. 이번 실험의 선택은 기존 2-CTA 배치에서 가중치 전송 단위를 K64→K128로 바꾼 구성이다.
Anthropic 유래 TMA/WGMMA primitives에 기반한 학습 확장이라는 기존 기조와 Apache-2.0 설명을 유지한다.

## 최종 같은 실행 비교

| 구현 | µs |
| --- | ---: |
| 이전 선택: 2-CTA, K64 weight stages | {t['ring_baseline']:.3f} |
| 새 선택: K128, ring12, producer32 | {t[name]:.3f} |
| K128, ring12, producer24 | {t['v12p24']:.3f} |
| K128, ring10, producer32 | {t['v10p32']:.3f} |
| 분리형 | {t['split']:.3f} |

이전 선택 대비 {gain:.2f}%, 분리형 대비 {splitgain:.2f}% 시간 단축. 5×160 교차 graph 측정의 통합 중앙값이다.
5개 라운드 모두 이전 선택보다 빨랐다. 과거 다른 GPU/클록 조건의 절대 µs와 섞어 비교하지 않는다.
252 µs·SoL90은 미달. B1–B4, 전체 모듈, L768의 새 측정은 아니다.

## 선택 배선

| 항목 | 이전 | 새 선택 |
| --- | --- | --- |
| derivative shared slots | 16 KiB × 4 = 64 KiB | 16 KiB × 2 = 32 KiB |
| weight shared slots | 16 KiB × 2 = 32 KiB | 32 KiB × 2 = 64 KiB |
| 별도 dGate 공간 | 16 KiB | 16 KiB |
| dynamic shared 합계 | 112 KiB | 112 KiB |
| weight ready/empty 단계 / 행 타일 | 16 | 8 |
| weight TMA load 명령 / 행 타일 | 16 | 16 |
| WGMMA commit / 주 dX 행 타일 | 16 | 8 |
| 주 dX WGMMA wait / 행 타일 | 8 | 8 |
| producer/compute register budget | 64/192 | 32/224 |

WGMMA N128, 누산 순서, 반올림, 수식, global ring, dW 장기 누적, 260 CTA/256 threads, hardware cluster2를 유지한다.
가중치의 논리 payload는 유지하며 두 load를 한 ready 단계로 묶어 다음 K128 묶음을 미리 읽는다.
raw-x/residual/output-gate 전송은 마지막 main GEMM 소비 완료 후 alias 공간에 넣는다.
기존 LN 입력 선행 전송과의 차이까지 포함한 실제 측정이다. 변경 각각의 단독 효과를 분리한 결과는 아니다.
occupancy API는 cluster2 132개까지 반환했고 260 CTA로 실행했다. 선택 cubin은 stack/spill 0, WGMMA serialization 경고 없음.

## 다른 구조 실험

총 {len(experiments)}개 측정 항목(대조군·재측정 포함)을 기록했다. 각 항목은 3~5개의 mutation/eager/graph/poison/flag 검사를 통과했다.

- BF16 포장, GP 결과 수명 단축, N64 gate, 레지스터 배분 변경만으로는 안정적인 이득을 얻지 못했다.
- 384-thread 제어군: N128 WGMMA는 80-register target에서 컴파일 실패. N64로 바꿔 실행한 제어군도 약 5% 느렸다.
- 두 compute WG의 weight 공유를 실제 구현했다. 첫 버전 1031 µs vs 같은 실행의 기존 340 µs. local spill 요청이 크게 증가했다.
- gate N32 및 LN fragment 수명을 줄인 버전은 581 µs vs 같은 실행의 기존 342 µs. 개선됐지만 선택 기준보다 느렸다.
- 추가 registers/1-CTA residency, K128 배치, WG 역할 template 특수화도 비교했고 더 빠르지 않았다.
- 두-WG 후보의 파이프라인·spill·명령 증가를 확인한 뒤 기존 배치에 K128만 적용해 위 이득을 얻었다.

## NCU

선택 cubin 별도 profile: SM {m['sm__throughput.avg.pct_of_peak_sustained_elapsed']:.2f}%, memory {m['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed']:.2f}%, L2 {m['lts__throughput.avg.pct_of_peak_sustained_elapsed']:.2f}%, HBM {m['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed']:.2f}%.
local load/store sectors: {m['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum']:.0f}/{m['l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:.0f}.
별도 측정의 hardware throughput 지표이며, 알고리즘의 달성 가능한 최소 시간 대비 효율을 측정한 값은 아니다.'''
readme+='''
profile latency와 위 CUDA-event latency를 직접 섞어 비교하지 않는다. `results.json`에 이전/두-WG/lowreg/선택 profile과 단위를 보관했다.

## 검증 / 제한

선택 cubin의 5개 변형 입력·7개 출력 검사, 반복 graph, scratch poison, counter/flag 초기화 검사 통과.
원래 relative-L2 한도를 유지했다. 같은 cubin의 memcheck/racecheck 모두 0 errors.
추가로 producer24 후보도 sanitizer를 통과했으며 `*-p24.*`로 기록을 보관했다.
production dispatch와 upstream selector는 변경하지 않았다. commit/push/온라인 게시 없음.

재현: `selected.json`의 환경으로 `Plan(..., clusters=10, mode=52)` 실행.
`run.slurm --configs 12:24,12:32,10:32 --cases 5 --rounds 5 --iterations 160`이 최종 비교다.
`prepare.py`는 후보 생성 기록이다. 직접 재현에는 현재 `sweep.py`와 `plan.py`를 사용한다.
'''
(R/'README.md').write_text(readme)
table=''.join(f'<tr><td>{html.escape(n)}</td><td>{v:.2f}</td></tr>' for n,v in [('분리형',t['split']),('이전 선택',t['ring_baseline']),('새 선택',t[name])])
trials=''.join(f'<tr><td>{html.escape(e["folder"].replace("trimul_b7_", ""))}<br>{e["variant"]}</td><td>{e["baseline_us"]:.2f}</td><td>{e["us"]:.2f}</td><td>{e["time_reduction_pct"]:+.2f}%</td></tr>' for e in experiments)
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B7–B12 K128 가중치 파이프라인</title><style>body{{max-width:1100px;margin:24px auto;padding:0 18px;background:#f5f7fa;color:#21384c;font:16px/1.65 system-ui}}section{{padding:20px;background:white;border:1px solid #d4dfeb;border-radius:10px;margin:18px 0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dae3ed}}a{{color:#1b60a0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.box{{background:#e7f2f8;padding:16px;border-radius:8px}}.good{{background:#e3f2e8}}.warning{{padding:14px;background:#fff1d7}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}table{{font-size:13px}}}}</style>
<a href="../../TRIMUL_STATUS.html">← 현황판</a><h1>B7–B12: 추가 {gain:.2f}% 시간 단축</h1><p class="warning">L384 양방향 B7–B12 한정. 252 µs·SoL90 미달. 전체 학습의 새 측정은 아닙니다.</p>
<section><h2>같은 실행의 5×160 교차 측정</h2><table><tr><th>구현</th><th>µs</th></tr>{table}</table><p>5개 라운드 모두 이전 선택보다 빨랐습니다. 과거 다른 조건의 절대 시간과 섞지 않습니다.</p></section>
<section><h2>선택: 총 shared memory를 유지하며 전송 단계를 재배치</h2><div class="grid"><div class="box"><b>이전 · K64</b><p>derivative: 16 KiB × 4</p><p>weight: 16 KiB × 2</p><p>dGate: 16 KiB</p><p>weight ready/empty 16단계</p></div><div class="box good"><b>새 선택 · K128</b><p>derivative: 16 KiB × 2</p><p>weight: 32 KiB × 2</p><p>dGate: 16 KiB</p><p>weight ready/empty 8단계</p></div></div><p>입력 ring → shared derivative + 미리 읽은 weight → N128 WGMMA → gate·LN 미분·residual → dX. source의 dp/dg·dW 계산은 유지합니다.</p><p>양쪽 모두 dynamic shared 112 KiB, 260 CTA, 단일 호출. weight 전송량과 TMA load 명령 수는 유지합니다. 선택 cubin의 stack/spill 0, WGMMA serialization 경고 없음.</p></section>
<section><h2>두 계산 그룹 공유도 구현·분석했습니다</h2><p>첫 후보 1031 µs, 임시값을 줄인 후보 581 µs. 각각의 같은 실행 기준은 약 340/342 µs였습니다. 두 번째 누산기를 shared로 옮기지 않아도 레지스터 부족에 따른 local spill과 WGMMA 직렬화가 생겼습니다.</p><p>NCU local load+store sectors는 첫 후보 약 47.29M → lowreg 약 4.32M으로 줄었지만 기존 선택보다 느렸습니다. 두-WG 구조는 채택하지 않았습니다.</p></section>
<section><h2>선택 cubin NCU·검증</h2><p>SM {m['sm__throughput.avg.pct_of_peak_sustained_elapsed']:.2f}% · memory {m['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed']:.2f}% · L2 {m['lts__throughput.avg.pct_of_peak_sustained_elapsed']:.2f}% · HBM {m['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed']:.2f}%.</p><p>이는 hardware throughput 지표이며 달성 가능한 알고리즘 상한 대비 효율과 동일하지 않습니다.</p><p>5개 변형 입력/7개 출력, eager/graph/poison/flag 검사 통과. 동일 cubin memcheck·racecheck 0 errors. production dispatch·commit·push 변경 없음.</p><p><a href="README.md">상세 설명</a> · <a href="selected.json">선택·SHA-256·환경</a> · <a href="results.json">모든 측정·NCU 원본 경로</a> · <a href="../trimul_b7_resource_design_20260922/index.html">앞선 설계안</a></p></section>
<section><h2>개별 실험: 각 행은 자체 기준과 비교</h2><p>대조군과 재측정을 포함합니다. µs를 행 사이에서 직접 비교하지 않습니다.</p><table><tr><th>실험</th><th>같은 실행 기준</th><th>후보 µs</th><th>시간 단축</th></tr>{trials}</table></section></html>'''
(R/'index.html').write_text(page)
status=B.parent/'TRIMUL_STATUS.html';s=status.read_text();marker='b7-weight-batch128-20260922'
if marker not in s:
 block=f'<aside id="{marker}"><b>최신 B7–B12:</b> K128 가중치 파이프라인. 같은 실행에서 {t["ring_baseline"]:.2f} → {t[name]:.2f} µs ({gain:.2f}% 단축). 정확도·memcheck·racecheck 통과, spill 0. 252 µs·SoL90 미달. <a href="runs/{R.name}/index.html">배선·성능·실패 구조·NCU</a></aside>'
 status.write_text(s.replace('</header>','</header>'+block,1))
print(json.dumps(dict(gain_pct=gain,split_gain_pct=splitgain,experiments=len(experiments),selected=selected['cubin'],ncu=m),indent=2))
