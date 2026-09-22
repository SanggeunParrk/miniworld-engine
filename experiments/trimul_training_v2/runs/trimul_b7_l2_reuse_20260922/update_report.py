from pathlib import Path
import csv
import hashlib
import json
import re

R = Path(__file__).resolve().parent
B = R.parent
groups = [
    ('TMA multicast / cluster 배치', 'trimul_b7_ring_multicast_20260922'),
    ('dX 두 행 가중치 재사용', 'trimul_b7_ring_dxpair_20260922'),
    ('dX 재사용 + LN prefetch', 'trimul_b7_ring_dxpair_prefetch_20260922'),
    ('source 두 채널 x_n 재사용', 'trimul_b7_ring_sourcepair_20260922'),
    ('source 재사용 pipeline', 'trimul_b7_ring_sourcepair_pipeline_20260922'),
    ('역할별 레지스터 배분', 'trimul_b7_ring_sourcepair_regs_20260922'),
    ('source 재사용 + SS WGMMA', 'trimul_b7_ring_sourcepair_ss_20260922'),
    ('기존 배선 + SS WGMMA', 'trimul_b7_ring_gp_ss_20260922'),
    ('초기 동기화 없는 cluster 배치', 'trimul_b7_ring_cluster_schedule_20260922'),
]
experiments = []
for label, folder in groups:
    for p in sorted((B/folder).glob('*.json')):
        if not p.name.startswith(('configs-', 'pipes-')):
            continue
        d = json.loads(p.read_text())
        if 'times' not in d:
            continue
        assert all(v['valid'] for case in d['checks'].values() for v in case.values())
        base = d['times']['ring_baseline']
        candidates = {n: dict(us=t, latency_change_pct=(t/base-1)*100)
                      for n,t in d['times'].items() if n not in ('split','ring_baseline','sourcepair_base')}
        experiments.append(dict(family=label, file=str(p.relative_to(B)), baseline_us=base,
                                checked_cases=len(d['checks']), candidates=candidates))
count = sum(len(x['candidates']) for x in experiments)
S = B/'trimul_b7_ring_cluster_schedule_20260922'
c = json.loads((S/'configs-1-10-10-0_2-10-10-0.json').read_text())
san = json.loads((S/'sanitizer-L384.json').read_text())
assert len(san) == 2 and all(x['returncode']==0 for x in san)
winner = 'c2g10u10m0'
t = c['times']
gain = (1-t[winner]/t['ring_baseline'])*100
split_gain = (1-t[winner]/t['split'])*100
paths = re.findall(r'CUBIN (\S+/trimul_b7_ring_cluster_schedule_20260922/build/\S+\.cubin)', (S/'memcheck-L384.log').read_text())
assert len(paths) == 1
cubin = Path(paths[0])
assert str(cubin) in (S/'job-15134.log').read_text()
assert str(cubin) in (S/'racecheck-L384.log').read_text()
assert str(cubin) in (S/'ncu-L384-mode52-p64-r12.log').read_text()
for f, h in c['source_sha256'][winner].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h
selection = dict(status='experimental_validated_small_gain', scope='B7-B12 bidirectional TriMul L384 only',
    plan='../trimul_b7_ring_cluster_schedule_20260922/plan.py', kwargs=dict(clusters=10,mode=52),
    environment=dict(B7_HW_CLUSTER='2',B7_MULTICAST='0',B7_CONSUMERS='10',B7_RING_DEPTH='12',B7_PRODUCER_REGS='64'),
    source_sha256=c['source_sha256'][winner], cubin=dict(path=str(cubin),sha256=hashlib.sha256(cubin.read_bytes()).hexdigest()),
    times_us=t, time_reduction_pct=gain, target_us=252,target_met=False,sol90_verified=False,
    benchmark='../trimul_b7_ring_cluster_schedule_20260922/configs-1-10-10-0_2-10-10-0.json',
    sanitizer='../trimul_b7_ring_cluster_schedule_20260922/sanitizer-L384.json',production_dispatch_changed=False)
(R/'selected.json').write_text(json.dumps(selection,indent=2)+'\n')

profile_files = {
    'previous': B/'trimul_b7_ring_depth_20260922/ncu-L384-mode52-p64-r12.csv',
    'dx_reuse': B/'trimul_b7_ring_dxpair_20260922/ncu-L384-mode52-p64-r20.csv',
    'source_reuse': B/'trimul_b7_ring_sourcepair_regs_20260922/ncu-L384-mode52-p24-r6.csv',
    'selected': S/'ncu-L384-mode52-p64-r12.csv',
}
profiles = {}
keys = ['gpu__time_duration.sum','lts__t_sectors.sum','dram__bytes_read.sum','dram__bytes_write.sum',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed','gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
        'lts__throughput.avg.pct_of_peak_sustained_elapsed','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed']
for name,p in profile_files.items():
    a=list(csv.DictReader(p.open())); profiles[name]={k:dict(value=a[-1][k],unit=a[0][k]) for k in keys if k in a[-1]}
(R/'results.json').write_text(json.dumps(dict(candidate_count=count,experiments=experiments,profiles=profiles),indent=2,ensure_ascii=False)+'\n')

readme = f'''# B7–B12: L2 재사용 구조 실험 및 CTA cluster 배치

최종 선택은 multicast 없이 2-CTA hardware cluster로 배치하고 불필요한 초기 cluster sync를 제거한 후보다.
수식, 저장값, ring 크기, WGMMA 연산 순서, 단일 cooperative kernel 범위를 유지한다.

## 최종 같은 실행 비교 (5×160회)

| 구현 | µs |
| --- | ---: |
| 이전 선택 | {t['ring_baseline']:.3f} |
| 1-CTA cluster 속성 | {t['c1g10u10m0']:.3f} |
| 새 2-CTA 배치 | {t[winner]:.3f} |
| 분리형 | {t['split']:.3f} |

이전 선택 대비 {gain:.2f}%, 분리형 대비 {split_gain:.2f}% 시간 단축.
5개 측정 라운드 모두 이전 선택보다 빨랐다. 크기가 작은 이득이며 과거 348.11 µs 기록과 절대 시간을 섞어 비교하지 않는다.
252 µs 목표는 미달. 전체 학습 또는 L768 결과가 아니다.

## 구조 실험

총 {count}개 설정/대조군을 비교했다. 각 3~5개 입력 변형에서 정확도·eager/graph·scratch poison·flag 재사용 검사를 통과했다.
다양한 구조 실험의 커널들은 보관하되 선택하지 않았다.

- TMA multicast: 2/4 CTA가 x_n을 공유. 준비 barrier와 cluster 동시 실행 한도 비용으로 전체 시간은 증가했다.
- dX 재사용: 한 CTA가 두 행 타일을 계산하며 주 경로 가중치 load를 공유. 두 번째 누산 결과는 shared에만 보관한다.
- source 재사용: 한 CTA가 두 채널 타일을 계산하며 x_n을 공유. 두 dW 누산기를 유지해 레지스터 압박이 증가했다.
- 위 구조들에서 CTA 비율, ring 크기, 레지스터 분배, GEMM pipeline, SS WGMMA를 비교했다.
- 기존 배선에 SS WGMMA만 적용한 후보도 더 느렸다.
- 4-CTA cluster 최초 설정은 occupancy API의 248 CTA 한도보다 커 실행 전에 제외했다. 240/224 CTA로 수정해 검증했다.

## NCU 해석

별도 프로파일에서 기존 L2 sectors 약 75.92M → dX 재사용 69.63M, source 재사용 75.81M.
dX의 특정 가중치 load는 절반이 되었지만 커널 전체 L2 트래픽 감소는 약 8.3%였다.
동시에 dX 재사용의 HBM write는 144.61 → 334.46 MB로 증가했다. source 재사용은 L2 총량이 거의 줄지 않았다.
따라서 반복 읽기를 줄이는 수식상의 계산만으로 전체 성능 이득을 주장하지 않는다.

새 선택의 NCU: SM 37.45%, memory 58.14%, L2 80.00%, HBM 40.15%.
L2 처리율 상승은 더 빠른 시간만의 효과가 아니다. L2 sectors도 약 80.86M로 늘었다.
SoL90은 미달이며 이 수치가 알고리즘의 최소 실행 시간 대비 달성률은 아니다.
프로파일 latency 339.296 µs는 위 graph 교차 측정과 별도이며 서로 비교하지 않는다.

## 검증·재현

새 선택은 5개 변형 입력의 7개 출력 검사 통과. 동일 cubin의 memcheck/racecheck 오류 0건.
선택 cubin의 ptxas spill 0 bytes. 소스·cubin SHA-256 및 환경: selected.json.
NCU·실험 원본 경로: results.json. production dispatch 변경 없음.
Anthropic 유래 Apache-2.0 TMA/WGMMA primitives에 기반한 학습 확장이다.
'''
(R/'README.md').write_text(readme)
trs=[]
for x in experiments:
    best=min(x['candidates'],key=lambda k:x['candidates'][k]['us']);v=x['candidates'][best]
    trs.append(f'<tr><td>{x["family"]}</td><td>{len(x["candidates"])}</td><td>{x["baseline_us"]:.2f}</td><td>{v["us"]:.2f}</td><td>{v["latency_change_pct"]:+.2f}%</td><td><a href="../{x["file"]}">원본</a></td></tr>')
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B7–B12 · L2 재사용 실험</title><style>body{{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui;color:#20374b;background:#f4f7fb}}section{{padding:20px;background:white;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ccd6df;text-align:left}}.flow{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.box{{background:#e9f4ef;padding:15px;border-radius:8px}}a{{color:#1564aa}}small{{color:#526779}}@media(max-width:700px){{.flow{{grid-template-columns:1fr}}}}</style>
<h1>B7–B12 · 재사용 구조 실험 후 CTA 배치만 채택</h1><p>L384 · H100 · 단일 kernel · 2026-09-22</p>
<section><h2>최종 반복 측정</h2><table><tr><th>구현</th><th>시간</th></tr><tr><td>이전 선택</td><td>{t['ring_baseline']:.2f} µs</td></tr><tr><td><b>새 2-CTA 배치</b></td><td><b>{t[winner]:.2f} µs</b></td></tr><tr><td>분리형</td><td>{t['split']:.2f} µs</td></tr></table><p>같은 실행에서 이전 선택보다 <b>{gain:.2f}%</b> 시간 단축. 5라운드 모두 개선. 252 µs 목표는 미달.</p><small>5×160회 교차 측정. 과거 348.11 µs는 다른 실행의 수치다. 전체 학습 성능이 아니다.</small></section>
<section><h2>무엇을 만들었나</h2><div class="flow"><div class="box"><b>실험: source 재사용</b><p>x_n 한 번 load → 두 채널의 projection/gate 재계산 → 두 dW 누산기 + ring dp/dg</p><p>레지스터 압박과 제어 비용이 증가했고, NCU L2 총량은 거의 줄지 않았다. 미채택.</p></div><div class="box"><b>실험: dX 재사용</b><p>가중치 한 번 load → 두 행 타일의 dX_n 계산 → shared에 한 행 누산값 보관 → 행별 LN 미분</p><p>주 경로 가중치 load는 절반. 전체 L2 sectors는 약 8.3% 감소했으나 HBM 쓰기가 증가했다. 미채택.</p></div></div><p><b>채택:</b> 기존 연산·데이터 이동 배선을 유지하고 CTA를 2개씩 hardware cluster로 배치. multicast와 초기 cluster 동기화는 사용하지 않는다.</p></section>
<section><h2>{count}개 설정 및 대조군</h2><p>각 행은 별도 실행. 시간은 µs이며 변화율은 같은 행의 기준 대비다. 양수는 느림.</p><table><tr><th>실험</th><th>설정 수</th><th>기준</th><th>가장 빠른 후보</th><th>시간 변화</th><th>자료</th></tr>{''.join(trs)}</table></section>
<section><h2>NCU·검증</h2><p>선택 후보: SM 37.45% · 메모리 58.14% · L2 80.00% · HBM 40.15%. SoL90 미달.</p><p>L2 처리율 상승만으로 효율이 개선됐다고 해석하면 안 된다. 선택 후보는 L2 요청량도 늘었으며, 성능 판단은 같은 실행의 시간 비교를 사용했다.</p><p>5개 입력/가중치/마스크 변형 검사, memcheck/racecheck 통과. 동일 cubin 확인. spill 없음.</p><p><a href="selected.json">선택 설정·SHA-256</a> · <a href="results.json">실험·프로파일 원본</a> · <a href="README.md">상세 기록</a> · <a href="../../TRIMUL_STATUS.html">전체 현황</a></p><small>Anthropic 유래 Apache-2.0 primitives 기반. production dispatch 변경 없음.</small></section></html>'''
(R/'index.html').write_text(page)
root=B.parent/'TRIMUL_STATUS.html';s=root.read_text();s=re.sub(r'<aside id="b7-l2-reuse-20260922">.*?</aside>','',s,flags=re.S)
note=f'<aside id="b7-l2-reuse-20260922"><b>최신 B7–B12:</b> 재사용 구조 실험 후 2-CTA 배치만 채택. 같은 실행에서 {t["ring_baseline"]:.2f} → {t[winner]:.2f} µs ({gain:.2f}% 단축). L2 80.00%, SoL90·252 µs 미달. <a href="runs/trimul_b7_l2_reuse_20260922/index.html">비교·구조·NCU·검증</a></aside>'
root.write_text(s.replace('</header>','</header>'+note,1))
print('variants',count,'gain_pct',gain,'split_gain_pct',split_gain)
