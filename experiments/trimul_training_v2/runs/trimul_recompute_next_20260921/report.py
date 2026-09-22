"""Build the private dashboard from frozen measurements, without changing baselines."""
from pathlib import Path
import csv,hashlib,json,re,shutil
R=Path(__file__).resolve().parent
SITE=R.parent/'anthropic_b1b4_pipeline_20260919/site-visuals';A=SITE/'dist/assets'
D={n:json.loads((R/('results-L%d.json'%n)).read_text()) for n in (384,768)}
labels={'saved':'Anthropic 파생 저장형','initial_recompute':'최초 내부 재계산','previous_recompute':'직전 재계산 최적화','optimized_recompute':'이번 선택'}
for j in D.values():
 for path,h in j['metadata']['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 for v in j['metadata']['cubins'].values():assert hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256']
 assert all(e['bit_exact'] for e in j['graph_vs_fresh_eager'].values())
 assert j['activation_restore_kernels']==0 and j['contraction_calls']==6
for name in ('memcheck-b1-L384.log','memcheck-module-L768.log'):
 assert 'ERROR SUMMARY: 0 errors' in (R/name).read_text(),name
assert 'RACECHECK SUMMARY: 0 hazards displayed' in (R/'racecheck-b1-L384.log').read_text()
T=[];K=[];B=[];NCU=[];C=[]
for n,j in D.items():
 for name,label in labels.items():T.append([str(n),label]+['%.3f'%(j['times'][s][name]['median_us']/1000) for s in ('forward','backward','forward_backward')])
 for kind,kn in [('B1–B4','b1_fused'),('B7–B12','front_b7b12')]:
  vals=[next(t['us'] for t in j['traces']['backward'][name] if t['name']==kn) for name in ('previous_recompute','optimized_recompute')]
  K.append([str(n),kind]+['%.1f'%v for v in vals]+['%.1f%%'%(100*(1-vals[1]/vals[0]))])
 for kind in ('b1','b7'):
  jj=json.loads((R/('paired-%s-L%d.json'%(kind,n))).read_text());B.append([str(n),kind]+['%s: %.2f μs'%(k,v['median_us']) for k,v in jj['times'].items()])
 raw=list(csv.DictReader((R/('ncu-b1-L%d-raw.csv'%n)).open()))[-1]
 det={x['Metric Name']:x for x in csv.DictReader((R/('ncu-b1-L%d-details.csv'%n)).open()) if x.get('Metric Name')}
 v=dict(L=n,region='B1–B4',dram=float(det['DRAM Throughput']['Metric Value']),sm=float(raw['sm__throughput.avg.pct_of_peak_sustained_elapsed']),tensor=float(raw['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']),occupancy=float(det['Achieved Occupancy']['Metric Value']),no_eligible=float(det['No Eligible']['Metric Value']))
 NCU.append(v)
 for k,cc in j['metadata']['cubins'].items():
  s=Path(cc['path']).with_suffix('.ptxas.log').read_text();assert not re.search(r'(?<!\d)[1-9]\d* bytes (?:spill|stack)',s)
  C.append(dict(L=n,kind=k,registers=[int(x) for x in re.findall(r'Used (\d+) registers',s)],stack_and_spills_zero=True,sha256=cc['sha256']))
(R/'ncu-summary.json').write_text(json.dumps(NCU,indent=2));(R/'compiler-summary.json').write_text(json.dumps(C,indent=2))
def mdtable(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(r)+' |\n' for r in rows)
def ht(head,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
head=['L','경로','Forward ms','Backward ms','전체 실측 ms'];kh=['L','구간','직전 μs','이번 μs','시간 감소'];nh=['L','DRAM','SM','Tensor pipe','Occupancy','No eligible']
N=[[str(x['L'])]+['%.1f%%'%x[k] for k in ('dram','sm','tensor','occupancy','no_eligible')] for x in NCU]
summary=[]
for n,j in D.items():
 t=j['times']['forward_backward'];new=t['optimized_recompute']['median_us'];prev=t['previous_recompute']['median_us'];orig=t['initial_recompute']['median_us'];saved=t['saved']['median_us']
 summary.append('L%d: 직전 대비 %.1f%% 시간 감소 (%.3f배), 최초 재계산 대비 %.3f배. 저장형 대비 지연은 아직 %.1f%% 더 큼.'%(n,100*(1-new/prev),prev/new,orig/new,100*(new/saved-1)))
report='''# TriMul 재계산 배선: dWproj 불필요한 gate 계산 제거 · 2026-09-21

## 결과

'''+ '\n\n'.join(summary)+'''

**1.7배 미달 / SoL90 미입증. Production 기본값은 변경하지 않았다.** 같은 H100 node02에서 네 경로를 동시에 비교한 결과다. 공개 Anthropic inference나 cuEquivariance 대비 새 성능 주장이 아니다.

- saved: Anthropic 파생 저장형 학습 forward + 기존 최적화 CUDA backward.
- initial_recompute: `trimul_fused_recompute_20260920` 최초 내부 재계산.
- previous_recompute: `trimul_recompute_optimized_20260920` 직전 선택.
- optimized_recompute: 이 디렉터리의 `selected-L384.json`, `selected-L768.json`.

BF16 / batch1 / C128 / outgoing·incoming 각 hidden128 / L384·768 / dropout25% / mask·residual / 모든 11개 gradient. 경로별·범위별 CUDA graph 600회 교대 측정. Live weight packing 포함. Optimizer, RNG 생성, CPU dispatch, 컴파일 제외. 전체 시간은 fwd+bwd를 별도 graph에서 직접 측정한 값이다.

'''+mdtable(head,T)+'''
## 변경한 배선

Forward는 동일한 Anthropic 파생 no-save K1 → cuBLAS 두 번 → no-save K3다. 유지하는 forward activation은 left/right/tri. x_n, projection/gate, LN 통계는 backward CUDA 안에서 다시 계산한다. Global activation 복원 커널은 0개다. Gradient 전달 텐서와 dW reduction scratch의 global 메모리는 계속 필요하다.

**B1–B4의 dWproj 역할에서만 재계산을 줄였다.** 기존에는 각 projection 출력 절반을 맡은 CTA에서도 output gate 128채널을 전부 계산했다. 이제 해당 CTA가 실제 쓰는 64채널만 계산한다. WG0은 input LN → 필요한 gate64 → dy·dropout·gate 곱을 처리한다. WG1은 그동안 output LN256을 처리한다. CTA 동기화 후 두 WG가 dWproj WGMMA를 실행한다. 재계산 activation의 새로운 HBM 저장은 없다.

CTA 배분은 dWgate36 / dWproj52 / dtri44에서 **36 / 48 / 48**로 바꿨다. Row64, 256 threads, 총132 CTA, shared227840 B. `DW_SPLITS=24`, `GATE_SPLITS=36`, `TRAIN_L=384/768`, `B1_DWPROJ_PIPE=1`. 새 B1 ptxas는255 registers, stack/spills0. Gate/projection 반올림과 미분 계산식을 유지한다. dW reduction partition 변경으로 weight gradient의 FP32 합산 묶음은 달라지며 수치 오차 검증으로 확인했다.

**B7–B12는 직전 선택을 유지한다.** 64 dW CTA +68 dX CTA, TMA producer1+consumer2, dX 방향별 resident packed weights128 KiB. 후보가 안정적인 추가 이득을 주지 못해 active B7 source와 plan 코드를 이전 스냅샷으로 되돌렸다. 실험 소스는 `rejected-b7/`에 별도 보관했다.

B5–B6 cuBLAS 네 번 유지. Forward 중간값의 논리적 유지량은 L384 226.492 MB, L768 905.970 MB로 직전 재계산과 같다. 저장형은719.585 /2878.341 MB. Input/weights/mask/output 제외 tensor-size 합계로, peak GPU memory 실측은 아니다.

## 구간별 trace

아래는 별도 profiler trace이며 graph 중앙값과 차이가 있을 수 있다.

'''+mdtable(kh,K)+'''
## 동일 버퍼 800회 교대 비교

'''+mdtable(['L','구간','후보1','후보2','후보3'],B)+'''

- B1 partition_only는 CTA 배분만 변경. projection_pipe가 실제 선택이다.
- B7 stats는 LN 통계 shared 재사용, stats_early_mask는 추가로 weight TMA 조기 발행 + mask hoist. Stats만의 차이는 양쪽 L 모두1% 미만. Combined 후보는 L768에서 역효과. 미채택.
- B7 consumer3은 ptxas spill 때문에 실행 전에 제외. Shared-input WGMMA로 register lifetime을 줄이는 시도도 consumer3 spill을 없애지 못했고 consumer2에서는 더 느렸다.
- B1 input-gradient / gate-weight 전용 중첩과12주기 dropout mask register cache도 더 느려 미채택. Config 공간을 모두 탐색했다는 주장은 하지 않는다.

## NCU: B1–B4 최종 커널

'''+mdtable(nh,N)+'''

`--set full`, 추가 tensor-pipe metric, cache-control none, clock-control none. NCU instrumented 시간 대신 위 CUDA graph 시간을 성능 판단에 쓴다. 낮은 eligible warp 비율과 tensor 활용률이 남는다. SM/DRAM 사용률을 알고리즘 SoL 비율과 동일시할 수 없으며 **SoL90 달성을 입증하지 못했다**. B7은 소스가 동일해 이번에는 다시 NCU를 수집하지 않았다. 직전 프로파일은 별도 이전 보고서에 유지한다.

## 재현

node02의 할당받은 H100에서 `runs/anthropic_adoption_20260919/env.sh`로 cu128 환경을 실행한다. `MINIWORLD_TRIMUL_TRAIN_BUILD_DIR`은 `/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build`로 지정한다. `bench.py --length 384` 또는 `--length 768`이 최종 선택 설정과 네 비교 경로를 로드한다. `paired_b1.py --length N`은 같은 버퍼의 B1 후보 비교다. B7 후보 소스는 active code에서 제외했으므로 당시 실험 재현에는 `rejected-b7/` 스냅샷이 필요하다. 검증 명령은 `verify.py`에 기록했다.

## 검증과 남은 제한

- 일반 입력: forward bit-exact. 11개 gradient 모두 독립 기준 상대L2≤0.05%.
- 구간 검사: B1 dg/dtri bit-exact. dW≤0.05%, LN parameter≤0.0005%.
- x/W/dy/dropout/mask를 바꾼 CUDA graph replay는 새 eager와 모든 출력 bit-exact.
- L384 B1 memcheck/racecheck, L768 전체 모듈 unfiltered memcheck: 오류0, race hazard0.
- 선택된 cubin: stack/spills0. 소스·설정·cubin SHA-256를 결과JSON에 고정했다.
- **L768 변경 입력의 dWL 독립 기준 검사는 실패가 남아 있다.** 이번0.0548456%, 저장형0.0541300%, 허용0.05%. 이전 재계산에서도 같은 문제다. 허용치를 올리지 않았고, 새 경로와 저장형 비교는 기준 안이다. 전체 검증 완료로 표현하지 않는다.

Anthropic Apache-2.0 native v5의 TMA/WGMMA, LN, layout 구현을 계승한 학습 확장이다. 다음 병목은 B7–B12 재계산과 WGMMA 의존성, 그리고 저장형보다 커진 backward 비용이다.
'''
(R/'README.md').write_text(report)
for src,dst in [('README.md','trimul-recompute-next.md'),('wiring.svg','trimul-recompute-next.svg'),('ncu-summary.json','trimul-recompute-next-ncu.json'),('compiler-summary.json','trimul-recompute-next-compiler.json')]:shutil.copy2(R/src,A/dst)
for n in D:
 shutil.copy2(R/('results-L%d.json'%n),A/('trimul-recompute-next-L%d.json'%n))
 for k in ('b1','b7'):shutil.copy2(R/('paired-%s-L%d.json'%(k,n)),A/('trimul-recompute-next-paired-%s-L%d.json'%(k,n)))
section='''<!-- ONCHIP_NEXT_BEGIN --><section class="panel" id="recompute-next"><span class="pill green">2026-09-21 · 추가 최적화 · 실험 경로</span>
<h2>B1–B4 약20% 단축 · 전체 학습 추가5% 단축</h2>
<p>dWproj가 실제 사용하는 gate64만 재계산하고, 입력 LN·gate 계산과 출력 LN 계산을 두 warp-group에 나눠 겹쳤다. Forward와 B7–B12의 선택은 유지했다.</p>
<div class="notice amber"><b>최초 재계산 대비 전체1.29–1.30배. 저장형보다는 아직20–21% 느리다.</b><br>1.7배 목표 미달 · SoL90 미입증. Production 기본값 유지. 공개 Anthropic 추론 / cuEq 대비 새 비교가 아니다.</div>
<p class="muted">H100 node02 · L384/768 · BF16 batch1/C128 · 양방향 · dropout25% · mask/residual · 11개 gradient · live packing 포함 · 각600회 교대 CUDA graph 측정.</p>
<div class="toolbar"><b>전체 fwd+bwd</b><button type="button" data-next-l="384" class="active" aria-pressed="true">L384</button><button type="button" data-next-l="768" aria-pressed="false">L768</button></div><div id="next-bars" class="panel" aria-live="polite"></div>
'''+ht(head,T)+'''<p>직전 대비 시간은 L384 <b>5.5%</b>, L768 <b>4.8%</b> 감소. 최초 재계산 / 직전 선택 / 이번 선택을 같은 실행에서 재측정했다. Forward는 같은 커널의 측정 변동이다.</p>
<h3>연산 → 실제 CUDA 커널 배선</h3><a href="assets/trimul-recompute-next.svg"><img src="assets/trimul-recompute-next.svg" alt="B1-B4 단일 CUDA 호출 내부에서 dWproj의 gate64와 출력 LN을 두 warp-group으로 나눈 배선" style="width:100%;height:auto;border-radius:12px;background:#f7fafc"></a>
<h3>구간별 별도 profiler trace</h3>'''+ht(kh,K)+'''
<p>B7–B12의 차이는 측정 변동이다. 동일 버퍼800회 비교에서 LN 통계 재사용은1% 미만, TMA 조기 발행을 더한 후보는 L768에서 역효과였다. Consumer3은 register spill, shared-input 재계산은 성능 저하로 미채택.</p>
<h3>NCU · B1–B4 최종 선택</h3>'''+ht(nh,N)+'''
<p>SM/DRAM 사용률은 알고리즘 SoL 비율이 아니다. 실행 가능한 warp가 없는 cycle이 여전히53–55%여서 SoL90이라고 판단하지 않는다.</p>
<h3>검증과 제한</h3><ul><li>일반 입력: forward bit-exact, 모든11개 gradient 상대L2≤0.05%. 변경 입력 graph replay는 새 eager와 bit-exact.</li><li>B1 memcheck/racecheck, L768 전체 unfiltered memcheck: 오류0. 선택 cubin stack/spills0.</li><li><b>L768 변경 입력 dWL 독립 기준 실패는 남아 있다.</b> 이번0.05485%, 저장형0.05413%, 허용0.05%. 기준을 완화하지 않았다.</li><li>Activation 복원 커널0개. Forward left/right/tri 유지. Gradient 전달과 reduction scratch는 global에 존재.</li></ul>
<p>Anthropic native v5 TMA/WGMMA·LN·layout을 계승한 학습 확장.</p><p><a href="assets/trimul-recompute-next.md">구현·후보·검증 보고서</a> · <a href="assets/trimul-recompute-next-L384.json">L384</a> · <a href="assets/trimul-recompute-next-L768.json">L768</a> · <a href="assets/trimul-recompute-next-ncu.json">NCU</a></p></section><!-- ONCHIP_NEXT_END -->'''
p=SITE/'dist/trimul.html';s=p.read_text()
if '<!-- ONCHIP_NEXT_BEGIN -->' in s:s=re.sub(r'<!-- ONCHIP_NEXT_BEGIN -->.*?<!-- ONCHIP_NEXT_END -->',lambda _:section,s,flags=re.S)
else:s=s.replace('<!-- ONCHIP_OPTIMIZED_BEGIN -->',section+'<details><summary>이전: 최초 재계산 대비 전체1.22배 단계</summary><!-- ONCHIP_OPTIMIZED_BEGIN -->',1).replace('<!-- ONCHIP_OPTIMIZED_END -->','<!-- ONCHIP_OPTIMIZED_END --></details>',1)
s=s.replace('<nav>','<nav><a href="#recompute-next">최신: gate 재계산 축소</a>',1) if 'href="#recompute-next"' not in s else s
plot={str(n):{k:round(j['times']['forward_backward'][k]['median_us']/1000,4) for k in labels} for n,j in D.items()}
script='''<script id="next-chart-script">(()=>{const data=DATA,labels=LABELS;const draw=n=>{const rows=data[n],max=Math.max(...Object.values(rows));document.getElementById('next-bars').innerHTML=Object.entries(rows).map(([k,v])=>`<div class="barline" style="grid-template-columns:170px 1fr 80px"><span>${labels[k]}</span><div class="track"><div class="fill ${k==='optimized_recompute'?'up':k==='initial_recompute'?'slow':''}" style="width:${v/max*100}%"></div></div><b>${v.toFixed(3)} ms</b></div>`).join('');document.querySelectorAll('[data-next-l]').forEach(b=>{const active=b.dataset.nextL===n;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});};document.querySelectorAll('[data-next-l]').forEach(b=>b.addEventListener('click',()=>draw(b.dataset.nextL)));draw('384');})();</script>'''.replace('DATA',json.dumps(plot)).replace('LABELS',json.dumps(labels,ensure_ascii=False))
if '<script id="next-chart-script">' in s:s=re.sub(r'<script id="next-chart-script">.*?</script>',lambda _:script,s,flags=re.S)
else:s=s.replace('</body>',script+'</body>')
p.write_text(s)
p=SITE/'dist/index.html';s=p.read_text();a=s.index('<!-- TRAINING_LINK_BEGIN -->');b=s.index('<!-- TRAINING_LINK_END -->',a)+len('<!-- TRAINING_LINK_END -->');s=s[:a]+'''<!-- TRAINING_LINK_BEGIN --><section class="notice"><b>최신: B1–B4 약20% 단축 · 전체 추가5% 단축</b> · <a href="trimul.html#recompute-next">배선·표·검증 보기 →</a><br>최초 재계산 대비1.29–1.30배. 저장형보다는20–21% 느림. 1.7배 미달 · SoL90 미입증.</section><!-- TRAINING_LINK_END -->'''+s[b:];p.write_text(s)
print('Report and dashboard generated from verified results')
