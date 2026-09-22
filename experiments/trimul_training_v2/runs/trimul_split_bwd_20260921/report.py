from pathlib import Path
import csv,hashlib,html,json,re,shutil
R=Path(__file__).resolve().parent;S=R.parent/'anthropic_b1b4_pipeline_20260919/site-visuals';A=S/'dist/assets'
D={n:json.loads((R/('results-L%d.json'%n)).read_text()) for n in (384,768)}
assert 'ERROR SUMMARY: 0 errors' in (R/'memcheck-L768.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (R/'racecheck-L384.log').read_text()
labels={'baseline':'직전 재계산형','split_no_save':'분리만 · LN 저장 없음','fused_xn':'입력 LN 저장 · 통합 B7','split_xn_pc2':'입력 LN 저장 · 분리 PC2','split_xn_pc1':'입력 LN 저장 · 분리 PC1'}
rows=[];details=[];checks=[];traces=[];ncu=[];compiler=[]
for n,j in D.items():
 for p,h in j['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 for name,units in j['cubins'].items():
  for c in units:
   p=Path(c['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==c['sha256']
   compiler.append(dict(L=n,variant=name,sha256=c['sha256'],ptxas=[s.strip() for s in p.with_suffix('.ptxas.log').read_text().splitlines() if 'spill' in s or 'Used ' in s]))
 for k,label in labels.items():
  f,b,t=[j['times'][s][k]['median_us'] for s in ('forward','backward','forward_backward')];base=j['times']['forward_backward']['baseline']['median_us'];rows.append([str(n),label,'%.3f'%(f/1000),'%.3f'%(b/1000),'%.3f'%(t/1000),'%+.2f%%'%(100*(t/base-1))])
  e=j['mutated_reference'][k];mx=max((v['relative_l2'],key) for key,v in e.items());checks.append([str(n),label,'%.5f%% %s'%(mx[0]*100,mx[1]),'통과' if j['mutated_reference_pass'][k] else '실패','통과' if j['mutated_vs_baseline_pass'][k] else '실패'])
  assert all(v['bit_exact'] for v in j['graph_vs_eager'][k].values())
 for k in ('baseline','fused_xn','split_xn_pc1'):
  tr=j['traces']['backward'][k];traces.append([str(n),labels[k],'%.2f'%sum(x['us'] for x in tr if x['name']=='b1_fused'),'%.2f'%sum(x['us'] for x in tr if x['name']=='front_b7b12')])
 for row in csv.DictReader((R/('ncu-b7-L%d-raw.csv'%n)).open()):
  if not row['ID']:continue
  def val(k):return float(row[k].replace(',',''))
  o=dict(L=n,role='dW' if row['ID']=='0' else 'dX',registers=val('launch__registers_per_thread'),threads=row['Block Size'],grid=row['Grid Size'],smem_dynamic_kb=val('launch__shared_mem_per_block_dynamic'),occupancy=val('sm__warps_active.avg.pct_of_peak_sustained_active'),sm=val('sm__throughput.avg.pct_of_peak_sustained_elapsed'),memory=val('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'),tensor=val('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),no_issue=100-val('smsp__issue_active.avg.pct_of_peak_sustained_active'),max_cta_registers=val('launch__occupancy_limit_registers'),max_cta_shared=val('launch__occupancy_limit_shared_mem'),profile_us=val('gpu__time_duration.avg'))
  ncu.append(o)
(R/'ncu-summary.json').write_text(json.dumps(ncu,indent=2));(R/'compiler-summary.json').write_text(json.dumps(compiler,indent=2))
head=['L','배선','fwd ms','bwd ms','fwd+bwd ms','전체 시간 변화']
def md(h,rs):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(r)+' |\n' for r in rs)
def ht(h,rs):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rs)+'</tbody></table></div>'
nrows=[[str(x['L']),x['role'],'%.1f%%'%x['occupancy'],'%.1f%%'%x['sm'],'%.1f%%'%x['memory'],'%.1f%%'%x['tensor']] for x in ncu]
nh=['L','커널','실측 occupancy','NCU SM','NCU memory','Tensor pipe']
report='''# Backward 분리 실험 · 2026-09-21

## 판단

B7–B12를 dW와 dX 역할의 두 CUDA 호출로 분리했다. **분리만 하면 전체 학습은 2–3% 느려진다. 입력 LN activation을 저장·재사용하면서 분리하면 직전 재계산형 대비 L384 4.51%, L768 5.70% 시간이 줄었다.** 저장만 하고 B7을 통합 유지하면 약2.6–2.8% 감소다. 분리 효과와 LN 저장 효과를 구분해야 한다.

PC1과 PC2 분리형 차이는 전체 시간0.1–0.2%로 작다. PC1은 dW의 실제 두 CTA/SM 배치를 확인한 개발 후보다. **L768 변경 입력의 독립 gradient 검증 미통과가 남아 있어 production 기본값으로 채택하지 않았다.** 목표1.7배 또는 알고리즘 SoL90 달성 결과가 아니다.

## 같은 실행에서 측정한 전체 모듈

node02 H100 두 장, BF16, batch1, C128, L384/768, 공유 출력 LN256의 양방향 TriMul, mask/dropout25%/residual. CUDA graph 경로별600회 교대. 매 호출 weight packing·forward 저장·전체11개 gradient·추가 커널 호출·최종 reduction 포함. RNG 생성, optimizer, CPU dispatch, 컴파일 제외. 합계 열은 fwd+bwd 한 호출 직접 측정이며 단독 중앙값의 합이 아니다.

'''+md(head,rows)+'''

기준은 runs/trimul_recompute_next_20260921의 직전 on-chip 재계산 구현이다. Anthropic 원본 추론이나 cuEquivariance 대비 수치가 아니다. 기존 activation 전체 저장형보다 이 재계산형이 빠르다는 뜻도 아니다. 과거 저장형의 다른 실행 수치를 이번 표에 섞지 않았다.

## 무엇을 분리했나

- 기존 단일 B7 커널도 dW CTA와 dX CTA가 별개였다. 한 thread가 두 역할의 최대 레지스터를 동시에 유지하던 구조는 아니다.
- 기존: dW64 CTA + dX68 CTA가 하나의132-CTA cooperative launch. 384 threads/CTA, compiled168 registers/thread, dynamic shared221184B를 모든 CTA에 예약.
- 새 dW:256 CTA,256 threads/CTA. TMA producer1 WG + WGMMA consumer1 WG. producer32/consumer224 register budget, compiled128 registers/thread. Dynamic shared65536B. NCU에서 register/shared 상한 모두2 CTA/SM, 실측 occupancy 약24%를 확인했다.
- 새 dX:132 CTA,256 threads/CTA. 두 active WG만 남김. Compiled180 registers/thread, dynamic shared221184B,1 CTA/SM. dX의 thread당 레지스터 수가 감소한 것은 아니다. 불필요한 세 번째 WG를 제거해 CTA 전체 자원과 작업 분배를 바꿨다.
- 두 호출은 같은 stream에서 순서대로 실행한다. 두 stream 동시 실행 벤치가 아니다. 각 호출 내부에서 해당 파라미터 미분 최종 reduction까지 끝낸다. 새 B7 dW/dX와 수정 B1 모두 ptxas stack/spill0.

## Forward 저장과 backward 재사용

K1 → cuBLAS2개 → K3를 유지. K3가 이미 계산하는 BF16 affine 입력 LN 출력 x_n(C128)만 TMA store한다. B1은 출력 gate 재계산에서, B7의 dW/dX는 입력 projection/gate 재계산에서 이를 읽는다. B1 출력 LN 재계산은 그대로다.

Left/right/tri는 기존처럼 유지한다. Mean/rstd, 출력 xn_out, projection, gate, pL/pR/gL/gR preactivation을 새로 저장하지 않는다. B7 입력 LN 미분은 raw x에서 통계를 다시 계산한다. 추가 forward activation은 L38437.75MB/L768150.99MB. dW partial scratch도 기존8.39MB에서16.78MB로 증가하므로 peak memory가 이 activation 증가분과 같다는 뜻은 아니다.

## Backward trace 보조 수치

'''+md(['L','경로','B1 μs','B7 두 역할 합 μs'],traces)+'''

Profiler trace5회 평균이며 위 event 중앙값과 계측 범위가 다르다.

## NCU

'''+md(nh,nrows)+'''

NCU memory는 L1/L2/DRAM 등 memory throughput 중 bottleneck 지표다. HBM만의 대역폭도, 알고리즘 최소 시간 대비 효율도 아니다. 두 커널 모두 SoL90에 근접했다고 볼 근거가 없다. dX는 여전히1 CTA/SM이고 WG 동기화·WGMMA 대기가 남아 있다. DRAM 왕복만으로 병목을 설명할 수 없다.

## 정확도 · 모든 실패 포함

초기 입력: 모든 경로의 y는 bit-exact,11개 gradient 상대L2≤5e-4. 저장 LN과 no-save forward는 같은 출력이다. x/WL/Wg/dy/dropout scale/mask를 변경한 graph replay는 각 경로의 새 eager 호출과 모든 출력 bit-exact.

변경 입력의 독립 reference 및 직전 재계산형과의 비교:

'''+md(['L','경로','독립 기준 최대 상대L2','독립 기준≤0.05%','직전 경로≤0.05%'],checks)+'''

L768 새 분리형 dWL0.05557%가0.05% 한도를 넘는다. 기존 경로도0.05485%로 실패한다. 새 분리형은 기존 경로와는0.05% 이내지만 이것만으로 독립 검증 통과라고 하지 않는다. 저장+통합 경로는 독립 기준을 통과하고 기존 경로와의 비교를 실패했다. 두 판단을 분리해 기록했다. 허용오차를 늘리지 않았다.

추가 누적 분할/CTA 후보를 검사했다.4개 누적 segment는 초기 입력부터 최대0.05770%로 미통과. PC1 DW splits6은 독립 검증을 통과하나 B7 약2.865ms로 느리다. splits10/12/14/15는 변경 입력 검증 미통과. splits17은 cooperative residency 한도를 넘어 launch가 거부되어 후보에서 제외했고 guard를 추가했다. 더 많은 후보를 검증했다고 주장하지 않는다.

L384 B7 racecheck0 hazards/0 errors/0 warnings. L768 전체 호출 unfiltered memcheck0 errors. 이는 수치 검증 미통과를 대체하지 않는다. GPU 실험 할당13363은 종료했고 기존 학습13228은 유지했다.

## 구현과 재현

- b7_roles.cu / b7_pipe_dw.inc: split-role 및 register/shared specialization.
- role_plan.py: config별 SHA-256 cubin, cooperative launch, separate reduction scratch.
- b1_fused.cu / saved_plans.py: saved x_n 재사용.
- bench.py: 원형/분리만/저장만/둘 다 전체 연결 및600회 paired timing.
- validate.py: 초기·변경 입력·graph/eager 독립 비교. 실패도 명시 저장.
- capture_ncu.py: 선택된 두 B7 launch만 profiler capture.

node02 할당 내 MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build 설정. bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_split_bwd_20260921/bench.py --length384에서 --length와384 사이 공백을 넣어 실행한다. L768도 동일. --only split_xn_pc1 --check-only는 초기 입력 검사만 수행하므로 변경 입력 검증을 대신하지 않는다.

Anthropic Apache-2.0 native v5 TMA/WGMMA·LN 구현을 계승한 학습 실험이다. 엔진 production dispatch는 변경하지 않았다. Source/cubin hashes는 results JSON에 고정했다.
'''
report=report.replace('--length384에서 --length와384 사이 공백을 넣어 실행한다','--length 384를 실행한다')
(R/'README.md').write_text(report)
selection=dict(candidate='split_xn_pc1',production_ready=False,reason='L768 mutated dWL relative_l2 exceeds unchanged 5e-4 independent-reference limit',L384_independent_pass=True,L768_independent_pass=False,baseline_has_same_failure=True,engine_default_changed=False)
(R/'selection.json').write_text(json.dumps(selection,indent=2))
for n in D:shutil.copy2(R/('results-L%d.json'%n),A/('trimul-split-bwd-L%d.json'%n))
for src,dst in [('README.md','trimul-split-bwd.md'),('ncu-summary.json','trimul-split-bwd-ncu.json'),('compiler-summary.json','trimul-split-bwd-compiler.json'),('selection.json','trimul-split-bwd-selection.json')]:shutil.copy2(R/src,A/dst)
section='''<!-- SPLIT_BWD_BEGIN --><section class="panel" id="split-bwd"><span class="pill amber">2026-09-21 · B7 dW/dX 분리 · 수치 검증 미완료 후보</span><h2>LN 저장 + backward 분리: 전체 시간 −4.5% / −5.7%</h2><p>직전 <b>재계산형</b> 대비. 분리만 하면 느려지고, 입력 LN 결과를 저장·재사용하면서 분리하면 빨라진다. B1–B4, cuBLAS, B7–B12, 저장 비용까지 전부 포함했다.</p><p class="muted">node02 H100 · BF16 · C128 · 양방향 · mask/dropout25%/residual · live packing · 각600회 CUDA graph 교대 측정. Anthropic 원본 / cuEq 대비 배율이 아니다.</p><div class="notice amber"><b>실험 후보로 유지.</b> L768 변경 입력의 dWL 상대L2가0.05557%로 기존0.05% 한도를 넘는다. 기준 경로도0.05485%로 실패한다. 허용오차와 production 기본값은 그대로다.</div>
<div class="toolbar"><b>전체 fwd+bwd</b><button type="button" data-split-l="384" class="active" aria-pressed="true">L384</button><button type="button" data-split-l="768" aria-pressed="false">L768</button></div><div id="split-bars" class="panel" aria-live="polite"></div>
'''+ht(head,rows)+'''
<h3>실제 연결: B7–B12를 역할별 두 커널로</h3><div class="flow"><div class="node"><b>Forward K3</b>기존 연산 + 입력 x_n TMA store<br>출력 LN·gate·projection 저장 없음</div><div class="arrow">→</div><div class="node purple"><b>B1–B4 + cuBLAS</b>B1의 gate 재계산에 x_n 재사용<br>cuBLAS contraction 미분4회</div><div class="arrow">→</div><div class="node"><b>CUDA ① dW</b>x_n으로 projection/gate 재계산<br>dW + 최종 reduction<br>256 threads · 64KiB shared · 2 CTA/SM</div><div class="arrow">→</div><div class="node"><b>CUDA ② dX</b>x_n으로 projection/gate 재계산<br>dX + 입력 LN 미분 + residual<br>256 threads · 216KiB shared · 1 CTA/SM</div></div>
<p>두 커널은 같은 stream에서 순차 실행한다. 기존 통합 커널도 역할별 CTA가 달랐다. 분리의 핵심은 <b>dW의 shared 예약량을 줄여 두 CTA를 배치</b>하고 <b>dX의 불필요한 세 번째 WG를 제거</b>한 것이다. 새 backward 커널은 ptxas spill0. 평균·역표준편차는 저장하지 않고 입력 LN 미분에서 재계산한다.</p>
<h3>NCU: 자원 분리 확인, SoL90은 아님</h3>'''+ht(nh,nrows)+'''
<p>NCU memory는 HBM만의 사용률이 아니며 알고리즘 최소시간 대비 효율도 아니다. dX는 여전히1 CTA/SM이다. 입력 LN 추가 유지량37.75/150.99MB 외에 dW partial scratch도 약8.39MB 늘어난다.</p>
<details><summary>변경 입력 수치 검증 — 실패 포함</summary>'''+ht(['L','경로','독립 기준 최대 상대L2','독립 기준≤0.05%','직전 경로≤0.05%'],checks)+'''<p>모든 경로 graph replay = fresh eager bit-exact. L384 B7 racecheck0 hazards, L768 전체 memcheck0 errors. 메모리 검증은 수치 오차 한도를 대체하지 않는다.</p></details>
<p><a href="assets/trimul-split-bwd.md">구현·조건·실패 후보 보고서</a> · <a href="assets/trimul-split-bwd-L384.json">L384 원시 결과</a> · <a href="assets/trimul-split-bwd-L768.json">L768 원시 결과</a> · <a href="assets/trimul-split-bwd-ncu.json">NCU</a></p></section><!-- SPLIT_BWD_END -->'''
p=S/'dist/trimul.html';s=p.read_text()
if '<!-- SPLIT_BWD_BEGIN -->' in s:s=re.sub(r'<!-- SPLIT_BWD_BEGIN -->.*?<!-- SPLIT_BWD_END -->',lambda _:section,s,flags=re.S)
else:s=s.replace('<main>','<main>'+section,1)
if 'href="#split-bwd"' not in s:s=s.replace('<nav>','<nav><a href="#split-bwd">최신: backward 분리</a>',1)
data={str(n):{k:round(j['times']['forward_backward'][k]['median_us']/1000,6) for k in labels} for n,j in D.items()}
script='''<script id="split-bwd-chart">(()=>{const data=DATA,labels=LABELS;const draw=n=>{const r=data[n],max=Math.max(...Object.values(r));document.getElementById('split-bars').innerHTML=Object.entries(r).map(([k,v])=>`<div class="barline" style="grid-template-columns:minmax(120px,220px) 1fr 84px"><span>${labels[k]}</span><div class="track"><div class="fill ${k==='split_xn_pc1'?'up':''}" style="width:${v/max*100}%"></div></div><b>${v.toFixed(3)} ms</b></div>`).join('');document.querySelectorAll('[data-split-l]').forEach(b=>{const a=b.dataset.splitL===n;b.classList.toggle('active',a);b.setAttribute('aria-pressed',String(a));});};document.querySelectorAll('[data-split-l]').forEach(b=>b.addEventListener('click',()=>draw(b.dataset.splitL)));draw('384');})();</script>'''.replace('DATA',json.dumps(data)).replace('LABELS',json.dumps(labels,ensure_ascii=False))
if '<script id="split-bwd-chart">' in s:s=re.sub(r'<script id="split-bwd-chart">.*?</script>',lambda _:script,s,flags=re.S)
else:s=s.replace('</body>',script+'</body>')
p.write_text(s)
p=S/'dist/index.html';s=p.read_text();s=re.sub(r'<!-- TRAINING_LINK_BEGIN -->.*?<!-- TRAINING_LINK_END -->','''<!-- TRAINING_LINK_BEGIN --><section class="notice"><b>최신: 입력 LN 저장 + B7 backward 분리</b> · <a href="trimul.html#split-bwd">성능·배선·NCU·검증 →</a><br>직전 재계산형보다 전체4.5%/5.7% 단축. L768 변경 입력 gradient 오차가 한도를 넘어 실험 후보로 유지한다.</section><!-- TRAINING_LINK_END -->''',s,flags=re.S);p.write_text(s)
print('Report and dashboard generated; candidate remains experimental')
