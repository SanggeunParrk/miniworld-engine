from pathlib import Path
import hashlib,html,json,re,shutil
R=Path(__file__).resolve().parent;S=R.parent/'anthropic_b1b4_pipeline_20260919/site-visuals';A=S/'dist/assets';D={n:json.loads((R/('results-L%d.json'%n)).read_text()) for n in (384,768)}
assert 'ERROR SUMMARY: 0 errors' in (R/'memcheck-L768.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (R/'racecheck-L384.log').read_text()
labels={'original_xn':'입력 x_n만','xn_in_stats':'입력 LN 통계 추가','xn_out_stats':'출력 LN 통계 추가','xn_both_stats':'입력·출력 LN 통계 추가','xn_stats_out_pg':'출력 projection·gate 추가','xn_stats_in_pg':'입력 projection·gate 추가','xn_stats_all_pg':'입력·출력 projection·gate 추가','both_ln_stats_all_pg':'출력 xn_out까지 전부 추가'}
statrows=[];pgrows=[];allrows=[];methods=[];compiler=[]
for n,j in D.items():
 for path,h in j['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 t=j['times'];base=t['original_xn']['median_us'];anchor=t['xn_both_stats']['median_us']
 for k in ('original_xn','xn_in_stats','xn_out_stats','xn_both_stats'):
  us=t[k]['median_us'];statrows.append([str(n),labels[k],'%.3f'%(us/1000),'%+.3f'%(us-base),'%+.2f%%'%(100*(us/base-1))])
 for k in ('xn_both_stats','xn_stats_out_pg','xn_stats_in_pg','xn_stats_all_pg'):
  us=t[k]['median_us'];pgrows.append([str(n),labels[k],'%.3f'%(us/1000),'%+.3f'%(us-anchor),'%+.2f%%'%(100*(us/anchor-1)),'%.2f'%((j['retained_bytes'][k]-j['retained_bytes']['xn_both_stats'])/1e6)])
 for k,label in labels.items():allrows.append([str(n),label,'%.3f'%(t[k]['median_us']/1000),'%.2f'%(j['retained_bytes'][k]/1e6)])
 for kind,info in j['tuning'].items():methods.append([str(n),kind,'%.3f'%info['times']['0']['median_us'],'%.3f'%info['times']['1']['median_us'],str(info['winner'])])
 for label,units in j['cubins'].items():
  for c in units:
   p=Path(c['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==c['sha256'];compiler.append(dict(L=n,variant=label,path=str(p),sha256=c['sha256'],ptxas=[s.strip() for s in p.with_suffix('.ptxas.log').read_text().splitlines() if 'spill' in s or 'Used ' in s]))
 for category in ('checks','mutated_checks'):
  for k,e in j[category].items():
   assert e['forward']['bit_exact'];assert all(v.get('relative_l2',0)<=v.get('limit',0) for v in e.values())
for p,h in json.loads((R/'derivation.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
(R/'compiler-summary.json').write_text(json.dumps(compiler,indent=2))
def md(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(row)+' |\n' for row in rows)
def ht(h,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
sh=['L','保存 조건'.replace('保存','저장'),'전체 fwd ms','x_n만 대비 μs','변화']
ph=['L','저장 조건','전체 fwd ms','통계 기준 대비 μs','변화','추가 MB']
report='''# TriMul forward: 통계 및 projection/gate 저장 비용 · 2026-09-21

## 결론

평균·역표준편차 저장 비용은 작다. 입력 LN 통계만 추가하면 L384 +0.64%, L768 +1.33%. 입력·출력 LN 통계를 모두 추가하면 +0.14%, +1.02%다. 출력 통계만 저장한 경우는 소폭 빨라졌으며 컴파일러 스케줄·spill 변화까지 포함된 실제 지연이다. 저장되는 바이트가 실행시간을 단조롭게 증가시킨다는 뜻은 아니다.

Projection/gate는 큰 비용이다. 입력 x_n + 양쪽 LN 통계를 저장하는 동일 기준에서 출력 쪽만 +10.57%/+9.75%, 입력 쪽만 +31.58%/+27.96%, 양쪽 모두 +41.43%/+38.16%다.

## 조건

node02 H100 두 장, BF16, B1/C128, 공유 출력 LN256 양방향 TriMul, L384/768, mask/dropout25%/residual. **Forward만** 비교. 각 경로600회 교대 CUDA graph, 매 호출 weight packing과 모든 저장 포함. RNG 생성, optimizer, backward, CPU dispatch, 컴파일 제외. 기존 학습13228과 독립 에이전트13364는 건드리지 않았고 실험 할당13365는 종료했다.

## 1. 평균·역표준편차

기존 input x_n BF16 저장에 FP32 mean/rstd만 추가한다. LN 내부에서 이미 계산한 값을 기록하며 재계산·별도 LN 커널을 추가하지 않는다. Input/output statistics 모두 K3에서 저장한다.

'''+md(sh,statrows)+'''

추가 크기: 입력 또는 출력 한 LN의 통계는 L3841.18MB/L7684.72MB, 두 LN 모두2.36MB/9.44MB. 세 측정 block에서 양쪽 통계 추가의 변화는 L384 +0.08~0.21%, L768 +0.85~1.09%. 정확히0 비용이라고 하지는 않는다.

## 2. Projection/gate 저장

이 표의 기준은 input x_n + 입력/출력 mean/rstd다. Output xn_out은 추가하지 않았다.

'''+md(ph,pgrows)+'''

- 입력: pL, pR와 gL/gR의 sigmoid 이전 gate logit. 모두 BF16이며 기존 backward와 같은 [g0,p0,g1,p1,...] × M interleaved layout. 네 값 전체1024채널을 저장한다.
- 출력: BF16 projection128채널 및 BF16 sigmoid(BF16 gate logit)128채널. 둘 다 저장하며 residual/dropout 반영 전 값이다.
- 입력 gate logit과 출력 sigmoid gate를 구분한다. 입력의 sigmoid 이후 값을 FP32로 저장하는 실험은 아니다.
- Left/right/tri는 원래 유지하던 값이다. 추가 MB는 이 기존 값과 allocator/peak memory를 제외한 새 tensor의 논리적 크기다.

## 전체 저장량

'''+md(['L','조건','fwd ms','기존 left/right/tri 외 추가 MB'],allrows)+'''

## 실제 CUDA 구현

K1은 Anthropic native v5 원본 fused LN + projection/gate + mask와 타일(2,64,8,2)을 유지했다. 추가 gate/projection을 이미 존재하는 FP32 accumulator에서 BF16으로 기록한다. 원래 a/b 계산과 반올림은 유지해 최종 출력이 bit-exact다.

K3도 기존 fused LN + 출력 projection/gate + dropout/residual 구조와 config(2,64,4,1,regs24/240,serialLN)를 유지했다. LN 통계는 ln_fragment의 반환값을 저장하고, projection/gate는 같은 epilogue에서 추가 저장한다. Full tile/config 재튜닝이 아니라 저장 스케줄 두 종류 비교다.

'''+md(['L','커널','방법0 μs','방법1 μs','선택'],methods)+'''

K1 방법0은 gate/projection 전용 staging을32KiB 추가해 a/b와 함께 TMA store, 방법1은 기존 stage를 재사용하며 TMA read 완료를 기다린다. K3 방법0은 별도32KiB staging과 TMA, 방법1은 register에서 vector STG. 둘 다 방법0을 선택했다. L768 K1의0.1% 차이는 유의미한 우위로 단정하지 않는다. 단독 시간과 전체 fwd 시간은 캐시 상태가 달라 차이를 그대로 더하지 않는다.

## Compiler 영향과 이전 설명 정정

원래 input x_n만 TMA 저장하는 K3에도 ptxas52B spill stores/68B spill loads가 있었다. 과거 현황판의 '저장ON 후보는 모두 spill0' 설명은 잘못되어 이번에 수정했다. 당시 compiler JSON에도 이 값이 남아 있었다. Input 통계만 추가한 후보는40B/56B, 양쪽 통계 및 선택한 projection/gate 저장 후보는0이다. 따라서 작은 ±1% 차이를 순수 HBM store 시간이라고 해석하지 않는다.

수정된 저장OFF 대조군과 실제 이전 커널도 함께 측정했다. L384 none285.152 vs original285.312μs, x_n294.624 vs original293.920μs. L768 none1157.072 vs original1158.320μs, x_n1187.280 vs original1186.080μs. 본 표의 통계 기준은 실제 이전 original_xn을 사용한다.

## 검증

- 모든 후보의 최종 y가 이전 forward와 bit-exact. K1 a/b도 bit-exact.
- 저장 projection/logit/gate는 FP32 PyTorch GEMM·sigmoid 기준과 비교해 상대L2≤3.82e-5(0.00382%). 한도5e-4를 그대로 유지했다.
- FP32 mean/rstd는 독립 mean/variance/rsqrt 수식과 상대L2≤2e-6. LN activation은 검증된 fused LN 기준과 비교.
- x, 입력/출력 gamma, WL/Wg, dropout scale, pair mask를 바꾼 동일 graph replay에서도 y와 저장값 검증 통과. 저장값이 초기 capture 값으로 고정되지 않는다.
- L384 새 save_K1/K3 필터 racecheck0 hazards/0 errors/0 warnings. L768 전체 check-only unfiltered memcheck0 errors. 두 저장 방법을 모두 검사했다.
- 이 결과는 forward 저장 비용이며 backward 연결·절감량을 검증한 결과가 아니다. 지난 B7 분리 후보의 L768 gradient 제한을 해결했다는 뜻도 아니다. Production dispatch는 그대로다.

## 재현

node02 H100 할당에서 MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build 설정 후:

    bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_save_cost_20260921/bench.py --length 384

L768도 동일. 저장 값 생성은 save_cost_core.py와 save_k1.cu/save_k3.cu, 원본에서의 추출·변경은 derive.py 및 derivation.json으로 기록했다. Source/cubin SHA-256은 결과 JSON에 있다. Anthropic Apache-2.0 native v5 개발을 계승한 저장 비용 분석이다.
'''
(R/'README.md').write_text(report)
for n in D:shutil.copy2(R/('results-L%d.json'%n),A/('trimul-save-cost-L%d.json'%n))
for src,dst in [('README.md','trimul-save-cost.md'),('compiler-summary.json','trimul-save-cost-compiler.json')]:shutil.copy2(R/src,A/dst)
section='''<!-- SAVE_COST_BEGIN --><section class="panel" id="save-cost"><span class="pill green">2026-09-21 · Forward 저장 비용 · 검증 완료</span><h2>LN 통계는 약0–1% · projection/gate 전부 저장은 +38–41%</h2><p>입력 <code>x_n</code> 저장을 기준으로 비교했다. <b>Forward만 실측</b>했으며 backward 절감량은 이 표에 포함하지 않는다.</p><p class="muted">node02 H100 · BF16 · C128 · 공유 출력 LN256 양방향 · mask/dropout25%/residual · live packing · 각600회 교대 CUDA graph.</p><h3>① 평균·역표준편차: 이미 계산한 FP32 값을 저장</h3>'''+ht(sh,statrows)+'''<p>두 LN 통계를 합해 추가2.36MB / 9.44MB다. 재계산이나 별도 LN 커널은 없다. 작은 시간 차이에는 컴파일러 스케줄·spill 변화도 포함된다.</p><h3>② Projection/gate: 입력 쪽 저장량이 더 크다</h3><p>기준: <b>입력 x_n + 입력/출력 LN 통계</b>. 출력 xn_out은 제외.</p>'''+ht(ph,pgrows)+'''<div class="toolbar"><b>전체 Forward</b><button type="button" data-savecost-l="384" class="active" aria-pressed="true">L384</button><button type="button" data-savecost-l="768" aria-pressed="false">L768</button></div><div id="savecost-bars" class="panel" aria-live="polite"></div><div class="flow"><div class="node"><b>K1 · 입력 projection/gate</b>기존 LN + GEMM + mask 유지<br>BF16 projection·gate logit4개 저장<br>추가302MB / 1208MB</div><div class="arrow">→</div><div class="node purple"><b>cuBLAS contraction</b>원래 left/right/tri 유지</div><div class="arrow">→</div><div class="node"><b>K3 · LN 통계 + 출력 값</b>mean/rstd FP32 직접 저장<br>출력 BF16 projection·sigmoid gate<br>추가75.5MB / 302MB</div></div><p>입력 gate는 <b>sigmoid 이전 logit</b>, 출력 gate는 <b>sigmoid 이후 BF16</b>이다. 두 저장 스케줄을 비교해 TMA를 선택했다. 전체 tile 공간 재튜닝 결과는 아니다.</p><details><summary>저장량·방법·검증 상세</summary>'''+ht(['L','조건','fwd ms','기존 값 외 추가 MB'],allrows)+ht(['L','커널','방법0 μs','방법1 μs','선택'],methods)+'''<p>모든 y/a·b bit-exact. Projection/gate 상대L2≤0.00382%. FP32 통계 한도2e-6 통과. 변경 입력 graph replay와 저장값 검사 통과. L384 새 커널 racecheck0, L768 전체 memcheck0.</p></details><div class="notice amber"><b>이전 compiler 설명 정정:</b> 입력 x_n만 저장하는 기존 K3는52B spill stores/68B loads였다. 저장ON 전부 spill0이라는 설명은 잘못됐다. 이번 양쪽 통계 및 선택한 projection/gate 저장 후보는0이다. 성능 수치에는 실제 compiler 변화도 포함했다.</div><p><a href="assets/trimul-save-cost.md">상세 보고서</a> · <a href="assets/trimul-save-cost-L384.json">L384 결과</a> · <a href="assets/trimul-save-cost-L768.json">L768 결과</a> · <a href="assets/trimul-save-cost-compiler.json">Compiler 기록</a></p></section><!-- SAVE_COST_END -->'''
p=S/'dist/trimul.html';s=p.read_text()
if '<!-- SAVE_COST_BEGIN -->' in s:s=re.sub(r'<!-- SAVE_COST_BEGIN -->.*?<!-- SAVE_COST_END -->',lambda _:section,s,flags=re.S)
else:s=s.replace('<main>','<main>'+section,1)
if 'href="#save-cost"' not in s:s=s.replace('<nav>','<nav><a href="#save-cost">최신: 통계·projection 저장</a>',1)
chart={str(n):{k:round(j['times'][k]['median_us']/1000,6) for k in ('original_xn','xn_both_stats','xn_stats_out_pg','xn_stats_in_pg','xn_stats_all_pg')} for n,j in D.items()}
script='''<script id="save-cost-chart">(()=>{const data=DATA,labels=LABELS;const draw=n=>{const r=data[n],max=Math.max(...Object.values(r));document.getElementById('savecost-bars').innerHTML=Object.entries(r).map(([k,v])=>`<div class="barline" style="grid-template-columns:minmax(120px,220px) 1fr 84px"><span>${labels[k]}</span><div class="track"><div class="fill ${k==='original_xn'?'up':'slow'}" style="width:${v/max*100}%"></div></div><b>${v.toFixed(3)} ms</b></div>`).join('');document.querySelectorAll('[data-savecost-l]').forEach(b=>{const a=b.dataset.savecostL===n;b.classList.toggle('active',a);b.setAttribute('aria-pressed',String(a));});};document.querySelectorAll('[data-savecost-l]').forEach(b=>b.addEventListener('click',()=>draw(b.dataset.savecostL)));draw('384');})();</script>'''.replace('DATA',json.dumps(chart)).replace('LABELS',json.dumps(labels,ensure_ascii=False))
if '<script id="save-cost-chart">' in s:s=re.sub(r'<script id="save-cost-chart">.*?</script>',lambda _:script,s,flags=re.S)
else:s=s.replace('</body>',script+'</body>')
s=s.replace('저장ON 후보는0.','입력만 TMA 저장은52B stores/68B loads, 둘 다 TMA 저장은0.')
p.write_text(s)
p=S/'dist/index.html';s=p.read_text();s=re.sub(r'<!-- TRAINING_LINK_BEGIN -->.*?<!-- TRAINING_LINK_END -->','''<!-- TRAINING_LINK_BEGIN --><section class="notice"><b>최신: 통계·projection/gate 저장 비용</b> · <a href="trimul.html#save-cost">표·저장 배선·검증 →</a><br>평균·역표준편차는 약0–1%. 입력·출력 projection/gate 전부 저장은 forward +38–41%.</section><!-- TRAINING_LINK_END -->''',s,flags=re.S);p.write_text(s)
# Correct the prior prose to match its already published compiler artifact.
for p in [A/'trimul-ln-only-save.md',R.parent/'trimul_ln_only_save_20260921/README.md',R.parent/'trimul_ln_only_save_20260921/report.py']:
 s=p.read_text().replace('저장ON 후보들은 이번 빌드에서 spill0이다.','저장ON도 설정에 따라 다르다. 입력 LN만 TMA 저장하는 경로는52B spill stores/68B loads, 입력·출력 둘 다 TMA 저장하는 경로는0이다.').replace('저장ON 후보는0.','입력만 TMA 저장은52B stores/68B loads, 둘 다 TMA 저장은0.');p.write_text(s)
print('Saved report, updated dashboard, corrected prior spill prose from compiler evidence')
