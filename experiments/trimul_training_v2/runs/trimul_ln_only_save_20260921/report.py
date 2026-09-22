from pathlib import Path
import hashlib,json,re,shutil
R=Path(__file__).resolve().parent;SITE=R.parent/'anthropic_b1b4_pipeline_20260919/site-visuals';A=SITE/'dist/assets'
D={n:json.loads((R/('results-L%d.json'%n)).read_text()) for n in (384,768)}
assert 'ERROR SUMMARY: 0 errors' in (R/'memcheck-L768.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed' in (R/'racecheck-L384.log').read_text()
labels={'0':'추가 LN 저장 없음','1':'입력 x_n만','2':'출력 xn_out만','3':'입력·출력 둘 다'}
rows=[];krows=[];memory=[];compiler=[]
for n,j in D.items():
 for path,h in j['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 base=j['full_times']['0']['median_us']
 for mode,label in labels.items():
  us=j['full_times'][mode]['median_us'];rows.append([str(n),label,'%.6f'%(us/1000),'%+.3f'%(us-base),'%+.2f%%'%(100*(us/base-1))])
  memory.append([str(n),label,'%.3f MB'%(j['retained_ln_bytes'][mode]/1e6)])
 for mode in range(1,4):
  t=j['k3_times'];krows.append([str(n),labels[str(mode)],'%.3f'%t['%d_tma'%mode]['median_us'],'%.3f'%t['%d_stg'%mode]['median_us']])
 for key,c in j['cubins'].items():
  p=Path(c['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==c['sha256'];s=p.with_suffix('.ptxas.log').read_text();compiler.append(dict(L=n,variant=key,sha256=c['sha256'],ptxas=[v.strip() for v in s.splitlines() if 'spill' in v or 'Used ' in v]))
 assert all(v['output_bit_exact'] for v in j['checks'].values())
 assert all(v['output_bit_exact'] for v in j['mutated_graph_checks'].values())
def md(h,r):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(x)+' |\n' for x in r)
def ht(h,r):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in r)+'</tbody></table></div>'
head=['L','추가 저장','전체 forward ms','증가 μs','증가율']
report='''# Forward: LN activation만 저장할 때의 비용 · 2026-09-21

## 결과

입력 LN 출력만 저장하면 전체 forward는 **L384 +3.61%, L768 +2.69%**. 출력 LN 출력만 저장하면 **+6.70%, +4.27%**. 둘 다 저장하면 **+10.77%, +7.96%**.

**저장하는 것은 affine 이후 BF16 `x_n`(C128), `xn_out`(C256)뿐이다. 평균·역표준편차, projection, gate, input preactivation은 저장하지 않는다.** 기존처럼 left/right/tri는 유지한다. 따라서 '저장 없음'은 LN 추가 저장 없음이라는 뜻이다.

H100 node02 GPU0/1, BF16 batch1/C128, bidirectional shared output LN256, L384/768, mask/dropout25%/residual. 기존 Anthropic 파생 no-save forward를 기준으로 같은 실행에서 측정했다. 각 variant마다 600회 교대 CUDA graph timing. Live weight packing 포함, RNG/optimizer/CPU dispatch/컴파일 제외. Forward만 비교했으며 backward 이득은 아직 측정하지 않았다.

'''+md(head,rows)+'''
## 통제한 구현

K1, cuBLAS contractions, K3의 LN/GEMM/반올림/epilogue는 유지한다. K3가 원래 계산하는 두 LN 결과를 필요한 경우에만 global buffer에 추가로 기록한다. 입력 LN 저장도 K3에서 수행하여 K1을 변경하거나 K3의 입력 LN 재계산을 없애지 않았다. 따라서 LN 분리형과 융합형을 다시 비교한 결과가 아니다.

새 activation 복원용 커널은 없다. K3 내부에서 기존 output staging shared memory를 재사용한 TMA store와 register→global vector store를 비교했다. **모든 저장 조합, 두 L에서 TMA가 더 빨랐다.** K3 config는 `(BI=2,BJ=64,slots=4,NACC=1,regs24/240,LN_serial=1)`로 공통 고정했다. 저장 스케줄 두 개를 비교했으며 전체 tile/config 공간 재튜닝 결과가 아니다.

수정된 커널의 저장OFF와 완전 미수정 기존 forward도 비교했다: L384 284.704 vs284.768 μs, L7681172.720 vs1174.032 μs. 약0.02%/0.11% 차이로, 저장OFF 대조군이 기존 경로와 같은 성능임을 확인했다.

## K3 단독: 같은 입출력 버퍼에서의 저장 방법 비교

'''+md(['L','저장','TMA μs','vector STG μs'],krows)+'''

K3 단독과 전체 forward는 캐시 상태와 타이밍 범위가 달라 증가량이 정확히 같지는 않다. 최종 답변에는 전체 forward 실측을 사용한다.

## 추가 유지 메모리

'''+md(['L','저장','추가 tensor 크기'],memory)+'''

논리적인 tensor 크기 합계다. peak GPU memory 실측은 아니다.

## 검증과 한계

- 모든7개 K3 후보에서 최종 y가 미수정 forward와 bit-exact.
- 저장된 LN 출력은 독립 FP32 수식→BF16 기준 상대L2 약0.000009–0.000011. 검사 한도0.0005. Input/output LN 각각 검사했다.
- x, LN gamma, dropout scale, mask를 바꾼 CUDA graph replay에서도 y는 미수정 경로와 bit-exact. 저장된 activation도 변경된 입력에 대한 독립 기준과 일치한다.
- L384 K3 racecheck: 오류0/hazard0. L768 K3 memcheck: 오류0. 필터는 `kns=infer_k3`; 전체 backward sanitizer가 아니다.
- 저장OFF K3는 기존 커널과 동일하게 ptxas16B spill stores/32B spill loads가 있다. 저장ON도 설정에 따라 다르다. 입력 LN만 TMA 저장하는 경로는52B spill stores/68B loads, 입력·출력 둘 다 TMA 저장하는 경로는0이다. 저장 비용뿐 아니라 컴파일러 스케줄 변화가 포함된 실제 실행시간 비교이며, 전 후보 spill0이라고 주장하지 않는다.
- 이 실험은 BF16 LN activation만 저장한다. Mean/rstd까지 저장하는 경우와 backward 연결/성능은 별도 평가가 필요하다. Production dispatch는 변경하지 않았다.

## 재현

node02에 할당된 H100에서 `MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build` 설정 후 `bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_ln_only_save_20260921/bench.py --length 384` 실행. L768도 동일하다. `--check-only`는 정확도만 검사한다.

Anthropic Apache-2.0 native v5 TMA/WGMMA·LN 구현을 계승한 저장 정책 실험. 기존 `no_save_k3.cu`에서의 변경은 derive.py와 derivation.json에 기록했다.
'''
(R/'README.md').write_text(report);(R/'compiler-summary.json').write_text(json.dumps(compiler,indent=2))
for n in D:shutil.copy2(R/('results-L%d.json'%n),A/('trimul-ln-only-save-L%d.json'%n))
shutil.copy2(R/'README.md',A/'trimul-ln-only-save.md');shutil.copy2(R/'compiler-summary.json',A/'trimul-ln-only-save-compiler.json')
section='''<!-- LN_ONLY_SAVE_BEGIN --><section class="panel" id="ln-only-save"><span class="pill green">2026-09-21 · Forward LN 저장 비용 실측</span><h2>LN activation만 저장: 입력 +3% · 출력 +4–7% · 둘 다 +8–11%</h2>
<p><b>x_n(C128), xn_out(C256)만 선택 저장.</b> 평균·역표준편차 / projection / gate / preactivation은 추가 저장하지 않는다. Left/right/tri는 기존처럼 유지한다. 현재 no-save forward와 같은 실행에서 비교했다.</p>
<p class="muted">H100 node02 · L384/768 · BF16 batch1/C128 · 양방향 · mask/dropout25%/residual · live packing 포함 · 각600회 교대 CUDA graph. Backward 성능은 이번에 측정하지 않았다.</p>
<div class="toolbar"><b>전체 forward</b><button type="button" data-lns-l="384" class="active" aria-pressed="true">L384</button><button type="button" data-lns-l="768" aria-pressed="false">L768</button></div><div id="lns-bars" class="panel" aria-live="polite"></div>
'''+ht(head,rows)+'''
<h3>실제 배선</h3><p><b>기존 K1 → cuBLAS ×2 → K3</b>를 유지한다. K3의 기존 LN 계산 결과에서 <b>x_n / xn_out → TMA store</b>만 선택적으로 추가했다. 입력 LN도 K3에서 저장한다. K3가 저장된 x_n을 다시 읽도록 바꾸거나 LN을 별도 커널로 분리한 실험은 아니다.</p>
<p>같은 tile/config에서 TMA와 vector STG를 비교했고 모든 경우 TMA가 빨랐다. 완전 미수정 forward와 새 저장OFF 대조군의 차이는0.02%/0.11%였다.</p>
<details><summary>K3 저장 방법 비교 및 추가 메모리</summary>'''+ht(['L','저장','TMA μs','vector STG μs'],krows)+ht(['L','저장','추가 tensor 크기'],memory)+'''</details>
<h3>검증</h3><ul><li>모든 후보 y bit-exact. 저장 LN activation은 독립 기준 상대L2≤0.000011.</li><li>변경 입력 graph replay 검증 통과. K3 memcheck/racecheck 오류0.</li><li>기존 저장OFF K3에16B spill stores/32B loads가 있고 입력만 TMA 저장은52B stores/68B loads, 둘 다 TMA 저장은0. 컴파일러 스케줄 변화도 포함된 실측이다.</li><li>Mean/rstd 저장, backward 연결·성능, 전체 config 재튜닝은 이 비교에 포함하지 않았다.</li></ul>
<p><a href="assets/trimul-ln-only-save.md">상세 보고서</a> · <a href="assets/trimul-ln-only-save-L384.json">L384 결과</a> · <a href="assets/trimul-ln-only-save-L768.json">L768 결과</a></p></section><!-- LN_ONLY_SAVE_END -->'''
p=SITE/'dist/trimul.html';s=p.read_text()
if '<!-- LN_ONLY_SAVE_BEGIN -->' in s:s=re.sub(r'<!-- LN_ONLY_SAVE_BEGIN -->.*?<!-- LN_ONLY_SAVE_END -->',lambda _:section,s,flags=re.S)
else:s=s.replace('<!-- ONCHIP_NEXT_BEGIN -->',section+'<!-- ONCHIP_NEXT_BEGIN -->',1)
if 'href="#ln-only-save"' not in s:s=s.replace('<nav>','<nav><a href="#ln-only-save">LN 저장 비용</a>',1)
data={str(n):{k:round(j['full_times'][k]['median_us']/1000,6) for k in labels} for n,j in D.items()}
script='''<script id="ln-save-chart">(()=>{const data=DATA,labels=LABELS;const draw=n=>{const rows=data[n],max=Math.max(...Object.values(rows));document.getElementById('lns-bars').innerHTML=Object.entries(rows).map(([k,v])=>`<div class="barline" style="grid-template-columns:170px 1fr 88px"><span>${labels[k]}</span><div class="track"><div class="fill ${k==='0'?'up':'slow'}" style="width:${v/max*100}%"></div></div><b>${v.toFixed(3)} ms</b></div>`).join('');document.querySelectorAll('[data-lns-l]').forEach(b=>{const active=b.dataset.lnsL===n;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});};document.querySelectorAll('[data-lns-l]').forEach(b=>b.addEventListener('click',()=>draw(b.dataset.lnsL)));draw('384');})();</script>'''.replace('DATA',json.dumps(data)).replace('LABELS',json.dumps(labels,ensure_ascii=False))
if '<script id="ln-save-chart">' in s:s=re.sub(r'<script id="ln-save-chart">.*?</script>',lambda _:script,s,flags=re.S)
else:s=s.replace('</body>',script+'</body>')
p.write_text(s)
p=SITE/'dist/index.html';s=p.read_text();a=s.index('<!-- TRAINING_LINK_BEGIN -->');b=s.index('<!-- TRAINING_LINK_END -->',a)+len('<!-- TRAINING_LINK_END -->');s=s[:a]+'''<!-- TRAINING_LINK_BEGIN --><section class="notice"><b>최신: LN activation만 저장하는 forward 비교</b> · <a href="trimul.html#ln-only-save">표·저장 배선·검증 →</a><br>입력만 +3%, 출력만 +4–7%, 둘 다 +8–11%. 다른 activation과 LN 통계는 추가 저장하지 않았다. <a href="trimul.html#recompute-next">Backward 최적화 현황</a></section><!-- TRAINING_LINK_END -->'''+s[b:];p.write_text(s)
print('Generated report and LN-only-save dashboard section')
