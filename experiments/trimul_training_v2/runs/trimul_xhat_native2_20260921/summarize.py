from pathlib import Path
import json,hashlib,csv,html
R=Path(__file__).resolve().parent;root=R.parents[1];site=root/'runs/anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows=[];ncu=[]
for n in (384,768):
 d=json.loads((R/('results-L%d.json'%n)).read_text());assert set(d['times'])=={'forward','b1','backward','forward_backward'}
 for name,digest in d['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
 for name,label in [('baseline','raw tri 재계산 기준'),('prior_xhat','이전 FP32 x̂'),('xhat_fp32','현재 native FP32 x̂')]:
  t=[d['times'][s][name]['median_us']/1000 for s in ('forward','b1','backward','forward_backward')];rows.append([str(n),label,*['%.4f'%v for v in t]])
 for variant in ('prior_xhat','xhat_fp32'):
  p=R/('ncu-%s-L%d-raw.csv'%(variant,n))
  if p.exists():
   a=list(csv.DictReader(p.open()));units,v=a[0],a[1]
   def value(k):return v.get(k,'?')+' '+units.get(k,'')
   ncu.append([str(n),variant,*[value(k) for k in ('gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed')]])
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+ '|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows)
head=['L','경로','FWD ms','B1–B4 ms','BWD ms','전체 ms']
text='''# FP32 정규화 값 저장 / raw tri 없는 B1–B4

2026-09-21 · 현재 학습 개발 방향. Anthropic 파생 CUDA / H100 node02 / 양방향 C128 / dropout 25%, mask, residual / L384·768.

## 실제 저장·계산 정책

- Forward K3에서 입력 affine `x_n`(BF16), 출력 pre-affine `xhat`(FP32), 출력 `rstd`(FP32)를 저장한다.
- B1은 `tri`도 평균도 받지 않는다. 평균·분산 reduction 및 `(tri-mean)*rstd` 재계산이 없다.
- `xn_out = xhat * gamma_out + beta_out`만 복원한다. Projection/gate 및 그 미분은 기존 융합 구조로 계산한다.
- LN 미분은 `q=dNorm*gamma`, `dTri=rstd*(q-mean(q)-xhat*mean(q*xhat))`를 사용한다.
- Input LN 값은 계속 저장한다. B7은 기존 split_xn_pc1 구현을 유지한다.
- 학습 어댑터: `training.py:Training`. Production dispatch를 바꾸지는 않았다.

## 구현 변경

1. Gate 가중치를 shared memory에 유지해 타일별 재로드를 없앴다.
2. FP32 xhat를 CUDA fragment 순서의 native 타일 배치로 저장한다. 물리 shape `[M/64,16,4,2,32,4]`, 논리 shape `[M,256]`.
3. Forward는 정렬된 128-bit vector store, backward는 16 KiB ×4의 H100 TMA bulk copy로 읽는다. 읽은 값은 shared memory의 vector load로 곧바로 소비한다. 별도 layout 변환 커널은 없다.
4. Forward 저장을 affine 계산의 의존 순서에 배치해 K3의 register spill을 제거했다. 재계산이나 저정밀 압축으로 바꾸지 않았다.
5. dTri는 LN 미분이 끝난 뒤 비게 된 shared memory를 재사용해 TMA로 출력한다.
6. 길이별 K3 후보64개 중 컴파일/파이프라인 조건을 통과한44개와 B1 20개를 검증·교대 측정했다. K3 `(BI,BJ,slots,acc,regs,serial)=(1,128,4,1,1,1)`, B1 132 CTA·unroll8 선택. 실제 탐색 결과는 `tune-L*.json`.

## 전체 모듈 측정

각 구간별 600회 CUDA graph 교대 측정 중앙값. weight packing, backward Wp 전치, 양방향 cuBLAS, B7, 11개 gradient를 포함한다. Optimizer/RNG 생성/컴파일/CPU dispatch는 제외한다. 구간별 중앙값을 더한 값과 전체 실측은 다를 수 있다.

'''+table(head,rows)+'''\n
비교 대상 `prior_xhat`는 이전 정규화 FP32 저장 경로이며, `baseline`은 raw tri에서 출력 LN을 재계산하는 shared B1 경로다. Anthropic 원본이나 cuEq 대비 수치가 아니다. raw tri 기준은 비교용이며 현재 선택 경로로 되돌리지 않는다.

## 검증과 한계

- L384·768 출력 및 11개 gradient: raw tri 기준과 bit-exact. 입력/가중치/mask/dropout/dy 변경 및 gamma_out=0 검사 포함.
- Forward 이후 raw tri를 NaN으로 덮어써도 모든 backward 결과 bit-exact. backward 보존 tensor에 tri가 없음을 확인했다.
- CUDA graph replay와 eager 결과 bit-exact.
- FP32 xhat+rstd 저장 payload: L384 151.58 MB, L768 606.34 MB. BF16 tri를 보존할 때보다 커지므로 LN 재계산 제거가 자동으로 전체 속도 향상을 의미하지 않는다.
- 기존 B7의 L768 dWL 독립 기준 상대 L2 0.055569% 문제(한도0.05%)는 이 작업과 별개로 남아 있다. 기존/신규가 같은 값이므로 production 승격은 보류한다.
'''
verification={}
for n in (384,768):
 p=R/('verification-L%d.json'%n)
 if p.exists():verification[str(n)]=json.loads(p.read_text())
text+='\nSanitizer/NCU 실행 상태: `verification-L*.json`.\n'
if all(len(verification.get(str(n),[])) >= (6 if n==384 else 5) and all(x['returncode']==0 for x in verification[str(n)]) for n in (384,768)):
 text+='\nL384·768 전체 memcheck 0 errors. L384 B1/K3 racecheck 0 hazards. NCU 두 길이 모두 정상 완료.\n'
if ncu:text+='\n## B1 NCU 실측\n\n'+table(['L','경로','시간','DRAM read','DRAM write','DRAM peak %','Tensor peak %'],ncu)+'\n\nNCU 시간은 replay/profiling 비용을 포함한 별도 실행이므로 위 CUDA event 벤치 시간과 섞지 않는다. peak 비율만으로 전체 알고리즘 SoL90 달성을 주장하지 않는다.\n'
(R/'README.md').write_text(text)
summary=dict(rows=rows,ncu=ncu,verification=verification,policy=dict(save=['input_affine_bf16_xn','output_pre_affine_fp32_xhat','output_fp32_rstd'],b1_reads_tri=False,b1_recomputes_normalization=False),production_ready=False)
(R/'summary.json').write_text(json.dumps(summary,indent=2))
for n in (384,768):
 for stem in ('results','verification'):
  p=R/('%s-L%d.json'%(stem,n))
  if p.exists():(site/'assets'/('trimul-xhat-native-%s-L%d.json'%(stem,n))).write_bytes(p.read_bytes())
(site/'assets/trimul-xhat-native.md').write_text(text);(site/'assets/trimul-xhat-native-summary.json').write_text(json.dumps(summary,indent=2))
tbody=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in rows)
section='''<!-- XHAT_NATIVE_BEGIN --><section class="panel" id="xhat-native"><span class="pill green">2026-09-21 · 現 학습 개발 경로</span><h2>정규화 x̂ + rstd 저장: B1에서 raw tri와 LN 정규화 재계산 제거</h2><p><b>K3 → FP32 x̂/rstd → B1.</b> B1은 저장한 pre-affine 정규화 값에 γ/β만 적용한다. 평균·분산 계산과 (tri−μ)×rstd는 없다. Projection/gate와 gradient는 같은 커널 안에서 계산한다.</p><p>Native 타일 배치 + 128-bit 저장 / H100 TMA bulk 읽기, gate 가중치 shared 상주, dTri TMA 출력. K3 spill 제거 후 길이별 K3 후보64개 중 실행 가능한44개·B1 20개 설정을 비교했다.</p><div class="table-wrap"><table><thead><tr>'''+''.join('<th>'+h+'</th>' for h in head)+'''</tr></thead><tbody>'''+tbody+'''</tbody></table></div><p class="muted">node02 H100 · 양방향 C128 BF16 · mask/dropout25%/residual · 600회 교대 CUDA graph 중앙값. 전체 = live packing + fwd + 전체 bwd. Anthropic 원본/cuEq 대비가 아니다.</p><div class="notice amber"><b>이전 FP32 저장 경로 대비 B1 −1.4/−2.8%, 전체 −0.55/−0.43%.</b> 작은 개선이며 raw tri 기준보다 전체 약6.4/6.8% 느리다. 정규화 값을 FP32로 보존하므로 BF16 tri보다 저장량이 크다. 현재 개발 방향은 정규화 값 저장으로 유지한다. L384/768 출력·11 gradients bit-exact, tri NaN poison 통과. 기존 B7의 L768 독립 기준 오차 문제로 production 승격은 보류.</div><p><a href="assets/trimul-xhat-native.md">구현·검증·NCU 보고서</a> · <a href="assets/trimul-xhat-native-summary.json">실측 원본 요약</a> · <a href="#current-wiring">새 수식 / HBM 배선도</a></p></section><!-- XHAT_NATIVE_END -->'''
p=site/'trimul.html';s=p.read_text()
if '<!-- XHAT_NATIVE_BEGIN -->' in s:
 a=s.index('<!-- XHAT_NATIVE_BEGIN -->');b=s.index('<!-- XHAT_NATIVE_END -->',a)+len('<!-- XHAT_NATIVE_END -->');s=s[:a]+section+s[b:]
else:s=s.replace('<main>','<main>'+section,1)
a=s.index('<!-- LN_POLICY_TUNED_BEGIN -->');b=s.index('<!-- LN_POLICY_TUNED_END -->',a)
h=s[a:b].replace('선택한 학습 개발 경로','이전 통계 저장 실험 · 현재는 위 x̂ 저장 경로').replace('출력 통계 TMA 저장 (선택)','출력 통계 TMA 저장 (당시 선택)').replace('<h2>출력 LN 통계만 저장:', '<h2>이력 · 출력 LN 통계만 저장:');s=s[:a]+h+s[b:]
s=s.replace('saved output stats + shared B1 + split_xn_pc1 B7','saved FP32 xhat/rstd + no-tri B1 + split_xn_pc1 B7')
s=s.replace('출력 LN 평균·역표준편차는 K3에서 저장하고 B1이 TMA로 읽는다. 출력 LN activation/projection/gate는 저장하지 않는다.','출력 pre-affine 정규화 x̂와 rstd를 K3에서 저장한다. B1은 raw tri를 읽지 않고 정규화도 재계산하지 않는다. affine/projection/gate만 재계산한다.')
p.write_text(s)
print(json.dumps(summary,indent=2))
