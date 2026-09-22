from pathlib import Path
import json,hashlib,html
R=Path(__file__).resolve().parent;root=R.parents[1];old=R.parent/'trimul_ln_policy_v4_20260921';site=root/'runs/anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows=[];gains={};evidence={}
for n in (384,768):
 d=json.loads((R/('results-L%d.json'%n)).read_text());prior=json.loads((old/('final-L%d.json'%n)).read_text())
 for rec in (d,prior):
  for f,h in rec['source_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 assert d['cubins']['tri_stats']==prior['cubins']['stats_only']
 assert all(v['bit_exact'] for v in d['checks']['tri_stats'].values()) and all(v['bit_exact'] for v in d['mutated']['tri_stats'].values())
 for name,label in [('baseline','tri + LN 통계 재계산'),('xhat_fp32','폐기: FP32 정규화 저장'),('tri_stats','현재: tri + 저장 통계')]:
  rows.append([str(n),label,*['%.4f'%(d['times'][k][name]['median_us']/1000) for k in ('forward','b1','backward','forward_backward')]])
 gains[str(n)]={name:{k:100*(1-d['times'][k]['tri_stats']['median_us']/d['times'][k][name]['median_us']) for k in ('forward','b1','backward','forward_backward')} for name in ('baseline','xhat_fp32')}
 evidence[str(n)]=dict(saved_policy=d['saved_policy'],same_b1_cubin_as_previous_sanitized_stats_path=True,prior_sanitizer_records=json.loads((old/('verification-L%d.json'%n)).read_text()),current_all_gradients_bit_exact=True)
def mdtable(head,rows):return '| '+' | '.join(head)+' |\n|'+ '|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows)
head=['L','경로','FWD ms','B1–B4 ms','BWD ms','전체 ms']
text='''# 현재 TriMul 학습 개발 정책: tri + 저장 통계

2026-09-21 · 사용자 지시에 따라 출력 LayerNorm activation 저장 정책을 폐기하고 BF16 tri 입력으로 복구했다.

## 저장 정책과 진입점

- 현재 진입점: `runs/trimul_training_current.py:Training`.
- 구현: `tri_policy.py:Training` → 기존 검증된 `trimul_ln_policy_v4_20260921/final.py:Replacement(...,-1)`.
- 입력 affine BF16 `x_n` 저장은 유지한다.
- 출력 쪽 큰 activation은 contraction이 이미 만든 BF16 `tri`만 보존한다. 복사하지 않는다.
- K3는 행별 FP32 평균 `mu_out`와 역표준편차 `rstd_out`만 추가 저장한다.
- 출력 pre-affine `xhat` 및 affine `z`는 저장하지 않는다. 출력 projection/gate도 저장하지 않는다.
- B1은 tri와 통계를 TMA로 읽고 shared memory에서 `h=(float(tri)-mu)*rstd`, `z=BF16(h*gamma+beta)`를 재구성한다. 평균·분산 reduction은 다시 하지 않는다.
- B4는 재구성 h와 저장 rstd로 LN 미분을 계산한다. B7 및 cuBLAS 연결은 유지한다.
- 정책 선택은 `runs/trimul_training_selection.json`에 명시했다. Production dispatch는 별개이며 기존 B7 정확도 문제로 승격하지 않았다.

## 재측정

node02 H100 GPU 2개에서 길이별 독립 실행. 양방향 C128/H256 BF16, mask, dropout25%, residual 포함. 구간별 600회 교대 CUDA graph 중앙값. Live packing, Wp 전치, cuBLAS, 전체11개 gradient 포함. Optimizer/RNG 생성/CPU dispatch/compile 제외. 구간 중앙값 합과 전체 실측은 다를 수 있다.

'''+mdtable(head,rows)+'\n\n'
for n,g in gains.items():text+='- L%s: 폐기한 FP32 저장 경로 대비 B1 지연 %.2f%%, BWD %.2f%%, 전체 %.2f%% 감소. 기존 통계 재계산 기준 대비 전체 %.2f%% 감소.\n'%(n,g['xhat_fp32']['b1'],g['xhat_fp32']['backward'],g['xhat_fp32']['forward_backward'],g['baseline']['forward_backward'])
text+='''
## 검증

- L384·768: 일반 입력, 가중치/입력/mask/dropout/dy 변경, gamma_out=0을 포함한 출력·11개 gradient가 기존 tri 기준과 bit-exact.
- CUDA graph replay와 eager bit-exact.
- 보존 tensor가 원본 BF16 tri와 같은 주소이며, 출력 LN activation allocation이 없고 통계만 FP32 `[2M]`임을 검사했다.
- 현재 B1 cubin은 기존 stats 경로의 cubin과 동일하다. 해당 소스 SHA-256도 검증했다. 기존 memcheck/racecheck 성공 기록을 재사용하며, 이번 새 배선의 수치 검사는 별도로 다시 실행했다.
- `current-entry-check.json`은 새 공통 진입점을 직접 실행한 별도 검사다.
- 기존 B7 L768 dWL 독립 기준 상대L2 0.055569% 문제(한도0.05%)는 남아 있다. 위 bit-exact는 기존 구현과의 일치이며 독립 기준 문제 해결을 뜻하지 않는다.

## 보존 메모리

출력 쪽 tri+통계: L384 76.68 MB, L768 306.71 MB. 폐기한 FP32 xhat+rstd의151.58/606.34 MB보다 작다. 입력 x_n 및 left/right는 동일하게 보존한다. MB는 십진 버퍼 크기이며 실측 DRAM 트래픽이 아니다.
'''
(R/'README.md').write_text(text)
summary=dict(policy=json.loads((R/'selection.json').read_text()),rows=rows,latency_reduction_percent=gains,verification=evidence)
if (R/'current-entry-check.json').exists():summary['entry_check']=json.loads((R/'current-entry-check.json').read_text())
(R/'summary.json').write_text(json.dumps(summary,indent=2))
for n in (384,768):(site/'assets'/('trimul-tri-restore-L%d.json'%n)).write_bytes((R/('results-L%d.json'%n)).read_bytes())
(site/'assets/trimul-tri-restore.md').write_text(text);(site/'assets/trimul-tri-restore-summary.json').write_text(json.dumps(summary,indent=2))
body=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
section='''<!-- TRI_RESTORE_BEGIN --><section class="panel" id="tri-restore"><span class="pill green">2026-09-21 · 현재 학습 개발 경로</span><h2>출력 LN activation 저장 폐기 → BF16 tri + 평균·역표준편차</h2><p><b>출력 x̂/z는 저장하지 않는다.</b> K3는 입력 x_n 저장을 유지하며 출력 mean/rstd만 추가 저장한다. B1은 contraction의 원본 tri를 읽어 (tri−mean)×rstd와 affine를 재구성한다. 평균·분산 reduction은 재계산하지 않는다.</p><div class="table-wrap"><table><thead><tr>'''+''.join('<th>'+x+'</th>' for x in head)+'''</tr></thead><tbody>'''+body+'''</tbody></table></div><p>폐기한 FP32 저장 경로 대비 전체 지연 <b>L384 −7.9%, L768 −7.7%</b>. 두 길이 출력·11 gradients 및 변경 입력 graph replay는 기존 tri 기준과 bit-exact다.</p><p class="muted">node02 H100 · 양방향 C128/H256 BF16 · mask/dropout25%/residual · 구간별600회 교대 CUDA graph 중앙값. 전체 = fwd + 전체 bwd + live packing. Anthropic 원본/cuEq 대비 수치가 아니다.</p><div class="notice amber"><b>개발 설정으로 선택.</b> 기존 B7 L768 독립 기준 정확도 문제는 남아 있어 production 승격과는 구분한다. 기존 검증된 동일 cubin을 재사용하고 새 배선·공통 진입점은 다시 검증했다.</div><p><a href="assets/trimul-tri-restore.md">현재 정책·측정 보고서</a> · <a href="assets/trimul-tri-restore-summary.json">원본 결과와 검증 근거</a> · <a href="#current-wiring">수정한 수식 / HBM 배선도</a></p></section><!-- TRI_RESTORE_END -->'''
p=site/'trimul.html';s=p.read_text()
if '<!-- TRI_RESTORE_BEGIN -->' in s:
 a=s.index('<!-- TRI_RESTORE_BEGIN -->');b=s.index('<!-- TRI_RESTORE_END -->',a)+len('<!-- TRI_RESTORE_END -->');s=s[:a]+section+s[b:]
else:s=s.replace('<main>','<main>'+section,1)
a=s.index('<!-- XHAT_NATIVE_BEGIN -->');b=s.index('<!-- XHAT_NATIVE_END -->',a);h=s[a:b].replace('現 학습 개발 경로','폐기 · 출력 activation 저장 실험').replace('현재 native FP32 x̂','당시 native FP32 x̂').replace('현재 개발 방향은 정규화 값 저장으로 유지한다.','이 저장 정책은 폐기했고 현재는 위 tri+통계 경로를 사용한다.');s=s[:a]+h+s[b:]
s=s.replace('saved FP32 xhat/rstd + no-tri B1 + split_xn_pc1 B7','BF16 tri + saved mean/rstd + shared B1 + split_xn_pc1 B7')
s=s.replace('출력 pre-affine 정규화 x̂와 rstd를 K3에서 저장한다. B1은 raw tri를 읽지 않고 정규화도 재계산하지 않는다. affine/projection/gate만 재계산한다.','출력 정규화/affine activation 저장은 폐기했다. K3는 평균·역표준편차만 저장하고, B1은 원본 BF16 tri로 정규화/affine/projection/gate를 재계산한다. 평균·분산 reduction은 생략한다.')
p.write_text(s)
print(json.dumps(gains,indent=2))
