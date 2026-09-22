from pathlib import Path
import hashlib,json,csv,html
R=Path(__file__).resolve().parent;root=R.parents[1];site=root/'runs/anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows=[];gains={};profiles=[];checks={};ablations=[]
for n in (384,768):
 d=json.loads((R/('results-L%d.json'%n)).read_text())
 for p,h in d['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 assert all(v['bit_exact'] for v in d['checks']['optimized'].values()) and all(v['bit_exact'] for v in d['mutated']['optimized'].values())
 for name,label in [('baseline','직전 tri+통계 B1'),('optimized','현재 복사 제거+조기 로드')]:
  rows.append([str(n),label,*['%.4f'%(d['times'][k][name]['median_us']/1000) for k in ('forward','b1','backward','forward_backward')]])
 gains[str(n)]={k:100*(1-d['times'][k]['optimized']['median_us']/d['times'][k]['baseline']['median_us']) for k in d['times']}
 for name in ('baseline','optimized'):
  a=list(csv.DictReader((R/('ncu-%s-L%d-raw.csv'%(name,n))).open()));u,v=a[0],a[1]
  profiles.append([str(n),name,*[v[k]+' '+u[k] for k in ('gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio')]])
 checks[str(n)]=json.loads((R/('verification-L%d.json'%n)).read_text())
 for t in json.loads((R/('tune-L%d.json'%n)).read_text()):
  f=t['config']['defines']
  if not f['PAIR_WP'] and not f['B1_PREFETCH_DY']:
   ablations.append([str(n),str(f['B1_DIRECT_DP']),str(f['B1_NO_REDUNDANT_SYNC']),str(f['B1_EARLY_RAW']),'%.3f'%t['times']['candidate'],'%.2f%%'%(100*(1-t['ratio']))])
def table(h,rs):return '| '+' | '.join(h)+' |\n|'+ '|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rs)
head=['L','경로','FWD ms','B1–B4 ms','BWD ms','전체 ms']
text='''# B1–B4: shared 복사 제거와 다음 tri 타일의 조기 로드

2026-09-21 · Anthropic 파생 CUDA/TMA/WGMMA 학습 개발 경로. 저장 정책은 BF16 tri + FP32 mean/rstd, 입력 affine x_n 저장을 그대로 유지한다. 출력 LayerNorm activation은 저장하지 않는다.

## 변경

1. **dProj 16 KiB shared 복사 제거.** B4의 dNorm GEMM이 dProj의 원래 shared 위치를 직접 읽는다. dNorm의32 KiB와 dProj의16 KiB는 겹치지 않는다.
2. **중복 CTA 동기화 제거.** 통계 저장 분기에서 gate 재계산 직후 연속 실행되던 두 fence/barrier 중 하나를 제거했다.
3. **다음 입력 TMA 조기 발행.** 현재 B4가 dNorm/dProj를 모두 소비한 후, 현재 dTri의 TMA store가 완료되기를 기다리는 동안 다음 tri/x_n/mean/rstd를 반대 슬롯으로 가져온다. 저장 통계는 현재 LN 미분에서 이미 소비한 뒤에 덮어쓴다.

수식·BF16 반올림·CTA별 reduction 순서·HBM 보존 텐서는 그대로다. 동일 CUDA launch의 Phase A/B 구조도 유지한다.

## 설정 탐색

직전의132 CTA, affine unroll L384=4/L768=8을 유지하고 복사 제거/중복 동기화 제거/dWproj 두 GEMM 묶기/조기 tri 로드/dy 프리패치의24개 유효 조합을 길이별로 비교했다. 모든 후보의 B1 여섯 출력이 bit-exact였다.

두 길이 모두 DIRECT_DP=1, NO_REDUNDANT_SYNC=1, EARLY_RAW=1, PAIR_WP=0, PREFETCH_DY=0을 선택했다. dWproj 묶기와 dy 프리패치는 추가 이득이 작거나 없었다. 이번 탐색은 구현 옵션24개이며 전체 타일/CTA 공간을 다시 탐색했다는 뜻은 아니다.

## 전체 학습 재측정

node02 H100 GPU2개에서 길이별 독립 실행. 양방향 C128/H256 BF16 · mask/dropout25%/residual · 구간별600회 교대 CUDA graph 중앙값. Live packing, Wp 전치, cuBLAS, B7, 전체11개 gradient 포함. Optimizer/RNG 생성/compile/CPU dispatch 제외. FWD는 같은 구현이며 작은 차이는 측정 변동이다.

'''+table(head,rows)+'\n\n'
for n,g in gains.items():text+='- L%s: B1 지연 %.2f%% 감소, BWD %.2f%% 감소, 전체 %.2f%% 감소.\n'%(n,g['b1'],g['backward'],g['forward_backward'])
text+='\n## 변경별 영향\n\n'+table(['L','dProj 복사 제거','중복 sync 제거','조기 raw 로드','B1 μs','해당 교대 기준 대비 감소'],ablations)+'\n\n각 행은 해당 후보와 기존 커널의 같은 시점 교대 측정이다. 작은 차이는 최적 조합의 전체 벤치로 다시 확인했다.\n'
text+='\n## NCU\n\n'+table(['L','경로','시간','DRAM read','DRAM write','DRAM peak %','Tensor peak %','long-scoreboard/issue'],profiles)+'''

DRAM 바이트는 사실상 동일하다. shared 복사와 대기를 줄여 동일한 작업을 더 빨리 처리한다. long-scoreboard/issue는 활성 발행당 메모리 의존 대기 지표이며 전체 시간의 백분율이 아니다. NCU profiling 시간은 위 CUDA event 벤치와 별도다. DRAM/Tensor peak 활용률만으로 알고리즘 SoL90을 주장하지 않는다.

## 검증

- 일반 입력 및 입력/가중치/mask/dropout/dy 변경·gamma_out=0에서 출력과11개 gradient가 직전 tri+통계 기준과 bit-exact.
- CUDA graph replay와 eager bit-exact. 원본 BF16 tri와 저장 FP32 통계만 남는 정책 검사 통과.
- 상세 sanitizer/NCU 실행 상태는 verification-L*.json.
- 기존 B7 L768 dWL 독립 기준 상대L2 0.055569%(한도0.05%) 문제는 남아 있다. 기존 구현과 일치한다는 검증이므로 production 승격을 뜻하지 않는다.
'''
complete=all(len(checks[str(n)])==(6 if n==384 else 5) and all(x['returncode']==0 for x in checks[str(n)]) for n in (384,768))
if complete:text+='\n새 B1: L384·768 memcheck 0 errors, L384 racecheck 0 hazards. 두 길이 NCU 완료.\n'
(R/'README.md').write_text(text)
sumry=dict(rows=rows,latency_reduction_percent=gains,ncu=profiles,validation=checks,verification_complete=complete,policy_unchanged=True,production_ready=False)
sumry['entry_check']=json.loads((R/'current-entry-check.json').read_text())
(R/'summary.json').write_text(json.dumps(sumry,indent=2))
for n in (384,768):
 for stem in ('results','verification'):(site/'assets'/('trimul-b1-tri-opt-%s-L%d.json'%(stem,n))).write_bytes((R/('%s-L%d.json'%(stem,n))).read_bytes())
(site/'assets/trimul-b1-tri-opt.md').write_text(text);(site/'assets/trimul-b1-tri-opt-summary.json').write_text(json.dumps(sumry,indent=2))
body=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
section='''<!-- B1_TRI_OPT_BEGIN --><section class="panel" id="b1-tri-opt"><span class="pill green">2026-09-21 · 현재 학습 개발 경로</span><h2>B1–B4 지연 −9.1% / −10.2%: dProj 복사 제거 + 다음 tri 조기 로드</h2><p><b>저장 정책은 그대로.</b> BF16 tri + FP32 mean/rstd, 입력 affine x_n 유지. dProj16KiB shared 복사를 없애고 연속 CTA 동기화를 줄였다. 현재 dTri를 TMA로 쓰는 동안 다음 tri/x_n/통계를 미리 읽는다.</p><div class="table-wrap"><table><thead><tr>'''+''.join('<th>'+x+'</th>' for x in head)+'''</tr></thead><tbody>'''+body+'''</tbody></table></div><p>구현 옵션24개를 길이별 비교. 최종 전체 학습 지연은 두 길이 모두 약1.9% 감소. FWD/cuBLAS/B7은 같은 구현이다. 출력·11 gradients는 변경 입력까지 bit-exact다.</p><p class="muted">node02 H100 · 양방향 C128/H256 BF16 · mask/dropout25%/residual · 구간별600회 교대 CUDA graph 중앙값. Anthropic 원본/cuEq 대비 수치가 아니다.</p><p>NCU에서 DRAM 바이트는 거의 같고, L768 DRAM 처리율은50.4→56.4%, long-scoreboard/issue는1.30→0.94. SoL90 달성을 뜻하지 않는다.</p><div class="notice amber">개발 설정. 기존 B7 L768 독립 기준 정확도 문제 때문에 production 승격은 보류한다.</div><p><a href="assets/trimul-b1-tri-opt.md">변경별 실험·검증·NCU</a> · <a href="assets/trimul-b1-tri-opt-summary.json">실측 요약</a> · <a href="#current-wiring">수식 / HBM 배선도</a></p></section><!-- B1_TRI_OPT_END -->'''
p=site/'trimul.html';s=p.read_text()
if '<!-- B1_TRI_OPT_BEGIN -->' in s:
 a=s.index('<!-- B1_TRI_OPT_BEGIN -->');b=s.index('<!-- B1_TRI_OPT_END -->',a)+len('<!-- B1_TRI_OPT_END -->');s=s[:a]+section+s[b:]
else:s=s.replace('<main>','<main>'+section,1)
a=s.index('<!-- TRI_RESTORE_BEGIN -->');b=s.index('<!-- TRI_RESTORE_END -->',a);h=s[a:b].replace('현재 학습 개발 경로','저장 정책 복구 이력 · 위에서 B1 추가 개선').replace('현재: tri + 저장 통계','복구 당시: tri + 저장 통계');s=s[:a]+h+s[b:]
s=s.replace('BF16 tri + saved mean/rstd + shared B1 + split_xn_pc1 B7','tri/stats + direct-dProj + early-prefetch B1 + split_xn_pc1 B7')
p.write_text(s)
print('VERIFICATION_COMPLETE',complete);print('GAINS',gains)
