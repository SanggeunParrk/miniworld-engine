from pathlib import Path
import json,csv,hashlib,html
R=Path(__file__).resolve().parent;root=R.parents[1];E=R.parent/'trimul_b1_sol90_20260921';site=root/'runs/anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows=[];profiles=[];bounds=[];gains={};validation={}
for n in (384,768):
 d=json.loads((R/('results-L%d.json'%n)).read_text())
 for p,h in d['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 assert all(v['bit_exact'] for v in d['checks']['optimized'].values()) and all(v['bit_exact'] for v in d['mutated']['optimized'].values())
 for name,label in [('baseline','직전 선택 B1'),('optimized','이번 최적화 B1')]:rows.append([str(n),label,*['%.4f'%(d['times'][k][name]['median_us']/1000) for k in ('forward','b1','backward','forward_backward')]])
 gains[str(n)]={k:100*(1-d['times'][k]['optimized']['median_us']/d['times'][k]['baseline']['median_us']) for k in d['times']}
 for name in ('baseline','optimized'):
  rs=list(csv.DictReader((R/('ncu-%s-L%d-raw.csv'%(name,n))).open()));u,v=rs[0],rs[1]
  keys=('gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio')
  profiles.append([str(n),name,*[v[k]+' '+u[k] for k in keys]])
  if name=='optimized':
   observed_us=float(v['gpu__time_duration.sum']);bytes_=(float(v['dram__bytes_read.sum'])+float(v['dram__bytes_write.sum']))*1e6
   mem_us=1800*n*n/3.35e6;gemm_us=262144*n*n/989.5e6
   bounds.append(dict(L=n,ideal_unique_payload_bytes=1800*n*n,dense_gemm_flops=262144*n*n,ideal_memory_us=mem_us,ideal_gemm_us=gemm_us,ideal_roofline_efficiency_pct=100*max(mem_us,gemm_us)/observed_us,measured_traffic_roofline_efficiency_pct=100*bytes_/3.35e6/observed_us,ncu_us=observed_us))
 validation[str(n)]=json.loads((R/('verification-L%d.json'%n)).read_text());assert len(validation[str(n)])==(6 if n==384 else 5) and all(v['returncode']==0 for v in validation[str(n)])
summary=dict(rows=rows,latency_reduction_percent=gains,ncu=profiles,roofline=bounds,validation=validation,verification_complete=True,policy_unchanged=True,production_ready=False,sol90_reached=False,baseline='trimul_b1_tri_opt_20260921 matched on node01',experimental_source=str(E))
(R/'summary.json').write_text(json.dumps(summary,indent=2))
def mdtable(h,rs):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rs)
head=['L','경로','FWD ms','B1–B4 ms','BWD ms','전체 ms']
text='''# B1–B4: node01 검증 완료, SoL90 최적화 진행 중

2026-09-21. Anthropic 파생 CUDA/TMA/WGMMA 학습 개발 경로. 기준은 직전 `trimul_b1_tri_opt_20260921`을 node01에서 함께 측정한 결과다. Anthropic 추론이나 cuEquivariance 대비 수치가 아니다.

## 결과

양방향 C128/H256 BF16, L384/L768, mask/dropout25%/residual. 구간별600회 교대 CUDA graph 중앙값. Live packing, Wp 전치, cuBLAS, B7, 전체11 gradients 포함. Optimizer/RNG 생성/컴파일/CPU dispatch 제외. FWD는 같은 구현이므로 작은 차이는 변동이다.

'''+mdtable(head,rows)+'\n\n'
for n,g in gains.items():text+='- L%s: B1 %.2f%%, BWD %.2f%%, 전체 %.2f%% 지연 감소.\n'%(n,g['b1'],g['backward'],g['forward_backward'])
text+='''
## 구현

- 다음 tri/x_n/mean/rstd의 TMA를 출력 LN 미분 중으로 앞당겼다. FP32 통계를 double buffer하여 현재 값이 덮이지 않게 했다.
- 출력 LN affine 재구성을 두 warp-group에 분산했다.
- 두 dNorm WGMMA 타일을 함께 발행했다.
- BF16 dNorm 조각을 레지스터에 유지하여 shared-memory 왕복을 줄였다. 정규화 tri 값은 계속 원본 tri와 mean/rstd에서 재구성한다.
- dTri TMA 저장 폭을 튜닝했다. CTA128/120/96은132보다 느려 채택하지 않았다.

저장 정책: 입력 affine x_n BF16, 원본 tri BF16, 출력 mean/rstd FP32. 출력 LN activation 저장 없음. HBM activation/gradient와 수식, 반올림, CTA별 reduction 순서는 그대로다. B7/cuBLAS도 같다. 상세 실험은 `../trimul_b1_sol90_20260921/README.md` 및 단계별 JSON에 있다.

## NCU

'''+mdtable(['L','경로','시간','DRAM read','DRAM write','DRAM peak','Tensor peak','long-scoreboard/issue'],profiles)+'''

NCU 시간은 위 CUDA event 벤치와 별개다. 처리율은 전체 커널의 SoL과 같지 않다.

## SoL90 미달

낙관적 streaming roofline은 `max(1800*L²/3.35TBps, 262144*L²/989.5TFps)`로 계산한다. 이는 큰 입력/출력을 한 번씩 읽고 쓰는 payload 기준이며 작은 parameter/scratch와 pointwise/shared-memory/instruction/dependency 비용을 생략한 단순 모델이다. 따라서 달성 가능한 정확한 최단시간이라고 주장하지 않는다. 현재 구현이 실제 이동한 DRAM 바이트를 사용하는 고정 스케줄 roofline도 별도로 표시한다. 추가 재읽기를 필수 연산량으로 둔갑시켜 SoL을 높게 표시하지 않는다.

'''+mdtable(['L','고유 payload 메모리 하한 μs','GEMM 하한 μs','단순 이상 모델 효율','실측 트래픽 roofline 효율'],[[str(b['L']),'%.2f'%b['ideal_memory_us'],'%.2f'%b['ideal_gemm_us'],'%.1f%%'%b['ideal_roofline_efficiency_pct'],'%.1f%%'%b['measured_traffic_roofline_efficiency_pct']] for b in bounds])+'''

[NVIDIA H100 공식 사양](https://www.nvidia.com/en-us/data-center/h100/): SXM 3.35TB/s, BF16 1979TF/s는 sparsity 포함이므로 dense989.5TF/s를 사용했다. node01 독립 대역폭 교정은3.105TB/s였다. 현 경로에는 dWgate를 위한 x_n/dGate 재읽기가 남아 있어, 스케줄을 더 바꿀 여지가 있다. 출력 LN 미분/저장 단계와 재계산 단계의 대기가 주요 후속 대상이다.

## 검증 및 남은 제한

일반 입력, 입력/가중치/mask/dropout/dy 변경, gamma_out=0, graph/eager 모두 출력과11 gradients가 직전 기준과 bit-exact다. memcheck L384/L768 0 errors, racecheck L384 0 hazards. 새 B1의 검증 완료이며, 기존 B7 L768 독립 기준 dWL 상대L2 0.055569%(한도0.05%) 문제는 남아 있다. Production 승격이나 SoL90 달성을 뜻하지 않는다.
'''
(R/'README.md').write_text(text)
(root/'TRIMUL_STATUS.md').write_text(text)
assets=site/'assets';(assets/'trimul-b1-sol90.md').write_text(text);(assets/'trimul-b1-sol90-summary.json').write_text(json.dumps(summary,indent=2))
for n in (384,768):
 for stem in ('results','verification'):(assets/('trimul-b1-sol90-%s-L%d.json'%(stem,n))).write_bytes((R/('%s-L%d.json'%(stem,n))).read_bytes())
(assets/'trimul-b1-sol90-calibration.json').write_bytes((E/'calibration-node01.json').read_bytes())
body=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
section='''<!-- B1_SOL90_BEGIN --><section class="panel" id="b1-sol90"><span class="pill green">2026-09-21 · 현재 학습 개발 경로 · node01</span><h2>B1–B4 추가 최적화 · SoL90은 진행 중</h2><p><b>직전 B1 대비 지연 L384 −%.2f%% / L768 −%.2f%%.</b> 다음 입력 TMA를 LN 미분 중에 발행하고, LN 재계산을 두 warp-group에 분산했다. dNorm WGMMA를 묶고 BF16 조각을 레지스터에 유지한다. 저장 정책은 입력 x_n + 원본 tri + 출력 mean/rstd 그대로다.</p><div class="table-wrap"><table><thead><tr>'''%(gains['384']['b1'],gains['768']['b1'])+''.join('<th>'+x+'</th>' for x in head)+'''</tr></thead><tbody>'''+body+'''</tbody></table></div><p class="muted">node01 H100 · 양방향 C128/H256 BF16 · dropout25%%/mask/residual · 구간별600회 교대 CUDA graph 중앙값. FWD는 같은 구현이다. 기준은 직전 자체 학습 커널이며 Anthropic 추론/cuEq 비교가 아니다.</p><p>실측 트래픽 roofline 효율은 L384 %.1f%% / L768 %.1f%%. 고유 입력·출력만 사용하는 낙관적 모델은 %.1f%% / %.1f%%다. 어느 쪽도90%%가 아니다. DRAM 사용률을 전체 SoL로 부르지 않는다.</p>'''%(bounds[0]['measured_traffic_roofline_efficiency_pct'],bounds[1]['measured_traffic_roofline_efficiency_pct'],bounds[0]['ideal_roofline_efficiency_pct'],bounds[1]['ideal_roofline_efficiency_pct'])+'''<div class="notice amber">11 gradients bit-exact, memcheck 두 길이와 racecheck L384 통과. 기존 B7 독립 기준 정확도 문제 때문에 production 승격은 보류한다.</div><p><a href="assets/trimul-b1-sol90.md">변경·검증·SoL 정의</a> · <a href="assets/trimul-b1-sol90-summary.json">실측 JSON</a> · <a href="assets/trimul-b1-sol90-calibration.json">대역폭 교정</a> · <a href="#current-wiring">현재 HBM/수식 배선도</a></p></section><!-- B1_SOL90_END -->'''
p=site/'trimul.html';s=p.read_text()
if '<!-- B1_SOL90_BEGIN -->' in s:
 a=s.index('<!-- B1_SOL90_BEGIN -->');b=s.index('<!-- B1_SOL90_END -->',a)+len('<!-- B1_SOL90_END -->');s=s[:a]+section+s[b:]
else:s=s.replace('<main>','<main>'+section,1)
a=s.index('<!-- B1_TRI_OPT_BEGIN -->');b=s.index('<!-- B1_TRI_OPT_END -->',a);s=s[:a]+s[a:b].replace('현재 학습 개발 경로','직전 구현 이력 · 위에서 추가 최적화')+s[b:];p.write_text(s)
print('FULLY_VERIFIED',gains)
