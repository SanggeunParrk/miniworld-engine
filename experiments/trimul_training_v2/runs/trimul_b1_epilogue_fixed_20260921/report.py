from pathlib import Path
import csv,json,hashlib,html
R=Path(__file__).resolve().parent;root=R.parents[1];site=root/'runs/anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows=[];gains={};profiles=[];bounds=[];checks={}
for n in (384,768):
 d=json.loads((R/('results-L%d.json'%n)).read_text())
 for p,h in d['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 assert all(v['bit_exact'] for v in d['checks']['optimized'].values()) and all(v['bit_exact'] for v in d['mutated']['optimized'].values())
 for name,label in [('baseline','직전 v49 B1'),('optimized','현재 LN 합산/TMA/동기화 개선')]:rows.append([str(n),label,*['%.4f'%(d['times'][k][name]['median_us']/1000) for k in ('forward','b1','backward','forward_backward')]])
 gains[str(n)]={k:100*(1-d['times'][k]['optimized']['median_us']/d['times'][k]['baseline']['median_us']) for k in d['times']}
 for name in ('baseline','optimized'):
  a=list(csv.DictReader((R/('ncu-%s-L%d-raw.csv'%(name,n))).open()));u,v=a[0],a[1]
  ks=('gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio')
  profiles.append([str(n),name,*[v[k]+' '+u[k] for k in ks]])
  if name=='optimized':
   us=float(v['gpu__time_duration.sum']);traffic=(float(v['dram__bytes_read.sum'])+float(v['dram__bytes_write.sum']))*1e6
   bounds.append(dict(L=n,ncu_us=us,unique_payload_bytes=1800*n*n,gemm_flops=262144*n*n,ideal_roofline_pct=100*max(1800*n*n/3.35e6,262144*n*n/989.5e6)/us,measured_traffic_roofline_pct=100*traffic/3.35e6/us))
 checks[str(n)]=json.loads((R/('verification-L%d.json'%n)).read_text());assert len(checks[str(n)])==(6 if n==384 else 5) and all(x['returncode']==0 for x in checks[str(n)]),checks
stress=json.loads((R/'delayed-cta-check.json').read_text());assert len(stress)==6 and all(all(x['bit_exact']) for x in stress)
summary=dict(baseline='trimul_b1_sol90_selected_20260921, paired on node01',rows=rows,latency_reduction_percent=gains,ncu=profiles,roofline=bounds,validation=checks,delayed_cta_checks=stress,verification_complete=True,production_ready=False,sol90_reached=False,policy_unchanged=True)
def tab(h,rs):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rs)
head=['L','경로','FWD ms','B1–B4 ms','BWD ms','전체 ms']
text='''# B1–B4: LN parameter 합산 / TMA 저장 중첩 / CTA 내부 Phase 전환

2026-09-21, node01 H100. 기준은 직전 v49 개발 경로 `trimul_b1_sol90_selected_20260921`. Anthropic 추론이나 cuEquivariance 대비 수치가 아니다. 저장은 입력 affine x_n BF16, 원본 tri BF16, 출력 mean/rstd FP32. 출력 LayerNorm activation은 저장하지 않는다.

## 검증된 전체 학습 측정

양방향 C128/H256 BF16, L384/L768, mask/dropout25%/residual. 구간별600회 교대 CUDA graph 중앙값. Packing/Wp 전치/cuBLAS/B7/전체11 gradients 포함. Optimizer/RNG 생성/컴파일/CPU dispatch 제외. FWD는 같은 구현이며 작은 차이는 변동이다.

'''+tab(head,rows)+'\n\n'
for n,g in gains.items():text+='- L%s: B1 %.2f%%, BWD %.2f%%, 전체 %.2f%% 시간 감소.\n'%(n,g['b1'],g['backward'],g['forward_backward'])
text+='''
## 바뀐 구현

1. 출력 LN dgamma/dbeta의 warp shuffle 합산을32KiB shared scratch를 통한 균형 이진 합산으로 바꿨다. 죽은 dy/current x_n 버퍼를 재사용한다. 기존 덧셈 순서를 유지하며 별도 HBM tensor를 추가하지 않는다.
2. dNorm WGMMA 직후 다음 tri/x_n/stats TMA를 발행한다. **두 warp-group의 완료를 CTA barrier로 확인한 후에만** dProj가 있던 반대 슬롯을 덮어쓴다.
3. dTri TMA 저장을 먼저 발행하고, LN parameter 합산 중에 저장을 진행한다. 타일을 재사용하기 전 완료를 기다린다.
4. Phase B는 같은 CTA가 Phase A에서 쓴 dGate 행들만 읽으므로 첫 전체 grid barrier를 제거했다. CTA fence와 TMA 완료 보장은 유지한다. **모든 CTA의 parameter partial을 합치는 마지막 grid barrier는 유지한다.**

BF16 반올림/합산 순서/보존 tensor는 그대로다. B7와 cuBLAS도 바꾸지 않았다.

## 발견하고 고친 경쟁 조건

초기 조기 TMA 후보는 일반 벤치와 gradient 비교를 통과했지만 memcheck 환경에서 dTri/LN gradient가 달라졌다. WGMMA wait는 warp-group 단위라서 group0이 다음 입력으로 dProj를 덮을 때 group1이 아직 읽을 수 있었다. CTA barrier를 추가한 이 수정본으로 전체 측정과 sanitizer를 다시 실행했다. 초기 후보 `trimul_b1_epilogue_20260921`는 선택하거나 게시하지 않았다. 아래 결과는 수정 후 결과다.

## 제외한 실험

- dWgate 단일 루프 및 accumulator shared parking: 레지스터 spill/추가 이동 비용으로 느려 제외.
- dWgate TMA 버퍼2~7개 및 소유 타일2~3개 묶기: 추가 이득이 작아 제외.
- LN scratch 전치 +128-bit shared load: 더 느려 제외. 현재는 행 배치와 scalar shared load를 사용한다.
- dy 선행 로드: 추가 이득이 없고 LN scratch와 겹치므로 사용하지 않는다.

## NCU와 SoL

'''+tab(['L','경로','시간','DRAM read','DRAM write','DRAM peak','Tensor peak','long-scoreboard/issue'],profiles)+'\n\n'+tab(['L','실측 트래픽 roofline 효율','낙관적 고유 payload 모델 효율'],[[str(b['L']),'%.1f%%'%b['measured_traffic_roofline_pct'],'%.1f%%'%b['ideal_roofline_pct']] for b in bounds])+'''

**SoL90 미달.** NCU DRAM 처리율은 전체 SoL과 같지 않다. 이전과 같은 단순 모델 `max(1800*L²/3.35TBps, 262144*L²/989.5TFps)`를 사용했다. 고유 큰 tensor의 단일 입출력과 GEMM을 계산하며 scalar math/shared-memory/instruction/dependency/작은 scratch 비용을 생략한 낙관적 모델이다. 실측 트래픽 모델은 별도 dWgate의 x_n/dGate 재읽기도 포함하므로 알고리즘 효율과 구분한다. NCU sampling의 중첩 inline source 귀속을 합산해 시간 비율처럼 표시하지 않았다.

[NVIDIA H100 사양](https://www.nvidia.com/en-us/data-center/h100/)의 SXM3.35TB/s와 dense BF16 989.5TF/s(표의 sparsity1979TF/s 절반)를 사용했다. 별도 node01 streaming 교정은3.105TB/s였다. 정확한 전체 알고리즘 최단시간이 증명된 것은 아니다.

## 검증

- 일반 및 변경 입력/가중치/mask/dropout/dy, gamma_out=0: 출력과11 gradients가 기준과 bit-exact.
- Graph replay/eager bit-exact. BF16 원본tri + FP32mean/rstd 정책 검사 통과.
- memcheck L384/L7680 errors, racecheck L3840 hazards.
- CTA 일부를 Phase B 직전에 고의로 지연: 두 길이×3seed×20graph replay에서 B1 여섯 출력 bit-exact.
- 기존 B7 독립 기준 L768 dWL 상대L2 0.055569%(한도0.05%) 문제는 남아 있다. 이번 검증은 개발 경로 개선이며 production 승격을 뜻하지 않는다.
'''
(R/'summary.json').write_text(json.dumps(summary,indent=2));(R/'README.md').write_text(text);(root/'TRIMUL_STATUS.md').write_text(text)
assets=site/'assets';(assets/'trimul-b1-epilogue.md').write_text(text);(assets/'trimul-b1-epilogue-summary.json').write_text(json.dumps(summary,indent=2))
for n in (384,768):
 for stem in ('results','verification'):(assets/('trimul-b1-epilogue-%s-L%d.json'%(stem,n))).write_bytes((R/('%s-L%d.json'%(stem,n))).read_bytes())
body=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
section='''<!-- B1_EPILOGUE_BEGIN --><section class="panel" id="b1-epilogue"><span class="pill green">2026-09-21 · 현재 학습 개발 경로 · node01</span><h2>B1–B4 LN 합산과 TMA 저장을 겹치기</h2><p><b>직전 v49 대비 B1 지연 L384 −%.2f%% / L768 −%.2f%%.</b> LN parameter 합산을 shared scratch로 바꾸고 dTri 저장을 겹쳤다. CTA가 자신의 dGate를 소비하는 경계의 전체 GPU 동기화를 제거했다. 마지막 parameter 합산의 전체 동기화는 유지한다.</p><div class="table-wrap"><table><thead><tr>'''%(gains['384']['b1'],gains['768']['b1'])+''.join('<th>'+x+'</th>' for x in head)+'''</tr></thead><tbody>'''+body+'''</tbody></table></div><p class="muted">node01 H100 · 양방향 C128/H256 BF16 · dropout25%%/mask/residual · 구간별600회 교대 CUDA graph. FWD는 같은 구현. 비교 기준은 직전 자체 학습 경로다.</p><p>SoL90은 아직 미달이다. 실측 트래픽 roofline 효율 %.1f%% / %.1f%%, 낙관적 고유 payload 모델 %.1f%% / %.1f%%. 숫자의 정의와 한계는 아래 분석에 명시했다.</p>'''%(bounds[0]['measured_traffic_roofline_pct'],bounds[1]['measured_traffic_roofline_pct'],bounds[0]['ideal_roofline_pct'],bounds[1]['ideal_roofline_pct'])+'''<p>초기 조기 TMA의 warp-group 간 경쟁 조건을 sanitizer 수치 비교로 발견해 CTA barrier를 추가했다. 이 표는 수정 후 전체 재검증 결과다. 출력·11 gradients bit-exact, memcheck/racecheck와 CTA 지연 검증 통과.</p><div class="notice amber">개발 설정. 기존 B7 독립 기준 정확도 문제로 production 승격은 보류한다.</div><p><a href="assets/trimul-b1-epilogue.md">구현·경쟁 조건 수정·NCU·SoL 정의</a> · <a href="assets/trimul-b1-epilogue-summary.json">실측 JSON</a> · <a href="#current-wiring">현재 HBM/수식 SVG</a></p></section><!-- B1_EPILOGUE_END -->'''
p=site/'trimul.html';s=p.read_text()
if '<!-- B1_EPILOGUE_BEGIN -->' in s:
 a=s.index('<!-- B1_EPILOGUE_BEGIN -->');b=s.index('<!-- B1_EPILOGUE_END -->',a)+len('<!-- B1_EPILOGUE_END -->');s=s[:a]+section+s[b:]
else:s=s.replace('<main>','<main>'+section,1)
a=s.index('<!-- B1_SOL90_BEGIN -->');b=s.index('<!-- B1_SOL90_END -->',a);s=s[:a]+s[a:b].replace('현재 학습 개발 경로','직전 v49 구현 이력')+s[b:];p.write_text(s)
print('VERIFIED',gains)
