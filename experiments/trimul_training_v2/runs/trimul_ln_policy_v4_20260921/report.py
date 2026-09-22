from pathlib import Path
import csv,json,shutil
R=Path(__file__).resolve().parent;root=R.parents[1];site=root/'runs/anthropic_b1b4_pipeline_20260919/site-visuals/dist'
metrics={'gpu__time_duration.sum':'time_us','dram__bytes_read.sum':'dram_read_MB','dram__bytes_write.sum':'dram_write_MB','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed':'dram_pct','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed':'tensor_pct','sm__warps_active.avg.pct_of_peak_sustained_active':'occupancy_pct','launch__registers_per_thread':'registers'}
ncu={};perf=[];pr=[];nr=[]
for n in (384,768):
 ncu[str(n)]={};d=json.loads((R/('final-L%d.json'%n)).read_text());base=d['times']['forward_backward']['baseline']['median_us']
 for k,label in [('baseline','직전 shared B1'),('stats_only','출력 통계 TMA 저장 (선택)'),('xhat_fp32','FP32 x̂ 대체 + TMA 저장')]:
  u,v=list(csv.DictReader((R/('ncu-%s-L%d-raw.csv'%(k,n))).open()));entry={}
  for raw,key in metrics.items():
   value=float(v[raw]);unit=u[raw]
   if key=='time_us':value*=dict(us=1,ms=1000,ns=.001,s=1e6)[unit]
   elif key.endswith('_MB'):value*=dict(byte=1e-6,Kbyte=.001,Mbyte=1,Gbyte=1000)[unit]
   entry[key]=value
  ncu[str(n)][k]=entry
  vals=[d['times'][s][k]['median_us']/1000 for s in ('forward','b1','backward','forward_backward')];delta=(vals[-1]*1000/base-1)*100
  cells=[str(n),label]+['%.4f'%x for x in vals]+['%+.2f%%'%delta];perf.append('| '+' | '.join(cells)+' |');pr.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
  nr.append('<tr>'+''.join('<td>'+c+'</td>' for c in [str(n),label,'%.2f'%entry['dram_read_MB'],'%.2f'%entry['dram_write_MB'],'%.1f%%'%entry['dram_pct'],'%.1f%%'%entry['tensor_pct'],str(int(entry['registers']))])+'</tr>')
(R/'ncu-summary.json').write_text(json.dumps(ncu,indent=2))
for n in (384,768):assert 'ERROR SUMMARY: 0 errors' in (R/('memcheck-L%d.log'%n)).read_text()
assert 'RACECHECK SUMMARY: 0 hazards' in (R/'racecheck-L384.log').read_text()
head=['L','경로','FWD ms','B1–B4 ms','BWD ms','전체 ms','전체 변화']
report='''# 출력 LN 저장 정책 최적화: 통계만 저장하고 TMA로 prefetch

2026-09-21 · node02 H100 · 양방향 BF16 C128/H256 · dropout25%, mask, residual.
선택한 **개발 기본값**: 입력 affine x_n 저장 유지 + 출력 LN 평균/역표준편차 FP32 저장.
원본 tri를 유지하고 B1의 큰 출력 측 입력으로 한 번 읽는다. 출력 x̂/affine LN activation은 저장하지 않는다.
학습 어댑터: `training.py:Training`, 구현 `save_k3.cu`, `b1_fused.cu`, `lowreg_stats.inc`, `replace_plan.py`.
기존 B7 independent-reference 이슈가 남아 있으므로 production 승격은 별개다.

## 최종 직접 측정

기준은 직전 shared B1 + split_xn_pc1 B7이며 Anthropic 원본 inference와의 비교가 아니다.

'''+ '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join(perf)+'''

Slurm 13474_0/1. 경로별600회(200×3) 교대 CUDA graph 중앙값. Live weight packing과 모든11개 gradient 포함, optimizer/RNG 생성/CPU dispatch/compile 제외.
구간 중앙값을 합산하지 않고 전체를 직접 측정했다. 검증·NCU 프로파일링 시간은 벤치 시간에 포함하지 않았다.

## 무엇을 바꿨나

- K3가 이미 계산한 출력 LN mean/rstd를 FP32 [L²]×2 버퍼에 저장. 추가 저장은 L384 1.18MB /L768 4.72MB.
- B1이 x_n/tri 입력 타일과 함께 두 통계를 TMA로 미리 로드한다. 기존 input barrier의 transaction에512B를 추가해 별도 kernel/전역 scalar load 대기를 피했다.
- B1은 저장 통계로 affine와 LN 미분을 계산한다. 출력 LN mean/variance reduction은 재계산하지 않는다. 출력 projection/gate 및 B7 수식과 저장 정책은 그대로다.
- 큰 출력 측 global 읽기는 tri 하나. tri와 출력 LN activation을 함께 읽는 경로가 아니다.
- CTA 96/112/120/128/132 × affine unroll 1/2/4/8의20개 조합을 각 길이에서 검증/튜닝. 기존 경로도 같은 공간을 점검했다.
- 최종 통계 경로: 132 CTA ×256threads, L384 unroll4 /L768 unroll8. 기존 기준은132 CTA/unroll4; 기존 L768 unroll8의 sweep 이득은 약0.15%로 작았다.
- K1/contraction/output GEMM/B7는 동일. 이 실험은 출력 LN 저장 정책과 B1 구현에 한정한다.

## NCU

원본 raw CSV 단위(Mbyte/Gbyte, us/ms)를 변환한 값은 `ncu-summary.json`에 있다. `--cache-control none --clock-control none`, 워밍업 후 B1 한 호출, full metrics. 프로파일링 latency는 CUDA graph 벤치 latency와 구분한다.
L768 B1 실제 DRAM 읽기: 기준924.06MB → 통계 저장928.69MB → FP32 x̂ 대체1228.34MB.
통계 경로는 DRAM bytes 절감이 아니라 재계산 제거와 prefetch로 빨라졌다. Register/thread는255→243, occupancy는12.5%로 같다. Tensor active는17.5→19.0% 정도다.
FP32 x̂ 대체의 추가 읽기도 실측으로 확인했으며, 이는 두 원본을 읽어서가 아니라 FP32로 원소 바이트가 늘어났기 때문이다.
이 지표들은 SoL90이나 이 구조의 성능 상한을 의미하지 않는다.

## 시도 후 제외한 후보

1. FP16 x̂ 직접 저장: gradient 상대L2 최대 약0.24%, 0.05% 기준 실패. BF16보다 정밀해도 affine를 BF16으로 재구성하는 경계에서 차이가 생긴다.
2. FP16 x̂ + mean/rstd로 원래 BF16 값을 복원: 오차는 줄었지만 L384 최대 약0.0743%로 실패하고, 나눗셈/복원 비용 때문에 느렸다. L768만 통과한 결과로 일반 채택하지 않았다.
3. 통계만 저장하되 일반 load: 정확하지만 초기에는 gate 계산과의 overlap 손실/통계 load 대기로 이득이 없었다. 병렬 배치를 복원한 뒤 TMA prefetch에서 개선됐다.
4. FP32 x̂ scalar store → TMA 타일 store: 정확도를 유지하고 forward 비용을 줄였지만, 최종 전체는 기준보다 약7.1~7.3% 느렸다. 64B swizzle 주소식이 잘못된 중간 V2는 정확도 실패로 제외하고 V3에서 수정했다.
5. FP32 B1 gate weight 재로드 제거: shared 공간을 다시 배치하고 gradient global store를 바꿔 정확도는 유지했으나 더 느렸다. 채택하지 않았다.

실험 원본은 sibling 폴더 `trimul_ln_policy_tune_20260921`(FP16/복원), `trimul_ln_policy_v2_20260921`(중간), `trimul_ln_policy_v3_20260921`(TMA 저장/기존 baseline tuning), `trimul_ln_policy_v5_20260921`(gate weight 유지)에 보관했다. V4가 최종 선택이다.

## 검증

- 일반 입력과 x/weight/gamma/beta/mask/dropout mask/dy 변경 후 전체 y 및11개 gradient가 기준과 bit-exact. gamma_out[0]=0 포함. CUDA graph replay도 eager와 bit-exact.
- 대체 FP32 경로는 tri NaN poison 검증 통과. 선택된 통계 경로는 의도적으로 tri를 유지하므로 이 검증의 대상이 아니다.
- 최종 memcheck L384/768 모두0 errors. L384 B1+K3 racecheck0 hazards (0 errors,0 warnings).
- 기존 B7의 L768 변형 입력 독립 reference dWL 오차0.055569%가0.05%를 넘는 문제는 해결한 것이 아니다. 기존과 동일한 gradient이므로 production 승격은 하지 않았다.
- 재현: node02 Slurm `final.sbatch`, `verify.sbatch`. 결과 JSON에 소스 SHA-256/cubin 경로/600개 timing sample 기록.
- 현재 SVG는 mean/rstd의 K3 WRITE → B1 TMA READ와 PyTorch 수식을 반영한다.
'''
(R/'README.md').write_text(report)
section='''<!-- LN_POLICY_TUNED_BEGIN --><section class="panel" id="ln-policy-tuned"><span class="pill green">2026-09-21 · 선택한 학습 개발 경로</span><h2>출력 LN 통계만 저장: B1 지연 −6.6/−7.1% · 전체 −1.5/−1.6%</h2><p><b>입력 x_n + 출력 LN 평균·역표준편차 저장.</b> 출력 LN activation은 저장하지 않고, B1의 큰 출력 측 입력은 원본 tri 하나를 유지한다. 통계512B를64행 입력 타일과 함께 TMA로 미리 가져온다.</p><div class="table-wrap"><table><thead><tr>'''+''.join('<th>'+h+'</th>' for h in head)+'''</tr></thead><tbody>'''+''.join(pr)+'''</tbody></table></div><p class="muted">node02 H100 · BF16 C128/H256 양방향 · dropout25%/mask/residual · 각600회 교대 CUDA graph · live packing + 전체11개 gradient · 구간과 전체 각각 직접 측정.</p><h3>현재 저장 배선</h3><pre>K3 → y + input x_n + output mean/rstd 저장
B1 ← x_n + tri + mean/rstd를 TMA로 읽기
   → 출력 LN mean/variance reduction 생략
   → affine / projection / gate / gradient 계산
추가 저장: L384 1.18MB /L768 4.72MB</pre><h3>NCU 실측 · B1 한 호출</h3><div class="table-wrap"><table><thead><tr><th>L</th><th>경로</th><th>DRAM read MB</th><th>DRAM write MB</th><th>DRAM %</th><th>Tensor %</th><th>register/thread</th></tr></thead><tbody>'''+''.join(nr)+'''</tbody></table></div><p>통계 경로는 DRAM 읽기가 줄어든 것이 아니다. 재계산 제거와 TMA prefetch로 빨라졌다. Occupancy는12.5%로 동일하며 SoL90을 뜻하지 않는다.</p><h3>선택과 제외</h3><ul><li>기존/통계 경로 모두 CTA 수×unroll 20개 조합 점검. 최종132 CTA, L384 unroll4 /L768 unroll8.</li><li>FP16 x̂ 및 FP16 복원은 두 길이를 모두 만족하는 정확도 확보 실패.</li><li>FP32 x̂ TMA 저장은 bit-exact지만 전체 약7% 느림. Gate weight 재로드 제거 후보도 더 느려 제외.</li></ul><div class="notice"><b>최종 검증:</b> 기존 y·11개 gradient bit-exact, 입력/가중치/dropout mask 변경 및 gamma=0, graph replay 통과. Memcheck 두 길이0 errors, B1/K3 racecheck L384 0 hazards.</div><div class="notice amber">학습 개발 기본값으로 연결. 기존 B7 독립 reference 이슈는 남아 production 승격은 별개다. 아래 이전 실험들의 “현재”는 각 실험 당시 상태이며 최신 선택은 이 섹션과 현재 SVG다.</div><p><a href="#current-wiring">업데이트된 HBM 배선 SVG</a> · <a href="assets/ln-policy-report.md">전체 개발 보고서</a> · <a href="assets/ln-policy-final-L384.json">L384 원본</a> · <a href="assets/ln-policy-final-L768.json">L768 원본</a> · <a href="assets/ln-policy-ncu.json">NCU 요약</a></p></section><!-- LN_POLICY_TUNED_END -->'''
p=site/'trimul.html';s=p.read_text();start='<!-- LN_POLICY_TUNED_BEGIN -->';end='<!-- LN_POLICY_TUNED_END -->'
if start in s:a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+section+s[b:]
else:s=s.replace('<main>','<main>'+section,1).replace('<nav>','<nav><a href="#ln-policy-tuned">최신: LN 통계 TMA</a>',1)
s=s.replace('<code>shared B1 + split_xn_pc1 B7</code>','<code>saved output stats + shared B1 + split_xn_pc1 B7</code>')
s=s.replace('평균·역표준편차·projection/gate 저장은 별도 forward 실험이며 이 backward에 연결하지 않았다.','출력 LN 평균·역표준편차는 K3에서 저장하고 B1이 TMA로 읽는다. 출력 LN activation/projection/gate는 저장하지 않는다.')
p.write_text(s)
for name,target in [('README.md','ln-policy-report.md'),('final-L384.json','ln-policy-final-L384.json'),('final-L768.json','ln-policy-final-L768.json'),('ncu-summary.json','ln-policy-ncu.json')]:shutil.copy2(str(R/name),str(site/'assets'/target))
print('report and dashboard updated')
