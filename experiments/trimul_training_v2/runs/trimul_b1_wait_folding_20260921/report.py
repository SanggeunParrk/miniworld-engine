"""Publish only measured, verified B1 wait-folding development results."""
from pathlib import Path
import csv, hashlib, html, json

R=Path(__file__).resolve().parent
ROOT=R.parents[1]
SITE=ROOT/'runs/anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows=[]; bounds=[]; validation={}; gains={}; gate=[]
for n in (384,768):
    d=json.loads((R/('results-L%d.json'%n)).read_text())
    for p,h in d['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    for field in ('checks','mutated','graph_vs_eager'):
        assert all(v['bit_exact'] for v in d[field]['optimized'].values())
    for key,label in [('baseline','직전 v50'),('optimized','대기 축소')]:
        rows.append([str(n),label,*['%.4f'%(d['times'][s][key]['median_us']/1000) for s in ('forward','b1','backward','forward_backward')]])
    gains[str(n)]={s:100*(1-v['optimized']['median_us']/v['baseline']['median_us']) for s,v in d['times'].items()}
    validation[str(n)]=json.loads((R/('verification-L%d.json'%n)).read_text())
    assert len(validation[str(n)])==(6 if n==384 else 5)
    assert all(v['returncode']==0 for v in validation[str(n)])
    unit,profile=list(csv.DictReader((R/('ncu-optimized-L%d-raw.csv'%n)).open()))[:2]
    assert unit['gpu__time_duration.sum']=='us'
    us=float(profile['gpu__time_duration.sum'])
    traffic=sum(float(profile[k])*1e6 for k in ('dram__bytes_read.sum','dram__bytes_write.sum'))
    bounds.append(dict(L=n,ncu_us=us,dram_peak_pct=float(profile['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed']),tensor_peak_pct=float(profile['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']),traffic_bytes=traffic,measured_traffic_roofline_pct=100*traffic/3.35e6/us,ideal_roofline_pct=100*max(1800*n*n/3.35e6,262144*n*n/989.5e6)/us))
    gdir=R.parent/'trimul_b1_gate_bound_20260921'
    g=json.loads((gdir/('results-L%d.json'%n)).read_text())
    assert g['gate_partial_bit_exact']
    _,p=list(csv.DictReader((gdir/('ncu-L%d-raw.csv'%n)).open()))[:2]
    gate.append(dict(L=n,isolated_us=g['times']['gate_only']['median_us'],ncu_us=float(p['gpu__time_duration.sum']),dram_peak_pct=float(p['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'])))
stress=json.loads((R/'delayed-cta-check.json').read_text())
assert len(stress)==6 and all(all(v['bit_exact']) for v in stress)
summary=dict(baseline='v50 trimul_b1_epilogue_fixed_20260921, paired node01',rows=rows,latency_reduction_percent=gains,roofline=bounds,gate_phase_diagnostic=gate,validation=validation,delayed_cta_checks=stress,verification_complete=True,production_ready=False,sol90_reached=False,policy_unchanged=True)
head=['L','경로','FWD ms','B1–B4 ms','BWD ms','전체 ms']
def table(h,r):
    return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(x)+' |' for x in r)
text='''# B1–B4 대기 축소: 검증된 개발 경로

2026-09-21 · node01 H100. 비교 기준은 직전 v50 자체 학습 경로이며 Anthropic 추론이나 cuEquivariance가 아니다.

양방향 C128/H256 BF16, dropout25%/mask/residual, L384/L768. 구간별600회 교대 CUDA graph 중앙값. 전체 수치는 live packing/Wp 전치/cuBLAS/B7/11 gradients를 포함하며 optimizer/RNG 생성/compile/CPU dispatch를 제외한다. FWD 구현은 동일하므로 그 차이는 측정 변동이다.

'''+table(head,rows)+'''

## 실제 변경

1. dGate TMA 저장 완료를 dNorm WGMMA 뒤로 옮겨 기존 CTA barrier와 합쳤다. dGate가 있던 shared scratch를 LN parameter 합산이 재사용하기 전에 저장 완료를 반드시 기다린다.
2. dTri TMA 저장 뒤의 warp-group barrier는 바로 다음 호출자의 CTA barrier가 포괄하므로 제거했다. 저장 완료 대기와 호출자의 CTA barrier는 유지한다.

**다음 raw TMA 전에 두 warp-group의 WGMMA 완료를 확인하는 CTA barrier를 유지한다.** 이전 경쟁 조건을 재도입하지 않았다. FP32 덧셈 순서, BF16 반올림, 입력 affine x_n 저장, 원본 tri BF16와 출력 mean/rstd FP32 저장 정책은 그대로다. 출력 LN activation은 저장하지 않는다. B7/cuBLAS는 변경하지 않았다.

## SoL90 진행 상황

'''+table(['L','NCU B1 μs','DRAM peak','실측 트래픽 roofline','고유 payload 모델'],[[str(b['L']),'%.3f'%b['ncu_us'],'%.2f%%'%b['dram_peak_pct'],'%.2f%%'%b['measured_traffic_roofline_pct'],'%.2f%%'%b['ideal_roofline_pct']] for b in bounds])+'''

**SoL90 미달이다.** DRAM 처리율을 전체 알고리즘 SoL로 표시하지 않는다. 이전과 같은 낙관적 모델 `max(1800*L²/3.35TBps, 262144*L²/989.5TFps)`은 scalar/shared/instruction/의존성 및 작은 scratch 비용을 생략한다. 실측 트래픽 모델에는 dWgate의 x_n/dGate 재읽기가 포함된다. 정확한 도달 가능한 최소시간을 증명한 수치는 아니다.

[H100 공식 사양](https://www.nvidia.com/en-us/data-center/h100/)의 SXM3.35TB/s, dense BF16 989.5TF/s를 사용한다. 공식1979TF/s는 sparsity 수치다. 별도 node01 streaming 교정은3.105TB/s였다.

### dWgate만 분리한 진단

'''+table(['L','단독 CUDA event μs','단독 NCU μs','단독 DRAM peak'],[[str(g['L']),'%.3f'%g['isolated_us'],'%.3f'%g['ncu_us'],'%.2f%%'%g['dram_peak_pct']] for g in gate])+'''

원래 CTA 소유 행/입력/부분합을 그대로 쓰고 shared reservation도 동일하게 유지한 진단이다. 최종 CTA 간 합산은 제외한다. 단독 부분합은 원래 B1과 bit-exact. 단독 측정은 전체 실행 중 해당 구간의 직접 타이밍과 다르며, 단독85%를 전체 SoL85%로 해석하면 안 된다.

## 검증

- 두 길이 출력·전체11 gradients: 기준과 bit-exact.
- 입력/가중치/dy/dropout/mask 변경 및 gamma_out=0: bit-exact.
- CUDA graph 재실행/eager, 원본tri·stats 저장 정책 검사 통과.
- memcheck 두 길이0 errors, racecheck L3840 hazards.
- CTA 일부 및 WG1을 의도적으로 지연: 두 길이×3seed×20replay의 B1 여섯 출력 bit-exact.

개발 경로이며 production 승격이 아니다. 기존 B7 독립 기준 L768 dWL 상대L2 0.055569%가 한도0.05%를 넘는 문제는 그대로 남아 있다.

## 추가 시도

- dWproj와 dNorm WGMMA를 연속 발행: 유효하지만 추가 이득이 작아 제외.
- 출력 LN affine과 gate 재계산, sigmoid와 projection 겹치기: 추가 이득이 작아 제외.
- LN parameter 합산 동기화를 warp-group으로 축소: 추가 이득이 없어 CTA 동기화 유지.
- 마지막 partial의 L1 우회/일반 읽기: 추가 이득 없어 기존 volatile 읽기 유지.
- 마지막 CTA partial 합산 unroll1~132: 기존 컴파일러 설정보다 유의한 개선 없음. 낮은 unroll은 오히려 느리다.
'''
(R/'summary.json').write_text(json.dumps(summary,indent=2))
(R/'README.md').write_text(text)
(ROOT/'TRIMUL_STATUS.md').write_text(text)
assets=SITE/'assets'
(assets/'trimul-b1-waits.md').write_text(text)
(assets/'trimul-b1-waits-summary.json').write_text(json.dumps(summary,indent=2))
for n in (384,768):
    for stem in ('results','verification'):
        (assets/('trimul-b1-waits-%s-L%d.json'%(stem,n))).write_bytes((R/('%s-L%d.json'%(stem,n))).read_bytes())
table_html='<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
section='''<!-- B1_WAITS_BEGIN --><section class="panel" id="b1-waits"><span class="pill green">현재 학습 개발 경로 · node01</span><h2>B1–B4: TMA 완료 대기를 기존 동기화와 합치기</h2><p>직전 v50 대비 B1 지연 L384 −%.2f%% / L768 −%.2f%%. 전체 학습은 −%.2f%% / −%.2f%%. 저장 정책과 수식은 그대로 유지했다.</p>'''%(gains['384']['b1'],gains['768']['b1'],gains['384']['forward_backward'],gains['768']['forward_backward'])+table_html+'''<p>dropout25%%/mask/residual · BF16 C128/H256 · 구간별600회 교대 측정. 출력과11 gradients bit-exact, memcheck/racecheck 및 CTA·warp-group 지연 검사 통과.</p><p><b>SoL90은 아직 미달.</b> 전체 B1의 실측 트래픽 roofline은 %.1f%% / %.1f%%. dWgate만 떼어낸 진단의 DRAM peak %.1f%% / %.1f%%는 전체 SoL이 아니다.</p>'''%(bounds[0]['measured_traffic_roofline_pct'],bounds[1]['measured_traffic_roofline_pct'],gate[0]['dram_peak_pct'],gate[1]['dram_peak_pct'])+'''<div class="notice amber">기존 B7 독립 기준 정확도 문제는 남아 있어 production 승격은 보류한다.</div><p><a href="assets/trimul-b1-waits.md">구현·NCU·검증·제외 후보</a> · <a href="assets/trimul-b1-waits-summary.json">실측 JSON</a> · <a href="#current-wiring">현재 HBM/수식 SVG</a></p></section><!-- B1_WAITS_END -->'''
p=SITE/'trimul.html';s=p.read_text()
if '<!-- B1_WAITS_BEGIN -->' in s:
    a=s.index('<!-- B1_WAITS_BEGIN -->');b=s.index('<!-- B1_WAITS_END -->',a)+len('<!-- B1_WAITS_END -->');s=s[:a]+section+s[b:]
else:
    s=s.replace('<main>','<main>'+section,1)
a=s.index('<!-- B1_EPILOGUE_BEGIN -->');b=s.index('<!-- B1_EPILOGUE_END -->',a)
s=s[:a]+s[a:b].replace('현재 학습 개발 경로','직전 v50 구현 이력')+s[b:];p.write_text(s)
print('VERIFIED',gains)
