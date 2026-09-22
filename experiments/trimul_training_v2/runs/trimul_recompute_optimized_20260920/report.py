"""Generate the report and private static dashboard from final measurements."""
from pathlib import Path
import csv,hashlib,json,shutil
R=Path(__file__).resolve().parent
SITE=R.parent/'anthropic_b1b4_pipeline_20260919/site-visuals'
D={n:json.loads((R/('results-L%d.json'%n)).read_text()) for n in (384,768)}
labels={'saved':'기존 저장형','previous_recompute':'재계산 최적화 전','optimized_recompute':'재계산 최적화 후'}
for j in D.values():
 for name,h in j['metadata']['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
 assert j['activation_restore_kernels']==0 and j['contraction_calls']==6
 assert all(e['bit_exact'] for e in j['graph_vs_fresh_eager'].values())
NCU=[]
for n in (384,768):
 for kind in ('b1','b7'):
  stem=('ncu-b1-' if kind=='b1' else 'ncu-final-b7-')+'L%d'%n
  raw=list(csv.DictReader((R/(stem+'-raw.csv')).open()))[-1]
  det={r['Metric Name']:r for r in csv.DictReader((R/(stem+'-details.csv')).open()) if r['Metric Name']}
  NCU.append(dict(L=n,region='B1–B4' if kind=='b1' else 'B7–B12',report=stem+'.ncu-rep',
   dram=float(det['DRAM Throughput']['Metric Value']),sm=float(raw['sm__throughput.avg.pct_of_peak_sustained_elapsed']),
   tensor=float(raw['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']),
   occupancy=float(det['Achieved Occupancy']['Metric Value']),no_eligible=float(det['No Eligible']['Metric Value']),
   duration=float(det['Duration']['Metric Value']),duration_unit=det['Duration']['Metric Unit']))
(R/'ncu-summary.json').write_text(json.dumps(NCU,indent=2,ensure_ascii=False))
T=[];G=[]
for n,j in D.items():
 for key,label in labels.items():
  T.append([str(n),label]+['%.3f'%(j['times'][scope][key]['median_us']/1000) for scope in ('forward','backward','forward_backward')])
 for region,kn in (('B1–B4','b1_fused'),('B7–B12','front_b7b12')):
  a=next(t['us'] for t in j['traces']['backward']['previous_recompute'] if t['name']==kn)
  b=next(t['us'] for t in j['traces']['backward']['optimized_recompute'] if t['name']==kn)
  G.append([str(n),region,'%.1f'%a,'%.1f'%b,'%.1f%%'%(100*(1-b/a))])
N=[[str(x['L']),x['region']]+['%.1f%%'%x[k] for k in ('dram','sm','tensor','occupancy')] for x in NCU]
def mdtable(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(r)+' |\n' for r in rows)
def htmltable(head,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
head=['L','경로','Forward ms','Backward ms','전체 실측 ms'];gh=['L','구간','최적화 전 μs','최적화 후 μs','시간 감소'];nh=['L','구간','DRAM','SM throughput','Tensor pipe','Occupancy']
report='''# TriMul 재계산 배선 최적화 · 2026-09-21

## 결과와 비교 기준

직전 on-chip 재계산 구현 대비 **전체 학습 1.22배**, 시간은 **18.1% / 18.3% 감소**했다. Backward 시간은 **22.2% / 21.7% 감소**했다. 그러나 **저장형 학습 경로보다 전체가 아직 25.6% / 26.4% 느리다**. 목표 1.7배와 SoL90은 달성하지 않았다. 실험 배선만 최적화했으며 production 기본값을 바꾸지 않았다.

기준을 세 가지로 고정했다. `saved`는 Anthropic 파생 저장형 학습 forward + 기존 최적화 CUDA backward다. `previous_recompute`는 `trimul_fused_recompute_20260920`의 직전 on-chip 재계산 구현이다. `optimized_recompute`는 이 디렉터리의 최종 선택이다. 공개 Anthropic 추론 또는 cuEquivariance와의 새 비교가 아니다.

H100 80GB / node02, BF16, batch1, C128, outgoing·incoming 각 hidden128, L384/768, dropout25%, mask, residual, 11개 gradient. 경로별·범위별 CUDA graph 600회 교대 측정. live weight packing 포함. optimizer, RNG 생성, CPU dispatch, 컴파일 제외. 전체 시간은 fwd/bwd 합산 추정이 아닌 별도 graph 실측이다.

'''+mdtable(head,T)+'''
## 최종 배선

Forward는 같은 Anthropic 파생 no-save K1 → cuBLAS 2회 → no-save K3다. dropout/residual과 BF16 반올림을 학습 경로에 맞춘 상태다. 입력·가중치·mask 참조 외의 forward 중간값은 **left/right/tri만 유지**한다.

- **B1–B4: `b1_fused.cu` 한 번.** 다음 행 타일의 TMA 입력 전송을 현재 LN/GEMM과 겹친다. 48 KiB 행 버퍼 두 개와 resident Wgate/Wproj를 쓴다. gate/projection을 레지스터에서 바로 미분에 소비하고 dg도 TMA로 쓴다. 역할별 작업량을 36 dWgate / 52 dWproj / 44 dX CTA로 나눴다. L별 dropout 인덱스 주기도 컴파일 때 정한다.
- **B5–B6: 기존 cuBLAS 4회.** 저장된 left/right로 contraction 미분. forward contraction을 반복하지 않는다.
- **B7–B12: `b7_warp_specialized.cu` 한 번.** dW CTA 64개는 TMA producer warp-group 1개와 계산 consumer 2개로 구성한다. 입력 LN → packed gate/projection WGMMA → GLU 미분 → dW를 각 consumer 안에서 수행한다. dX CTA 68개는 한 방향의 packed 가중치 128 KiB를 shared memory에 두고 gate와 projection 미분 양쪽에서 재사용한다. 입력 LN 미분·residual까지 같은 호출에서 끝난다.
- 두 영역 모두 재계산 activation의 HBM 쓰기·복원 커널은 없다. gradient 전달 텐서와 dW reduction partial은 global scratch다. 이를 HBM 접근 자체가 없다는 뜻으로 해석하면 안 된다.

Forward 중간값의 논리적 유지량: L384 719.585 → 226.492 MB, L768 2878.341 → 905.970 MB. 이전 재계산 버전과 같다. 입력/가중치/mask/출력 제외한 tensor-size 합계이며 **학습 peak GPU memory 실측이 아니다**.

## 어떤 구간이 빨라졌나

아래는 같은 실행의 별도 profiler trace다. 합계가 CUDA graph 중앙값과 정확히 일치하지 않을 수 있다.

'''+mdtable(gh,G)+'''
## 설정과 후보 선택

- B1: row tile64, 256 threads, 132 CTAs, `DW_SPLITS=26`, `GATE_SPLITS=36`, `TRAIN_L=384/768`, dynamic shared227840 B. ptxas254 registers, stack/spills0.
- B7: row64, DW hidden32와 native gate32/projection32 pack, 384 threads, 132 CTAs, `DW_SPLITS=4`, `PIPE_DW=1`, `DX_RESIDENT=1`, dynamic shared221184 B. ptxas168 static registers/thread; DW producer40 / consumers232 each, DX compute232 / idle32 via setmaxnreg. Stack/spills0.
- `selected-L384.json`와 `selected-L768.json`을 실제 전체 벤치가 읽는다. 빌드 캐시는 source/header/compiler/config SHA256 기반이다. config를 plan별로 캡처해 다른 plan 설정이 섞이지 않게 했다.
- 추가 resident-weight 후보를 동일 버퍼 주소로 800회씩 교대 비교: B7 612.992→592.640 μs (384), 2394.352→2254.944 μs (768). 3.3% / 5.8% 개선으로 채택했다.
- dW 결과 TMA-store는 추가 이득 0.2% 이하여서 미채택. DW GEMM과 다음 재계산의 추가 중첩, hidden64 producer 배선, 다른 LN affine-chain 설정도 최종 선택에 포함하지 않았다. 부동소수 누적 순서를 gate/projection 교대로 바꾸는 후보는 엄격한 LN parameter-gradient 허용치를 넘겨 폐기했다.
- 초기 warp-specialized 후보의 레지스터 요청 합계가 CTA pool을 넘어서 멈추던 문제를 수정했다. 최종 요청량은 pool 이하다. 모든 실패·미채택 후보가 production과 분리되어 있다.
- 이것은 검증한 구조/분배/파이프라인 후보의 선택이다. 모든 타일 공간의 exhaustive autotuning 완료를 주장하지 않는다.

## NCU

Full set, warm data, cache/clock control none; 별도 프로파일 실행. 아래 hardware 활성도를 알고리즘의 최적 상한 비율로 읽으면 안 된다.

'''+mdtable(nh,N)+'''
두 커널은 각각 1 CTA/SM이며 eligible warp가 없는 scheduler cycle이 약52–55%다. TMA와 Tensor Core를 썼지만 **HBM 또는 Tensor Core 상한에 가까운 상태는 아니다**. 의존성, 낮은 occupancy, 재계산 중복이 여전히 남아 있다. NCU만으로 SoL90을 주장하지 않는다. `.ncu-rep`, details/raw CSV, `ncu-summary.json`에 근거를 남겼다. NCU 종료 후 환경의 utf-8-sig helper 경고가 있지만 report 저장·CSV export는 성공했다.

## 검증과 남은 수치 한계

- 일반 입력의 forward는 bit-exact; 11개 gradient가 독립 기존 reference 대비 상대 L2 ≤0.05%를 통과한다. 각 구간 테스트는 dx/dtri2e-5, LN parameter5e-6 등 더 엄격한 기존 허용치를 유지한다.
- 입력·가중치·dy·dropout·mask를 바꾼 graph replay는 새 eager 결과와 전 출력 bit-exact다. 새 경로와 저장형도 ≤0.05%다.
- **변경 입력 L768 dWL은 독립 reference 대비 여전히 실패**한다: 새 재계산0.05485%, 저장형0.05413%, 허용0.05%. 기존 재계산과 동일한 한계다. 허용치를 높이거나 새 커널의 검증 완료로 숨기지 않았다. production 승격 전에 해결해야 한다.
- B1 L384 memcheck/racecheck, 최종 B7 L384 racecheck, 최종 전체 모듈 L768 unfiltered memcheck: 모두 error0, race warning0. 이전 B7 warp 후보도 별도 memcheck 통과. 검사 로그를 보존했다.
- 최종 trace: B1–B4 1회, B7–B12 1회, cuBLAS 전체6회, activation 복원0회. source/cubin SHA256을 결과 JSON과 다시 대조했다.

## 재현

node02의 할당된 H100에서 실행한다. L384와 L768를 별도 GPU에서 병렬 검증했다. 기존 학습/다른 에이전트 잡은 중지하지 않았다.

```bash
export MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_recompute_optimized_20260920/bench.py --length 384
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_recompute_optimized_20260920/bench.py --length 768
```

`probe.py`는 개별 후보, `paired_b7.py`는 같은 버퍼 주소의 최종 B7 후보 비교, `check.py`는 구간 정확도·sanitizer용이다. 비교용 saved/previous 구현과 Anthropic upstream 소스는 기존 runs 디렉터리들을 참조한다. 독립 설치 가능한 production 패키지는 아니다.

## 출처

Anthropic Apache-2.0 native v5의 TMA/WGMMA, LN fragment, matrix layout을 계승하고 Miniworld의 backward 수식과 결합했다. 이 작업의 기여는 추론 커널을 기반으로 한 학습 재계산·스케줄링 확장이다. 원본 추론 커널을 자체 개발로 표기하지 않는다.
'''
(R/'README.md').write_text(report)
A=SITE/'dist/assets'
for n in D:shutil.copy2(R/('results-L%d.json'%n),A/('trimul-recompute-optimized-L%d.json'%n))
shutil.copy2(R/'ncu-summary.json',A/'trimul-recompute-optimized-ncu.json')
shutil.copy2(R/'README.md',A/'trimul-recompute-optimized.md')
for n in D:shutil.copy2(R/('paired-b7-L%d.json'%n),A/('trimul-recompute-paired-b7-L%d.json'%n))
section='''<!-- ONCHIP_OPTIMIZED_BEGIN --><section class="panel" id="recompute-optimized">
<span class="pill green">2026-09-21 · 재계산 배선 최적화 · 실험 경로</span>
<h2>전체 학습 1.22배: 재계산 변경 전보다18% 단축</h2>
<p>Forward는 같은 Anthropic 파생 inference K1/K3다. B1–B4와 B7–B12의 <b>TMA 전송·재계산·미분 실행 순서와 가중치 재사용</b>을 바꿨다. left/right/tri만 forward 중간값으로 유지하고 나머지는 backward의 shared memory·register에서 다시 계산한다.</p>
<div class="notice amber"><b>기존 저장형보다 아직25–26% 느리다. Production 기본값은 변경하지 않았다.</b><br>공개 Anthropic 추론이나 cuEq를 이겼다는 수치가 아니다. 아래 세 경로를 같은 실행에서600회씩 교대 측정했다. 1.7배와 SoL90은 미달이다.</div>
<p class="muted">H100 node02 · BF16 · batch1/C128 · outgoing/incoming 각 H128 · dropout25% · mask/residual · 11개 gradient · live packing 포함. optimizer/RNG/CPU dispatch/compile 제외.</p>
<div class="toolbar"><b>전체 학습 시간</b><button type="button" class="active" data-opt-l="384" aria-pressed="true">L384</button><button type="button" data-opt-l="768" aria-pressed="false">L768</button></div><div id="opt-bars" class="panel" aria-live="polite"></div>
'''+htmltable(head,T)+'''
<p>Forward 시간은 동일한 커널의 측정 노이즈 범위다. <b>Backward 시간은22.2% / 21.7% 감소</b>했다. 전체는 별도 graph로 직접 측정했다.</p>
<h3>실제 연산 → 커널 배선</h3>
<a href="assets/trimul-recompute-optimized.svg"><img src="assets/trimul-recompute-optimized.svg" alt="저장 없는 Anthropic forward, B1-B4 이중 버퍼, B7-B12 TMA producer와 resident weight의 실제 연산 배선" style="width:100%;height:auto;background:#f6f9fb;border-radius:12px"></a>
<h3>어디가 개선됐나</h3>
'''+htmltable(gh,G)+'''
<ul><li><b>B1–B4:</b> 행 버퍼2개로 다음 타일 TMA를 현재 LN/GEMM과 겹친다. 레지스터의 gate/projection을 미분에 직접 소비한다. CTA는 dWgate36 / dWproj52 / dX44.</li>
<li><b>B7–B12:</b> dW에 TMA producer1 + 계산 consumer2. dX에서는 방향별 packed 가중치128 KiB를 shared memory에 한 번 두고 gate·projection 미분에 함께 쓴다. dW64 / dX68 CTA.</li>
<li><b>유지한 것:</b> cuBLAS 미분4회, BF16 반올림·누적 순서, dropout/mask/residual, gradient scratch. activation 복원 커널은0개.</li></ul>
<p>Forward 유지량: L384 <b>719.6→226.5 MB</b>, L768 <b>2878.3→906.0 MB</b>. 이전 재계산 버전과 같으며 학습 peak 메모리가 아닌 tensor-size 합계다.</p>
<h3>NCU · 아직 상한 근처가 아니다</h3>
'''+htmltable(nh,N)+'''
<p>1 CTA/SM, scheduler가 실행할 warp를 못 찾는 cycle 약52–55%. 전송·계산의 의존성과 재계산 중복을 더 줄일 여지가 있다. 각 지표는 하드웨어 사용률이며 알고리즘 SoL 비율이 아니다.</p>
<h3>검증과 제한</h3><ul><li>일반 입력: forward bit-exact, 11개 gradient 상대 L2≤0.05%. 변경 입력 graph는 새 eager와 bit-exact.</li>
<li>B1 memcheck/racecheck, 최종 B7 racecheck, 최종 L768 전체 unfiltered memcheck: 오류0. ptxas stack/spills0.</li>
<li><b>L768 변경 입력의 dWL 독립 기준 검사 실패는 남아 있다.</b> 새0.05485%, 저장형0.05413%, 허용0.05%. 허용치를 올리지 않았다.</li>
<li>resident 가중치의 추가 이득은 같은 버퍼로800회 교대 확인했다. dW TMA-store의0.2% 이하 차이는 채택하지 않았다. 전체 config 공간을 다 탐색했다고 주장하지 않는다.</li></ul>
<p>Anthropic의 Apache-2.0 TMA/WGMMA·LN·layout 구현을 계승한 학습 확장이다.</p>
<p><a href="assets/trimul-recompute-optimized.md">구현·검증 보고서</a> · <a href="assets/trimul-recompute-optimized-L384.json">L384 측정</a> · <a href="assets/trimul-recompute-optimized-L768.json">L768 측정</a> · <a href="assets/trimul-recompute-optimized-ncu.json">NCU</a></p>
</section><!-- ONCHIP_OPTIMIZED_END -->'''
p=SITE/'dist/trimul.html';s=p.read_text()
if '<!-- ONCHIP_OPTIMIZED_BEGIN -->' in s:
 a=s.index('<!-- ONCHIP_OPTIMIZED_BEGIN -->');b=s.index('<!-- ONCHIP_OPTIMIZED_END -->',a)+len('<!-- ONCHIP_OPTIMIZED_END -->');s=s[:a]+section+s[b:]
else:
 s=s.replace('<!-- ONCHIP_RECOMPUTE_BEGIN -->',section+'<details><summary>이전: 커널 내부 재계산 최초 구현</summary><!-- ONCHIP_RECOMPUTE_BEGIN -->',1).replace('<!-- ONCHIP_RECOMPUTE_END -->','<!-- ONCHIP_RECOMPUTE_END --></details>',1)
 s=s.replace('<nav><a href="#onchip-recompute">새 BWD 내부 재계산</a>', '<nav><a href="#recompute-optimized">최신 재계산 최적화</a><a href="#onchip-recompute">최초 재구현</a>',1)
plot={str(n):{k:round(j['times']['forward_backward'][k]['median_us']/1000,4) for k in labels} for n,j in D.items()}
script='''<script id="opt-chart-script">(()=>{const data=DATA,labels=LABELS;const draw=n=>{const rows=data[n],max=Math.max(...Object.values(rows));document.getElementById('opt-bars').innerHTML=Object.entries(rows).map(([k,v])=>`<div class="barline" style="grid-template-columns:150px 1fr 80px"><span>${labels[k]}</span><div class="track"><div class="fill ${k==='optimized_recompute'?'up':k==='previous_recompute'?'slow':''}" style="width:${v/max*100}%"></div></div><b>${v.toFixed(3)} ms</b></div>`).join('');document.querySelectorAll('[data-opt-l]').forEach(b=>{const active=b.dataset.optL===n;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});};document.querySelectorAll('[data-opt-l]').forEach(b=>b.addEventListener('click',()=>draw(b.dataset.optL)));draw('384');})();</script>'''.replace('DATA',json.dumps(plot)).replace('LABELS',json.dumps(labels,ensure_ascii=False))
if '<script id="opt-chart-script">' in s:
 a=s.index('<script id="opt-chart-script">');b=s.index('</script>',a)+len('</script>');s=s[:a]+script+s[b:]
else:s=s.replace('</body>',script+'</body>')
p.write_text(s)
p=SITE/'dist/index.html';s=p.read_text();a=s.index('<!-- TRAINING_LINK_BEGIN -->');b=s.index('<!-- TRAINING_LINK_END -->',a)+len('<!-- TRAINING_LINK_END -->')
s=s[:a]+'''<!-- TRAINING_LINK_BEGIN --><section class="notice"><b>최신: TriMul 재계산 배선 최적화 · 전체1.22배</b> · <a href="trimul.html#recompute-optimized">배선·측정·제한 보기 →</a><br>이전 재계산 버전보다 전체18% 단축. 저장형보다는 아직25–26% 느리며, production은 변경하지 않았다.</section><!-- TRAINING_LINK_END -->'''+s[b:];p.write_text(s)
print('Generated README, NCU summary, assets, and dashboard sections')
