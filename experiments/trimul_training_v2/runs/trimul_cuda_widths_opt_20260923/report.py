from pathlib import Path
import json,hashlib,html,re,csv
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
rows=[];validation=[]
for D in (64,128,256,384,512):
 for L in (384,768):
  pair=json.loads((R/f'paired-D{D}-L{L}.json').read_text());assert pair['complete']
  file=R/(f'latest-D128-L{L}.json' if D==128 else f'check-D{D}-L{L}.json');v=json.loads(file.read_text());assert v['complete']
  a=json.loads((R/f'autograd-D{D}-L{L}.json').read_text());assert a['complete'] and int(a['job'])>=15895
  if D>=384:assert v['route']['gp_shared_a']==1
  if D!=128:
   assert v['route']['fused'] and v['trace'].get('width_b7')==1 and not v['trace'].get('native_gp')
   for k,e in v['checks']['graph_mutation'].items():assert e['relative_l2'] <= (5e-6 if k.startswith(('dgamma','dbeta')) else 0)
  t=v['times']['training'];f=v['times']['training_forward' if D==128 else 'forward']
  row=dict(D=D,L=L,previous_us=pair['times']['previous']['median_us'],current_us=pair['times']['current']['median_us'],speedup=pair['speedup'],time_reduction_pct=100*(1-1/pair['speedup']),triton_us=t['triton']['median_us'],cuda_us=t['cuda']['median_us'],vs_triton=t['triton']['median_us']/t['cuda']['median_us'],forward_us=f['cuda']['median_us'],job=pair['job']);rows.append(row);validation.append(a)
for suffix in ['D64','D256','D384','D512','D128-L768']:
 assert 'ERROR SUMMARY: 0 errors' in (R/f'memcheck-{suffix}.log').read_text()
 assert '0 hazards displayed (0 errors, 0 warnings)' in (R/f'racecheck-{suffix}.log').read_text()
summary=dict(date='2026-09-23',scope='H100, bidirectional TriMul, BF16, B1, direction hidden=D, total hidden=2D, dropout25%, mask, residual, all 11 gradients',rows=rows,comparison='Previous CUDA is the immediately preceding all-width port. It is not Anthropic, cuEquivariance, or the original pre-adoption engine.',entry='runs/trimul_training_current.py',production_dispatch_changed=False,sol90_verified=False,validation_shapes=10)
profiles={}
for D in (256,512):
 data=list(csv.DictReader((R/f'ncu-D{D}-L384.csv').open()));unit,v=data[0],data[1]
 keys=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
 profiles[D]={k:dict(value=v[k],unit=unit[k]) for k in keys}
summary['ncu']=profiles
(R/'ncu-summary.json').write_text(json.dumps(profiles,indent=2))
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
mt='\n'.join('| %d | %d | %.3f | %.3f | %.2fx | %.1f%% |'%(v['D'],v['L'],v['previous_us']/1000,v['current_us']/1000,v['speedup'],v['time_reduction_pct']) for v in rows)
tt='\n'.join('| %d | %d | %.3f | %.3f | %.2fx |'%(v['D'],v['L'],v['triton_us']/1000,v['cuda_us']/1000,v['vs_triton']) for v in rows)
md=f'''# TriMul D별 CUDA 최적화 · 2026-09-23

D64/128/256/384/512 × L384/768을 연결·검증했다. D128 L384는 기존 선택을 유지했다. 나머지는 이전 CUDA 포트보다 빨라졌다. **D128 외 네 폭은 여전히 Triton보다 느리며, 성능 개발 완료나 SoL90 달성을 주장하지 않는다.**

## 왜 D128 L768의 배속이 떨어졌나

L384는 단일 B7, L768은 분리 B7이었다. 이전 L768 trace에서 뒤쪽 두 커널은 825 + 1207 = 2032us를 차지했다. 단일 후보는 dWL 상대 오차0.051913%가 기존 기준0.050000%를 초과해 선택하지 못했다.

단일 커널 내부에서 dW의 긴 FP32 누적을 두 구간으로 나누고 마지막에 합치도록 수정했다. 기존 x_n 저장·재계산 정책과 미분 수식은 유지했다. dWL 오차는0.013900%로 낮아졌다. 입력·weight 변경, gamma=0, graph/eager, strict LN 기준을 통과해 L768도 단일 B7로 연결했다. B1과 forward는 기존 경로다.

## 다른 D는 왜 느렸나 / 이번 변경

앞선 것은 D128의 최적화 수준을 확장한 구현이 아니라, 폭을 지원하도록 만든 기능 포트였다. 큰 전역 작업 공간과 반복적인 입력/가중치 로드, GEMM 사이의 동기화가 남아 있었다. D가 커지면 GEMM 연산량도 D²에 비례해 커진다.

- 입력 projection/gate 재계산에 Anthropic K1의 register operand + TMA producer + WGMMA consumer를 차용했다. D64/256은 입력 x_n 타일을 레지스터에 유지한다. D384/512는 shared memory에 유지하고 WGMMA SS가 직접 읽어 register spill을 없앤다. 가중치는 순서대로 공급한다.
- dLeft/dRight를 TMA로 읽고 dp/dg를 TMA로 저장한다. 이 작업 공간은 channel-major로 바꿔 dX/dW GEMM이 같은 값을 소비한다.
- NCU에서 register-operand D512의 local load 요청202,205,424 sectors, ptxas spill load436bytes를 발견했다. 무작정 register 한도를 풀면 CTA 상주 수가 줄어 더 느렸다. 큰 폭의 x_n을 shared operand로 바꿔 같은 두 CTA/SM에서 ptxas spill 0을 달성했다.
- 한 weight block 전체가 끝날 때까지 기다리는 대신, 필요한 K 조각의 WGMMA가 끝나면 그 shared slot을 즉시 반환한다. 가중치 버퍼를 줄여 큰 D에서도 B7 CTA 두 개가 SM에 상주할 수 있게 했다.
- **B7은 하나의 CUDA kernel launch다.** 성능을 진단한 분리 producer 후보는 최종 연결에 넣지 않았다. dX와 dW를 별도 kernel로 분리하지 않았다. 다만 단일 launch 내부의 global dp/dg·partial dW 작업 공간은 남아 있다.
- forward/B1은 이전의 검증된 CUDA cubin을 재사용한다. B7의 자원 요구량 때문에 forward/B1까지 occupancy가 줄지 않게 grid를 별도로 계산한다.
- m64n128, 더 큰 warp-group 타일, projection accumulator 압축 등도 시험했으나 전체가 느려져 선택하지 않았다.

## 같은 GPU에서 이전 CUDA와 교대 측정

이전은 바로 앞 D별 CUDA 포트다. Anthropic 또는 cuEquivariance 대비 배속이 아니다. D128 L384의 차이는 잡음 수준이다.

| D | L | 이전 CUDA ms | 현재 CUDA ms | 배속 | 시간 감소 |
|---:|---:|---:|---:|---:|---:|
{mt}

## 별도의 같은 실행 Triton/CUDA 교대 측정

위 표와 별도 실행이다. 각 행의 두 값은 같은 프로세스에서 측정했다. 1x 미만이면 CUDA가 느리다. Triton 기준은 기존 벤치의 정적 compile 경로이며 cache miss 시 heuristic24 설정을 사용하는 경우가 있어, 모든 config를 재튜닝한 최상 성능이라고 주장하지 않는다.

| D | L | Triton ms | CUDA ms | Triton 대비 배속 |
|---:|---:|---:|---:|---:|
{tt}

## 실제 배선 / HBM 흐름

```
forward: x -> Anthropic-derived K1 -> left/right + saved x_n
         left/right -> cuBLAS x2 -> tri -> existing CUDA output -> y
backward: tri, x_n, dy -> existing CUDA B1 -> dTri, dGate, output weight/LN grads
          dTri, left/right -> cuBLAS x4 -> dLeft/dRight
          x, x_n, dLeft/dRight, dGate, dy -> one CUDA B7 -> dx, input weight/LN grads
B7 internal: TMA x_n / weights / dLeft / dRight
             -> projection/gate recompute -> dp/dg global workspace
             -> dX and split-K dW -> LN derivative + residual + partial reductions
```

출처: Anthropic 추론 커널·TMA/WGMMA 구현을 계승한 학습 확장이다. 원본 파일은 수정하지 않았고 파생 파일과 SHA를 보관했다. 입력 LN 출력 x_n은 저장하며 출력 LN activation 저장 정책은 사용하지 않는다. D512 forward 입력 LN은 기존처럼 별도 경로다.

## 선택한 큰 폭 B7 설정

| D | producer token tile | weight slot 수 | slot당 K64 chunk | CTA/SM | dW split 수 |
|---:|---:|---:|---:|---:|---:|
|64|64|8|1|2|128|
|256|64|1|4|2|32|
|384|64|1|6|2|32|
|512|64|1|4|2|32|

D128은 기존 특화 ring 기반 구현을 사용한다. 폭별 커널 config와 선택 근거는 `gp-tune-*`, `fused-tune-*`, `tune-*` JSON 및 job 로그에 보관했다. 모든 가능한 config를 탐색했다는 뜻은 아니다.

## NCU 진단 · D512 L384 B7

register-operand → shared-operand 변경 후 NCU kernel 시간은7.302 →6.131ms, tensor pipe 활성률은26.8% →31.7%였다. local load/store 요청은202,205,424 /58,952,892 sectors →0/0으로 감소했다. 총 DRAM 읽기+쓰기는 약8.04GB →8.00GB로 거의 같다. 즉 개선은 큰 HBM 트래픽 감소가 아니라 spill 처리 비용을 없앤 효과다. 이 활성률을 SoL32%라고 부르지 않는다.

## 검증과 측정 범위

- public PyTorch autograd 진입점 10shape, 모든11개gradient, 두 번 forward 후 backward의 저장값 독립성, in-place 변경 탐지 통과.
- 각 shape에서 PyTorch/Triton BF16 비교: output 상대L2<0.005, 각 gradient<0.01. 이는 교차 구현 기준이며 D128 같은 수식의 엄격 LN5e-6 기준을 완화한 것이 아니다.
- 입력/weight 변경 후 CUDA graph 검증. 새 네 폭은 graph/eager 일반 output/gradient bit-exact, atomic LN gradient 상대L2≤5e-6.
- 네 새 폭 L384와 수정된 D128 L768에서 compute-sanitizer memcheck/racecheck 모두0errors/0hazards.
- 측정은 CUDA graph replay, dropout25% 고정 mask/scale, residual 포함. RNG 생성·optimizer·CPU dispatch·compile·plan 생성은 제외했다.
- 개발 진입점만 변경했다. production 자동 dispatch나 원격 push는 하지 않았다. 선택 B7의 NCU D256/D512 L384 자료를 `ncu-*`에 기록했다. 파이프 활성률은 roofline/SoL90 달성률과 다르다. Cache와 clock을 강제 고정하지 않은 진단 측정이다.

다음 병목은 넓은 D의 B1과 B7 GEMM·전역 작업 공간 전달이다. 특히 D512 개선폭은 작다. 단일 launch 자체만으로 이 비용이 사라지지는 않는다.

[HTML](index.html) · [수치](summary.json) · [재현·SHA](manifest.json)
'''
(R/'README.md').write_text(md)
def table(head,rs):return '<div class="scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rs)+'</table></div>'
pt=table(['D','L','이전 CUDA ms','현재 CUDA ms','배속','시간 감소'],[[v['D'],v['L'],f"{v['previous_us']/1000:.3f}",f"{v['current_us']/1000:.3f}",f"{v['speedup']:.2f}×",f"{v['time_reduction_pct']:.1f}%"] for v in rows])
tt=table(['D','L','Triton ms','CUDA ms','Triton 대비 배속'],[[v['D'],v['L'],f"{v['triton_us']/1000:.3f}",f"{v['cuda_us']/1000:.3f}",f"{v['vs_triton']:.2f}×"] for v in rows])
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TriMul D64–512 CUDA 최적화</title><style>body{max-width:1150px;margin:24px auto;padding:0 16px;font:16px/1.6 system-ui;color:#dce6f2;background:#111923}section{background:#1c2937;padding:20px;border-radius:12px;margin:20px 0}a{color:#7dc4ff}.notice{border-left:4px solid #ffc15c;padding:16px;background:#392f20}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:right;white-space:nowrap;border-bottom:1px solid #405061}pre{overflow:auto;padding:14px;background:#12201f;color:#9ce3c4}.flow{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.box{border:1px solid #69bba0;padding:12px;border-radius:8px}small{color:#acbbcc}</style><h1>TriMul · D64–512 CUDA 최적화</h1><p>2026-09-23 · H100 · 양방향 · B1 · BF16 · hidden=방향별 D · dropout25% · residual · 모든 gradient</p><p class="notice">D128 L768은 단일 B7로 연결했습니다. 네 다른 폭도 이전 CUDA보다 빨라졌습니다. <b>D128 외에는 여전히 Triton보다 느립니다.</b> SoL90/최적화 완료를 주장하지 않습니다.</p><section><h2>D128 L768: 다른 경로였던 이유</h2><p>기존 L768은 dWL 정확도 문제로 분리 B7을 사용했습니다. dW 누적을 두 구간으로 나눠 FP32로 합쳐 오차를 0.051913% → 0.013900%로 낮추고, 단일 B7로 전환했습니다. 기존 strict LN 기준을 유지했습니다.</p></section><section><h2>이전 CUDA → 현재 CUDA</h2><p>같은 GPU에서 교대 측정. 바로 이전 폭별 CUDA 포트 대비이며 Anthropic/cuEquivariance 대비가 아닙니다. D128 L384는 기존 선택 유지입니다.</p>'''+pt+'''</section><section><h2>기존 Triton 대비</h2><p>별도 실행에서 Triton/CUDA를 교대 측정했습니다. 1× 미만이면 CUDA가 느립니다. 기존 cache miss 시 heuristic24를 사용하며 전체 config 재튜닝 결과는 아닙니다.</p>'''+tt+'''</section><section><h2>선택된 배선</h2><div class="flow"><div class="box">기존 CUDA B1<br>tri + x_n + dy<br>→ dTri / dGate / output grads</div><b>→</b><div class="box">cuBLAS ×4<br>dLeft / dRight</div><b>→</b><div class="box">단일 CUDA B7<br>TMA / WGMMA<br>→ dx / input grads</div></div><h3>B7 내부</h3><pre>x_n → TMA → D64/256: registers · D384/512: shared operand
  큰 폭의 register spill을 없애며 projection/gate마다 HBM 재로드하지 않음
weights → 작은 shared slot → WGMMA → 사용한 K chunk 즉시 반환
dLeft/dRight → TMA → dp/dg 생성 → channel-major global workspace
workspace → dX와 dW 계산 → LN 미분 + residual + partial reduction</pre><p>Anthropic K1의 producer/consumer 구조를 차용한 학습 확장입니다. B7은 하나의 kernel launch입니다. dp/dg 및 partial workspace의 HBM 왕복은 남아 있습니다. forward/B1은 이전 검증 cubin을 재사용하고 B7과 별도로 occupancy를 계산합니다.</p><p>큰 GEMM 타일, N128, accumulator 압축 후보는 전체에서 느려져 제외했습니다. 남은 병목은 넓은 폭의 B1/B7 GEMM과 작업 공간 전달입니다. 특히 D512 개선폭이 작습니다.</p></section><section><h2>NCU · D512 L384 B7</h2><p>local load/store: 202,205,424 / 58,952,892 sectors → 0 / 0. 커널 진단 시간 7.302 → 6.131ms, tensor pipe 활성률 26.8% → 31.7%. DRAM 전송은 약8GB로 비슷합니다. spill 제거에 따른 개선이며, 이를 SoL32%라고 해석하지 않습니다.</p></section><section><h2>검증·적용 범위</h2><p>10shape public autograd / 11gradient / 여러 forward 저장값 / version check / 입력 변경 CUDA graph 통과. 새 네 폭 L384와 D128 L768에서 memcheck·racecheck 0errors/0hazards.</p><p>개발 진입점 <code>runs/trimul_training_current.py</code>. production 자동 dispatch 및 push는 하지 않았습니다. CUDA graph 시간에서 optimizer·RNG 생성·CPU dispatch·compile·plan 생성은 제외했습니다. D256/D512의 NCU 진단도 첨부했습니다. 파이프 활성도를 roofline/SoL90 달성률로 해석하지 않습니다.</p><p><a href="README.md">상세 설명·기준</a> · <a href="summary.json">수치 JSON</a> · <a href="manifest.json">SHA·재현</a></p></section></html>'''
(R/'index.html').write_text(page)
# Refresh prior bookmarked experiment URLs; preserve their numerical evidence.
for name in ('trimul_widths_20260922','trimul_cuda_widths_20260923'):
 p=R.parent/name/'index.html';p.write_text(page.replace('<meta charset=',f'<base href="../{R.name}/"><meta charset=',1))
status=ROOT/'TRIMUL_STATUS.html';s=status.read_text();sec='<section id="trimul-widths"><h2>2026-09-23 · 모든 폭 CUDA 최적화</h2><p>D128 L768 단일 B7 전환·strict 검증 완료. D64/256/384/512도 이전 CUDA보다 개선했으나 여전히 Triton보다 느립니다.</p><a href="runs/'+R.name+'/index.html">전체 측정표·배선·검증 보기</a></section>';s=re.sub(r'<section id="trimul-widths">.*?</section>',sec,s,flags=re.S);s=s.replace('L768 새 단일 B7 적용·엔진 production 승격','L768 새 단일 B7 적용은 완료했습니다. 엔진 production 승격');s=s.replace('TriMul · 개발 마무리','TriMul · 개발 현황').replace('2026-09-22 · 이번 최적화 실험 종료 · L384 양방향 H100','2026-09-23 · D64–512 학습 CUDA 업데이트 · 양방향 H100').replace('L768 최신 전체 수치는 없습니다.','당시 기록입니다. 현재 L768은 위의 최신 D별 표를 참고하세요.').replace('trimul_training_current.py의 L384는 수정된 단일 B7. 다른 길이는 기존 경로 유지.','trimul_training_current.py: 다섯 폭, 두 길이를 지원하며 B7은 단일 커널입니다.').replace('수정 후 runs/trimul_ln_gradient_20260922/policy.py의 Fixed. 과거 표는 수정 전 Latest 기록.','최신 선택은 runs/trimul_training_current.py. 아래 L384 고정 표는 과거 기록입니다.')
status.write_text(s)
p=ROOT/'TRIMUL_STATUS.md';s=p.read_text().replace('# TriMul 개발 현황 · 2026-09-22','# TriMul 개발 현황 · 2026-09-23').replace('남은 범위: L768 새 단일 B7, production 승격','남은 범위: 큰 폭의 추가 최적화, production 승격');needle='## TriMul D별 실험';s=s[:s.index(needle)]+f'## TriMul D별 최적화 · 2026-09-23\n\n[최신 표·배선·검증](runs/{R.name}/index.html). D128 L768은 단일 B7로 전환했다. D64/256/384/512도 이전 CUDA 대비 개선했지만 Triton보다 느리다. 10shape autograd와 수정 커널 sanitizer 검증 통과.\n' if needle in s else s;p.write_text(s)
files=[R.parent/'trimul_training_current.py',R/'summary.json',R/'README.md',R/'index.html']
files += list(R.glob('ncu-*.csv'))+list(R.glob('ncu-*.log'))+[R/'ncu-summary.json']
files += list(R.glob('*.py'))+list(R.glob('*.cu'))+list(R.glob('*.cuh'))+list((R/'fixed128').glob('*.cu'))+list((R/'fixed128').glob('*.py'))+list((R/'fixed128').glob('*.inc'))+list((R/'gp_headers').rglob('*.cuh'))
for pattern in ('check-D*.json','latest-D128*.json','autograd-D*.json','paired-D*.json','memcheck-D*.log','racecheck-D*.log'):files+=list(R.glob(pattern))
for D in (64,256,384,512):
 for L in (384,768):
  v=json.loads((R/f'check-D{D}-L{L}.json').read_text());files += [Path(v['cubin']),Path(v['route']['b7_cubin'])]
for L in (384,768):
 v=json.loads((R/f'latest-D128-L{L}.json').read_text())
 files += [Path(c['path']) for group in v['cubins'].values() for c in group]
manifest=dict(summary=summary,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(files)},commands=['sbatch '+str(R/'final.sbatch'),'sbatch '+str(R/'paired.sbatch'),'sbatch '+str(R/'autograd.sbatch'),'sbatch '+str(R/'sanitize.sbatch'),'sbatch '+str(R/'sanitize128.sbatch')]);(R/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('REPORT COMPLETE',len(rows))
