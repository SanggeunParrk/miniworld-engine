from pathlib import Path
import csv,json,hashlib,html,re
R=Path(__file__).resolve().parent

def profile(p):
 rows=list(csv.reader(p.open()));i=next(i for i,r in enumerate(rows) if 'Kernel Name' in r)
 units=dict(zip(rows[i],rows[i+1]));vals=dict(zip(rows[i],rows[i+2]))
 return {k:dict(value=float(v.replace(',','')),unit=units[k]) for k,v in vals.items() if k.startswith(('gpu__','dram__','lts__','sm__','smsp__'))}
def scaled(d,k,unit):
 x=d[k];factor={'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9,'ns':1e-3,'us':1,'ms':1e3}
 return x['value']*factor[x['unit']]/(1e6 if unit=='MB' else 1)
def fmt(x):return '%.3f'%x

report=dict(selected=True,baseline='B1 v51',scope='B1-B4 CUDA, BF16 C128/H256, dropout .25, node01',sol90_achieved=False,measurements={},sanitizers={},experiments={})
tables=[];mdrows=[];ncrows=[];profiles={};resources={}
for n in (384,768):
 d=json.loads((R/('results-L%d.json'%n)).read_text());report['measurements'][str(n)]=d
 assert set(d['valid'])=={'baseline','late_only','optimized'}
 for group in ('checks','mutated','graph_vs_eager'):
  assert all(v['bit_exact'] for vals in d[group].values() for v in vals.values())
 for scope in ('b1','backward','forward_backward'):
  t={k:v['median_us'] for k,v in d['times'][scope].items()};gain=100*(1-t['optimized']/t['baseline'])
  mdrows.append('| %d | %s | %.3f | %.3f | %.3f | %.2f%% |'%(n,scope,t['baseline'],t['late_only'],t['optimized'],gain))
  tables.append('<tr><td>%d</td><td>%s</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.2f%%</td></tr>'%(n,scope,t['baseline'],t['late_only'],t['optimized'],gain))
 for variant in ('baseline','optimized'):
  p=profile(R/('ncu-L%d-%s.csv'%(n,variant)));profiles['%d-%s'%(n,variant)]=p
  ncrows.append('| %d | %s | %.3f | %.3f | %.3f | %.2f%% |'%(n,variant,scaled(p,'gpu__time_duration.sum','us'),scaled(p,'dram__bytes_read.sum','MB'),scaled(p,'dram__bytes_write.sum','MB'),p['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed']['value']))
 cubin=Path(d['cubins']['optimized']['b1']);log=cubin.with_suffix('.ptxas.log').read_text().split('Function properties for b1_fused',1)[1]
 resources[str(n)]=dict(cubin=str(cubin),sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),ptxas=log)
 s=json.loads((R/('sanitizers-L%d.json'%n)).read_text())
 assert len(s)==2 and all(x['returncode']==0 for x in s)
 for x in s:
  check=json.loads((R/('%s-L%d.json'%(x['tool'],n))).read_text())
  assert check['sha256']==resources[str(n)]['sha256']
  x['probe']=check
 report['sanitizers'][str(n)]=s
 c=json.loads((R/('current-check-L%d.json'%n)).read_text())
 assert c['b7_unchanged'] and c['b1_cubin']==str(cubin)
 report.setdefault('current_adapter_checks',{})[str(n)]=c
report['profiles']=profiles;report['resources']=resources
report['current_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'policy.py',R.parent/'trimul_training_current.py',R/'replace_plan.py',*R.glob('*.cu'),*R.glob('*.cuh'),*R.glob('*.inc')]}
for name in ('chunk_phases','chunk_carry','weight_store_late','gate_reverse','tma_cache_priority','gate_demote'):
 root=R.parent/('trimul_b1_'+name+'_20260922')
 report['experiments'][name]={str(n):json.loads((root/('tune-L%d.json'%n)).read_text()) for n in (384,768)}
(R/'summary.json').write_text(json.dumps(report,indent=2))
readme='''# B1–B4: dW 저장 순서와 TMA 캐시 재사용 개선

2026-09-22 · node01 H100 · 양방향 BF16 C128/H256 · dropout 25%, mask, residual.

## 선택된 변경

1. dWproj FP32 누산값을 dWgate 단계가 끝날 때까지 레지스터에 유지한다.
   기존 global partial 버퍼에 쓰는 시점만 늦춘다. 새 activation·scratch를 추가하지 않는다.
2. dGate TMA 출력에는 L2 evict_last 힌트를 준다.
3. dWgate가 소비한 dGate는 evict_first로 읽는다. L384에서는 x_n 읽기도 함께 낮춘다.
   캐시 힌트가 지켜져야만 맞는 코드가 아니며, 기존 TMA 완료 대기와 CTA/grid barrier를 유지한다.

계산식, FP32 누산 순서, BF16 반올림, 입력 x_n·원본 tri·출력 LN mean/rstd 저장 정책은 그대로다.
단일 CUDA 호출, 두 계산 warpgroup, dynamic shared 228352B와 132 CTA를 유지한다.
L384 B1_LATE_WEIGHT=3 / B1_TMA_PRIORITY=2 / B1_GATE_DEMOTE=3,
L768은 B1_GATE_DEMOTE=1을 사용한다. 설정은 policy.py에서 고정했다.

## 동일 실행 3자 비교

단위 µs. 각 구간 5라운드 × 200회 교차 CUDA graph 측정의 pooled median.
B1 구간에서 5개 라운드 모두 새 후보가 v51보다 빨랐다.
전체 구간은 기존 v51 회귀 벤치의 B7를 그대로 둔 측정이며 최신 B7 후보 비교는 아니다.
각 구간은 별도로 측정하므로 시간을 더해 전체를 만들지 않는다.

| L | 구간 | v51 | 저장만 지연 | 저장 지연 + 캐시 정책 | 시간 단축 |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(mdrows)+'''

전체 학습 시간 변화는 작다. 위 B1 개선율을 전체 fwd+bwd 개선율로 사용하지 않는다.
forward는 동일한 코드다. live weight packing과 전체 11 gradients를 포함하며,
optimizer·RNG 생성·컴파일 시간·CPU dispatch는 제외한다.

## NCU: HBM 읽기 감소

별도 profiler 실행. cache-control none, clock-control none, 5회 워밍업 후 1회 수집.
출력 쓰기의 일부가 캐시에 남으면 kernel 범위 DRAM write counter에 포함되지 않을 수 있다.
아래 실제 바이트 감소를 논리적 출력 버퍼 삭제로 해석하지 않는다.

| L | 구현 | NCU µs | HBM 읽기 MB | HBM 쓰기 MB | HBM peak 비율 |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(ncrows)+'''

캐시 재사용으로 HBM 읽기를 줄이면서 속도가 개선됐다. 전체 알고리즘의 최소 시간을
입증한 것은 아니다. HBM peak 비율이 낮아졌어도 해야 할 연산이 늘거나 느려졌다는 뜻은 아니다.
**SoL90 미달 / 전체 알고리즘 SoL90 달성 근거 없음.**

## 검증

- 일반·변경 입력, weights, dy, mask/dropout, gamma_out[0]=0: forward 및 11 gradients bit-exact.
- graph/eager bit-exact, 원본 tri·통계·x_n 저장 pointer/dtype 확인.
- 별도 B1 probe: 3개 입력 변형, 6개 출력 bit-exact, NaN scratch poison, 반복 replay, counter 0.
- L384/L768 각각 동일 cubin의 memcheck·racecheck 통과. SHA-256은 summary.json에 포함.
- 현재 개발 runs/trimul_training_current.py에도 새 B1을 연결했다. 기존 B7 class를 유지했고,
  일반·변경 입력에서 전체 출력/11 gradients 및 graph/eager bit-exact를 별도로 확인했다.
- 기존 B7의 독립 수학 참조 정확도 문제를 해결한 실험은 아니다. 생산용 dispatch로 승격하지 않았다.

## 함께 구현하고 제외한 구조

- 2/4/8/16 타일씩 주 계산과 dWgate 교대: FP32 partial 왕복으로 느려짐.
- dWproj를 레지스터에 유지하는 묶음 교대: 손해는 줄었지만 기본보다 느림.
- dWgate 역순 타일 방문: FP32 누산 순서가 바뀌어 일부 입력에서 상대 L2 0.01% 한도를 초과.
  성능을 채택하지 않고 제외했다. 이번 선택은 역순 방문을 사용하지 않는다.
- x_n·tri·dTri까지 캐시 우선순위를 바꾼 조합은 추가 이득이 없어 제외했다.

상세 모든 후보·원본 결과·소스/cubin 식별자는 summary.json 및 각 실험 폴더에 보관했다.
역순 구현의 최초 job15403은 전처리 줄바꿈 오류로 실패했고 수정 후 job15408을 검사했다.
개발 연결 검사 job15435는 Python 모듈 이름 충돌로 시작 단계에서 실패했고,
절대 파일 경로 import로 수정한 job15438을 사용했다.

PTX cache-policy 의미와 구문은 NVIDIA 공식 문서를 확인했다:
https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
성능 수치는 이 문서가 아니라 위 H100 실험의 결과다.
'''
(R/'README.md').write_text(readme)
body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B1–B4 캐시 재사용 최적화</title><style>body{max-width:1150px;margin:24px auto;padding:0 18px;background:#f5f7fa;color:#21384c;font:16px/1.65 system-ui}section{padding:20px;background:white;border:1px solid #d4dfeb;border-radius:10px;margin:18px 0}table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:left;border-bottom:1px solid #dae3ed}.scroll{overflow:auto}a{color:#1b60a0}.notice{padding:14px;background:#fff1d7}pre{white-space:pre-wrap;font:inherit}.flow{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.box{padding:14px;background:#e7f2f8;border-radius:8px}.save{background:#fce5ca}.cache{background:#def1df}</style>
<a href="../../TRIMUL_STATUS.html">← 현황판</a><h1>B1–B4: 저장 시점·캐시 재사용 개선</h1>
<p class="notice">B1 L384 4.38% / L768 0.75% 단축 · SoL90 미달 · 전체 학습 개선율과 구분</p>
<section><h2>한 CUDA 호출 안에서 바뀐 순서</h2><p>큰 중간 activation의 저장 정책과 수식은 유지합니다.</p>
<b>기존 v51</b><div class="flow"><div class="box">주 계산<br>dGate·dTri·LN grads</div>→<div class="box save">dWproj partial 저장</div>→<div class="box">x_n·dGate 다시 읽기<br>dWgate 계산</div>→<div class="box">최종 합산</div></div>
<b>새 선택</b><div class="flow"><div class="box">주 계산<br>dWproj 누산값 레지스터 유지</div>→<div class="box cache">dGate 캐시 보존<br>소비한 타일 우선순위 낮춤</div>→<div class="box">dWgate 계산</div>→<div class="box save">dWproj partial 저장</div>→<div class="box">최종 합산</div></div></section>
<section><h2>같은 실행의 비교</h2><div class="scroll"><table><tr><th>L</th><th>구간</th><th>v51 µs</th><th>저장 지연 µs</th><th>새 선택 µs</th><th>시간 단축</th></tr>'''+''.join(tables)+'''</table></div><p>전체 구간은 기존 B7를 유지한 회귀 벤치입니다. 전체 학습 변화는 작으며 B1 개선율로 대신 표시하지 않습니다.</p></section>
<section><h2>구현·NCU·검증·실패한 후보</h2><pre>'''+html.escape(readme)+'''</pre><p><a href="summary.json">측정·검증·SHA-256</a> · <a href="README.md">Markdown</a> · <a href="policy.py">선택 개발 adapter</a></p></section></html>'''
(R/'index.html').write_text(body)
p=R.parent.parent/'TRIMUL_STATUS.html';s=p.read_text();marker='b1-cache-reuse-20260922'
if marker not in s:
 s=s.replace('</header>','</header><aside id="'+marker+'"><b>최신 B1–B4:</b> dW 저장 지연 + TMA 캐시 재사용. 동일 실행에서 L384 190.53 → 182.18 µs (4.38%), L768 640.96 → 636.13 µs (0.75%). 정확도·memcheck·racecheck 통과. <a href="runs/'+R.name+'/index.html">배선·성능·NCU·검증</a></aside>',1)
 p.write_text(s)
print('Report written; both exact selected cubins passed memcheck/racecheck.')
