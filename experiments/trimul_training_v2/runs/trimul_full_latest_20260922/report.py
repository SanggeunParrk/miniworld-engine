from pathlib import Path
import json,statistics,html
R=Path(__file__).resolve().parent;ROOT=R.parent.parent
r=json.loads((R/'results.json').read_text());assert r['complete']
labels={'historical_1048':'이전 개선 B7 조합','regression_1185':'구형 B7을 둔 검사 코드','latest_combined':'최신 B1 + 최신 단일 B7'}
t={s:{n:v['median_us'] for n,v in vals.items()} for s,vals in r['times'].items()}
old=t['full']['historical_1048'];new=t['full']['latest_combined'];pct=100*(1-new/old);speed=old/new
failed=[]
for c,models in r['checks'].items():
 for name,checks in models.items():
  for grad,e in checks['graph'].items():
   if not e['valid']:failed.append(dict(case=c,model=name,gradient=grad,relative_l2=e['relative_l2'],limit=e['limit']))
assert all(x['model']=='latest_combined' and x['gradient'] in ('dgamma_in','dbeta_in') for x in failed)
telemetry={}
for scope in t:
 rows=[list(map(float,x['csv'].split(','))) for x in r['telemetry'] if x['phase']==scope and x['returncode']==0 and len(x['csv'].split(','))==5]
 telemetry[scope]={n:statistics.median(x[i] for x in rows) for i,n in enumerate(['sm_mhz','memory_mhz','power_w','power_limit_w','temperature_c'])}
verified=json.loads((R/'historical-cubin-verification.json').read_text());assert all(x['matches'] for x in verified.values())
summary=dict(times_us=t,speedup_vs_historical=speed,time_reduction_pct=pct,failures=failed,telemetry=telemetry,job=r['job'],latest_correctness_passed=False,latest_timing_is_diagnostic=True,production_dispatch_changed=False,current_adapter_changed=False)
(R/'summary.json').write_text(json.dumps(summary,indent=2))
rows=['| 구간 (µs) | 이전 개선 B7 조합 | 구형 B7 검사 코드 | 최신 B1+단일 B7 후보 |','|---|---:|---:|---:|']
scopes={'full':'전체 fwd+bwd 직접 측정','forward':'forward','backward':'backward','b1':'B1–B4 단독','b7':'B7–B12 단독'}
for s,label in scopes.items():rows.append('| %s | %s |'%(label,' | '.join('%.3f'%t[s][n] for n in labels)))
text='''# 최신 B1+B7 양방향 TriMul 전체 재측정 · 2026-09-22

**L384 전체 fwd+bwd: 최신 조합 %.3f µs. 같은 실행의 이전 개선 B7 조합 %.3f µs 대비 %.2f%% 단축, %.4f배.**

**최신 조합의 수치는 정확도 검증이 남은 후보의 진단값이다.** 입력 LN gamma/beta gradient가 변형 입력에서 기존 상대L2 한도5e-6를 초과했다. 허용치를 완화하거나 기본 경로에 적용하지 않았다.

## 조건과 시간

job%s · node01 H10080GB · L384/C128/양방향 H256 · BF16 · dropout25%% · pair mask·residual.
같은 프로세스/GPU의 3자 교차 CUDA graph 비교. 전체6×300회, 구간별5×300회.
full은 매 replay마다 fresh forward activation을 생성하는 전체 그래프를 직접 측정했다. 별도 구간 시간을 더한 값이 아니다.
전체6개 라운드에서 최신 후보가 이전 개선 B7 조합보다 빨랐다.

%s

포함: 양방향, B1–B12, cuBLAS contraction, live weight packing, 11개 gradient.
제외: optimizer, dropout RNG 생성, compilation, CPU dispatch. 전체 MiniWorld 학습 step이 아닌 TriMul 모듈 시간이다.

## 왜 이전1,048µs / 직전1,185µs와 다른가

- 과거1,047.808µs 및1,075.888µs 조합의 B1/B7 cubin을 SHA-256으로 대조해 동일함을 확인했다. 이번 실행에서는1,065.824µs다.
- 직전1,185µs 검사는 오래된 split_xn_pc1 B7을 사용했다. 이 검사 조합은 이번 실행에서1,222.288µs다.
- 최신 후보는 선택된 cache-policy B1과 K128/ring12/producer32/cluster2 단일 B7이다. B7 cubin SHA는 selected.json과 일치한다.
- 이번 full 구간의 3자 혼합 workload 중앙 SM clock은%.0fMHz, 메모리는%.0fMHz, 전력%.2fW다. 과거1,048µs 측정은1920MHz,1,076µs 측정은1845MHz였다. 개별 arm의 클록으로 해석하지 않는다.
- 과거와 이번 절대 시간을 직접 나눠 개선율을 계산하지 않았다. %.2f%%는 이번 동일 실행 비교다.

## 정확도

일반 입력 및2개 weights/input/mask/dropout 변형 검사. 모든 경로 graph/eager bit-exact이며 모든 출력은 finite다.
forward, 출력측 gradient는 정확히 일치했다. 최신 후보의 dX/가중치 gradient는 기존 한도내지만 입력 LN 파라미터 gradient는 아래 한도를 넘었다.

| case | gradient | 상대L2 | 기존 한도 |
|---|---|---:|---:|
%s

이전 개선 B7 조합 및 구형 B7 검사 코드는 모든3case에서 기존 검증을 통과했다. 최신 후보는 case1/2가 실패했으며 성능 진단만 진행했다.
이는 새 전체 연결 검사에서 드러난 한계다. 개별 B7의 과거 sanitizer/입력 검사 통과를 전체 정확도 통과로 대체하지 않았다.
job15522는 첫 실패에서 중단했다. 실패 기록을 보존하고 job%s에서 한도를 유지한 채 모든case와 진단 시간을 수집했다.

## 실제 실행과 적용 상태

Profiler에서 이전 개선 조합은 front_b7b12_dw + front_b7b12_dx, 최신 후보는 b7_joint 한 회를 확인했다.
둘 다 infer_k1/save_k3, b1_fused 한 회 및 cuBLAS contraction을 포함한다. 결과JSON에 전체kernel 목록·cubin hash·원본samples·telemetry를 보존했다.
측정용 policy.py에 최신 조합을 구성했으며 생산 dispatch와 runs/trimul_training_current.py는 변경하지 않았다.
최신 조합의 정확도 문제를 해결하기 전까지 검증된 기본 경로로 승격하지 않는다. 새 SoL 주장은 없다.

[원본결과](results.json) · [요약](summary.json) · [과거 cubin 대조](historical-cubin-verification.json) · [벤치](bench.py)
'''%(new,old,pct,speed,r['job'],'\n'.join(rows),telemetry['full']['sm_mhz'],telemetry['full']['memory_mhz'],telemetry['full']['power_w'],pct,'\n'.join('| %s | %s | %.9g | %.1g |'%(x['case'],x['gradient'],x['relative_l2'],x['limit']) for x in failed),r['job'])
(R/'README.md').write_text(text)
tr=''.join('<tr><td>%s</td>%s</tr>'%(label,''.join('<td>%.3f</td>'%t[s][n] for n in labels)) for s,label in scopes.items())
bars=[]
for i,(n,label) in enumerate(labels.items()):
 y=30+i*48;v=t['full'][n];color='#d28d2c' if n=='latest_combined' else '#497faf'
 bars.append('<text x="8" y="%d">%s</text><rect x="255" y="%d" width="%.1f" height="24" fill="%s"/><text x="680" y="%d">%.1f µs</text>'%(y+17,label,y,v*.31,color,y+17,v))
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TriMul 최신 전체 재측정</title><style>body{max-width:1080px;margin:25px auto;padding:0 16px;font:16px/1.65 system-ui;color:#24364c;background:#f4f7fb}section{background:white;padding:18px;margin:18px 0;border-radius:10px}.notice{background:#fff0d2;padding:14px;border-left:5px solid #bc7718}table{border-collapse:collapse;width:100%%}td,th{padding:8px;border-bottom:1px solid #dae2ed;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}svg{max-width:100%%;height:auto}a{color:#1766a1}</style><h1>最新 B1+B7 · L384 전체 학습 재측정</h1><p>2026-09-22 · node01 H100 · BF16 C128/H256 · 양방향 · dropout25%% / mask / residual</p><h2>최신 후보 %.3f ms · 이전 조합 대비 %.2f%% 단축</h2><p class="notice"><b>성능 진단값 — 전체 정확도 검증 미완료.</b> 입력 LN γ·β gradient가 기존 상대L2 한도5×10⁻⁶를 초과했습니다. 최대9.535×10⁻⁶. 기준 완화·기본 경로 승격 없이 기록합니다.</p><section><h2>같은 실행의 전체 그래프 시간</h2><svg viewBox="0 0 815 205" role="img" aria-label="이전 조합, 구형 검사 코드, 최신 후보의 전체 시간"><g font-family="system-ui" font-size="14">%s</g></svg><p>과거1,048µs였던 조합을 동일 cubin으로 재현했습니다. 이번 실행에서는%.3fµs입니다. 과거1,185µs는 다른 B7을 둔 검사 코드였고 이번에는%.3fµs입니다.</p></section><section><h2>구간별 직접 측정 · µs</h2><div class="scroll"><table><tr><th>구간</th><th>이전 개선 B7 조합</th><th>구형 B7 검사 코드</th><th>최신 B1+단일 B7 후보</th></tr>%s</table></div><p>전체6×300회 / 구간5×300회 교차 CUDA graph. 전체는 구간 합계가 아닌 직접 측정값입니다. 전체MiniWorld 학습step이 아닌 TriMul 모듈입니다.</p></section><section><h2>배선·검증</h2><p>최신: cache-policy B1 → cuBLAS → K128/ring12/producer32 단일 B7. 저장은 입력 affine BF16 x_n + 원본 BF16 tri + 출력 LN FP32 mean/rstd.</p><p>3case의 forward는 정확히 일치, 모든 경로 graph/eager bit-exact. 최신의 입력 LN gradient만 기존 한도초과. profiler에서 b1_fused·b7_joint 각각1회, 원래 선택 cubin SHA 일치 확인.</p><p>생산dispatch 및 기존 개발 진입점은 변경하지 않았습니다. 정확도 문제가 남은 후보입니다.</p></section><p><a href="README.md">측정 조건·실패 수치·구성 근거</a> · <a href="results.json">원본</a> · <a href="../../TRIMUL_STATUS.html">전체 현황판</a></p></html>'''%(new/1000,pct,''.join(bars),old,t['full']['regression_1185'],tr)
(R/'index.html').write_text(page.replace('最新','최신'))
print('REPORT',new,pct,speed,'diagnostic',failed)
