from pathlib import Path
import json,html,csv,hashlib
R=Path(__file__).resolve().parent
v=json.loads((R/'validation.json').read_text());b=json.loads((R/'block.json').read_text());t=json.loads((R/'tune.json').read_text());san=json.loads((R/'sanitizer.json').read_text())
assert v['complete'] and b['complete'] and all(x['returncode']==0 for x in san)
rows=[]
for L,x in v['times'].items():
 a=x['baseline']['median_us'];z=x['selected']['median_us'];rows.append((f'Transition backward · L{L}',a,z,(1-z/a)*100))
a=b['times']['baseline']['median_us'];z=b['times']['selected']['median_us'];rows.append(('MiniPairformer fwd+bwd · L384',a,z,(1-z/a)*100))
ln=max(e['relative_l2'] for c in v['cases'].values() for n,e in c['errors'].items() if n in ('dgamma','dbeta'))
exact=all(e['bit_exact'] for c in v['cases'].values() for n,e in c['errors'].items() if n not in ('dgamma','dbeta'));assert exact
ncu={}
for name in ('baseline','parallel_transpose'):
 ncu[name]=[{k:r[k] for k in ('Kernel Name','gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed','smsp__warps_eligible.avg.per_cycle_active')} for r in csv.DictReader((R/('ncu-'+name+'.csv')).open()) if r.get('Kernel Name')]
summary=dict(comparison='Previous hand-CUDA Transition vs same code with parallel LN reducer and tiled dWs store',rows=[dict(scope=n,before_us=a,after_us=z,time_reduction_pct=p) for n,a,z,p in rows],ln_relative_l2_max=ln,non_ln_bit_exact=exact,ncu=ncu,validation_jobs=[v['job'],b['job'],san[0]['job']],installation=json.loads((R/'installed.json').read_text()))
(R/'summary.json').write_text(json.dumps(summary,indent=2))
table='\n'.join(f'| {n} | {a:.3f} | {z:.3f} | {p:.2f}% |' for n,a,z,p in rows)
readme=f'''# Transition backward 부분합 최적화 · 2026-09-22

기준은 **직전 hand-CUDA Transition**이다. Anthropic/Triton과의 새 비교가 아니다. Anthropic v5의 TMA/WGMMA primitives를 계승한 학습 구현의 부분합 처리를 개선했다.

## 결과

H100 SXM, BF16, D128/H512. 같은 프로세스에서 순서를 번갈아 CUDA graph 실행, 5×250 표본의 중앙값. 전체 블록은 정확도를 수정한 동일 TriMul + Transition, L384, dropout25%, 두 residual과 모든 파라미터 gradient 포함. RNG 생성·optimizer·CPU dispatch·compile 제외.

| 범위 | 이전 µs | 변경 µs | 시간 감소 |
|---|---:|---:|---:|
{table}

Standalone job{v['job']}; block job{b['job']}. 표의 행들은 서로 다른 실행의 측정이며 단독 시간 합으로 전체 시간을 산출하지 않는다.

## 채택

- main backward 본체는 그대로 유지: 64 dW CTA + 68 dX CTA, 255 registers/thread, spill 0.
- LN gradient: 한 CTA에서 채널당 544개 부분합을 순차 합산하던 작업을 4 CTA × 8개 부분합 그룹으로 병렬화. 고정된 합산 순서, atomic 없음.
- dWs: 16×16 tile을 shared memory에서 전치하여 출력의 인접 주소에 저장. dW의 합산 순서와 BF16 반올림 지점 유지.
- reducer launch grid: 769 → 772 CTA. forward, 저장 정책, backward 본체의 fusion과 입력·출력은 동일.
- 적용: `{summary['installation']['file']}`. 이 Transition 개발 작업 트리의 D128/H512 경로에 반영. 중앙 main 레포로 병합/push한 것은 아니다.

## 검증

- L384/768 각 12조건, 총24조건: 실제 forward saves에서 계산. 작은 입력, gamma=0 채널, x=0, dy=0, 가중치 변경 포함.
- dX와 세 dW 모두 기존 native와 bit-exact. LN 최대 상대L2 **{ln:.3e}**, 기존 제한5e-6 유지.
- 전체 블록8조건 통과. graph/eager는 모든 gradient에서 bit-exact.
- 두 길이 memcheck0errors, racecheck0hazards (job{san[0]['job']}). 필터는 main backward와 reducer를 모두 포함.
- PyTorch/cuEq와의 새로운 비교는 이번 범위가 아니다. 기존 CUDA 연산 계약을 유지하는 변경을 검증했다.

## NCU

NCU는 cache-control none / clock-control none, kernel replay14passes. latency 벤치는 별도 수행.

| 커널 | 이전 µs | 변경 µs |
|---|---:|---:|
| 본체 |390.624|392.288|
| 부분합 축약 |17.920|5.568|

본체 Tensor Core elapsed utilization 약56%. 이는 전체 알고리즘 SoL 비율이 아니다. HBM read/write 합 약165MB, 해당 replay 약391µs. 본체의 PC sampling은 barrier16.5%, wait20.5%, warpgroup-arrive5.8%; 이 비율은 표본 비율이며 시간 절감량이 아니다. 255register와 1CTA/SM, eligible warps/scheduler 약0.60인 상태여서 본체의 스케줄·동기화 개선이 다음 과제다. SoL90 미달 여부를 정량 확정할 calibrated bound는 이번에 산출하지 않았다.

## 제외한 후보

- per-warp global 부분합 float2 벡터화: 뚜렷한 이득 없음.
- CTA 내 부분합 누적 후 한 번 저장: L384 이득은 작고 L768에서는 추가 barrier 비용 때문에 선택안보다 느림.
- DW_REPL7/9: L384 약485/465µs, 선택안426µs보다 느림. L768에서도 느림.

## 재현과 증거

`prepare.py`, `next_candidates.py`, `tune.sbatch`는 직접 cubin 후보 비교. `integrate.py parallel_transpose 8`로 선택 native snapshot 생성 후 `validate.sbatch`, `block.sbatch`, `sanitize.sbatch`, `verify_install.sbatch` 실행. 기본 env는 `runs/anthropic_adoption_20260919/env.sh`.

- [정확도·standalone시간](validation.json), [전체블록](block.json), [7개 후보×2길이](tune.json)
- [NCU이전](ncu-baseline.csv), [NCU선택안](ncu-parallel_transpose.csv), [sanitizer](sanitizer.json)
- [선택소스](selected/transition_fused_bwd_sm90a_kernel.cu), [설치SHA](installed.json)
- [설치경로검증](installed_validation.json), [구조·결과HTML](index.html)
'''
(R/'README.md').write_text(readme)
trs=''.join(f'<tr><td>{n}</td><td>{a:.3f}</td><td>{z:.3f}</td><td>{p:.2f}%</td></tr>' for n,a,z,p in rows)
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Transition backward · 부분합 최적화</title><style>
body{{font:16px/1.65 system-ui;background:#101b2b;color:#e8eff8;max-width:1100px;margin:24px auto;padding:0 18px}}section{{background:#1a2b40;padding:20px;margin:20px 0;border-radius:12px}}h1,h2{{line-height:1.3}}a{{color:#85d4ff}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #405269}}.scroll{{overflow:auto}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.box{{border:1px solid #5d829f;padding:14px;border-radius:8px}}.new{{border-color:#52c7a5}}code{{color:#b9eddc}}small{{color:#b3c4d8}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>Transition backward 부분합 최적화</h1><p>2026-09-22 · H100 · D128/H512 · 이전 hand-CUDA 대비</p>
<section><h2>실제 연결 결과</h2><div class="scroll"><table><tr><th>범위</th><th>이전 µs</th><th>변경 µs</th><th>시간 감소</th></tr>{trs}</table></div><p>동일 프로세스에서 5×250회 교차 CUDA graph 측정. 전체 블록은 같은 정확도 수정 TriMul, dropout25%, 모든 gradient·residual 포함. optimizer·RNG 생성·compile 제외. standalone과 block은 별도 잡.</p></section>
<section><h2>HBM 입력 → 계산 → HBM 출력</h2><div class="box"><b>유지: backward 본체</b><p>입력: dy, x, 저장된 x_n/rstd/mean×rstd, gamma, Wa/Wb/Ws</p><p>64 dW CTA: dh/a/b 재계산 → SwiGLU 미분 → dWa/dWb/dWs 부분합<br>68 dX CTA: dh/a/b 재계산 → dXn → LN 미분 + residual</p><p>출력: dX, dW 부분합 [64,3,64,128] FP32, LN 부분합 [68,8,256] FP32</p></div><p style="text-align:center">↓ HBM 부분합 → reducer → 최종 dWa/dWb/dWs + dγ/dβ</p><div class="grid"><div class="box"><b>이전 reducer · 769 CTA</b><p>LN: 채널당 544개 항 직렬 합산<br>dWs: 출력 주소가 stride512로 떨어짐</p><strong>NCU 17.920µs</strong></div><div class="box new"><b>선택 reducer · 772 CTA</b><p>LN: 4 CTA, 8개 그룹이 병렬 합산<br>dWs: 16×16 shared 전치 → 인접 주소 저장<br>고정 합산 순서·atomic 없음</p><strong>NCU 5.568µs</strong></div></div><p>새로운 큰 중간 텐서나 커널 호출을 추가하지 않았다. 이전과 동일하게 본체1회 + 축약1회.</p></section>
<section><h2>정확도 및 적용</h2><p>두 길이24조건 + 전체 블록8조건 통과. dX·dW는 bit-exact, LN 최대 상대L2 {ln:.3e} (한도5e-6). Graph/eager bit-exact, memcheck0errors, racecheck0hazards.</p><p>엔진 <code>miniworld-engine-tbwd</code> 작업 트리에 backward 파일만 반영했다. main 레포 병합/push는 하지 않았다.</p><p><a href="installed.json">설치 SHA</a> · <a href="installed_validation.json">설치경로 검증</a> · <a href="validation.json">정확도</a> · <a href="block.json">블록 시간</a> · <a href="sanitizer.json">sanitizer</a></p></section>
<section><h2>왜 전체 이득은 작은가?</h2><p>이번에 줄인 것은 약18µs의 마지막 축약이다. 약391µs의 본체는 거의 그대로이며 NCU Tensor Core 사용률은 약56%다. 이 사용률을 알고리즘 SoL 비율로 해석하면 안 된다.</p><p>본체는 255register/thread, 1CTA/SM, eligible warps/scheduler 약0.60. PC sampling에서 barrier16.5%, wait20.5%, warpgroup-arrive5.8%. 다음에는 본체의 동기화·명령 발행 간격을 줄여야 한다.</p><p>CTA 내 LN 누적과 dX/dW 배분 변경도 시험했다. 전자는 L768에서 손해, 후자는 두 길이 모두 느려 제외했다.</p><p><a href="tune.json">모든 후보 결과</a> · <a href="ncu-baseline.csv">이전 NCU</a> · <a href="ncu-parallel_transpose.csv">선택 NCU</a></p></section>
<p>Anthropic v5 TMA/WGMMA primitives를 계승한 학습 구현. 이번 표는 이전 CUDA와의 비교이며 Anthropic/Triton 대비 새로운 성능 주장에 해당하지 않는다.</p><p><a href="README.md">재현·자세한 설명</a> · <a href="../../TRIMUL_STATUS.html">전체 현황</a></p></html>'''
(R/'index.html').write_text(page)
section=f'<section id="transition-bwd-upgrade"><h2>추가 개선 · Transition backward</h2><p>2026-09-22 · 기존 hand-CUDA 대비. LN 부분합 병렬화 + dWs 연속 저장.</p><div class="scroll"><table><tr><th>범위</th><th>이전 µs</th><th>변경 µs</th><th>시간 감소</th></tr>{trs}</table></div><p>엄격한 gradient 검증24조건 + 블록8조건, memcheck/racecheck 통과. Transition 개발 작업 트리에 반영. <a href="runs/transition_bwd_upgrade_20260922/index.html">구조·NCU·전체 결과</a></p></section>'
p=R.parents[1]/'TRIMUL_STATUS.html';s=p.read_text()
import re
s=re.sub(r'<section id="transition-bwd-upgrade">.*?</section>','',s,flags=re.S);s=s.replace('<section id="minipairformer-baselines">',section+'<section id="minipairformer-baselines">',1);p.write_text(s)
p=R.parents[1]/'TRIMUL_STATUS.md';s=p.read_text();start='## Transition backward 추가 개선 · 2026-09-22';s=s.split(start)[0].rstrip();p.write_text(s+'\n\n'+start+'\n\n| 범위 | 이전 µs | 변경 µs | 시간 감소 |\n|---|---:|---:|---:|\n'+table+'\n\n기존 hand-CUDA 대비. 엄격 검증·memcheck/racecheck 통과, Transition 개발 작업 트리에 반영. [구조·결과](runs/transition_bwd_upgrade_20260922/index.html).\n')
print('REPORT_DONE')
