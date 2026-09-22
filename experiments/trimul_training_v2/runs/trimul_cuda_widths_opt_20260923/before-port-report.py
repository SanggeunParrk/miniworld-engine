from pathlib import Path
import json,html,hashlib,re
R=Path(__file__).resolve().parent;root=R.parents[1]
selection=json.loads((R/'selection.json').read_text());rows=[];training=[];summaries=[]
for D in (64,128,256,384,512):
 for L in (384,768):
  cfg=(dict(D=D,L=L,k1=[2,64,8,2,1],k3=[2,64,4,1],input_ln='fused',emit_xn=False,evidence='native-D128-nosave.json',tuning='Existing D128 configs, no retuning') if D==128 else selection[f'{D}-{L}']);native=json.loads((R/cfg['evidence']).read_text());nr=native['results'][str(L)]
  if D in (384,512):
   comp=json.loads((R/f'ln-compare-D{D}.json').read_text());v=comp['results'][str(L)]['times'];old=v['triton']['median_us'];new=v['separate_ln' if D==512 else 'fused_ln']['median_us'];job=comp['job']
  else:old=nr['times']['triton'];new=nr['times']['native'];job=native['job']
  rows.append((D,L,old/1000,new/1000,old/new));b=json.loads((R/f'bench-D{D}-L{L}.json').read_text());assert b['complete'] and not b['failures']
  times=b['times'];tf=times['training_forward']['triton']['median_us'];total=times['training']['triton']['median_us'];training.append((D,L,tf/1000,total/1000,times['training']['pytorch']['median_us']/1000,times['training']['cueq']['median_us']/1000))
  win=next(v for v in nr['k1'] if v['config']==nr['selected_k1']);log=win['ptxas'];spill=re.findall(r'(\d+) bytes spill (?:stores|loads)',log)
  summaries.append(dict(D=D,L=L,inference_before_us=old,inference_after_us=new,speedup=old/new,job=job,training_job=b['job'],config=cfg,k1_spill_bytes=spill,k1_smem=None,forward_relative_l2=nr['relative_l2']))
current_training=[]
for D,L,tf,total,pt,cq in training:
 if D==128:
  evidence=json.loads((R/f'latest-D128-L{L}.json').read_text());assert evidence['complete'] and all(all(v['valid'] for v in c.values()) for c in evidence['checks'].values())
  old=evidence['times']['training']['triton']['median_us']/1000;new=evidence['times']['training']['cuda']['median_us']/1000;fwd=evidence['times']['training_forward']['cuda']['median_us']/1000
  route='최신 CUDA B1 + 단일 B7' if L==384 else '최신 CUDA B1 + 분리 CUDA B7';job=evidence['job']
 else:old=total;new=total;fwd=tf;route='기존 Triton · 최신 CUDA 미포팅';job=None
 current_training.append(dict(D=D,L=L,old_ms=old,current_ms=new,forward_ms=fwd,speedup=old/new,route=route,job=job))
ct='\n'.join(f"| {v['D']} | {v['L']} | {v['old_ms']:.3f} | {v['current_ms']:.3f} | {v['speedup']:.2f}× | {v['route']} |" for v in current_training)
cth=''.join(f"<tr><td>{v['D']}</td><td>{v['L']}</td><td>{v['old_ms']:.3f}</td><td>{v['current_ms']:.3f}</td><td>{v['speedup']:.2f}×</td><td>{v['route']}</td></tr>" for v in current_training)
summary=dict(scope='B1, bidirectional TriMul, pair width D, hidden D per direction / concatenated 2D',inference=summaries,training=[dict(D=d,L=l,forward_ms=f,total_ms=t,pytorch_compile_total_ms=p,cueq_total_ms=c) for d,l,f,t,p,c in training],current_training=current_training,training_route='Current comparison: D128 latest CUDA B1+B7; other widths existing Triton, CUDA not ported',tuning='Triton miss cap24; native enumerated feasible BI/BJ, weight-ring slots, SKCH, occupancy; not exhaustive global tuning')
(R/'summary.json').write_text(json.dumps(summary,indent=2))
it='\n'.join(f'| {d} | {l} | {o:.3f} | {n:.3f} | {s:.2f}× |' for d,l,o,n,s in rows)
tt='\n'.join(f'| {d} | {l} | {f:.3f} | {t:.3f} | {p:.3f} | {c:.3f} |' for d,l,f,t,p,c in training)
md=f'''# 양방향 TriMul 폭별 실험 · 2026-09-22

D64/256/384/512 폭별 실험에 D128을 같은 조건으로 재측정해 추가했다. D128 추론은 기존 Anthropic K1/K3 설정을 유지했으며 재튜닝하지 않았다. **D는 입력 pair 폭이며, outgoing/incoming hidden도 각각 D다. 합친 hidden과 출력 LN 폭은 2D.** B1, L384/768, BF16, mask 및 residual 포함. 학습 dropout25%, 추론 dropout0. 모두 실제 GPU 실행 결과다.

## 추론: 폭별 CUDA 후보

기준은 같은 실행의 기존 Triton/cuBLAS 경로. 타이밍은 static compile + 수동 CUDA graph, 가중치 packing 포함. optimizer, dropout RNG, compile 및 CPU dispatch 제외. 각 행은 같은 GPU 프로세스의 교대 측정이다. 폭/길이 사이에는 잡과 클럭이 다르다.

| D | L | Triton ms | 선택 후보 ms | 배속 |
|---:|---:|---:|---:|---:|
{it}

| D | 입력 LN / K1 | 출력 | 선택 근거 |
|---|---|---|---|
|64|Anthropic 파생 fused K1, 사용하지 않는 x_n 저장 제거|Anthropic K3|두 길이 약1.5배, spill0|
|128|기존 Anthropic K1 설정, x_n 저장 없음|Anthropic K3|기존 설정 그대로 재측정; 재튜닝 없음|
|256|Anthropic 파생 fused K1, x_n 저장|기존 Triton LN/projection/gate|최소 K3 shared memory246.25KiB > H100227KiB, spill0|
|384|Anthropic 파생 fused K1, x_n 저장|기존 Triton LN/projection/gate|LN 분리는 전체0.3–0.5% 차이뿐. 단순한 융합형 유지; K1 spill store/load8B|
|512|별도 Triton LN (torch.compile) → Anthropic 파생 K1|기존 Triton LN/projection/gate|1CTA/SM 허용. LN 분리로 K1 spill store744/load976B →0; 전체 약2–3% 추가 단축|

Native K1은 Anthropic v5 body를 계승한다. upstream source는 그대로 두고 복사본의 MINB(최소 CTA/SM)만 컨피그로 바꿨다. D512는 기존 64-token tile의 최소2CTA/SM 제약으로 shared memory 예산을 만족할 수 없었으며, 1CTA/SM을 허용한 후보를 실행·검증했다. D512 LN 분리에는 이미 존재하는 LNM=0 경로를 사용한다.

튜닝축: BI/BJ, ring slots2/4/6/8, SKCH1/2/4/6/8 중 D를 나누는 값, 최소 CTA1/2. 자원 한도를 먼저 검사한 뒤 모든 남은 조합을 실행했다. K3(D64)는 slots4/6/8, accumulator1/2. [선택 config·cubin SHA](selection.json).

**D256 이상은 전체 CUDA 단일 경로가 아닌 혼합 경로다.** 출력 K3는 두 방향을 합친 H=2D 기준 최소 shared memory가 D256/384/512에서246.25/361.25/476.25KiB여서, 기존 구조의 compile-time 제약에 걸린다. 새 K3를 완성했다고 주장하지 않는다.

## 학습: 최신 연결 결과와 기존 대비 배속

D128은 최신 cache-policy B1–B4를 연결했다. L384 B7–B12는 LN gradient 수정 단일 커널, L768은 기존 분리형 CUDA 커널이다. L384/768 각각 기존 Triton과 같은 GPU 프로세스에서 교대로 측정했다. dropout25%, residual, 매 실행 forward와 저장, 실시간 가중치 packing, 11개 gradient 포함. profiler에서 두 길이의 `b1_fused`와 L384의 단일 `b7_joint` 실행을 확인했다. 두 길이 모두 이전 CUDA 대비 엄격한 수치 검증, Triton 교차 구현 검증, 입력·가중치 변경 후 graph 검증을 통과했다.

| D | L | 기존 Triton 전체 ms | 현재 전체 ms | 기존 대비 배속 | 실제 학습 경로 |
|---:|---:|---:|---:|---:|---|
{ct}

L768 새 단일 B7은 dWL relative L2 0.0519%로 엄격한 기준 0.0500%를 초과해 선택하지 않았다. 허용 오차를 완화하지 않았으며 [실패 증거](latest-D128-L768-single-rejected.json)를 남겼다.

다른 D의 1.00×는 최신 CUDA가 같은 속도라는 뜻이 아니라, **아직 포팅하지 않아 기존 Triton을 유지**한다는 뜻이다. D128 기존 시간은 아래의 과거 기준선 수치 대신 최신 CUDA와 같은 실행에서 재측정한 값을 썼다.

## 참고: 기존 경로만의 기준 측정

아래 Triton은 **기존 폭 일반화 경로**다. D128 행 역시 동일한 기존 Triton 기준선이며, 최신 CUDA 결과는 위의 별도 학습 비교표에 있다. 다른 폭으로 해당 최신 커널을 포팅한 결과도 아니다. 또한 위 추론 후보는 training autograd를 제공하지 않는다.

| D | L | Triton train fwd ms | Triton fwd+bwd ms | PyTorch compile fwd+bwd ms | cuEq fwd+bwd ms |
|---:|---:|---:|---:|---:|---:|
{tt}

cuEq는 동일한 양방향 수식의 primitives 조합이다. 공개 단방향 TMU 두 번 호출과 다르다. PyTorch도 동일 수식의 reference를 static compile한 결과이며 일반적인 모든 PyTorch 구현의 최선 성능을 뜻하지 않는다. 이전 D128의 다른 실행 수치와 배속을 계산하지 않는다.

Triton의 신규 shape cache miss는 24개 후보를 탐색했다. 전체 config 공간을 완전히 튜닝한 상한 성능은 아니다. 원본 로그에 해당 cache miss 기록을 남겼다.

## D512·L768 주소 오류 수정

기존 `_input_dual_bwd_kernel`의 `rk`가 int32여서 `k*fs1`이 행의 int64 주소에 더해지기 **전에** overflow했다. KP4096, M589824에서 최대 열 offset은2,415,329,280 elements로2^31-1을 넘는다. `rk`를 int64로 바꿔 곱셈부터64비트로 계산하도록 수정했다.

- 원본 memcheck: out-of-bounds read 재현, 종료99. 수정본:0errors.
- 수정 후 D512/L768 전체 추론·학습 forward·모든 gradient 검증 통과.
- `miniworld-engine-k1k3`, `miniworld-engine-tbwd`의 해당 커널에 반영. [반영 파일·SHA](installed-offset-fix.json).
- 실험용 engine snapshot은 덮어쓰지 않았다. 벤치에서는 동일한 수정 커널을 명시적으로 연결했다. [원본/수정 검사](offset-sanitizer.json).

## 검증·적용 범위

모든 기존 경로는 forward relativeL2≤.005, gradient≤.01의 BF16 교차 구현 기준을 통과. 입력 변경 후 graph/eager는 일반 출력 bit-exact, 기존 atomic LN gradient는5e-6 기준을 유지했다. 새 추론 후보도 전체 출력≤.005, 변경된 입력/가중치가 graph에 반영되는지 확인했다. 기존 D64/256/384/512 native 선택 후보는 두 길이에서 mask=0, memcheck/racecheck를 모두 통과했다. 추가 D128은 전체 forward 및 graph 입력 변경 검증을 수행했으며 이번 추가 측정에서 sanitizer는 재실행하지 않았다. 해당 sanitizer JSON과 로그에 결과를 기록했다.

실험용 [selected.py](selected.py)의 `Inference`에 D64/256/384/512의 폭/길이별 선택을 모았다. D128 표 행은 별도 `native_d128.py`로 측정했다. `torch.no_grad()`에서 사용하며 input/weight 내용 변경을 지원한다. graph replay 중 tensor storage 교체는 지원하지 않는다. **엔진 auto-dispatch에 추론 후보를 승격하거나 push하지 않았다.** 엔진에 적용한 변경은 주소 overflow 수정이다.

다음 우선순위: D512의 streamed-K K1 및 출력 K3 새 설계, 넓은 폭의 training backward 설계. D64도 최신 B1/B7을 새 폭에 맞게 포팅해야 한다. 지금의 학습 성능표만으로 D128 통합 알고리즘의 폭별 성능을 판단할 수 없다.

## 증거와 재현

`latest.sbatch`: D128 최신 CUDA와 기존 Triton 학습 비교. `d128.sbatch`: D128 기준선 두 길이 및 기존 CUDA 추론 재측정. `bench.sbatch`: 나머지 4폭×2길이, PyTorch/Triton/cuEq. `native.sbatch`, `native-nosave.sbatch`, `native-separate.sbatch`: Native 튜닝. `compare-ln.sbatch`: D384/512 같은 실행 LN 융합/분리 비교. `offset.sbatch`: 주소 오류 재현/수정. `native-sanitize*.sbatch`, `check-selected.sbatch`: 메모리·race·최종 adapter 검증. H100 node01, 공통 env `runs/anthropic_adoption_20260919/env.sh`.

[요약 JSON](summary.json) · [구조/전체 표 HTML](index.html) · [최종 선택](selection.json). NCU 기반 roofline/SoL은 이번 폭별 실험에서 측정하지 않았고, ptxas resource 정보 및 torch profiler kernel breakdown을 기록했다.
'''
(R/'README.md').write_text(md)
itr=''.join(f'<tr><td>{d}</td><td>{l}</td><td>{o:.3f}</td><td>{n:.3f}</td><td>{s:.2f}×</td></tr>' for d,l,o,n,s in rows)
ttr=''.join(f'<tr><td>{d}</td><td>{l}</td><td>{f:.3f}</td><td>{t:.3f}</td><td>{p:.3f}</td><td>{c:.3f}</td></tr>' for d,l,f,t,p,c in training)
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>양방향 TriMul · D별 실험</title><style>body{{font:16px/1.65 system-ui;background:#101c2b;color:#e9eff7;max-width:1200px;margin:24px auto;padding:0 16px}}section{{background:#1b2d43;padding:20px;border-radius:12px;margin:20px 0}}h1,h2{{line-height:1.3}}a{{color:#7ed6ff}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #40536b}}.scroll{{overflow:auto}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.card{{border:1px solid #4b8c80;border-radius:9px;padding:16px}}.arrow{{color:#7cd3b1}}.note{{background:#443b23;padding:14px;border-radius:8px}}code{{color:#9aebcf}}@media(max-width:650px){{.cards{{grid-template-columns:1fr}}}}</style>
<h1>양방향 TriMul · D64 / 128 / 256 / 384 / 512</h1><p>2026-09-22 · H100 · B1 · L384/768 · BF16 · D=pair폭=방향별hidden, 합친hidden=2D</p>
<p class="note">추론 CUDA 후보와 기존 학습 경로를 구분합니다. D128 최신 통합 backward를 넓은 폭으로 이식한 결과는 아직 없습니다.</p>
<section><h2>추론: 같은 실행의 Triton 대비</h2><div class="scroll"><table><tr><th>D</th><th>L</th><th>Triton ms</th><th>선택 후보 ms</th><th>배속</th></tr>{itr}</table></div><p>mask·residual·실시간 가중치 packing 포함. static compile + CUDA graph. D256 이상은 Anthropic 파생 입력 CUDA + 기존 Triton 출력 혼합 경로입니다.</p></section>
<section><h2>선택한 연산 배선 · HBM 중심</h2><div class="cards">
<div class="card"><h3>D64 · D128</h3><p>x <span class="arrow">→</span> CUDA K1 [LN+projection+gate] <span class="arrow">→</span> left/right</p><p>left/right <span class="arrow">→</span> cuBLAS outgoing/incoming <span class="arrow">→</span> tri</p><p>tri+x <span class="arrow">→</span> CUDA K3 [LN+projection+gate+residual] <span class="arrow">→</span> y</p><b>x_n 저장 없음 · D128은 기존 설정 유지</b></div>
<div class="card"><h3>D256 · D384</h3><p>x <span class="arrow">→</span> CUDA K1 [LN+projection+gate] <span class="arrow">→</span> left/right + x_n</p><p>left/right <span class="arrow">→</span> cuBLAS <span class="arrow">→</span> tri</p><p>tri+x_n+x <span class="arrow">→</span> Triton [LN_out+projection+gate+residual] <span class="arrow">→</span> y</p><b>x_n은 출력 gate가 읽음 · D384 spill8B</b></div>
<div class="card"><h3>D512</h3><p>x <span class="arrow">→</span> 별도 Triton LN <span class="arrow">→</span> x_n</p><p>x_n <span class="arrow">→</span> CUDA K1 [projection+gate] <span class="arrow">→</span> left/right <span class="arrow">→</span> cuBLAS <span class="arrow">→</span> tri</p><p>tri+x_n+x <span class="arrow">→</span> Triton 출력 <span class="arrow">→</span> y</p><b>1CTA/SM 허용 · LN 분리로 spill0 · 융합형보다 추가2–3% 단축</b></div>
<div class="card"><h3>학습 · 현재 연결</h3><p>D128: Anthropic 파생 forward → 최신 CUDA B1–B4 → cuBLAS → CUDA B7–B12 (L384 단일 / L768 분리)</p><p>기타 D: 기존 Triton LN/front/저장 <span class="arrow">→</span> cuBLAS contraction <span class="arrow">→</span> Triton 출력</p><p>기존 Triton + cuBLAS backward · dropout25% · 모든gradient</p><b>D128 최신 B1–B4 / B7–B12의 width 포팅은 미완료</b></div></div></section>
<section><h2>학습 · 최신 CUDA 연결 및 기존 대비 배속</h2><div class="scroll"><table><tr><th>D</th><th>L</th><th>기존 Triton ms</th><th>현재 ms</th><th>배속</th><th>실제 경로</th></tr>{cth}</table></div><p>D128: 최신 cache-policy B1–B4. L384는 수정 단일 B7, L768은 기존 분리형 CUDA B7. profiler로 실행을 확인했습니다. L768 단일 B7 후보는 dWL 오차0.0519%가 기준0.0500%를 초과해 제외했습니다. 두 길이 모두 엄격한 이전 CUDA 대비 검증, Triton 비교, graph 입력·가중치 변경 검증 통과. dropout25%, residual, 전체 gradient와 가중치 packing 포함.</p><p>D128 기준선은 같은 실행에서 재측정했습니다. 나머지 D의1.00×는 최신 CUDA 미포팅으로 기존 Triton을 유지한다는 뜻입니다.</p></section>
<section><h2>참고 · 기존 경로만의 기준 측정</h2><div class="scroll"><table><tr><th>D</th><th>L</th><th>Triton fwd ms</th><th>Triton 전체 ms</th><th>PyTorch compile 전체 ms</th><th>cuEq 전체 ms</th></tr>{ttr}</table></div><p>D128도 동일한 기존 Triton 기준선입니다. 최신 D128 CUDA 결과는 위의 학습 비교표에 있습니다. cuEq는 같은 양방향 수식의 primitives 구성. Triton 신규 shape는 heuristic24개 탐색으로, 전체 config 공간 튜닝 완료를 뜻하지 않습니다.</p></section>
<section><h2>발견·수정한 실제 오류</h2><p>D512/L768 입력 gradient의 <code>k * stride</code>가32비트에서 overflow했습니다. 최대offset2,415,329,280 elements. 곱셈 전에64비트로 승격했습니다.</p><p>원본memcheck OOB 재현 → 수정본0errors → 전체학습·gradient 통과. 두 엔진 개발 작업 트리에 반영했습니다.</p><a href="offset-sanitizer.json">원본/수정 검사</a> · <a href="installed-offset-fix.json">반영파일·SHA</a></section>
<section><h2>폭을 늘렸을 때의 제약</h2><p>합친hidden=2D인 기존 K3의 최소shared memory: D256 246.25KiB, D384 361.25KiB, D512 476.25KiB. H100 한도227KiB를 넘어 기존 구조로는 지원하지 못합니다. D512 입력 K1도 기존최소2CTA/SM 제약을1CTA로 완화해야 했습니다.</p><p>D384 LN 분리는 전체0.3–0.5% 차이여서 융합 유지. D512는 spill을 없애고 약2–3% 단축해 분리 채택. 다음 후보는 streamed-K 입력/출력 타일과 넓은 폭의 training backward입니다.</p></section>
<section><h2>증거·적용 범위</h2><p>D64/256/384/512 추론 후보는 <code>selected.py:Inference</code>의 실험 adapter이며 D128은 <code>native_d128.py</code>로 기존 설정을 재측정했습니다. production auto-dispatch 변경이나 push는 하지 않았습니다. 계산 코드와 벤치, shape별config, ptxas 로그, cubin SHA를 남겼습니다.</p><p><a href="selection.json">선택config·SHA</a> · <a href="summary.json">전체수치</a> · <a href="README.md">검증·재현</a> · <a href="selected.py">추론adapter</a></p><p>D64/256/384/512 native 후보의 mask0·memcheck/racecheck 결과는 <code>native-sanitizer-D*.json</code>. 최종adapter검증은 <code>selected-check-D*.json</code>. 추가 D128은 전체 출력·graph 입력변경 검증을 통과했으며 이번에는 sanitizer를 재실행하지 않았습니다. NCU roofline/SoL은 이번폭별실험에서 측정하지 않았습니다.</p></section><p>Anthropic v5 구현을 계승한 파생커널이며 기존발표에 대한성능우위 주장이 아닙니다. <a href="../../TRIMUL_STATUS.html">전체현황</a></p></html>'''
(R/'index.html').write_text(page)
section=f'<section id="trimul-widths"><h2>추가 실험 · TriMul D64/128/256/384/512</h2><p>H100, 방향별 hidden=D, L384/768. 폭별 추론 후보와 기존 대비 배속을 기록했습니다. D128 학습은 최신 CUDA B1/B7을 연결해 두 길이에서 기존 Triton과 재측정했고, 다른 D의 최신 CUDA는 미포팅입니다. D512/L768 학습의 32비트 주소 overflow를 수정했습니다.</p><p><a href="runs/trimul_widths_20260922/index.html">폭별 성능표·연산배선·제약·검증</a></p><p>최신 D128 B1/B7의 넓은 폭 포팅과 production 추론 승격은 아직 하지 않았습니다.</p></section>'
p=root/'TRIMUL_STATUS.html';s=p.read_text();s=re.sub(r'<section id="trimul-widths">.*?</section>','',s,flags=re.S);s=s.replace('<section id="transition-bwd-upgrade">',section+'<section id="transition-bwd-upgrade">',1);p.write_text(s)
p=root/'TRIMUL_STATUS.md';s=p.read_text();mark='## TriMul D별 실험 · 2026-09-22';s=s.split(mark)[0].rstrip();p.write_text(s+'\n\n'+mark+'\n\n[폭별 성능표·구조·검증](runs/trimul_widths_20260922/index.html). D64/128/256/384/512, L384/768. D128 최신 CUDA B1/B7 학습을 연결하고 기존 Triton 대비 배속을 기록했다. 다른 D는 최신 CUDA 미포팅으로 기존 경로 유지. D512/L768의 주소 overflow를 수정했다. D128 최신 통합 backward의 폭별 포팅은 아직 하지 않았다.\n')
print('REPORT_DONE')
