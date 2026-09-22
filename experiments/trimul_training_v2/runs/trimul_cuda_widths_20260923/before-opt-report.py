from pathlib import Path
import json,hashlib,re,html
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'trimul_widths_20260922'
previous=json.loads((R/'before-port-summary.json').read_text());rows=[];checks=[]
for D in (64,128,256,384,512):
 for L in (384,768):
  a=json.loads((R/f'autograd-D{D}-L{L}.json').read_text());assert a['complete'] and a['version_check'];checks.append(a)
  if D==128:
   p=OLD/f'latest-D128-L{L}.json';v=json.loads(p.read_text());scope='training_forward';route='B1 latest + B7 single' if L==384 else 'B1 latest + B7 split'
  else:
   p=R/f'check-D{D}-L{L}.json';v=json.loads(p.read_text());scope='forward';route='Width CUDA B1 + B7'
   assert v['trace'].get('width_b1')==v['trace'].get('width_b7')==1
   for name,e in v['checks']['graph_mutation'].items():assert e['relative_l2']<=(5e-6 if name.startswith(('dgamma','dbeta')) else 0)
  assert v['complete'];t=v['times']['training'];f=v['times'][scope];old=t['triton']['median_us']/1000;new=t['cuda']['median_us']/1000
  rows.append(dict(D=D,L=L,triton_ms=old,cuda_ms=new,speedup=old/new,triton_forward_ms=f['triton']['median_us']/1000,cuda_forward_ms=f['cuda']['median_us']/1000,forward_speedup=f['triton']['median_us']/f['cuda']['median_us'],route=route,job=v['job'],evidence=str(p.relative_to(ROOT))))
sanitizers={}
for D in (64,256,384,512):
 m=(R/f'memcheck-D{D}.log').read_text();r=(R/f'racecheck-D{D}.log').read_text();assert 'ERROR SUMMARY: 0 errors' in m and '0 hazards displayed (0 errors, 0 warnings)' in r
 sanitizers[D]=dict(memcheck=True,racecheck=True,length=384)
summary=dict(scope='Bidirectional TriMul; B1; BF16; D pair channels; H=2D total; dropout25%; residual; all 11 gradients',training=rows,inference=previous['inference'],autograd_shapes=10,sanitizers=sanitizers,connection='runs/trimul_training_current.py:Training and bidirectional_trimul_cuda',limitations=['Wide ports are slower than Triton and are not performance-complete','Wide kernels retain the recomputation policy and B1/B7 single-launch grouping but use full-size transient global scratch instead of the D128 specialized streaming schedule','Manual CUDA graph timings exclude plan construction, compilation, CPU dispatch, optimizer and dropout RNG','One visible H100 GPU per process, B1, L384/768 only','No automatic production engine dispatch or push'])
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
mt='\n'.join(f"| {v['D']} | {v['L']} | {v['triton_ms']:.3f} | {v['cuda_ms']:.3f} | {v['speedup']:.2f}× |" for v in rows)
ft='\n'.join(f"| {v['D']} | {v['L']} | {v['triton_forward_ms']:.3f} | {v['cuda_forward_ms']:.3f} | {v['forward_speedup']:.2f}× |" for v in rows)
it='\n'.join(f"| {v['D']} | {v['L']} | {v['inference_before_us']/1000:.3f} | {v['inference_after_us']/1000:.3f} | {v['speedup']:.2f}× |" for v in previous['inference'])
md=f'''# 양방향 TriMul · D64–512 CUDA backward 연결

**다섯 폭 모두 현재 개발 학습 진입점에서 CUDA backward로 연결했다.** 이전 표처럼 D128만 CUDA로 측정하고 나머지는 기존 Triton을 현재 결과로 표시하지 않는다. 다만 **D64/256/384/512 CUDA 포트는 아직 Triton보다 느리다.** 연결·정확성 검증과 성능 최적화 완료는 별개다.

## 학습 전체: 같은 실행의 기존 Triton 대비

B1, H100, BF16, hidden은 방향별 D / 합친 폭2D. dropout25%, pair mask, residual, 모든11개gradient, 실시간 weight packing 포함. static compile + 수동 CUDA graph. 각 행의 기존/현재는 같은 GPU 프로세스 교대 측정. optimizer·dropout RNG·compile·초기 plan 생성·CPU dispatch 제외. 1×보다 작으면 CUDA가 느리다.

| D | L | 기존 Triton ms | 현재 CUDA ms | 기존 대비 배속 |
|---:|---:|---:|---:|---:|
{mt}

## 학습 forward

| D | L | 기존 Triton ms | 현재 CUDA ms | 배속 |
|---:|---:|---:|---:|---:|
{ft}

## 추론 forward · 앞선 측정 유지

추론 전용 설정이다. 이번에 다시 측정한 학습 forward와 다른 경로이며 학습의 dropout은25%, 추론은0이다.

| D | L | 기존 Triton ms | 선택 추론 경로 ms | 배속 |
|---:|---:|---:|---:|---:|
{it}

## 실제 연결

- `runs/trimul_training_current.py:Training(a)`가 D로 분기한다. D128은 기존에 검증한 cache-policy B1과 B7, 나머지 네 폭은 새 CUDA 포트다.
- 같은 파일의 `bidirectional_trimul_cuda(...)`는 PyTorch autograd 진입점이다. 모든 D에서 11개 gradient를 반환한다. 폭 미지원 시 예외를 내며 Triton backward로 조용히 대체하지 않는다.
- D128 L384는 수정된 단일 B7, L768은 검증된 분리형 CUDA B7을 유지한다. L768 단일 후보의 dWL 오차0.0519%가 기준0.0500%를 초과한 기록은 이전 실험에 남겨 두었다.
- D64/256/384/512의 backward는 `width_b1` 1회 → cuBLAS contraction backward4회 → `width_b7` 1회. profiler의 실제 커널 이름으로 확인했다.
- 새로운 폭별 커널은 Anthropic의 TMA/WGMMA primitives와 기존 재계산 수식을 차용했다. D128의 저수준 스트리밍 배선을 그대로 복제한 것은 아니다. 큰 폭에서는 shared memory 한도를 지키기 위해 일시적인 전역 작업 공간을 사용한다.
- 입력 projection/gate는 앞서 튜닝한 Anthropic 파생 K1을 사용한다. 입력 x_n은 저장하며 backward가 재사용한다. projection/gate는 backward에서 재계산한다. 두 방향 contraction은 cuBLAS다.

## 넓은 폭의 HBM 흐름

1. Forward: x → K1 → left/right + x_n → cuBLAS → tri → 출력 CUDA → y. D512 입력 LN은 별도 경로다. 출력 CUDA는 내부 phase 사이에서 임시 normalized tri 버퍼를 쓴다.
2. B1 입력: tri, x_n, dy, dropout scale, output weights/affine. LN_out·projection·gate 재계산 → dp/dg → dNorm/dW → LN 미분. 출력: dTri, dGate, dWproj, dWgate, dgamma_out, dbeta_out.
3. cuBLAS: dTri + left/right → dLeft/dRight.
4. B7 입력: x, x_n, dLeft/dRight, dGate, dy, input weights/affine, mask. input projection/gate 재계산 → dp/dg → dx_n와 dW → LN 미분+residual. 출력: dx, 네 input dW, dgamma_in, dbeta_in.

**현재 느린 이유:** 한 CUDA launch 안에 묶었지만 normalized tri, dp/dg, dx_n 등의 중간값을 전역 workspace에 써서 phase 간 전달한다. D128 스트리밍 구현처럼 메모리 왕복을 충분히 줄이지 못했다. 이 상태를 D128과 동등하게 튜닝됐다고 주장하지 않는다. 다음 개선은 전역 작업 공간을 작은 ring 또는 shared memory 타일로 줄이는 것이다.

## 타일·자원

GEMM은 m64n64k16 WGMMA를 사용하며 K는64단위로 이동한다. D64는 1 warp-group/N64, 나머지는2 warp-group/N128이 A 타일을 공유한다. TMA는2stage로 계산과 겹친다. dW split-K는 D64에서128, 나머지에서32. cooperative grid는 드라이버의 실제 occupancy 한도 내에서 선택한다. 이 값들은 현재 검증한 설정이며 넓은 autotune 공간 탐색을 완료한 상태는 아니다.

## 검증

- 실제 public autograd 경로: D64/128/256/384/512 × L384/768, **10개 shape 통과**.
- 출력 relativeL2≤0.005, 각 gradient≤0.01: BF16 교차 구현 기준. 이 수치는 기존 D128 동일 수식 비교의 엄격한5e-6 LN 검증 기준을 대체하지 않는다.
- Forward를 두 번 수행한 후 각 backward의 저장값 독립성, forward 후 in-place 변경 탐지 통과.
- CUDA graph에서 입력·가중치 변경이 반영된다. 새 폭별 경로 graph/eager는 일반 출력·gradient bit-exact, atomic LN parameter gradient relativeL2≤5e-6를 별도 확인했다.
- 새 네 폭 L384에서 compute-sanitizer memcheck와 racecheck 모두0errors/0hazards. L768도 수치·graph·autograd 검증을 수행했다.
- 실제 TMA/WGMMA cubin, source 및 측정 JSON SHA를 [manifest.json](manifest.json)에 기록한다. NCU SoL을 새로 측정했다는 주장은 하지 않는다.

## 재현·범위

`final.sbatch`: 새 네 폭 두 길이의 같은 실행 Triton/CUDA 비교. `autograd.sbatch`: 현재 개발 진입점10shape. `sanitize.sbatch`: memcheck/racecheck. 공통환경은 `runs/anthropic_adoption_20260919/env.sh`.

이 진입점은 **개발용**이며 production 엔진의 자동 dispatch 변경이나 push는 하지 않았다. 한 프로세스에 H100 한 장이 보이는 환경, B1, L384/768, BF16을 지원한다. [HTML](index.html) · [수치 JSON](summary.json).
'''
(R/'README.md').write_text(md)
def table(header,body):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in header)+'</tr></thead><tbody>'+body+'</tbody></table></div>'
tr=''.join(f"<tr><td>{v['D']}</td><td>{v['L']}</td><td>{v['triton_ms']:.3f}</td><td>{v['cuda_ms']:.3f}</td><td class=\"{'fast' if v['speedup']>1 else 'slow'}\">{v['speedup']:.2f}×</td><td>{v['route']}</td></tr>" for v in rows)
fr=''.join(f"<tr><td>{v['D']}</td><td>{v['L']}</td><td>{v['triton_forward_ms']:.3f}</td><td>{v['cuda_forward_ms']:.3f}</td><td>{v['forward_speedup']:.2f}×</td></tr>" for v in rows)
ir=''.join(f"<tr><td>{v['D']}</td><td>{v['L']}</td><td>{v['inference_before_us']/1000:.3f}</td><td>{v['inference_after_us']/1000:.3f}</td><td>{v['speedup']:.2f}×</td></tr>" for v in previous['inference'])
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TriMul · 모든 D CUDA backward</title><style>body{background:#111d2c;color:#eaf0f8;max-width:1300px;margin:25px auto;padding:0 16px;font:16px/1.6 system-ui}section{background:#1b2d43;padding:20px;border-radius:12px;margin:18px 0}a{color:#8dd5ff}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #40546d;text-align:left;white-space:nowrap}.scroll{overflow:auto}.fast{color:#80e6ac}.slow{color:#ffbe85}.note{padding:15px;background:#463522;border-radius:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.box{border:1px solid #659aab;padding:14px;border-radius:8px}.hbm{color:#ffce87}code{color:#a9dfbd}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>양방향 TriMul · D64 / 128 / 256 / 384 / 512</h1><p>H100 · B1 · L384/768 · BF16 · hidden=방향별D / 합친2D</p>
<p class="note"><b>모든 폭 CUDA backward 연결 및 autograd 검증 완료.</b> D64/256/384/512 포트는 아직 Triton보다 느립니다. 아래1×미만 수치는 성능 회귀를 뜻합니다. D128과 같은 수준으로 튜닝됐다는 뜻이 아닙니다.</p>
<section><h2>학습 전체 · 같은 실행의 기존 Triton 대비</h2>'''+table(['D','L','기존 ms','CUDA ms','배속','실제 경로'],tr)+'''<p>dropout25%, mask·residual·11개gradient·실시간 weight packing 포함. CUDA graph 교대 측정. 초기 plan 생성, compile, CPU dispatch, optimizer와 RNG 생성 제외.</p></section>
<section><h2>학습 forward</h2>'''+table(['D','L','기존 ms','CUDA ms','배속'],fr)+'''</section><section><h2>추론 forward · 앞선 추론 측정 유지</h2>'''+table(['D','L','기존 ms','추론 후보 ms','배속'],ir)+'''<p>dropout0. 학습 forward와 경로가 다릅니다.</p></section>
<section><h2>실제 호출 경로</h2><p><code>runs/trimul_training_current.py</code>의 <code>Training(a)</code>와 <code>bidirectional_trimul_cuda(...)</code>가 모든 폭을 지원합니다. 최신 CUDA가 미포팅이라는 이전 표의 표시는 폐기했습니다.</p><p>D128은 기존 특화 B1과 B7을 유지합니다. L384 단일B7, L768 분리CUDA B7. 나머지 폭은 신규 WGMMA/TMA 기반 cooperative CUDA 포트로, profiler에서 <code>width_b1</code>과 <code>width_b7</code> 각1회 실행을 확인했습니다. production 자동dispatch 변경이나 push는 하지 않았습니다.</p></section>
<section><h2>새 폭별 포트 · HBM 중심</h2><div class="grid">
<div class="box"><h3>Forward</h3><p>x → Anthropic 파생 K1 → <span class="hbm">left/right + x_n</span> → cuBLAS → <span class="hbm">tri</span> → 출력CUDA → y</p><p>입력x_n 저장. projection/gate는 backward 재계산. 출력CUDA의 normalized tri는 일시적 전역workspace.</p></div>
<div class="box"><h3>B1–B4 · CUDA 1회</h3><p>입력: tri, x_n, dy, dropout, output weights/affine</p><p>LN_out·proj·gate 재계산 → <span class="hbm">norm·dp·dg workspace</span> → dNorm/dW → LN 미분</p><p>출력: dTri, dGate, dWproj, dWgate, dgamma_out, dbeta_out</p></div>
<div class="box"><h3>Contraction backward</h3><p>dTri + left/right → cuBLAS 4회 → <span class="hbm">dLeft/dRight</span></p><p>outgoing과incoming 모두 포함.</p></div>
<div class="box"><h3>B7–B12 · CUDA 1회</h3><p>입력: x, x_n, dLeft/dRight, dGate, dy, input weights/affine, mask</p><p>proj·gate 재계산 → <span class="hbm">dp/dg workspace</span> → dx_n와dW → LN 미분+residual</p><p>출력: dx, 네input dW, dgamma_in, dbeta_in</p></div></div><p class="note">한 launch에 합쳤지만 phase 사이의 전역workspace 왕복이 남았습니다. 이것이 D128 스트리밍 구현과 다른 점이며 현재 성능 한계입니다. 다음 최적화는 작은 ring/shared-memory 타일로 workspace를 줄이는 것입니다.</p></section>
<section><h2>검증과 자원</h2><p>10shape public autograd, 두 번 forward의 저장 독립성, in-place 변경 탐지 통과. 새 네 폭은 L384 memcheck/racecheck0errors/0hazards. L768 수치·graph 검증 통과. BF16 교차구현 기준: 출력0.005 / gradient0.01 relativeL2. graph/eager 일반값bit-exact, atomic LN gradient≤5e-6.</p><p>D64: 1warp-group/N64; D256/384/512: 2warp-group/N128, 입력 타일 공유. K64, TMA2stage. dW split-K128/32. Cooperative grid는 실제occupancy 한도로 제한합니다. 넓은 autotune 완료나 NCU SoL 달성을 주장하지 않습니다.</p><p><a href="README.md">상세 범위·재현</a> · <a href="summary.json">수치JSON</a> · <a href="manifest.json">검증·SHA</a> · <a href="widths.cu">CUDA소스</a> · <a href="width_autograd.py">autograd연결</a> · <a href="../../TRIMUL_STATUS.html">전체현황</a></p></section><p>Anthropic의 Hopper 구현과 기존발표를 계승한 학습 확장입니다. Anthropic 대비 우위 주장이 아닙니다.</p></html>'''
(R/'index.html').write_text(page)
# Existing bookmarked URL now shows the actual current CUDA comparison.
(OLD/'index.html').write_text(page.replace('<head>','<head>') .replace('<meta charset="utf-8">','<meta charset="utf-8"><base href="../trimul_cuda_widths_20260923/">',1))
(OLD/'README.md').write_text('# 현재 CUDA 폭별 결과\n\n최신 결과와 실제 연결은 [모든 폭 CUDA backward 보고서](../trimul_cuda_widths_20260923/README.md)를 보세요. 모든 D를 CUDA backward로 연결했지만, D128 외 포트는 현재 Triton보다 느립니다.\n\n| D | L | 기존 Triton ms | 현재 CUDA ms | 배속 |\n|---:|---:|---:|---:|---:|\n'+mt+'\n')
(OLD/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
section='<section id="trimul-widths"><h2>TriMul · D64/128/256/384/512 CUDA backward 연결</h2><p>10shape autograd 통과. D128은 기존 특화 경로, 나머지 폭은 새 CUDA B1/B7 포트. 새 포트는 아직 Triton보다 느리며 실제 배속을 기록했습니다.</p><p><a href="runs/trimul_cuda_widths_20260923/index.html">현재 성능표·HBM 배선·검증</a></p></section>'
p=ROOT/'TRIMUL_STATUS.html';s=p.read_text();s=re.sub(r'<section id="trimul-widths">.*?</section>',section,s,flags=re.S);p.write_text(s)
p=ROOT/'TRIMUL_STATUS.md';s=p.read_text();mark='## TriMul D별 실험 · 2026-09-22';s=s.split(mark)[0].rstrip();p.write_text(s+'\n\n'+mark+'\n\n[모든 폭 CUDA backward 연결·실제 배속](runs/trimul_cuda_widths_20260923/index.html). D64/128/256/384/512 × L384/768, autograd10shape 통과. D128 외 새 CUDA 포트는 현재 Triton보다 느리다.\n')
manifest=dict(complete=True,autograd_shapes=10,cuda_training_shapes=10,sanitizers=sanitizers,training_jobs=[dict(D=v['D'],L=v['L'],job=v['job']) for v in rows],checks=checks,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'widths.cu',R/'width_plan.py',R/'width_autograd.py',R/'fixture.py',R/'check.py',R/'check_autograd.py',R/'report.py',R/'README.md',R/'index.html',R/'summary.json',ROOT/'runs/trimul_training_current.py',*R.glob('check-D*.json'),*R.glob('autograd-D*.json'),*R.glob('*check-D*.log')]})
(R/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(mt);print('ALL_WIDTH_REPORT_DONE')
