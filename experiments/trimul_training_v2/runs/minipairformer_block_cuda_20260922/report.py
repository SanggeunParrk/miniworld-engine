"""Publish the completed whole-block CUDA Transition comparison."""
from pathlib import Path
import json

R = Path(__file__).resolve().parent
r = json.loads((R / 'results.json').read_text())
assert r['complete']
t = {s: {n: v['median_us'] for n, v in arms.items()} for s, arms in r['times'].items()}
scopes = {'inference': '추론', 'training_forward': '학습 forward', 'training': '학습 fwd+bwd'}
arms = ('previous_triton', 'previous_h100', 'latest_trimul_only', 'latest_all')
speed = {s: {n: t[s][n] / t[s]['latest_all'] for n in arms[:-1]} for s in scopes}
summary = dict(job=r['job'], times_us=t, speedup=speed,
               scope='B1/L384/C128, one bidirectional TriMul + Transition block',
               latest_training_diagnostic=True, production_dispatch_changed=False)
(R / 'summary.json').write_text(json.dumps(summary, indent=2))
headers = ['모드', '이전 Triton TriMul + Triton Transition', '이전 H100 TriMul + Triton Transition',
           '최신 TriMul + Triton Transition', '최신 TriMul + 새 CUDA Transition']
mdrows = []
htmlrows = []
for s, label in scopes.items():
    vals = ['%.3f' % (t[s][n] / 1000) for n in arms]
    mdrows.append('| ' + ' | '.join([label, *vals]) + ' |')
    htmlrows.append('<tr><th>' + label + '</th>' + ''.join('<td>' + v + '</td>' for v in vals) + '</tr>')
table = '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + '\n'.join(mdrows)
effects = '\n'.join('- %s: 이전 Triton 조합 대비 %.2f배, 이전 H100 TriMul 조합 대비 %.2f배; 새 Transition 연결 자체는 %.2f배.' %
                    (label, speed[s]['previous_triton'], speed[s]['previous_h100'], speed[s]['latest_trimul_only'])
                    for s, label in scopes.items())
details = '''## 배선과 비교 범위

- 한 블록은 양방향 TriMul → Transition이다. backward도 Transition의 실제 입력 gradient를 TriMul로 전달한다. 개별 모듈 시간의 합계가 아니다.
- 최신 Transition은 `/home/psk6950/miniworld-engine-tbwd`의 D128 hand-CUDA 구현을 그대로 복사한 `transition_snapshot/`이다. 원본 경로와 SHA-256은 `transition_sources.json`에 보존했다.
- Forward는 LN → expand/SwiGLU → squeeze/residual을 `transition_fwd_fused`에 융합한다. 학습에서는 x_n과 LN 통계를 저장하고 추론에서는 저장하지 않는다.
- Backward는 `transition_bwd_fused`와 작은 `reduce_partials`로 입력 및 모든 파라미터 gradient를 계산한다. 기존 Triton backward로 돌아가지 않았는지 profiler trace로 확인한다.
- 최신 TriMul은 Anthropic 유래 forward + 선택한 cache-policy B1 + K128 단일 B7 후보다. 추론은 infer_k1/infer_k3를 사용한다.
- 이전 TriMul은 보존한 Anthropic 도입 전 Triton/H100 경로다. 두 이전 조합의 Transition은 앞선 벤치와 같은 Triton full-K/residual 경로다. 과거 전체 설치 환경을 재현한 비교는 아니다.
- 네 조합 모두 동일 입력과 가중치, 동일 정밀도를 쓴다. 최신 TriMul + 기존 Transition 행을 추가하여 새 Transition만의 기여를 분리했다.

## 측정 조건

H100 node01, B=1/L=384/C=128, BF16, TriMul hidden128 per direction, Transition expansion4.
학습은 고정 row dropout25%, 추론은 dropout0. pair mask, 두 residual, live weight packing, 입력과 15개 파라미터 gradient를 포함한다.
Transition projection은 BF16, LN 파라미터는 FP32, squeeze 가중치는 nonzero다.
static compile + 수동 CUDA graph로 모드마다 5×250회 교차 측정한다.
optimizer, dropout RNG 생성, 컴파일, CPU dispatch는 제외한다. 일부 기존 Triton cache miss는 측정 전 최대24개 후보로 처리한다.

## 정확도와 적용 상태

경로 간 BF16 비교는 forward 상대L2 0.5%, gradient 1% 이내를 검사한다.
입력과 squeeze 가중치를 바꿔 graph/eager도 비교한다. 기존 LN atomic 누산에는 5e-6 한도를 쓰고 나머지는 bit-exact를 요구한다.
**최신 TriMul의 앞선 엄격 검사에서 입력 LN gradient가 최대9.535e-6로 5e-6 한도를 넘은 문제는 남아 있다.**
이 비교의 통과가 그 문제를 해결한 것은 아니다. 최신 학습 시간은 개발 후보의 진단 결과이며, production 기본 배선으로 승격하지 않았다.

## 재현 및 근거

- [벤치 코드](bench.py) · [Slurm 제출 파일](bench.sbatch)
- [원본 결과와 kernel trace 목록](results.json) · [요약](summary.json)
- [Transition 소스 SHA-256](transition_sources.json)
- [앞선 Transition 고정 비교](../minipairformer_block_20260922/index.html)

저장소 루트에서 `sbatch runs/minipairformer_block_cuda_20260922/bench.sbatch`.
'''
(R / 'README.md').write_text('# MiniPairformer 1블록: 최신 TriMul + 새 CUDA Transition\n\njob' + r['job'] + ' · 2026-09-22 · 단위 ms\n\n' + table + '\n\n' + effects + '\n\n' + details)
page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniPairformer · 새 CUDA Transition 연결</title>
<style>body{max-width:1200px;margin:24px auto;padding:0 16px;font:16px/1.65 system-ui;color:#24364c;background:#f3f6fa}section{background:white;padding:20px;margin:16px 0;border-radius:10px}.notice{background:#fff0d4;padding:14px;border-left:4px solid #bc7718}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #d7e1eb;text-align:right}th:first-child{text-align:left}.scroll{overflow:auto}.flow{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.box{padding:15px;border:1px solid #82aeb4;background:#ecf8f2;border-radius:8px}a{color:#1765a4}</style>
<h1>MiniPairformer 1블록 · 새 CUDA Transition까지 연결</h1>
<p>H100 · B1/L384/C128 · 학습 dropout25% / 추론 dropout0</p>
<section><h2>최신 조합의 실제 배선</h2><div class="flow"><div class="box">양방향 TriMul<br>Anthropic 유래 forward<br>cache-policy B1 + 단일 B7</div><b>→</b><div class="box">새 CUDA Transition<br>LN + expand/SwiGLU + squeeze/residual<br>학습: x_n + LN 통계 저장</div></div><p>Backward: Transition 융합 CUDA + 작은 부분합 축약 → TriMul B1–B12. 두 모듈을 실제 연결해 전체 시간을 잽니다.</p></section>
<section><h2>동일 실행 비교 · ms</h2><div class="scroll"><table><tr>HEADERS</tr>ROWS</table></div><p>EFFECTS</p></section>
<p class="notice">최신 학습은 정확도 검증이 남은 개발 후보입니다. TriMul 입력 LN gradient의 엄격 한도 초과 문제는 아직 해결하지 않았으며 production 기본 경로로 승격하지 않았습니다.</p>
<section><h2>비교 범위</h2><p>두 이전 행은 Anthropic 도입 전 TriMul과 기존 Triton Transition의 조합입니다. 최신 TriMul + 기존 Transition 행과 마지막 행의 차이가 새 CUDA Transition 연결 효과입니다. 과거 전체 설치 환경의 재현은 아닙니다.</p><p>mask, 두 residual, 입력과 15개 파라미터 gradient를 포함합니다. optimizer/RNG 생성/CPU dispatch/컴파일은 제외합니다. 모드마다 5×250회 교차 CUDA graph 측정.</p></section>
<p><a href="README.md">검증·재현·한계</a> · <a href="results.json">원본 결과</a> · <a href="summary.json">요약</a> · <a href="../../TRIMUL_STATUS.html">전체 현황</a></p></html>'''
page = page.replace('HEADERS', ''.join('<th>' + h + '</th>' for h in headers)).replace('ROWS', ''.join(htmlrows)).replace('EFFECTS', effects.replace('\n', '<br>'))
(R / 'index.html').write_text(page)
print(json.dumps(summary, indent=2))
