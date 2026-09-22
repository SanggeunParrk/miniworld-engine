from pathlib import Path
import json, html

R=Path(__file__).resolve().parent
r=json.loads((R/'results.json').read_text());assert r['complete']
t={scope:{n:v['median_us'] for n,v in arms.items()} for scope,arms in r['times'].items()}
labels={'pytorch_eager':'PyTorch eager','pytorch_compile':'PyTorch compile',
        'cueq_torch':'cuEquivariance TriMul + PyTorch Transition (compile)',
        'cueq_cuda':'cuEquivariance TriMul + 새 CUDA Transition (compile)',
        'latest_all':'최신 TriMul + 새 CUDA Transition'}
scopes={'inference':'추론','training_forward':'학습 forward','training':'학습 fwd+bwd'}
speed={s:{n:t[s][n]/t[s]['latest_all'] for n in labels if n!='latest_all'} for s in scopes}
checks={s:{n:max(v['relative_l2'] for v in checks.values()) for n,checks in arms.items()} for s,arms in r['checks'].items()}
summary=dict(job=r['job'],times_us=t,speedup_vs_latest=speed,versions=r['versions'],cross_path_max_relative_l2=checks,
             cueq='matched bidirectional composition of vendor primitives; not public unidirectional TMU',
             latest_training_diagnostic=True,production_dispatch_changed=False)
(R/'summary.json').write_text(json.dumps(summary,indent=2))
headers=['구성',*scopes.values()]
mdrows=[];htmlrows=[]
for n,label in labels.items():
 vals=['%.3f'%(t[s][n]/1000) for s in scopes]
 mdrows.append('| '+' | '.join([label,*vals])+' |')
 htmlrows.append('<tr><th>'+html.escape(label)+'</th>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
table='| '+' | '.join(headers)+' |\n|---|---:|---:|---:|\n'+'\n'.join(mdrows)
effects='\n'.join('- %s 대비: 추론 %.2f배, 학습 forward %.2f배, 학습 fwd+bwd %.2f배.'%(labels[n],*(speed[s][n] for s in scopes)) for n in labels if n!='latest_all')
details='''## 비교 대상

블록 하나는 양방향 TriMul → Transition이며, 전체를 실제 연결해 측정했다. 파라미터와 입력은 모든 조합에서 동일하다.

- **PyTorch eager:** 엔진 PyTorch 경로의 수식을 plain PyTorch 연산으로 작성. FP32 LayerNorm → BF16 출력, projection/gate, 두 방향 contraction, 공유 256채널 output LN, dropout/residual, SwiGLU Transition/residual이다.
- **PyTorch compile:** 위 전체 블록을 static/fullgraph `torch.compile`한다. 수동 CUDA graph를 사용하므로 Inductor의 자체 cudagraphs는 껐다.
- **cuEquivariance + PyTorch Transition:** cuEq input/output LN과 fused sigmoid-gated dual GEMM을 사용하고 contraction/output projection은 PyTorch로 구성한다. Transition은 PyTorch다. 전체 블록을 compile한다.
- **cuEquivariance + 새 CUDA Transition:** 동일한 cuEq TriMul에 최신 native CUDA Transition을 붙인다. Transition을 공통으로 두어 TriMul 변경 효과를 비교한다.
- **최신 조합:** Anthropic 유래 TriMul forward + cache-policy B1 + K128 단일 B7, 최신 native CUDA Transition forward/backward다.

설치된 cuEquivariance에는 이 블록에 대응하는 Transition API가 없다. cuEq 공개 TMU는 단방향이어서 두 번 호출하면 공유 output LN을 가진 MiniWorld 양방향 TriMul과 수식이 달라진다. 따라서 이 표는 **같은 수식을 가진 cuEq primitives 조합**과의 비교이며, NVIDIA 공개 TMU 모듈 또는 전체 Pairformer 제품의 벤치라고 해석하면 안 된다. cuEq 조합은 이전 비교에서 사용한 `compare_cueq_training.py:cueq_forward`와 동일하며, 추론에서는 dropout 곱셈을 생략한다.

## 조건

node01 H10080GB, B=1/L=384/C=128, BF16 projection과 activation, FP32 LN params, TriMul hidden128 per direction, Transition expansion4.
학습 고정 row dropout25%, 추론 dropout0. pair mask, 두 residual, live weight packing, 입력 및 15개 파라미터 gradient를 포함한다.
모드마다 5×250회 교차 CUDA graph; optimizer, RNG 생성, CPU dispatch, 컴파일은 제외한다. PyTorch eager도 CUDA graph를 쓰므로 eager CPU launch overhead는 포함하지 않는다.
cuEq는 설치된 버전의 init_triton_cache 및 기본 tuning 정책을 사용한다. 버전과 환경값은 results.json에 기록했다.

## 검증과 한계

PyTorch compile 대비 forward 상대L2 0.5%, gradient 1% 한도로 모든 출력과 gradient를 확인한다. 입력 및 squeeze 가중치를 바꾼 graph/eager 비교도 수행한다. LN gradient는 5e-6, 나머지는 bit-exact를 요구한다.
**이전에 남은 최신 TriMul 입력 LN gradient 엄격 검사(최대9.535e-6 >5e-6)는 해결되지 않았다.** 이번 BF16 경로 간 비교가 그 검사를 대체하지 않는다. 최신 학습 시간은 개발 후보의 진단 결과다. production 기본 배선을 바꾸지 않았다.

## 자료

- [원본 결과·시간표본·kernel trace 목록](results.json) · [요약](summary.json)
- [벤치 코드](bench.py) · [Slurm 제출](bench.sbatch)
- [Transition 원본 경로와 SHA-256](transition_sources.json)
- [이전 엔진과 비교](../minipairformer_block_cuda_20260922/index.html)

재현: `sbatch runs/minipairformer_block_baselines_20260922/bench.sbatch`
'''
(R/'README.md').write_text('# MiniPairformer 1블록: PyTorch·cuEquivariance 비교\n\njob'+r['job']+' · 단위 ms\n\n'+table+'\n\n'+effects+'\n\n'+details)
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MiniPairformer · PyTorch / cuEquivariance 비교</title><style>body{max-width:1150px;margin:24px auto;padding:0 16px;font:16px/1.65 system-ui;color:#24364c;background:#f3f6fa}section{background:white;padding:20px;margin:16px 0;border-radius:10px}.notice{background:#fff0d4;padding:14px;border-left:4px solid #bc7718}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #d7e1eb;text-align:right}th:first-child{text-align:left}.scroll{overflow:auto}a{color:#1765a4}</style><h1>MiniPairformer 1블록 · PyTorch / cuEquivariance 비교</h1><p>H100 · B1/L384/C128 · 양방향 TriMul → Transition · 학습 dropout25%</p><section><h2>동일 실행 전체 시간 · ms</h2><div class="scroll"><table><tr>HEADERS</tr>ROWS</table></div><p>EFFECTS</p></section><section><h2>cuEquivariance 비교의 정확한 의미</h2><p>공유 256채널 output LN을 유지하는 cuEq primitives 조합입니다. 두 단방향 공개 TMU 호출과는 다른 수식입니다. 설치된 cuEq에는 이 Transition API가 없으므로 PyTorch Transition을 붙인 행과 최신 CUDA Transition을 붙인 행을 모두 표시했습니다.</p><p>PyTorch compile과 cuEq 조합은 블록 전체를 static compile합니다. 모든 행에 수동 CUDA graph를 적용해 CPU dispatch를 제외하고 GPU 실행을 비교했습니다.</p></section><p class="notice">최신 학습은 엄격한 TriMul LN gradient 검증이 남은 개발 후보입니다. production 기본 경로로 승격하지 않았습니다.</p><p><a href="README.md">조건·검증·재현·한계</a> · <a href="results.json">원본 결과</a> · <a href="../minipairformer_block_cuda_20260922/index.html">이전 엔진 비교</a> · <a href="../../TRIMUL_STATUS.html">전체 현황</a></p></html>'''
page=page.replace('HEADERS',''.join('<th>'+h+'</th>' for h in headers)).replace('ROWS',''.join(htmlrows)).replace('EFFECTS',html.escape(effects).replace('\n','<br>'))
(R/'index.html').write_text(page)
print(json.dumps(summary,indent=2))
