from pathlib import Path
import json,shutil
R=Path(__file__).resolve().parent;root=R.parents[1];site=root/'runs/anthropic_b1b4_pipeline_20260919/site-visuals/dist'
head=['L','방식','정확도','FWD ms','B1 ms','BWD ms','전체 ms'];rows=[];hrs=[]
for n in (384,768):
 d=json.loads((R/('results-L%d.json'%n)).read_text())
 for k,label in [('baseline','기존 tri BF16'),('xhat_bf16','x̂ BF16 + rstd'),('xhat_fp32','x̂ FP32 + rstd')]:
  status='bit-exact' if k in d['valid'] else '실패 (진단 시간)'
  cells=[str(n),label,status]+['%.4f'%(d['times'][s][k]['median_us']/1000) for s in ('forward','b1','backward','forward_backward')]
  rows.append('| '+' | '.join(cells)+' |');hrs.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
race=R/'racecheck-13450_0.log';rs=race.read_text() if race.exists() else '';race_status='통과' if 'RACECHECK SUMMARY: 0 hazards' in rs else '진행 중'
intro='''# tri를 정규화 값으로 대체: backward 원본 읽기 제거 검증

2026-09-21 · node02 H100 · BF16 C128/H256 · 양방향 · mask/dropout25%/residual · Slurm 13445_0/1.
기준은 직전 shared B1 + split_xn_pc1 B7 학습 개발 경로다. 원본 Anthropic 추론과의 비교가 아니다.

## 이전 결론 정정

이전 실험은 tri와 affine LN_out(tri)를 둘 다 읽었다. 따라서 그 결과만으로 tri를 대체하는 저장 정책을 평가할 수 없었다.
원본 tri는 수학적으로 필수가 아니다. gamma/beta 적용 전 x̂=(tri−mean)*rstd와 rstd로 LN backward를 계산할 수 있다.
이번 두 후보는 실제로 tri를 backward 인자와 저장 activation 목록에서 제거했다.

'''
body='''
## 검증 결과

- forward 이후 원본 tri를 NaN으로 덮어쓴 뒤에도 각 후보의 전체11개 gradient가 덮어쓰기 전과 bit-exact. 이는 원본에 의존하지 않음을 검증한다. BF16 후보의 기준 대비 정확도와는 별개다.
- FP32 x̂: 일반 입력 및 x/weights/gamma/beta/mask/dropout mask/dy 변경 후 y와11개 gradient 모두 기존과 bit-exact. gamma_out[0]=0 포함. Graph replay도 eager와 bit-exact.
- BF16 x̂: forward는 기존과 동일하지만, 저장 반올림 뒤 affine를 다시 만들면서 backward 중간값이 달라진다. 최대 gradient 상대L2가 L384 약0.3523%, L768 약0.3545%(변형 포함)로 허용치0.05%를 초과한다. 유효한 학습 최적화 후보에서 제외했다.
- Memcheck 13447_0/1: 두 길이 모두0 errors.
- L384 B1 racecheck 13450_0: RACE_STATUS. 첫13448은 필터 문법 오류로 검사 미실행; 올바른 kns=b1_fused 필터로 재실행했다.
- 기존 경로 대비 동등성 검사이며, 기존 L768 B7 독립-reference 오차 문제 해결을 의미하지 않는다. Production 승격 없음.

## 실제 읽기와 저장 정책

| 방식 | B1 출력 측 global 입력 | 크기 L384 /L768 |
|---|---|---|
| 기존 | BF16 tri | 75.50 /301.99 MB |
| BF16 대체 | BF16 x̂ + FP32 rstd | 76.09 /304.35 MB |
| FP32 대체 | FP32 x̂ + FP32 rstd | 151.58 /606.34 MB |

위 크기는 논리적 입력/저장 텐서 크기이며 NCU DRAM byte 실측값이 아니다. 공통 입력 x_n, 가중치와 scratch는 제외했다.
BF16 대체는 큰 텐서 읽기량을 그대로 유지하고 rstd만 추가한다. FP32는 원본 tri를 읽지 않지만 대체 텐서의 원소당 바이트가 두 배다.
Forward에서 tri는 contraction 출력으로 일시적으로 존재하고 K3가 소비한다. 이후 backward를 위해 유지하지 않는다. 실제 모델 peak-memory는 측정하지 않았다.

### B1 구현

1. 입력 x_n으로 gate 재계산.
2. 저장된 x̂에서 affine BF16 값 fma(x̂,gamma,beta)를 shared memory에 구성. 출력 LN 평균/분산 재계산 없음.
3. projection, dWproj, dTri와 LN parameter gradient 계산. dx = rstd*(h−mean(h)−x̂*mean(h*x̂)).
4. 기존과 같은 후속 dWgate 단계 및 전체 gradient reduction.

FP32 구현은 64 KiB x̂ 타일을 한 번 읽어 shared memory에 유지한다. 공간 확보를 위해 gate GEMM 후 gate weight의32KiB shared 영역을 x̂의 절반으로 재사용한다. 이에 따라 다음 타일에서 gate weight32KiB를 다시 로드한다. 가중치는 반복 재사용되지만 실제 L2/DRAM 적중률은 NCU로 확인하지 않았다. 이 구현의 오버헤드를 저장 정책 자체의 필연적인 비용으로 일반화하면 안 된다.
BF16은 원래 tri의32KiB 영역만 대체하며 gate weight 재로드 없음.

## 성능 해석과 범위

FP32 후보의 전체 시간은 기존보다 L384 +8.57%, L768 +8.60%. 현재 구현을 기본값으로 채택할 이유는 없다.
그러나 이번 목적은 **tri 제거 가능성과 정확도 검증**이다. 새 K3는 scalar store, B1은 새 shared layout과 scalar read를 사용하며 재튜닝하지 않았다. 이 수치가 저장 정책의 최적 성능이나 하한을 뜻하지 않는다.
동일한 저장 형식/타일 최적화 수준에서 우열을 확정한 것이 아니다. BF16과 FP32 두 형식만 시험했으며 다른 저장/압축 형식이 불가능하다는 결론도 아니다.

- 경로별600회(200×3) 교대 CUDA graph 중앙값. live packing 및 전체11개 gradient 포함.
- optimizer/RNG 생성/CPU dispatch/compile 제외. 각 구간과 전체를 독립 측정했으므로 구간 중앙값의 합과 전체는 다를 수 있다.
- JSON: checks/mutated/poison/graph_vs_eager, raw timing samples, cubin 경로, source SHA-256.
- 현재 기본 경로와 SVG는 입력 x_n만 저장하는 기존 정책을 유지한다.
'''.replace('RACE_STATUS',race_status)
report=intro+'| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join(rows)+'\n'+body
(R/'README.md').write_text(report)
section='''<!-- REPLACE_TRI_BEGIN --><section class="panel" id="replace-tri"><span class="pill">2026-09-21 · tri 대체 저장 검증</span><h2>B1에서 원본 tri 읽기 제거 가능 · FP32 x̂ 경로 bit-exact</h2><p><b>정정:</b> 이전 실험은 tri와 affine LN_out을 둘 다 읽었다. 이번에는 gamma/beta 적용 전 정규화 값 x̂와 역표준편차만 저장하고, tri를 B1 인자와 backward 저장 목록에서 제거했다.</p><div class="notice"><b>직접 검증:</b> Forward 이후 tri를 NaN으로 덮어써도 backward 결과 불변. FP32 후보는 두 길이 모두 기존 y 및11개 gradient와 bit-exact. gamma=0 및 mask/dropout mask 변경, graph replay 포함.</div><div class="table-wrap"><table><thead><tr>'''+''.join('<th>'+h+'</th>' for h in head)+'''</tr></thead><tbody>'''+''.join(hrs)+'''</tbody></table></div><p class="muted">H100 node02 · 양방향 BF16 C128/H256 · dropout25% · 600회 교대 CUDA graph · 전체11개 gradient/live packing 포함.</p><h3>원본 tri를 읽는 화살표 없음</h3><pre>FWD: contraction → tri → K3 → y
                            ├→ input x_n 저장
                            └→ x̂ + rstd 저장
BWD: x̂ + rstd → affine BF16 재구성 → projection/dWproj
              └→ LN 미분 → dTri, dgamma, dbeta
tri는 K3 소비 뒤 backward 저장 목록에서 제외</pre><p>출력 측 논리적 읽기: 기존 BF16 tri 75.50/301.99MB → BF16 x̂+rstd 76.09/304.35MB, FP32 x̂+rstd 151.58/606.34MB. 실제 DRAM 전송량은 별도 프로파일링 필요.</p><div class="notice amber"><b>정확도:</b> BF16 저장은 최대 gradient 상대L2 약0.35%로 허용치0.05% 초과. 표의 BF16 시간은 진단용이며 유효한 속도 개선으로 취급하지 않는다.</div><div class="notice"><b>현재 판단:</b> FP32 대체는 정확하지만 이번 구현 전체는 약8.6% 느림. 새 scalar store/shared layout은 재튜닝하지 않았고, FP32 B1은 shared 공간 확보를 위해 gate weight를 타일마다 다시 읽는다. 저장 방식 자체의 상한·하한으로 해석하지 않는다. 현재 기본값 유지.</div><p>Memcheck L384/L768 0 errors · L384 B1 racecheck: '''+race_status+'''. 기존 B7 independent-reference 이슈는 별개.</p><p><a href="assets/replace-tri-report.md">상세 구현·검증</a> · <a href="assets/replace-tri-L384.json">L384 원본</a> · <a href="assets/replace-tri-L768.json">L768 원본</a></p></section><!-- REPLACE_TRI_END -->'''
p=site/'trimul.html';s=p.read_text();start='<!-- REPLACE_TRI_BEGIN -->';end='<!-- REPLACE_TRI_END -->'
if start in s:a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+section+s[b:]
else:s=s.replace('<main>','<main>'+section,1).replace('<nav>','<nav><a href="#replace-tri">tri 대체 검증</a>',1)
old='<h2>출력 LN 저장: B1은 빨라지지만 전체 학습 이득 없음</h2>'
s=s.replace(old,old+'<p class="notice amber">이전 기록: 원본 tri를 유지하고 출력 LN을 추가 저장한 실험이다. <a href="#replace-tri">tri 읽기를 실제로 제거한 후속 검증</a>을 별도로 확인해야 한다.</p>') if '이전 기록: 원본 tri를 유지하고' not in s else s
p.write_text(s)
shutil.copy2(str(R/'README.md'),str(site/'assets/replace-tri-report.md'))
for n in (384,768):shutil.copy2(str(R/('results-L%d.json'%n)),str(site/('assets/replace-tri-L%d.json'%n)))
p=root/'TRIMUL_STATUS.md';s=p.read_text();marker='## tri 대체 저장 검증 (2026-09-21)'
if marker not in s:s+='\n'+marker+'\n\n이전 출력 LN 추가 저장 실험은 tri도 읽는 경로였다. 후속 실험에서 tri를 B1 인자 및 backward 저장 목록에서 제거하고 NaN poison 검사를 통과했다. 정규화 값 FP32+rstd는 기존과 bit-exact, BF16은 약0.35% gradient 오차로 실패. FP32 첫 구현의 전체 시간은 약8.6% 증가했으나 재튜닝 전이며 정책 자체의 최적성 결론은 아니다. [상세 결과](runs/trimul_replace_tri_20260921/README.md). 기본값/SVG는 기존 경로 유지.\n'
p.write_text(s)
print('wrote report; racecheck',race_status)
