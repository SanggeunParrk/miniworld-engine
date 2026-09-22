from pathlib import Path
import json,re,shutil
R=Path(__file__).resolve().parent;S=R.parent/'anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows=[]
for n in (384,768):
 j=json.loads((R.parent/'trimul_b7_nextrow_20260921'/('results-L%d.json'%n)).read_text())
 for scope,label in [('b7','B7–B12'),('backward','전체 BWD'),('forward_backward','FWD + BWD')]:
  t=j['times'][scope];a,b=t['baseline']['median_us'],t['optimized']['median_us']
  rows.append('<tr><td>%d</td><td>%s</td><td>%.3f μs</td><td>%.3f μs</td><td>%.3f×</td><td>−%.2f%%</td></tr>'%(n,label,a,b,a/b,(1-b/a)*100))
section='''<!-- B7_SOL90_BEGIN --><section class="panel" id="b7-sol90"><span class="pill">최신 · B7–B12 CUDA · 2026-09-21</span><h2>뒤쪽 미분 1.38–1.45배 · 전체 학습 1.14–1.17배 · SoL90은 미달</h2>
<p>비교 기준은 <b>이번 B7 개발을 시작하기 직전의 v51 전체 학습 경로</b>다. Anthropic 추론이나 예전 saved-all B7과의 비교가 아니다. B1은 v51, forward는 기존 Anthropic 파생 경로를 유지했다.</p>
<p>dW는 두 행 타일을 묶어 다음 projection/gate GEMM과 현재 GLU 미분을 겹친다. dX는 전용 TMA producer와 한 compute warp group으로 구성한다. 두 커널 모두 shared 112 KiB, SM당 2 CTA다. L768에서는 다음 행 x_n과 첫 gate 타일도 선행 전송한다.</p>
<div class="table-wrap"><table><thead><tr><th>L</th><th>구간</th><th>개발 전</th><th>현재 선택</th><th>속도 배율</th><th>시간 감소</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p>node01 H100 · C128 / 양방향 H256 · dropout 25%, mask, residual · scope마다 별도 600회 교대 graph 측정. live packing과 모든11 gradients 포함; optimizer/RNG 생성/컴파일/CPU dispatch 제외. 구간별 시간을 더해 전체 시간을 만들지 않는다.</p>
<a href="assets/trimul-b7-current.svg"><img src="assets/trimul-b7-current.svg" alt="B7의 두 CUDA launch: HBM 입력, shared/register 계산, PyTorch 수식과 HBM 출력" style="width:100%;height:auto"/></a>
<p><b>검증:</b> 일반·변경 입력의 forward와 9개 gradients는 개발 전과 bit-exact. 입력 LN dgamma/dbeta는 CTA 수 변경에 따른 합산 순서 차이로 상대 L2 약3–4.3e-7이며 기존 한도5e-6 이내다. 두 길이 dW/dX memcheck 오류0, racecheck hazard0, 변경 입력 graph replay 및 counter 초기화 확인. 현재 개발 entrypoint에 반영했다. production dispatch는 별도다.</p>
<p><b>SoL:</b> 최종 L768 NCU tensor-pipe activity는 dW 52.95%, dX 43.21%; DRAM active cycles는 41.28%, 51.94%. 이것을 전체 알고리즘 SoL로 표시하지 않는다. SoL90 달성을 입증하지 못했다.</p>
<details><summary>수치 기준 재검증: 기존 cuBLAS 기준도 완전한 정답은 아니다</summary><p>x_n, 재계산한 BF16 projection/gate, 실제 dW에 넣는 GLU 미분까지 기존과 bit-exact였다. 차이는 GEMM 누적과 BF16 반올림에서 생긴다. L768 변경 입력의 dWL을 FP64 GEMM 후 BF16 반올림한 값과 비교하면 CUDA 0.020176%, 기존 기준 0.056992%였다.</p><p>그렇다고 모든 수치 검증을 통과한 것은 아니다. 기존 기준 대비 dWL 0.053318%는 기존0.05% 조건을 초과한다. FP64-rounded 기준에서도 L768 초기 dWR은 CUDA0.055680%, 기존기준0.059545%였다. 기준과의 불일치 및 허용오차 검토를 그대로 남기며, 한도를 바꾸거나 production-ready로 표시하지 않는다.</p></details>
<p><a href="assets/trimul-b7-sol90.md">변경·탈락 후보·NCU·검증 범위</a> · <a href="assets/trimul-b7-sol90.json">전체 근거 JSON</a> · <a href="#current-wiring">전체 FWD/BWD 배선</a></p></section><!-- B7_SOL90_END -->'''
p=S/'trimul.html';s=p.read_text();s=re.sub(r'<!-- B7_SOL90_BEGIN -->.*?<!-- B7_SOL90_END -->','',s,flags=re.S);s=s.replace('<main>','<main>'+section,1)
if 'href="#b7-sol90"' not in s:s=s.replace('<nav>','<nav><a href="#b7-sol90">최신: B7–B12 최적화</a>',1)
p.write_text(s)
p=S/'index.html';s=p.read_text();notice='<!-- B7_LATEST_BEGIN --><section class="notice"><b>최신: TriMul B7–B12 학습 최적화</b><p>이번 개발 전 대비 뒤쪽 미분 1.38–1.45배, 전체 학습 1.14–1.17배. HBM 배선·코드·검증과 SoL90 미달 상태를 함께 기록했다.</p><a href="trimul.html#b7-sol90">현재 결과와 실제 커널 배선 보기 →</a></section><!-- B7_LATEST_END -->';s=re.sub(r'<!-- B7_LATEST_BEGIN -->.*?<!-- B7_LATEST_END -->','',s,flags=re.S);s=s.replace('<main>','<main>'+notice,1);p.write_text(s)
for name,target in [('b7-current.svg','trimul-b7-current.svg'),('README.md','trimul-b7-sol90.md'),('report.json','trimul-b7-sol90.json')]:shutil.copyfile(str(R/name),str(S/'assets'/target))
print('B7 status section and assets updated')
# Keep selected wiring separate from rejected diagnostic fusion experiments.
J=R.parent/'trimul_b7_joint8_bulk_20260922'
j=json.loads((J/'report.json').read_text())
tr=[]
for name,label in [('trimul_b7_joint_cluster_20260921','4 CTA / spill 발생'),('trimul_b7_joint8_20260922','8 CTA / spill 제거'),('trimul_b7_joint8_bulk_20260922','8 CTA / bulk DSMEM')]:
 for v in j['candidates'][name]:
  t=v.get('times',{})
  tr.append('<tr><td>%s</td><td>%s</td><td>%.1f μs</td><td>%.1f μs</td><td>LN gradient 미달</td></tr>'%(label,v['L'],t['baseline'],t['joint']))
new='''<!-- B7_JOINT_BEGIN --><section class="panel" id="b7-joint"><span class="pill">2026-09-22 · 단일 커널 실험</span><h2>하나로 합칠 수 있다. 현재 통합 시제품은 채택하지 않았다.</h2>
<p><b>왜 두 개인가:</b> dW는 pair 위치를 따라 합산하고, dX는 hidden channel을 따라 합산한다. 현재 선택 경로는 두 작업에 서로 다른 타일과 레지스터 예산을 준다. 두 커널 사이에 dP/dG를 HBM으로 주고받는 버퍼는 없지만, projection/gate와 미분의 재계산·입력 읽기는 중복된다.</p>
<p><b>실제 통합:</b> P/G와 dP/dG를 한 번 계산해 dW·dX가 shared memory에서 공유하는 CUDA 커널을 만들었다. launch는 하나이며 추가 HBM activation 버퍼는 없다. 첫 후보의 spill을 없앤 뒤에도 느렸다. 다음 후보에서 작은 원격 읽기를 bulk DSMEM tree 전송으로 바꿨지만 개선 폭이 부족했다.</p>
<div class="table-wrap"><table><thead><tr><th>통합 시제품</th><th>L</th><th>동일 실행의 개발 전 split</th><th>단일 launch</th><th>검증</th></tr></thead><tbody>'''+''.join(tr)+'''</tbody></table></div>
<p><b>위 표는 수치 검증에 실패한 후보의 병목 진단용 시간이다.</b> 유효한 학습 성능으로 취급하지 않는다. 두 번째 열의 비교 기준은 선택된 nextrow가 아니라 B7 개발 전 split이다. 초기 입력 dX/dW는 기존 한도 이내, 입력 LN dgamma/dbeta는 기존5e-6 기준을 초과했다. 허용오차를 바꾸지 않았으며 기본 경로는 검증된 두 커널 그대로다.</p>
<p><b>NCU:</b> spill 없는 bulk 후보에서 tensor-pipe activity는 L384/L768 각각4.80%/4.83%, warp-active barrier stall 지표는54.25%/54.32%다. 이는 전체 시간의 비율이나 SoL 수치가 아니다. 반복적인 CTA 동기화가 큰 비용인 이 시제품은 탈락이며, 단일 커널 자체가 불가능하다는 증거는 아니다.</p>
<h3>현재 선택 경로의 하드웨어 roofline: 51.6% / 60.4%</h3>
<p>L384 51.6%, L768 60.4%. 각 커널의 dense BF16 HGMMA·HBM·L2 하한 중 최댓값을 구한 뒤 두 launch의 하한과 실측 시간을 각각 합산했다. NCU 실측 clock을 사용했다. scalar GLU/LN·의존성·동기화 비용은 모델에 없어 <b>도달 가능한 알고리즘 상한을 측정한 값은 아니며, SoL90 달성을 주장하지 않는다.</b> 중복 traffic을 줄이면 더 빨라져도 이 비율이 낮아질 수 있다.</p>
<p><a href="assets/trimul-b7-joint.json">단일 커널 소스 해시·오차·시간·NCU</a> · <a href="assets/trimul-b7-joint.md">구현과 다음 설계 제약</a> · <a href="assets/trimul-b7-roofline.json">Roofline 계산 근거</a> · <a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/">NVIDIA NCU 정의</a></p></section><!-- B7_JOINT_END -->'''
p=S/'trimul.html';s=p.read_text();s=re.sub(r'<!-- B7_JOINT_BEGIN -->.*?<!-- B7_JOINT_END -->','',s,flags=re.S);s=s.replace('<main>','<main>'+new,1);p.write_text(s)
for src,target in [(J/'report.json','trimul-b7-joint.json'),(J/'README.md','trimul-b7-joint.md'),(R.parent/'trimul_b7_roofline_20260921/summary.json','trimul-b7-roofline.json')]:shutil.copyfile(str(src),str(S/'assets'/target))
