from pathlib import Path
from html import escape
import json,xml.etree.ElementTree as ET
p=Path(__file__).resolve().parent
site=p.parent/'anthropic_b1b4_pipeline_20260919/site-visuals/dist'
for label in ('storepipe','ring96cache3'):
 checks=json.loads((p/(label+'-sanitizers.json')).read_text())
 assert len(checks)==6 and all(v['result']=='passed' for v in checks.values())
 assert 'ERROR SUMMARY: 0 errors' in (p/(label+'-initcheck-unfiltered-L64.log')).read_text()
 assert len(json.loads((p/(label+'-production-check.json')).read_text()))==6
out=['''<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="1390" viewBox="0 0 1240 1390" role="img" aria-labelledby="title desc"><title id="title">B7–B12 CUDA training: validated paths by training bucket</title><desc id="desc">L384 recomputes B7 in two CTA roles. L768 computes B7 once and passes BF16 derivatives through a 12 MiB global ring intended for L2. Both finish parameter reductions in the same CUDA launch and retain fused forward saves.</desc><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#496a7b"/></marker></defs><style>text{font-family:system-ui,sans-serif;fill:#183849}.h{font-size:23px;font-weight:700}.b{font-size:18px;font-weight:650}.t{font-size:15px}.s{font-size:13px;fill:#496474}.line{fill:none;stroke:#496a7b;stroke-width:2;marker-end:url(#a)}</style><rect width="1240" height="1390" fill="#f3f7f8"/>''']
def text(x,y,s,cl='t'):out.append('<text x="%s" y="%s" class="%s">%s</text>'%(x,y,cl,escape(s)))
def rect(x,y,w,h,c='#fff'):out.append('<rect x="%s" y="%s" width="%s" height="%s" rx="10" fill="%s" stroke="#9cb9c5"/>'%(x,y,w,h,c))
def arrow(d):out.append('<path d="%s" class="line"/>'%d)
text(30,38,'Bidirectional TriMul · B7–B12 · ONE CUDA launch','h')
text(30,65,'H100 · BF16 · dropout 25% · C128 · left/right H256 each · Anthropic v5 training extension')
rect(30,83,1180,80)
text(48,111,'Forward remains fused: input LN + projection/gate; saved values remain unchanged','b')
text(48,139,'Saved x_n, mean, inverse std, gate/projection preactivations; original x retained. B1–B6 feed this stage.')
for y,n,ring in [(185,384,False),(730,768,True)]:
 rect(30,y,1180,522,'#edf7f4');text(49,y+31,'L%d · %s'%(n,'B7 computed once → bounded L2 ring' if ring else 'B7 recomputation + overlapped dx store'),'h')
 text(49,y+57,'264 CTAs × 256 threads · two CTAs/SM · 112 KiB dynamic shared/CTA · zero stack/spills')
 rect(50,y+80,465,350);rect(540,y+80,650,350)
 text(68,y+108,('%d CTAs · B7 + B8 · weight gradients'%(160 if ring else 104)),'b')
 text(68,y+135,'8 hidden groups × %d splits; 2 FP32 accumulation segments'%(20 if ring else 13),'s')
 rect(68,y+150,429,82,'#edf4fb');text(84,y+178,'B7 · mask + sigmoid/GLU backward','b')
 text(84,y+205,'TMA preactivation + upstream; BF16 derivatives')
 arrow('M280 %dV%d'%(y+232,y+251))
 rect(68,y+255,429,88,'#e5f5ed');text(84,y+282,'B8 · WGMMA: dG/dP × saved x_n','b')
 text(84,y+309,'Separate gate/projection warpgroups; N128 MMA')
 text(68,y+371,'Global dW partials: %d MiB'%(20 if ring else 13))
 text(68,y+398,'No full d_concat; four direct weight tensor maps','s')
 text(560,y+108,'%d CTAs · B9–B12 · input gradients'%(104 if ring else 160),'b')
 rect(560,y+130,610,60,'#edf4fb');text(577,y+155,'B9 · output-gate WGMMA → BF16 contribution','b')
 text(577,y+178,'Retained in registers until B10 add','s')
 arrow('M866 %dV%d'%(y+190,y+210))
 rect(560,y+213,610,77,'#edf4fb');text(577,y+240,('B10 · TMA-read ring → WGMMA' if ring else 'B7 + B10 · recompute derivatives → WGMMA'),'b')
 text(577,y+267,'Lg → Lp → Rg → Rp; add B9; BF16 dx_n in registers')
 arrow('M866 %dV%d'%(y+290,y+308))
 rect(560,y+310,610,75,'#f3eefb');text(577,y+337,'B11 + B12 · input LN backward + residual','b')
 text(577,y+362,'Saved mean/rstd; BF16 pair access; TMA dx output')
 text(560,y+411,('Ring handoff: full TMA completion → release/acquire → safe reuse' if ring else 'Next gate / current LN inputs / prior dx store overlap computation'),'s')
 if ring:
  arrow('M497 %dH528V%dH560'%(y+187,y+251))
  text(49,y+454,'12 MiB global ring: 96 × 64 rows × 1024 BF16; intended for L2, can reach HBM','b')
 else:text(49,y+454,'B7 runs in both roles; the large intermediate tensors are not materialized.','b')
 text(49,y+483,'Same launch: completion barrier → final dW, dγ, dβ reduction → counters reset')
 text(49,y+506,'LN partials: %d KiB. No forward/save-policy change; production dispatch unchanged.'%(104 if ring else 160),'s')
rect(30,1273,1180,95)
text(49,1302,'B7–B12: L384 617.216 → 364.848 µs (1.692×) · L768 2365.632 → 1429.136 µs (1.655×)','b')
text(49,1328,'600 alternating graph samples/path. ≥1.7× and defensible SoL90 remain unachieved.')
text(49,1352,'NCU L768: L2 inter-partition fabric 90.27%, data sectors 57.04%; not proof of algorithmic SoL90.','s')
out.append('</svg>');svg='\n'.join(out);ET.fromstring(svg)
(p/'front-b7b12-ring-checkpoint.svg').write_text(svg);(site/'assets/front-b7b12.svg').write_text(svg)
html=(site/'trimul.html').read_text();a=html.index('<!-- B7B12_BEGIN -->');b=html.index('<!-- B7B12_END -->')+len('<!-- B7B12_END -->')
section='''<!-- B7B12_BEGIN --><section class="panel" id="b7b12-fusion"><span class="pill green">2026-09-20 · B7–B12 CUDA 융합</span>
<h2>B7–B12 단일 CUDA 호출: 1.69× / 1.66×</h2>
<p><b>입력 LN 융합 forward와 기존 저장 정책을 유지했다.</b> L384는 B7을 가중치·입력 미분 역할에서 각각 계산하며, L768은 B7을 한 번 계산하고 12 MiB 순환 버퍼로 전달한다. 최종 파라미터 미분 reduction까지 같은 커널 호출에서 끝난다.</p>
<div class="notice amber"><b>목표: 기존 기준 대비 최소 1.7×, 검증된 SoL90. 아직 달성하지 못했다.</b></div>
<div class="notice">Anthropic native v5의 TMA/WGMMA를 계승한 학습 확장이다. 아래 기준은 고정한 Triton/cuBLAS 학습 경로다. 실험용 전체 backward 연결·검증을 마쳤으며 production dispatch는 변경하지 않았다.</div>
<div style="overflow:auto"><a href="assets/front-b7b12.svg"><img src="assets/front-b7b12.svg" alt="L384의 B7 재계산과 L768의 단일 B7 및 순환 버퍼를 구분한 실제 CUDA 배선" style="display:block;width:100%;min-width:850px;height:auto"></a></div>
<h3>dropout 25% · CUDA graph 600회/경로 · 중앙값</h3>
<div class="table-wrap"><table><thead><tr><th>범위</th><th>L</th><th>기준 µs</th><th>새 CUDA µs</th><th>속도 배율</th><th>시간 감소</th></tr></thead><tbody>
<tr><td>B7–B12</td><td>384</td><td>617.216</td><td>364.848</td><td>1.692×</td><td>40.89%</td></tr>
<tr><td>B7–B12</td><td>768</td><td>2365.632</td><td>1429.136</td><td>1.655×</td><td>39.59%</td></tr>
<tr><td>전체 backward</td><td>384</td><td>1092.304</td><td>843.104</td><td>1.296×</td><td>22.81%</td></tr>
<tr><td>전체 backward</td><td>768</td><td>4338.912</td><td>3405.008</td><td>1.274×</td><td>21.52%</td></tr>
<tr><td>전체 forward + backward</td><td>384</td><td>1530.240</td><td>1273.920</td><td>1.201×</td><td>16.75%</td></tr>
<tr><td>전체 forward + backward</td><td>768</td><td>6068.656</td><td>5146.272</td><td>1.179×</td><td>15.20%</td></tr>
</tbody></table></div>
<p>전체 forward + backward는 각 replay 안에서 동일한 Anthropic 파생 입력-LN 융합 forward를 실행하고 저장값을 새로 만든 직접 측정이다. B1–B6는 동일하며 B7–B12만 교체했다. optimizer, RNG 생성, CPU autograd dispatch와 가중치 packing은 포함하지 않는다.</p>
<p class="muted">H100 node02 · BF16 · B1/C128/left·right 각 H256. 전체 backward의 B1–B6는 양쪽 동일하며, Claude가 개발 중인 B1–B4와 합친 결과는 아니다. 기준 input-dual Triton은 기존 cache miss 시 648개 중 3개 heuristic 후보를 사용한다. exhaustive 재튜닝 비교가 아니다. allocation·컴파일·host descriptor 생성은 replay에서 제외한다.</p>
<h3>실제 구현과 저장 위치</h3>
<div class="table-wrap"><table><thead><tr><th></th><th>L384</th><th>L768</th></tr></thead><tbody>
<tr><td>CUDA source</td><td>front_prefetch_lnpair_storepipe</td><td>front_ring96_cache3</td></tr>
<tr><td>CTA 역할</td><td>dW 104 / dX 160</td><td>dW 160 / dX 104</td></tr>
<tr><td>B7</td><td>두 역할에서 재계산</td><td>dW에서 한 번, TMA 순환 버퍼 전달</td></tr>
<tr><td>추가 global 작업 버퍼</td><td>dW 13 MiB + LN 160 KiB</td><td>ring 12 MiB + dW 20 MiB + LN 104 KiB</td></tr>
<tr><td>최적화</td><td>다음 gate·현재 LN 읽기·이전 dx 쓰기 겹침</td><td>ring 전송·GLU 겹침, 재사용 버퍼 L2 우선순위</td></tr>
</tbody></table></div>
<p>두 경로 모두 전체 크기의 <code>d_concat</code>·<code>dx_n</code> global tensor를 없앴다. 네 입력 가중치를 TMA descriptor로 직접 읽고 cat/transpose 복사를 피한다. ring은 <b>global memory에 할당</b>하며 L2에 머물도록 유도한다. HBM 접근이 전혀 없다는 뜻은 아니다. slot별 generation counter와 release/acquire, TMA 쓰기 완료 확인으로 재사용을 동기화한다.</p>
<h3>정확도와 검증</h3>
<p><b>각 선택 경로 6개 구성·24개 비교 통과:</b> L64/384/768 × dropout 0/25%, 최초 결과·입력 변경 graph replay 2회·가중치 변경 replay. 허용 오차 dx 2e-5, dW 5e-4, LN 파라미터 5e-6을 유지했다. 전체 backward 11개 출력도 확인했다.</p>
<p id="b7-sanitizer-status"><b>각 경로 sanitizer 7건 통과:</b> memcheck/racecheck/synccheck L64·384, 독립 host 입력 unfiltered initcheck L64. 이전 전체 fixture의 upstream 진단까지 해결했다는 주장은 아니다. ptxas stack/spills는 0이며, 0이 아닌 후보는 실행 전에 거부한다.</p>
<h3>NCU: L2 사용률을 세부 경로로 분해</h3>
<div class="table-wrap"><table><thead><tr><th>동일한 breakdown 측정</th><th>L384 재계산</th><th>L768 순환 버퍼</th></tr></thead><tbody><tr><td>Profile duration</td><td>365.728 µs</td><td>1426.688 µs</td></tr><tr><td>L2 구획 간 통신 경로 (LTC fabric)</td><td>78.63%</td><td>90.27%</td></tr><tr><td>L2 data sectors</td><td>53.23%</td><td>57.04%</td></tr><tr><td>L2 tag sectors</td><td>45.30%</td><td>56.65%</td></tr></tbody></table></div>
<p><b>L768에서 약 90%인 항목은 L2 구획 사이의 통신 경로다. 전체 L2 데이터 대역폭이나 알고리즘 SoL90과 같지 않다.</b> NCU throughput은 여러 하위 지표 중 가장 높은 값을 보고하므로 breakdown을 별도로 수집했다. 원래 full profile은 1.445 ms, SM 38.43%, 관측 DRAM 2.648 GB였다. 두 profile의 수치를 같은 실행으로 섞지 않는다.</p>
<p>재사용 가중치 전송과 중간 버퍼 트래픽까지 포함된 지표이므로, 필수 트래픽과 병목별 상한을 따로 검증해야 한다. 고정 1.7× 목표 시간은 L384 ≤361.769 µs, L768 ≤1388.857 µs다. <a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/">NVIDIA NCU 지표 정의</a></p>
<details><summary>후속 실험과 현재 선택 이유</summary><p>동적 dX 큐와 큰 가중치 TMA를 결합한 후보도 반복 정확도·sanitizer를 통과했다. L768 1426.576µs로 현재 1429.136µs와 차이가 작아 더 단순한 경로를 유지한다. 작은 ring, 단일 dW 누적 구간, 추가 캐시 힌트, 3 CTA/SM, 4 warpgroup, compiler register 설정 등은 안정적인 추가 이득을 주지 못했다. cluster DSM GLU 전달은 초기 정확도를 통과했지만 느렸다. 최적화는 계속 진행 중이다.</p></details>
<p>추가 통제 실험: 동일한 버퍼 주소로 dW/dX CTA 배분을 다시 비교해 L384 split 13, L768 split 20을 확인했다. TMA L2 promotion 64/128/256 B 및 없음은 유의미한 개선을 주지 못했다. 서로 다른 작업 버퍼 주소만으로 약 1% 차이가 관측되어, 작은 후보 차이는 확정 개선으로 취급하지 않는다.</p>
<h3>후속 최적화 실험 · 선택 경로 유지</h3>
<p>각 행은 같은 실행에서 같은 작업 버퍼 주소를 공유하고, 순서를 바꿔 600회 측정한 중앙값이다. 여기의 기준은 위 표의 기존 Triton이 아니라 <b>현재 선택한 CUDA 커널</b>이다. 서로 다른 실험 행의 절대 시간을 직접 비교하지 않는다.</p>
<div class="table-wrap"><table><thead><tr><th>시도</th><th>L</th><th>선택 CUDA µs</th><th>후보 µs</th><th>판단</th></tr></thead><tbody>
<tr><td>dX 두 행 타일의 가중치 공유</td><td>768</td><td>1456.512</td><td>1464.736</td><td>미채택 · ring/CTA/LN 재튜닝 후에도 이득 없음</td></tr>
<tr><td>x_n TMA multicast · 8 CTA</td><td>384</td><td>364.880</td><td>640.768</td><td>미채택 · 공유 비용이 큼</td></tr>
<tr><td>x_n TMA multicast · 8 CTA</td><td>768</td><td>1422.240</td><td>2052.688</td><td>미채택 · 별도 barrier로 B7과 겹쳐도 느림</td></tr>
<tr><td>다음 B7을 현재 dW WGMMA와 겹침</td><td>384</td><td>363.648</td><td>509.696</td><td>미채택</td></tr>
<tr><td>B7/dW 겹침 + ring 준비 즉시 알림</td><td>768</td><td>1461.744</td><td>1446.000</td><td>검증 중 · 반복 통제 실험에서는 0.6% 차이</td></tr>
</tbody></table></div>
<p>두 행 공유는 NCU L2 sector 수를 약 10% 줄였지만, 동기화 비용과 ring 지역성의 손해가 이득을 상쇄했다. multicast 분리 실험에서는 240 CTA 일반 배치 1484.880µs, 클러스터 배치만 적용 1481.408µs, multicast까지 적용 2070.640µs였다. 배치 자체보다 공유를 위한 대기·전송 비용이 컸다. 커널을 합치거나 읽기 횟수를 줄였다는 사실만으로 가속을 주장하지 않는다.</p>
<p>현재 선택한 두 CUDA 소스와 검증된 성능 수치는 그대로다. 새 후보의 작은 차이는 목표 달성이나 확정 개선으로 취급하지 않으며, 재현 정확도·sanitizer·전체 backward 검증을 거쳐야 한다.</p>
<p class="muted">재현 코드·원시 측정은 <a href="https://github.com/SanggeunParrk/miniworld-engine/tree/3f1c7a77a4af1a107257f87c24645255c294abb2/experiments/trimul_b7b12">miniworld-engine main의 실험 패키지</a>에 공개했다. 후속 미검증 후보는 선택 경로에 포함하지 않았다. 초기 약 1.15× → 1.38/1.37× → 1.60/1.54× → 현재 1.69/1.66×. 각 체크포인트는 자체 paired baseline과 비교한다.</p></section><!-- B7B12_END -->'''
# Both follow-up sanitizer sets and initchecks must exist before saying they passed.
assert all(v['result']=='passed' for v in json.loads((p/'ringqueuewtma128-sanitizers.json').read_text()).values())
assert 'ERROR SUMMARY: 0 errors' in (p/'ringqueuewtma128-initcheck-unfiltered-L64.log').read_text()
(site/'trimul.html').write_text(html[:a]+section+html[b:])
print('Updated only B7B12 section and wiring SVG')
