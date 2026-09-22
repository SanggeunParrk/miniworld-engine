from pathlib import Path
from html import escape
import json,shutil,hashlib,xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;a=r.parent/'anthropic_adoption_20260919';root=r.parent.parent
rows=json.loads((r/'final-results.json').read_text());prof=json.loads((r/'ncu-final-summary.json').read_text())
# Static, code-native architecture diagram. Operation ownership and HBM boundaries are explicit.
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 1130" role="img" aria-labelledby="title desc"><title id="title">TriMul B1–B4 CUDA · one cooperative kernel</title><desc id="desc">Two warpgroups per CTA share B1 inputs, accumulate dW in FP32 registers, compute dnorm and LayerNorm backward, prefetch the next tile, then reduce global partials inside the same cooperative launch.</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10" fill="#5b7585"/></marker></defs><style>text{font-family:system-ui,sans-serif;fill:#173548;font-size:17px}.t{font-size:24px;font-weight:700}.b{font-weight:700}.small{font-size:14px;fill:#506879}.edge{fill:none;stroke:#5b7585;stroke-width:2;marker-end:url(#arrow)}</style><rect width="1160" height="1130" fill="#f3f7f8"/><text x="35" y="42" class="t">B1–B4 · CUDA 단일 커널의 실제 배선</text><text x="35" y="70" class="small">H100 · C128 / packed H256 · dropout 25% · 132 CTAs × 256 threads · 각 tile은 64 rows</text>']
def box(x,y,w,h,title,lines,color='#fff'):
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#b8ccd5"/>')
 svg.append(f'<text x="{x+18}" y="{y+29}" class="b">{escape(title)}</text>')
 for i,line in enumerate(lines):svg.append(f'<text x="{x+18}" y="{y+55+i*25}" class="small">{escape(line)}</text>')
def edge(d):svg.append(f'<path d="{d}" class="edge"/>')
box(35,92,1090,102,'HBM · forward 저장값과 upstream gradient',['dy · gate · projection · x_n · normalized_tri · raw tri · output-LN mean/rstd/gamma · Wp','Forward의 융합·저장 정책 유지. RNG·할당·weight packing은 이 측정 밖.'])
svg.append('<rect x="25" y="216" width="1110" height="615" rx="16" fill="#eaf2f5" stroke="#6b8ba0" stroke-dasharray="7 5"/><text x="45" y="246" class="b">Persistent CTA · shared memory에서 타일을 재사용</text>')
box(55,268,800,105,'B1 · dropout + output gate 미분',['dy_eff = dy × dropout scale → BF16 dp, dg','128-bit mask 읽기와 dg 쓰기. dp는 HBM에 저장하지 않음.'],'#fff3d8')
edge('M455 194V268');box(890,268,215,105,'HBM output · dg',['나중 B9에서 사용','필수 출력은 유지']);edge('M855 320H890')
box(55,410,505,150,'WGMMA · dWg + dWp',['WG0: dWg 2 tiles + dWp 1 tile','WG1: dWp 3 tiles','각 warpgroup의 FP32 accumulator를','여러 row tile에 걸쳐 레지스터에 유지'],'#e3f4ec')
box(590,410,515,150,'WGMMA · dnorm + LN row statistics',['dnorm = BF16(dp @ Wp)','WG0 / WG1이 각각 128 channels 담당','GEMM fragment에서 LN backward 합산','dnorm도 shared memory 안에 유지'],'#e8edfc')
edge('M265 373V410');edge('M690 373V410')
box(590,600,515,137,'B4 · output LayerNorm backward',['채널 방향으로 읽어 dtri, dgamma, dbeta 계산','dtri는 TMA로 HBM에 출력','다음 입력 80 KiB의 TMA 전송과 겹침'],'#e8edfc');edge('M845 560V600')
box(55,600,505,137,'다음 타일 준비 · TMA overlap',['dy + gate + x_n + normalized_tri 미리 읽기','현재 LN epilogue가 읽지 않는 공간 재사용','projection / raw tri는 현재 타일이 끝난 뒤 읽기'],'#e2f3f5')
svg.append('<path d="M307 600V585H455V373" fill="none" stroke="#148771" stroke-dasharray="6 5" stroke-width="2" marker-end="url(#arrow)"/>')
svg.append('<text x="65" y="784" class="small">타일 반복 종료 → FP32 부분 dW와 dgamma/dbeta를 global workspace에 기록</text>')
box(55,858,505,112,'HBM · FP32 partial workspace',['132 CTAs 기준 약 25 MiB','입력 반복 읽기는 줄였지만 부분합 HBM은 존재'],'#fff')
box(590,858,515,112,'같은 launch · cooperative grid barrier',['모든 CTA가 부분합을 게시한 뒤','전체 CTA가 출력 원소를 나눠 최종 reduction'],'#f0eafa');edge('M310 831V858');edge('M560 913H590')
box(55,1000,1050,98,'최종 출력',['dg · dtri · BF16 dWg / dWp · FP32 dgamma / dbeta','FP32 dW 누적 → 마지막에 한 번 BF16 반올림. residual gradient는 기존 dy alias.'],'#e3f4ec');edge('M850 970V1000')
svg.append('</svg>');(r/'b1b4-shared.svg').write_text(''.join(svg));ET.parse(r/'b1b4-shared.svg')
# Reproducibility manifest: actual selected source and final benchmark.
files=['dual.cu','dual_primitives.cuh','dual.py','final_bench.py','final-results.json','validation.json','ncu-final-summary.json','memcheck-persistent.log','racecheck-persistent.log']
manifest={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in files};(r/'manifest.json').write_text(json.dumps(manifest,indent=2))
trs=[]
for z in rows:
 t=z['times'];b=t['B1-B4 Triton/cuBLAS']['median_us'];c=t['B1-B4 CUDA one kernel']['median_us'];fb=t['whole backward baseline']['median_us'];fc=t['whole backward new']['median_us']
 trs.append(f'<tr><td>L{z["L"]}</td><td>{b:.2f}</td><td>{c:.2f}</td><td><b>{b/c:.3f}×</b></td><td>{fb:.2f} → {fc:.2f}</td><td>{fb/fc:.3f}×</td><td>{b/1.7:.2f} µs 이하</td></tr>')
section='''<!-- B1B4_SHARED_BEGIN --><section class="panel" id="b1b4-shared"><span class="pill amber">최신 실측 · 1.7× 목표 미달</span><h2>B1–B4 CUDA · 입력 공유 + 단일 커널 reduction</h2><p><b>기존 Triton/cuBLAS 대비 L384 1.18×, L768 1.36×.</b> Anthropic의 TMA/WGMMA·fragment 구현을 계승하고 학습용 dW/dX 스케줄을 새로 구현했다. 이전의 느린 CUDA 시제품을 기준으로 목표 달성을 주장하지 않는다.</p><div class="table-wrap"><table><thead><tr><th>shape</th><th>기존 B1–B4 µs</th><th>새 CUDA µs</th><th>구간 속도</th><th>전체 backward µs</th><th>전체 속도</th><th>1.7×에 필요한 시간</th></tr></thead><tbody>'''+''.join(trs)+'''</tbody></table></div><p>node02 H100 · BF16 C128/H256 · dropout 25% · 동일 저장값/공통 mask · CUDA Graph 80회 × 20 교대 라운드. 전체 backward는 B1–B4만 교체했다. 나머지 공통 Triton 경로 한 곳은 cache miss로 3개 후보 fallback을 사용했으므로, 전체 시간은 나머지 경로까지 완전히 재튜닝한 결과가 아니다.</p><h3>어디가 합쳐졌나</h3><p>dp/dnorm의 중간 HBM 저장을 없애고, 같은 입력을 dW 타일마다 다시 읽지 않는다. 다음 타일 TMA를 LN epilogue와 겹치고, dW의 최종 합산도 cooperative launch 안에서 처리한다. <b>부분 dW의 global workspace는 남는다.</b></p><a href="assets/b1b4-shared.svg"><img src="assets/b1b4-shared.svg" alt="B1-B4 단일 CUDA 커널의 연산 배선과 HBM 경계" style="display:block;width:100%;height:auto;border-radius:12px"></a><h3>검증 및 현재 한계</h3><ul><li>L64/72/384/768: 수치 비교, dropout 0/1, 입력 변경 후 graph replay와 counter reset 통과.</li><li>전체 backward 11개 gradient 비교. 최종 큰 shape 최대 상대 L2 0.000286. 추가 입력 변경 검사 최대 0.000427.</li><li>L72 / 4 CTAs로 실제 여러 타일을 반복: memcheck 0 errors, racecheck 0 hazards.</li><li>NCU L384: HBM 활용률 47.6%, tensor 활성률 11.2%, active warps 12.5%. local load/store와 register spill은 0. 루프라인에 가깝다고 판단할 근거가 없다.</li><li>남은 병목: 255 registers/thread, dynamic shared 226 KiB로 CTA당 자원이 큼. 메모리 의존 대기와 barrier 비용이 남는다. 1.7×는 계속 미달이다.</li><li>실험용 전체 backward에 연결해 검증했으며 production 기본 배선·autotune cache로 승격하지 않았다.</li></ul><p><a href="assets/b1b4-shared-results.json">최종 paired 결과</a> · <a href="assets/b1b4-shared-validation.json">수치 검증</a> · <a href="assets/b1b4-shared-ncu.json">NCU</a> · <a href="assets/b1b4-shared-README.md">설계·실패한 시도·재현</a> · <a href="assets/b1b4-shared-manifest.json">SHA-256</a></p><p class="muted">아래 B1–B4 표들은 이전 시제품의 실험 기록이다. 현재 결과는 이 섹션을 기준으로 본다.</p></section><!-- B1B4_SHARED_END -->'''
site=a/'site'/'dist';html=(site/'trimul.html').read_text();import re
html=re.sub(r'<!-- B1B4_SHARED_BEGIN -->.*?<!-- B1B4_SHARED_END -->','',html,flags=re.S)
html=html.replace('<main>','<main>'+section,1)
if 'href="#b1b4-shared"' not in html:html=html.replace('<nav>','<nav><a href="#b1b4-shared">현재 B1–B4 CUDA</a>',1)
for p in (site/'trimul.html',a/'web'/'trimul.html',root/'ANTHROPIC_TRIMUL.html'):p.write_text(html)
assets={'final-results.json':'b1b4-shared-results.json','validation.json':'b1b4-shared-validation.json','ncu-final-summary.json':'b1b4-shared-ncu.json','README.md':'b1b4-shared-README.md','manifest.json':'b1b4-shared-manifest.json','b1b4-shared.svg':'b1b4-shared.svg'}
for src,dst in assets.items():shutil.copyfile(r/src,site/'assets'/dst)
print('Report, SVG and three HTML copies updated; SVG XML validated.')
