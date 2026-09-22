"""Publish the static audit state separately from prior measured results."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from html import escape

r = Path(__file__).resolve().parent
a = r.parent / 'anthropic_adoption_20260919'
root = r.parent.parent
site = a / 'site' / 'dist'

svgpath = r / 'b1b4-shared.svg'
backup = r / 'b1b4-shared-before-audit.svg'
if not backup.exists():
    shutil.copyfile(svgpath, backup)
svg = backup.read_text().replace('0 0 1160 1130', '0 0 1160 1230').replace('height="1130"', 'height="1230"')
svg = svg.replace('HBM · FP32 partial workspace', 'Global · FP32 partial workspace')
svg = svg.replace('입력 반복 읽기는 줄였지만 부분합 HBM은 존재', 'L2/HBM 경유 · 52 MB 전부 HBM인 것은 아님')
svg = svg.replace('</svg>', '''<rect x="35" y="1120" width="1090" height="86" rx="12" fill="#fff3d8" stroke="#d9bb6f"/>
<text x="53" y="1149" class="b">2026-09-20 · 시간 분해 준비 완료 / 새 GPU 측정 전</text>
<text x="53" y="1176" class="small">B1–B4 본체 → partial 저장 → grid 대기 → reduction을 별도 측정. 기존 융합·저장 정책 유지.</text>
</svg>''')
ET.fromstring(svg)
svgpath.write_text(svg)

section = '''<!-- B1B4_AUDIT_BEGIN --><section class="panel" id="b1b4-audit">
<span class="pill amber">2026-09-20 · 정적 점검 / 새 실측 전</span>
<h2>B1–B4: 1.7× 목표를 위한 측정 준비</h2>
<p>현재 커널 작업에 할당된 GPU가 없어 새 컴파일·벤치·NCU는 아직 실행하지 않았다.
node02의 기존 학습 GPU 4장은 유지한다. 아래 표의 성능 수치는 이전 실측이다.</p>
<div class="table-wrap"><table><thead><tr><th>점검</th><th>확인한 내용</th></tr></thead><tbody>
<tr><td>기준 소스</td><td>선택 버전은 dual.cu. dual_dspref.cu는 별도 L1 prefetch 실험.</td></tr>
<tr><td>기존 NCU</td><td>L384 HBM 47.6%, tensor 11.2% (elapsed 기준). 새 profile은 대기.</td></tr>
<tr><td>partial 왕복</td><td>global 52.45 MB. 기존 DRAM은 이론 최소보다 7.68 MB 많음. 전체를 HBM 비용으로 계산할 수 없음.</td></tr>
<tr><td>작은 shape 검사</td><td>기존 wrapper가 L64/count66·132를 막음. 새 harness는 empty CTA를 검사하도록 준비.</td></tr>
<tr><td>수치 검증</td><td>dg bit-exact, dtri 2e-5, dγ/dβ 5e-6, dW 5e-4. 입력 변경 후 graph replay 포함.</td></tr>
<tr><td>다음 실험</td><td>본체 / partial 저장 / grid 대기 / reduction 시간 분해. PART_ONLY=1·2와 계측 오버헤드 비교.</td></tr>
</tbody></table></div>
<p><b>성능 개선·1.7× 달성 주장은 아직 없다.</b> GPU 측정 후 첫 최적화 후보를 선택한다.</p>
<details><summary>상세 분석·메모리 배치·가설·재현 명령 (9개 항목)</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'''
section += escape((r / 'AUDIT_20260920.md').read_text())
section += '''</pre></details><p><a href="assets/b1b4-audit-20260920.md">분석 원문</a> ·
<a href="assets/b1b4-audit-manifest.json">준비 파일 SHA-256</a></p></section><!-- B1B4_AUDIT_END -->'''
html = (site / 'trimul.html').read_text()
html = re.sub(r'<!-- B1B4_AUDIT_BEGIN -->.*?<!-- B1B4_AUDIT_END -->', '', html, flags=re.S)
html = html.replace('<main>', '<main>' + section, 1)
if 'href="#b1b4-audit"' not in html:
    html = html.replace('<nav>', '<nav><a href="#b1b4-audit">현재 분석·측정 준비</a>', 1)
for path in (site / 'trimul.html', a / 'web' / 'trimul.html', root / 'ANTHROPIC_TRIMUL.html'):
    path.write_text(html)
files = ['dual.cu', 'dual_dspref.cu', 'dual.py', 'dual_primitives.cuh', 'core.py',
         'prepare_timing.py', 'dual_timing.cu', 'dual_dspref_timing.cu',
         'dual_experiment.py', 'check_experiment.py', 'measure_experiment.py',
         'profile_experiment.py', 'AUDIT_20260920.md']
manifest = dict(status='static audit; CUDA compile/GPU measurements pending',
                sha256={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in files})
(r/'audit-manifest.json').write_text(json.dumps(manifest,indent=2))
for src, dest in [('b1b4-shared.svg','b1b4-shared.svg'),
                  ('AUDIT_20260920.md','b1b4-audit-20260920.md'),
                  ('audit-manifest.json','b1b4-audit-manifest.json')]:
    shutil.copyfile(r/src,site/'assets'/dest)
print('Audit section and SVG updated; previous measured results preserved.')
