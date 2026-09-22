"""Publish the measured matched-cuEq comparison without altering other sections."""
from pathlib import Path
import hashlib
import json

p = Path(__file__).resolve().parent
site = p.parent / 'anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows = [json.loads((p / ('cueq-training-L%d.json' % n)).read_text()) for n in (384, 768)]
for row in rows:
    for scope in ('forward_backward', 'forward_training'):
        assert all(len(t['samples_us']) == 600 for t in row['times'][scope].values())
    for checked in (row['checks'], row['replay_checks']['forward_backward']):
        for path in checked.values():
            assert len(path) == 12
            assert all(v['finite'] and v['relative_l2'] <= v['limit'] for v in path.values())

tr = []
md = ['# Matched cuEquivariance bidirectional training comparison', '',
      '2026-09-20, node02 H100 80GB, BF16, B1/C128/H128 per direction, dropout 25%.',
      'cuEquivariance torch 0.9.1 / ops-cu12 0.10.0, ONDEMAND tuning, static fullgraph compile.',
      'Every timed path uses an explicit single-call CUDA graph. 600 alternating samples/path.', '',
      '| Scope | L | cuEq compiled ms | Current CUDA ms | Speedup |',
      '| --- | --- | --- | --- | --- |']
for scope, label in [('forward_backward', 'Forward + backward'), ('forward_training', 'Training forward')]:
    for row in rows:
        n = row['L']
        ts = row['times'][scope]
        ref, new = ts['cueq_compile']['median_us'], ts['cuda']['median_us']
        tr.append('<tr><td>%s</td><td>%d</td><td>%.3f</td><td>%.3f</td><td>%.3f×</td><td>%.2f%%</td></tr>' %
                  (label, n, ref/1000, new/1000, ref/new, 100*(1-new/ref)))
        md.append('| %s | %d | %.3f | %.3f | %.3fx |' % (label, n, ref/1000, new/1000, ref/new))

md += ['', '## Scope and numerical differences', '',
       '- Matched cuEq primitive composition: input LN, gated dual GEMM, outgoing + incoming contractions, shared 256-channel output LN, projection/gate, supplied dropout and residual. The public single-direction TMU called twice has different normalization semantics.',
       '- Current CUDA: Anthropic-derived saved forward, unchanged Triton/cuBLAS B1-B6, CUDA B7-B12. Separate Claude B1-B4 work is excluded.',
       '- Fresh saves and all 11 gradients are produced inside every timed training call. Weight packing/layout conversions are included for both paths. No optimizer, RNG generation, CPU/autograd dispatch or compilation time.',
       '- Forward timings preserve training saves and autograd; total timings are directly measured, not sums of independent kernel timings.',
       '- cuEq keeps its own BF16 rounding and saves. Cross-backend gradients do not meet the strict same-saves CUDA error contract. This comparison is mathematical equivalence with reported differences, not bitwise equivalence or convergence validation.',
       '- Eager cuEq is a secondary diagnostic: separate BF16 rounding at sigmoid/multiply/dropout/residual gives ~0.25% forward relative L2. Primary compiled comparison uses a 0.1% forward and 1% gradient cross-backend bound. Existing CUDA same-saves bounds remain unchanged.', '',
       '| L | Compiled forward relative L2 | Max compiled gradient relative L2 | cuEq eager total ms | Current prepacked total ms |',
       '| --- | --- | --- | --- | --- |']
for row in rows:
    checks = row['checks']['cueq_compile']
    forward = checks['forward']['relative_l2']
    grad = max(v['relative_l2'] for k, v in checks.items() if k != 'forward')
    ts = row['times']['forward_backward']
    md.append('| %d | %.8g | %.8g | %.3f | %.3f |' %
              (row['L'], forward, grad, ts['cueq_eager']['median_us']/1000,
               ts['cuda_prepacked']['median_us']/1000))
md += ['', '## Sources', '',
       '- NVIDIA public API: https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html',
       '- Benchmark: compare_cueq_training.py; source and benchmark SHA-256 are recorded in each JSON.',
       '- Raw results: cueq-training-L384.json, cueq-training-L768.json.',
       '- L384 source: front_prefetch_lnpair_storepipe.',
       '- L768 source: front_ring96_cache3_glu_ahead_u4_early_writer32 (validated experimental candidate, not production default).', '']
report = '\n'.join(md)
(p / 'CUEQ_TRAINING_COMPARISON.md').write_text(report)
(site / 'assets/b7b12-cueq-training.md').write_text(report)
for row in rows:
    name = 'cueq-training-L%d.json' % row['L']
    (site / 'assets' / ('b7b12-' + name)).write_bytes((p / name).read_bytes())

section = '''<!-- CUEQ_TRAINING_BEGIN --><section class="panel" id="cueq-training">
<span class="pill green">2026-09-20 · cuEquivariance와 실제 fwd+bwd 비교</span>
<h2>튜닝·compile한 cuEq 대비 학습 1.57× / 1.52×</h2>
<p><b>동일한 양방향 수식의 직접 측정:</b> cuEq 입력 LN·gated GEMM·출력 LN과 두 contraction을 조합해 shared 256-channel 출력 LN을 맞췄다. <a href="https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html">공개 단방향 TMU</a> 두 번 호출과는 정규화 수식이 다르다.</p>
<div class="table-wrap"><table><thead><tr><th>범위</th><th>L</th><th>cuEq tuned + compile ms</th><th>현재 CUDA ms</th><th>속도 배율</th><th>시간 감소</th></tr></thead><tbody>''' + ''.join(tr) + '''</tbody></table></div>
<p>H100 node02 · BF16 · batch 1 · C128 · 방향별 H128 · dropout 25%. cuEq torch 0.9.1 / ops-cu12 0.10.0, ONDEMAND 튜닝과 static fullgraph compile. <b>모든 경로 explicit CUDA graph, 경로당 600회 교대 측정</b>. 전체 시간은 fwd와 bwd를 한 번에 측정한 중앙값이며 단독 시간의 합이 아니다.</p>
<p>현재 구현 = <b>Anthropic 파생 학습 forward + 기존 Triton/cuBLAS B1–B6 + 새 CUDA B7–B12</b>. L768은 검증된 GLU lookahead/writer32 후보를 사용했다. 아래 기존 체크포인트 표와는 버전·측정 범위가 다르다. Claude가 별도 개발하는 B1–B4는 포함하지 않았다.</p>
<p>양쪽 모두 매 호출에서 저장값을 새로 생성하고 입력·가중치·LN 파라미터의 11개 gradient를 계산한다. <b>매 스텝 weight 재배열 비용도 포함</b>. optimizer·RNG 생성·CPU autograd dispatch·컴파일은 제외한다. Training forward 행도 backward용 저장을 유지한다.</p>
<div class="notice amber"><b>BF16 수치 차이:</b> compiled cuEq의 forward 상대 L2는 L384 0.00404%, L768 0.00482%. 11개 gradient 중 최대 상대 L2는 각각 0.8053%, 0.5119%다. 수식은 같지만 중간 반올림·저장이 달라 strict same-saves 검증과 구분하며, bitwise 일치나 학습 수렴을 검증했다는 뜻은 아니다.</div>
<details><summary>eager 및 weight 사전 packing 보조 수치</summary><p>cuEq eager 전체: L384 2.900ms, L768 10.997ms. 현재 구현에서 weight packing을 제외하면 1.287ms / 5.089ms. 주 비교는 더 빠른 tuned + compiled cuEq와 packing 포함 CUDA다. eager forward는 개별 BF16 연산 반올림으로 약 0.25% 차이가 있어 별도 진단으로 기록했다.</p></details>
<p><a href="assets/b7b12-cueq-training.md">조건·검증 보고서</a> · <a href="assets/b7b12-cueq-training-L384.json">L384 원시 측정</a> · <a href="assets/b7b12-cueq-training-L768.json">L768 원시 측정</a></p>
</section><!-- CUEQ_TRAINING_END -->'''
path = site / 'trimul.html'
html = path.read_text()
if '<!-- CUEQ_TRAINING_BEGIN -->' in html:
    start = html.index('<!-- CUEQ_TRAINING_BEGIN -->')
    end = html.index('<!-- CUEQ_TRAINING_END -->') + len('<!-- CUEQ_TRAINING_END -->')
    html = html[:start] + section + html[end:]
else:
    html = html.replace('<!-- B7B12_BEGIN -->', section + '<!-- B7B12_BEGIN -->', 1)
if '<a href="#cueq-training">cuEq 학습 비교</a>' not in html:
    html = html.replace('<nav>', '<nav><a href="#cueq-training">cuEq 학습 비교</a>', 1)
path.write_text(html)
print('Added matched-cuEq comparison; existing kernel sections preserved.')
