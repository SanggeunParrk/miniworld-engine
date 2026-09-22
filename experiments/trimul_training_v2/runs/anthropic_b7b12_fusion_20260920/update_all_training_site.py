"""Render the complete training integration comparison and provenance."""
from pathlib import Path
import hashlib
import json

p = Path(__file__).resolve().parent
site = p.parent / 'anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows = [json.loads((p/('all-training-L%d.json' % n)).read_text()) for n in (384, 768)]
replays = [json.loads((p/('all-training-replay-L%d.json' % n)).read_text()) for n in (384, 768)]
for row, replay in zip(rows, replays):
    assert replay['changed_input_weight_upstream'] and replay['restored_forward_exact'] and replay['counters_zero']
    for checks in [*row['checks'].values(), *row['replay_checks'].values(), replay['changed'], replay['restored']]:
        assert all(v['finite'] and v['relative_l2'] <= v['limit'] for v in checks.values())
    assert all(len(t['samples_us']) == 600 for scope in row['times'].values() for t in scope.values())
    assert row['metadata']['benchmark_sha256'] == hashlib.sha256((p/'compare_all_training.py').read_bytes()).hexdigest()

lines = ['# Complete bidirectional TriMul training: previous Miniworld versus current', '',
         '2026-09-20. H100 node02; batch1/C128/H128 per direction; BF16 weights/activations, FP32 LN parameters; supplied dropout25% and pair mask.',
         'All variants use explicit single-call CUDA Graph; 3 alternating blocks of 200 samples, pooled median.', '',
         '| Scope | L | Previous Triton ms | Previous H100 ms | Current all CUDA ms | vs Triton | vs H100 | cuEq compiled ms | vs cuEq |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
html_rows = []
cueq_rows = []
for scope, label in [('forward_training', '학습 Forward'), ('forward_backward', 'Forward + backward')]:
    for row in rows:
        ts = {k: v['median_us']/1000 for k,v in row['times'][scope].items()}
        n = row['L']
        a, b, c, q = [ts[k] for k in ('old_triton', 'old_h100', 'current_all', 'cueq_compile')]
        html_rows.append('<tr><td>%s</td><td>%d</td><td>%.3f</td><td>%.3f</td><td><b>%.3f</b></td><td>%.3f×</td><td>%.3f×</td></tr>' % (label,n,a,b,c,a/c,b/c))
        lines.append('| %s | %d | %.3f | %.3f | %.3f | %.3fx | %.3fx | %.3f | %.3fx |' % (label,n,a,b,c,a/c,b/c,q,q/c))
        if scope == 'forward_backward':
            cueq_rows.append('<tr><td>%d</td><td>%.3f</td><td>%.3f</td><td>%.3f×</td></tr>' % (n,q,c,q/c))

lines += ['', '## What is connected', '',
    '- Forward: selected Anthropic-derived fused input LN + gate/projection, packed bidirectional cuBLAS contractions, fused output, all training saves.',
    '- Backward B1-B4: dual_ln_prefetch, selected and retained by the Claude SoL report.',
    '- B5-B6: unchanged cuBLAS contraction backward.',
    '- B7-B12: L384 front_prefetch_lnpair_storepipe; L768 front_ring96_cache3_glu_ahead_u4_early_writer32.',
    '- Fresh forward saves are rebound into both backward plans each call. All 11 gradients and per-step weight layout conversions, including B1 Wp transpose, are timed.',
    '- Excludes optimizer, stochastic-mask generation, CPU/autograd dispatch and compilation. This is one TriMul operation, not full-model training.',
    '- The inference-only K1/K3 payload does not provide these training saves and is not included. Forward rows are training-mode forward with saves and dropout, not inference.', '',
    '## Baselines', '',
    '- Previous Miniworld algorithms are rerun from the retained routes in the current checkout, with output_backend=triton. This is not a recreation of an old environment or whole historical commit.',
    '- Previous Triton/H100 and cuEq use static fullgraph compile, dynamic=False, Inductor CUDA graphs disabled; explicit CUDA Graph wraps every timed path.',
    '- Old H100 enables front/f567/dual_bwd with the previously measured configs in trimul_sm90_round2_20260917/module/measured-configs-L*.json. Actual resolver calls and configs are in the JSON.',
    '- Triton uses the available cache and default 24 heuristic candidates on a miss. The input-dual backward misses its tuned cache. This is not a full-grid retuning claim.',
    '- cuEq uses the same shared-256-channel-LN bidirectional primitive composition, ONDEMAND tuning, torch0.9.1 / ops-cu12 0.10.0.', '',
    '## Validation and numerical scope', '',
    '- B1 and B7 regional contracts are separately checked with identical inputs. B1 dg is bit-exact. Existing tight per-output limits remain unchanged.',
    '- Applying the B7-only limits to a whole chain after changing B1 initially failed for dx/input-LN gradients. B1 changes propagate downstream. The combined chain is assessed with the existing B1 full-backward contract (relative L2 <=5e-4), while preserving the stricter regional checks.',
    '- Combined eager and captured outputs: forward exact vs identical saved-forward reference; all 11 gradients <=5e-4. Maximum gradient relative L2: L384 2.9234e-4; L768 4.9876e-4.',
    '- The exact measured combined graph also passed after modifying input x, an input weight and upstream dy in place. Restoring the inputs reproduces the forward exactly; both cooperative counters reset to zero.',
    '- Cross-backend comparisons use their own BF16 rounding/saves and are not bitwise. Old Miniworld gradient differences versus the saved-forward reference peak at0.0727%/0.0600%; cuEq peaks at0.8054%/0.5119%. No convergence claim.', '',
    '## Results interpretation', '',
    '- Complete training is 1.381x/1.401x faster than old Triton and 1.298x/1.304x faster than old H100.',
    '- Training forward versus old H100: L384 is0.7% slower, L768 is2.1% faster; the main gain is backward.',
    '- Adding B1-B4 to the previous B7-only experiment reduces total times from1.296 to1.168ms and5.158 to4.607ms (same run), a further1.110x/1.119x.',
    '- With both backward regions included, cuEq total comparison is1.747x/1.729x. This supersedes the earlier B7-only current row for complete-integration claims.', '',
    '## Reproduction and artifacts', '',
    '- compare_all_training.py --length384 or --length768 (space before the number in actual CLI). Use the existing anthropic_adoption_20260919/env.sh environment.',
    '- validate_all_replay.py --length384 or --length768 validates the exact captured callable with changed inputs; it does not overwrite timing records.',
    '- all-training-L*.json:600 raw samples/path, all block medians, precision checks, source hashes and old SM90 configs.',
    '- all-training-replay-L*.json:changed/restored-input checks.', '']
report = '\n'.join(lines).replace('--length384','--length 384').replace('--length768','--length 768')
(p/'ALL_TRAINING_COMPARISON.md').write_text(report)
(site/'assets/trimul-all-training.md').write_text(report)
for n in (384,768):
    for stem in ('all-training', 'all-training-replay'):
        name = '%s-L%d.json' % (stem,n)
        (site/'assets'/('trimul-'+name)).write_bytes((p/name).read_bytes())

section = '''<!-- ALL_TRAINING_BEGIN --><section class="panel" id="all-training">
<span class="pill green">2026-09-20 · 학습 CUDA 전체 연결 · B1–B4 포함</span>
<h2>이전 H100 경로 대비 전체 학습 약 1.30×</h2>
<p><b>Anthropic 파생 학습 forward → B1–B4 CUDA → cuBLAS B5–B6 → B7–B12 CUDA</b>를 한 학습 호출로 연결했다. 양방향 TriMul 한 개의 실측이며 모델 전체 step 시간이 아니다.</p>
<div class="table-wrap"><table><thead><tr><th>범위</th><th>L</th><th>이전 Triton ms</th><th>이전 자체 H100<br>Anthropic 미포함 ms</th><th>현재 전체 연결<br>Anthropic 파생 fwd ms</th><th>Triton 대비</th><th>H100 대비</th></tr></thead><tbody>''' + ''.join(html_rows) + '''</tbody></table></div>
<p>이전 H100 열은 <code>output_backend=triton</code>에 자체 CuTe <code>front/f567/dual_bwd</code>만 활성화한 경로이며 <b>Anthropic 커널을 포함하지 않는다</b>. Anthropic 파생 forward는 현재 전체 연결 열에 포함된다.</p>
<p><b>Forward는 이전 H100과 거의 같다:</b> L384 0.7% 느림, L768 2.1% 빠름. 전체 이득은 주로 backward에서 나온다. 위 Forward는 dropout과 backward용 저장을 포함한 <b>학습 forward</b>다. 별도 개발 중인 추론 전용 K1/K3 payload의 시간을 섞지 않았다.</p>
<p>H100 node02 · BF16 · B1/C128/방향별 H128 · dropout25%. 이전 경로와 cuEq는 static fullgraph compile, 모든 경로 explicit CUDA graph600회 교대 측정. <b>매 호출 새 저장값·11개 gradient·weight 재배열(B1 Wp transpose 포함)</b>을 계산했다. optimizer·RNG 생성·CPU dispatch·컴파일은 제외한다.</p>
<h3>cuEquivariance 비교도 B1–B4를 포함해 갱신</h3>
<div class="table-wrap"><table><thead><tr><th>L</th><th>cuEq tuned + compile fwd+bwd ms</th><th>현재 전체 연결<br>Anthropic 파생 fwd ms</th><th>속도 배율</th></tr></thead><tbody>''' + ''.join(cueq_rows) + '''</tbody></table></div>
<p>cuEq torch0.9.1 / ops0.10.0, ONDEMAND 튜닝. 동일한 공유256채널 LN 양방향 연산 조합이다. B7만 연결한 직전 경로는 같은 실행에서1.296/5.158ms였고, B1–B4 추가 후1.168/4.607ms로 줄었다.</p>
<details><summary>연결·정확도·이전 경로의 범위</summary>
<p>B1–B4는 Claude 최종 보고서의 선택안 <code>dual_ln_prefetch</code>. B7–B12는 L384 <code>front_prefetch_lnpair_storepipe</code>, L768 <code>front_ring96_cache3_glu_ahead_u4_early_writer32</code>. 두 CUDA plan 모두 이번 forward의 저장값으로 descriptor를 다시 묶는다. Production 기본 dispatch를 변경했다는 뜻은 아니다.</p>
<p>B1/B7은 동일 구간 입력에서 기존 엄격한 허용오차를 각각 통과했다. B1까지 바꾼 전체 체인에는 기존 전체-backward 기준5e-4를 적용했고, 11개 gradient의 최대 상대L2는0.0292%/0.0499%다. 구간 전용 기준과 혼동하지 않는다. 입력·가중치·upstream을 변경한 동일 graph replay도 통과했고 복원 후 forward가 정확히 재현됐다.</p>
<p>이전 Miniworld는 보존된 알고리즘 경로를 같은 환경에서 재실행했다. H100은 기존 측정된 front/f567/dual_bwd 설정을 사용했다. Triton cache miss는 기본24개 heuristic 후보이며 전체 config 재튜닝은 아니다. cuEq와는 BF16 중간 반올림·저장이 달라 gradient 상대L2가 최대0.8054%/0.5119% 차이 난다.</p>
</details>
<p><a href="assets/trimul-all-training.md">전체 조건·배선·검증 보고서</a> · <a href="assets/trimul-all-training-L384.json">L384 원시 측정</a> · <a href="assets/trimul-all-training-L768.json">L768 원시 측정</a></p>
</section><!-- ALL_TRAINING_END -->'''
path = site/'trimul.html'
html = path.read_text()
if '<!-- ALL_TRAINING_BEGIN -->' in html:
    a = html.index('<!-- ALL_TRAINING_BEGIN -->')
    b = html.index('<!-- ALL_TRAINING_END -->')+len('<!-- ALL_TRAINING_END -->')
    html = html[:a]+section+html[b:]
else:
    html = html.replace('<!-- CUEQ_TRAINING_BEGIN -->', section+'<!-- CUEQ_TRAINING_BEGIN -->',1)
if '<a href="#all-training">전체 학습 비교</a>' not in html:
    html = html.replace('<nav>', '<nav><a href="#all-training">전체 학습 비교</a>',1)
html = html.replace('<h2>튜닝·compile한 cuEq 대비 학습 1.57× / 1.52×</h2>',
                    '<h2>이전 중간 측정 · B1–B4 제외: cuEq 대비 1.57× / 1.52×</h2>',1)
path.write_text(html)
print('Complete training comparison added; earlier partial result labelled as history.')
