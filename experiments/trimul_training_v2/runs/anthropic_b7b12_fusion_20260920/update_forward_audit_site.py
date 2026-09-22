"""Publish the verified forward correction without rewriting older evidence."""
from pathlib import Path
import hashlib
import json

p = Path(__file__).resolve().parent
site = p.parent/'anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows = [json.loads((p/('forward-corrected-L%d.json'%n)).read_text()) for n in (384,768)]
traces = [json.loads((p/('forward-audit-L%d.json'%n)).read_text()) for n in (384,768)]
html_rows, trace_rows, config_rows = [], [], []
md = ['# Training forward audit and correction', '',
      'H100 node02; BF16; batch1/C128/H128 per direction; dropout25%; pair mask, residual, all backward saves and live weight packing included.',
      'All paths explicit CUDA Graph. Old routes static fullgraph compile. Six paths alternated for 600 event samples each. No RNG, optimizer, CPU dispatch or compilation time.', '',
      '| L | Old Triton ms | Old H100 ms | Reported adapter ms | Fixed packing ms | Fixed + selected configs ms | Speedup vs old H100 |',
      '|---|---|---|---|---|---|---|']
for row, trace in zip(rows,traces):
    n = row['L']
    assert all(row['checks'].values())
    assert all(len(t['samples_us']) == 600 for t in row['times'].values())
    for name,h in row['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == h, name
    ts = {k:v['median_us']/1000 for k,v in row['times'].items()}
    a,b,c,d,e = [ts[k] for k in ('old_triton','old_h100','reported_current','packing_fixed','corrected_tuned')]
    html_rows.append('<tr><td>%d</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td><b>%.3f</b></td><td>%.3f×<br>시간 %.1f%% 감소</td></tr>'%(n,a,b,c,d,e,b/e,(1-e/b)*100))
    md.append('| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3fx |'%(n,a,b,c,d,e,b/e))
    config_rows.append('<tr><td>%d</td><td>%s</td><td>%s</td></tr>'%(n,tuple(row['corrected_K1']),tuple(row['corrected_K3'])))
    def dur(path, needles):
        return sum(t['total_us_per_replay'] for t in trace['traces'][path] if any(s in t['name'] for s in needles))
    for label,old,new in [
        ('입력 LN + front',dur('old_h100',('layer_norm_fwd_fused','FrontSingleWarpgroup')),dur('current',('mw_saved_front',))),
        ('출력 LN + projection/gate',dur('old_h100',('_ln_mat_kernel','ParityF567')),dur('current',('mw_k3_train',)))]:
        trace_rows.append('<tr><td>%d</td><td>%s</td><td>%.1f</td><td>%.1f</td><td>%.1f</td></tr>'%(n,label,old,new,old-new))
    for stem in ('forward-audit','forward-corrected','training-k3-audit'):
        name='%s-L%d.json'%(stem,n)
        (site/'assets'/('trimul-'+name)).write_bytes((p/name).read_bytes())

md += ['', '## Findings', '',
    '- Actual graph traces confirm old H100 uses our CuTe front/f567 and separate input/output LN. It does not contain Anthropic. Current uses mw_saved_front and mw_k3_train.',
    '- The reported current adapter launched six copies, two concatenations and one stack conversion per call. Old static compile merged its packing. Untimed-prepack control removes about24us from the current adapter.',
    '- Corrected adapter rebuilds live weights every call into one contiguous allocation, exposing all layout conversions to one compiled packing operation. It retains the original shapes through views, and passes original Wg directly to K3 instead of transposing it twice.',
    '- L768 had used the L384 K1 schedule. The recorded L768 schedule is restored, but its effect was small and varied across paired runs. No claim of a large gain from this setting.',
    '- All24 feasible configurations in the existing saved-training K3 search space were compiled and checked at both lengths. Five screening leaders plus the starting schedule were confirmed with600 alternating samples. L384 selects (1,128,4,2,232,1); L768 retains (2,64,4,1,232,1). This is exhaustive only within the existing bounded space.',
    '- Both K3 output and BF16 saves must be bit-exact to the starting schedule; FP32 statistics require relative L2<=1e-6. All admitted candidates satisfy these tests.',
    '- Packing correction preserves every output and saved tensor bit-for-bit. Corrected graph was replayed after modifying x, WL and Wg: all outputs/saves match a fresh call exactly. Restoring inputs restores exact forward output.',
    '- Current learned-training derivative is not the upstream inference-only K1/K3 payload. Backward saves/dropout exist here; old H100 also saves its backward intermediates. Saving alone is not a sufficient explanation for the small relative improvement.',
    '- The raw pre-fix trace shows core region savings around16us/56us at L384/L768. Packing explains the erased margin at384; it does not hide a huge forward improvement.',
    '- This correction remeasures training FORWARD. Earlier directly measured full forward+backward results remain historical pre-correction results. Do not estimate a new full total by subtracting standalone forward times.',
    '- Only the experimental adapter/benchmark changes here; no production default dispatch change.', '',
    '## Artifacts', '',
    '- audit_training_forward.py / forward-audit-L*.json: actual old/new kernel traces and initial timing.',
    '- training_forward_adapter.py: corrected live packing and original-gate-weight adapter.',
    '- tune_training_k3_audit.py / training-k3-audit-L*.json: all24 configurations and held-out finalist samples.',
    '- audit_corrected_forward.py / forward-corrected-L*.json:600 samples/path, source hashes, changed-input replay checks and corrected kernel traces.', '']
report='\n'.join(md)
(p/'TRAINING_FORWARD_AUDIT.md').write_text(report)
(site/'assets/trimul-training-forward-audit.md').write_text(report)
section='''<!-- FORWARD_AUDIT_BEGIN --><section class="panel" id="forward-audit">
<span class="pill green">2026-09-20 · 학습 Forward 비교 정정</span>
<h2>같은 커널이 아니었다. 현재 adapter의 재배열 비용이 개선분을 지웠다.</h2>
<p>이전 H100에 Anthropic이 들어 있었다는 뜻이 아니다. CUDA graph의 실제 실행 커널을 확인했다. 이전은 자체 CuTe + 별도 LN, 현재는 Anthropic 파생 <code>mw_saved_front → cuBLAS 2개 → mw_k3_train</code>이다.</p>
<div class="table-wrap"><table><thead><tr><th>L</th><th>이전 Triton ms</th><th>이전 자체 H100 ms</th><th>직전 adapter 재측정 ms</th><th>재배열만 수정 ms</th><th>재배열·설정 수정 ms</th><th>이전 H100 대비</th></tr></thead><tbody>'''+''.join(html_rows)+'''</tbody></table></div>
<p>BF16 · dropout25% · mask/residual · backward 저장 · 매 호출 weight 재배열을 포함한다. 모든 경로 CUDA graph, 600회 교대 측정. <b>추론 전용 커널 성능을 학습 forward 성능으로 대체하지 않았다.</b></p>
<h3>직전 결과의 원인</h3><ol>
<li>현재 adapter는 복사6회 + cat2회 + stack1회를 따로 실행했다. 이전 경로는 compile로 재배열이 합쳐져 있었다. 현재에서 재배열을 제외한 진단 실험은 약24μs 짧았다.</li>
<li>재배열을 하나의 연속 버퍼로 합쳐 compile하고, K3에 원본 Wg를 넘겨 불필요한 transpose 왕복을 제거했다. 가중치는 여전히 매 호출 다시 읽는다.</li>
<li>L768 K1 설정 누락을 수정했다. K3는 기존 검색 공간의 가능한24개 설정을 모두 검증하고 상위 후보를 재측정했다. L768은 초기 K3 설정이 그대로 선택됐다.</li>
</ol>
<h3>커널 자체의 개선폭도 아직 작다</h3>
<div class="table-wrap"><table><thead><tr><th>L</th><th>구간</th><th>이전 H100 μs</th><th>직전 학습 파생 μs</th><th>절감 μs</th></tr></thead><tbody>'''+''.join(trace_rows)+'''</tbody></table></div>
<p>위 구간 표는 수정 전 profiler trace의 실행시간 합으로 원인을 설명한다. 모듈 지연은 위쪽 CUDA event 표를 사용한다. 입력+출력 구간의 절감은 약16/56μs다. 재배열 문제를 고쳐도 숨겨진 큰 폭의 개선이 드러나는 상태는 아니다. 이를 성능 상한으로 판단할 근거도 없다.</p>
<details><summary>설정 및 정확도 확인</summary>
<table><thead><tr><th>L</th><th>K1 (BI,BJ,NSLOT,SKCH,SCHED)</th><th>K3 (BI,BJ,NSLOT,NACC,REGS,LNSERIAL)</th></tr></thead><tbody>'''+''.join(config_rows)+'''</tbody></table>
<p>재배열만 바꾸면 출력과 모든 저장값이 bit-exact다. K3 후보는 BF16 출력·저장값 완전 일치, FP32 통계 상대L2≤1e-6를 요구했다. 입력 x·WL·Wg를 바꾼 동일 graph replay도 모든 저장값까지 새 호출과 정확히 일치했고, 복원하면 원래 출력이 재현됐다.</p>
</details>
<p><b>아래 전체 fwd+bwd 표는 수정 전 직접 측정값이다.</b> 이번에는 forward를 정정했다. 전체 학습 시간을 standalone forward 차이로 추정해 바꾸지 않았다. Production 기본 dispatch 변경도 아니다.</p>
<p><a href="assets/trimul-training-forward-audit.md">분석 보고서</a> · <a href="assets/trimul-forward-corrected-L384.json">L384 원시 결과</a> · <a href="assets/trimul-forward-corrected-L768.json">L768 원시 결과</a></p>
</section><!-- FORWARD_AUDIT_END -->'''
path=site/'trimul.html'
html=path.read_text()
if '<!-- FORWARD_AUDIT_BEGIN -->' in html:
    a=html.index('<!-- FORWARD_AUDIT_BEGIN -->'); b=html.index('<!-- FORWARD_AUDIT_END -->')+len('<!-- FORWARD_AUDIT_END -->')
    html=html[:a]+section+html[b:]
else:
    html=html.replace('<!-- ALL_TRAINING_BEGIN -->',section+'<!-- ALL_TRAINING_BEGIN -->',1)
html=html.replace('<nav>', '<nav><a href="#forward-audit">Forward 비교 정정</a>',1) if '<a href="#forward-audit">' not in html else html
html=html.replace('<h2>이전 H100 경로 대비 전체 학습 약 1.30×</h2>', '<h2>재배열 수정 전 실측 · 전체 학습 약 1.30×</h2>')
html=html.replace('<p><b>Forward는 이전 H100과 거의 같다:</b>', '<p><b>수정 전 Forward 측정값 — <a href="#forward-audit">위 정정 결과 참조</a>:</b>')
path.write_text(html)
print('Forward audit rendered; earlier full-training totals preserved as historical measurements.')
