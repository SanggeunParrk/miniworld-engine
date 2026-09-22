"""Render the measured checkpoint policy, with its scope explicitly labelled."""
from pathlib import Path
import hashlib
import json

R=Path(__file__).resolve().parent
site=R.parent/'anthropic_b1b4_pipeline_20260919/site-visuals/dist'
rows=[];memrows=[];bars=[]
md=['# TriMul: save all versus full activation recomputation', '',
    'Node02 H10080GB; BF16; batch1; C128; outgoing/incoming hidden128 each; shared output LN; mask; dropout25%; residual.',
    'Both routes execute identical B1-B12 CUDA/cuBLAS kernels, all eleven gradients, and live weight-layout conversions. Each timing is a single CUDA Graph replay; 3 blocks x200 interleaved samples per route. Excludes optimizer, RNG generation, compilation and CPU/autograd dispatch.', '',
    '**Scope:** this is whole-module activation checkpointing. The no-save forward retains no intermediate activation. Backward first re-executes the saved forward, materializing all intermediates in HBM, then runs the unchanged backward. It is NOT an implementation that fuses recomputation into backward registers or eliminates backward HBM writes.', '',
    '| L | Route | Forward ms | Backward including recompute ms | Joint forward+backward ms |',
    '|---|---|---:|---:|---:|']
for n in (384,768):
    d=json.loads((R/('results-L%d.json'%n)).read_text());m=json.loads((R/('memory-L%d.json'%n)).read_text())
    for p,h in d['metadata']['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    assert all(v['bit_exact'] and v['finite'] for v in d['checks']['recompute_vs_saved'].values())
    assert all(v['bit_exact'] and v['finite'] for checks in d['replay_checks'].values() for v in checks.values())
    t=d['times'];save=t['forward_backward']['saved']['median_us'];re=t['forward_backward']['recompute']['median_us']
    for name,label in [('saved','기존: 중간값 저장'),('recompute','전부 재계산: checkpoint')]:
        f,b,total=[t[k][name]['median_us']/1000 for k in ('forward','backward','forward_backward')]
        rows.append('<tr><td>%d</td><td>%s</td><td>%.3f</td><td>%.3f</td><td><b>%.3f</b></td></tr>'%(n,label,f,b,total))
        md.append('| %d | %s | %.3f | %.3f | %.3f |'%(n,name,f,b,total))
    md+=['', 'L%d: forward %.3fx faster; joint training time increases %.2f%%.'%(n,t['forward']['saved']['median_us']/t['forward']['recompute']['median_us'],100*(re/save-1)), '']
    ybytes=n*n*128*2
    retained=m['saved']['retained_including_y_bytes']-ybytes
    assert m['recompute']['retained_including_y_bytes']==ybytes
    memrows.append('<tr><td>%d</td><td>%.1f MB</td><td>0 MB</td><td>%.1f → %.1f MB</td></tr>'%(n,retained/1e6,m['saved']['forward_peak_increment_bytes']/1e6,m['recompute']['forward_peak_increment_bytes']/1e6))
    md+=['L%d saved intermediate activations: %.1f MB ->0. Forward peak incremental allocation including output: %.1f MB ->%.1f MB. This is NOT whole-training peak memory.'%(n,retained/1e6,m['saved']['forward_peak_increment_bytes']/1e6,m['recompute']['forward_peak_increment_bytes']/1e6),'']
    for kind in ('results','memory'):(site/'assets'/('trimul-recompute-%s-L%d.json'%(kind,n))).write_bytes((R/('%s-L%d.json'%(kind,n))).read_bytes())
    # Scale each shape independently; use jointly measured total, not fwd+bwd sum.
    for name,label,color in [('saved','저장형','#22c55e'),('recompute','재계산형','#f59e0b')]:
        val=t['forward_backward'][name]['median_us']
        bars.append('<div style="margin:8px 0"><span>L%d %s · %.3f ms</span><div style="height:18px;border-radius:5px;background:%s;width:%.1f%%"></div></div>'%(n,label,val/1000,color,90*val/re))
md+=['## Correctness and interpretation','',
    '- Forward output and all eleven gradients are bit-identical between save-all and recompute. CUDA Graph replay also passed after changing input, two weights, dy and dropout scale in place.',
    '- New no-save K3 passed compute-sanitizer memcheck at L384 (0 errors) and racecheck at L768 (0 errors, 0 warnings). Each ran dropout25, zero dropout-scale, and changed-mask/input/weight cases. Kernel filter: regex=infer_k3; this is not a sanitizer audit of unchanged backward kernels.',
    '- Forward K1 is the unmodified Anthropic inference body. No-save K3 preserves its fused input/output LN and TMA/WGMMA pipeline, with the training rounding/dropout/residual epilogue. Upstream revision f4f62fa6592ae4938d49b1757bea0cfeff9f468e; Apache-2.0.',
    '- Saved forward uses the selected per-shape K1 and previously tuned K3. No-save K1/K3 use the shipped H256 tile shapes; this experiment is not exhaustive retuning.',
    '- Recompute pays for the complete forward twice, including both triangular cuBLAS contractions. The second pass still writes intermediates needed by the existing backward. This is why eliminating retention does not eliminate total training HBM traffic.',
    '- Recomputed output y from the second pass is unused; this baseline re-executes the full saved forward, not an early-stop or register-local rematerialization kernel.',
    '- Memory measurements are actual PyTorch allocated bytes in eager forward after warmup, excluding shared inputs/weights and allocator reserve. No claim is made about full-model or backward peak memory.',
    '- Speed-only winner here: save-all. Full checkpointing trades more compute for less forward-retained memory. A fused backward-recomputation kernel or a selective-save policy is a separate experiment.',
    '- The experimental adapter is local; production dispatch is unchanged.', '',
    '## Reproduction', '', '```bash',
    'bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_all_recompute_20260920/bench.py --length 384',
    'bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_all_recompute_20260920/bench.py --length 768',
    '```', 'Run inside an allocated node02 H100 job, one GPU per process.', '']
text='\n'.join(md);(R/'REPORT.md').write_text(text);(site/'assets/trimul-all-recompute.md').write_text(text)
panel='''<!-- ALL_RECOMPUTE_BEGIN --><section class="panel" id="all-recompute">
<span class="pill green">2026-09-20 · 저장 / 전체 재계산 실측</span>
<h2>모두 다시 계산하면: forward는 빨라지고 전체 학습은 약24% 느려진다</h2>
<p>동일한 H100·BF16·L384/768·dropout25%·mask/residual·양방향 조건이다. 입력, 가중치, mask만 유지하고 <b>중간 activation 저장을 전부 제거</b>했다. 출력과11개 gradient는 저장형과 비트 단위로 일치한다.</p>
<div class="table-wrap"><table><thead><tr><th>L</th><th>경로</th><th>Forward ms</th><th>Backward ms</th><th>합계 실측 ms</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p>Backward 열에는 재계산 비용이 포함된다. 합계는 forward+backward를 한 그래프로 직접 측정했다. 개별 열의 단순 합과는 캐시 상태·측정 변동으로 조금 다르다.</p>
'''+''.join(bars)+'''
<h3>이번에 구현한 재계산 방식</h3>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px">
<div style="border:1px solid #546078;border-radius:10px;padding:16px"><b>저장형</b><p>Forward + activation 저장<br>↓<br>B1–B12가 저장값을 읽음</p></div>
<div style="border:1px solid #546078;border-radius:10px;padding:16px"><b>전체 checkpoint</b><p>저장 없는 Forward<br>↓<br>Backward 시작: Forward 재실행 + activation 재생성<br>↓<br>동일한 B1–B12 실행</p></div></div>
<p><b>이는 backward 커널 안에서 레지스터로 재계산하는 구현은 아니다.</b> 재계산한 값은 기존 backward에 전달하기 위해 HBM에 다시 쓴다. 삼각 cuBLAS 행렬곱도 두 번 실행된다. 따라서 이 결과로 재계산 융합 커널까지 느리다고 결론내릴 수 없다.</p>
<h3>메모리: forward가 끝난 뒤 유지하는 activation은0</h3>
<div class="table-wrap"><table><thead><tr><th>L</th><th>저장형 중간 activation</th><th>재계산형 중간 activation</th><th>Forward 추가 peak · 출력 포함</th></tr></thead><tbody>'''+''.join(memrows)+'''</tbody></table></div>
<p>입력·가중치·mask는 별도로 유지한다. 위 peak는 단일 eager forward의 추가 할당 실측이며, 전체 학습 peak가 아니다. Backward에서는 activation을 다시 생성한다.</p>
<p><b>현재 속도 기준은 저장형 유지.</b> 메모리가 필요할 때 전체 checkpoint를 선택할 수 있다. 필요한 값만 남기는 선택적 저장과 backward 안에서 재계산하는 융합 구현은 별도 비교 대상이다.</p>
<p>새 no-save K3: L384 memcheck 오류0, L768 racecheck 오류·경고0. dropout scale을0으로 만들어 residual만 남긴 경우와 mask·입력·가중치를 변경한 경우도 출력 일치를 확인했다.</p>
<p><a href="assets/trimul-all-recompute.md">구현 범위·재현 보고서</a> · <a href="assets/trimul-recompute-results-L384.json">L384 원시 결과</a> · <a href="assets/trimul-recompute-results-L768.json">L768 원시 결과</a></p>
</section><!-- ALL_RECOMPUTE_END -->'''
p=site/'trimul.html';s=p.read_text()
if '<!-- ALL_RECOMPUTE_BEGIN -->' in s:
    a=s.index('<!-- ALL_RECOMPUTE_BEGIN -->');b=s.index('<!-- ALL_RECOMPUTE_END -->')+len('<!-- ALL_RECOMPUTE_END -->');s=s[:a]+panel+s[b:]
else:
    assert '<!-- FORWARD_CAUSE_BEGIN -->' in s
    s=s.replace('<!-- FORWARD_CAUSE_BEGIN -->',panel+'<!-- FORWARD_CAUSE_BEGIN -->',1)
if '<a href="#all-recompute">' not in s:s=s.replace('<nav>','<nav><a href="#all-recompute">저장 vs 전체 재계산</a>',1)
p.write_text(s)
print('Report and all-recompute status panel rendered.')
