"""Publish measured prototype status into the existing static status page."""
from pathlib import Path
import hashlib,json,re,shutil
R=Path(__file__).resolve().parent
SITE=R.parent/'anthropic_b1b4_pipeline_20260919/site-visuals/dist'
data={n:json.loads((R/('results-L%d.json'%n)).read_text()) for n in (384,768)}
ncu=json.loads((R/'ncu-summary.json').read_text())
def tm(n,scope,k):return data[n]['times'][scope][k]['median_us']/1000
def regional(n,k,entry):return next(x['us'] for x in data[n]['traces']['forward_backward'][k] if x['name']==entry)
rows=[];hrows=[]
for n in (384,768):
 for k,label in [('saved','직전 저장형'),('fused_recompute','새 on-chip 재계산형')]:
  ts=[tm(n,s,k) for s in ('forward','backward','forward_backward')]
  rows.append('| %d | %s | %.3f | %.3f | %.3f |'%(n,label,*ts))
  hrows.append('<tr><td>%d</td><td>%s</td><td>%.3f</td><td>%.3f</td><td><b>%.3f</b></td></tr>'%(n,label,*ts))
region=[];hregion=[]
for n in (384,768):
 for label,old,new in [('B1–B4','dual_b1b4','b1_fused'),('B7–B12','front_b7b12','front_b7b12')]:
  a,b=regional(n,'saved',old),regional(n,'fused_recompute',new)
  region.append('| %d | %s | %.1f | %.1f | %.2fx slower |'%(n,label,a,b,b/a))
  hregion.append('<tr><td>%d</td><td>%s</td><td>%.1f</td><td>%.1f</td><td>%.2f× 느림</td></tr>'%(n,label,a,b,b/a))
metrics=[];hmetrics=[]
for x in ncu:
 m=x['metrics'];v=lambda key:m[key]['value'];label='B1–B4' if x['kernel']=='b1_fused' else 'B7–B12'
 vals=(v('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'),v('sm__throughput.avg.pct_of_peak_sustained_elapsed'),v('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'))
 metrics.append('| %d | %s | %.1f%% | %.1f%% | %.1f%% |'%(x['L'],label,*vals))
 hmetrics.append('<tr><td>%d</td><td>%s</td><td>%.1f%%</td><td>%.1f%%</td><td>%.1f%%</td></tr>'%(x['L'],label,*vals))
report='''# TriMul: inference forward + on-chip backward recomputation

## Outcome

Implemented CUDA/TMA/WGMMA B1–B4 and B7–B12 prototypes with activation reconstruction inside their consuming kernels. The measured prototype is slower than the saved-activation training path and is **not enabled in production**. This is not an optimized limit of the recomputation design.

Comparison baseline: the immediately preceding **Anthropic-derived saved training forward + optimized CUDA B1–B4/B7–B12**, with unchanged cuBLAS contraction backward. This is neither public Anthropic inference nor cuEquivariance.

H100 80GB node02, batch1, BF16 C128, outgoing/incoming hidden128 each, L384/768, dropout25%, mask and residual enabled. CUDA graphs, 600 interleaved measurements per path/scope. Includes live weight packing, all 11 gradients; excludes optimizer, RNG generation, CPU dispatch and compilation. Separate kernel traces are diagnostic and need not sum exactly to graph medians.

| L | Path | Forward ms | Backward ms | Measured total ms |
|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## Exact storage and fusion policy

Forward: Anthropic-derived no-save K1 → cuBLAS outgoing/incoming → no-save K3 with matching training dropout/residual and BF16 rounding. Retain `left/right/tri`, input/weight/mask references and packed weights. No `x_n`, raw input projection/gate, output-normalized tri, output projection/gate, mean or reciprocal-standard-deviation activation buffers.

B1–B4: reconstruct input LN and output gate; reconstruct output LN/projection only where the derivative consumes them. DX skips the output projection GEMM; dW-projection roles also skip it. Reconstructed values stay in registers/shared memory. Output LN statistics are computed inside the same kernel.

B5–B6: unchanged cuBLAS contraction gradients. Trace confirms 2 forward + 4 backward contractions, no repeated forward contraction.

B7–B12: reconstruct input LN and projection/gate, immediately apply GLU backward, input/weight derivatives and input-LN backward. Input LN statistics are recomputed inside this kernel. One cooperative launch per fused region; no standalone activation-restoration or gradient-reduction launches. Weight-gradient partial reduction buffers are still global scratch, not forward activations. `dg/dtri/dleft/dright` remain gradient outputs between regions.

Logical retained intermediate activation bytes (input/weights/mask/output excluded):

| L | Saved path MB | New retained left/right/tri MB | Removed MB |
|---|---:|---:|---:|
|384|719.585|226.492|493.093|
|768|2878.341|905.970|1972.371|

These are tensor-size accounting, **not total training peak-memory measurements**. The comparison harness keeps both implementations and reference tensors live.

## Backward region traces

| L | Region | Saved CUDA us | New CUDA us | Ratio |
|---|---|---:|---:|---|
'''+ '\n'.join(region)+'''

## NCU

Warm-cache, full section set, cache/clock control disabled; profiling is a separate run. Percentages below are hardware throughput/activity metrics, not a computed percentage of the theoretical optimal complete algorithm. Full raw reports and CSVs are in this run directory.

| L | Region | DRAM throughput | SM throughput | Tensor pipe activity |
|---|---|---:|---:|---:|
'''+ '\n'.join(metrics)+'''

Both kernels have about12.5% active-warp occupancy (one256-thread CTA/SM). B1 uses255 registers/thread; B7 uses238. Ptxas reports **zero register spills**. A two-role B1 candidate with spills was rejected, even though it was numerically correct.

Source inspection explains the next optimization work: B1 uses a single row buffer with load → LN → recompute GEMM → derivative GEMM dependencies; it no longer retains the original two-slot row prefetch. DW/DX roles repeat some reconstruction. B7's extra x_n shared tile and register pressure prevent the old two-CTA occupancy and repeatedly stage weights. This profile does not support a DRAM- or Tensor-Core-roofline claim, much less SoL90. Reducing global activation bytes alone did not compensate for the lost overlap and repeated computation.

## Configuration search

B1:64-row tiles,256 threads,132 CTAs; DW_SPLITS={20,24,28,32,36,40}, with3 DW groups; winner28 at both lengths. B7:64-row/64-hidden tiles,256 threads,132 CTAs; DW_SPLITS={4,6,8,10,12,14}, with8 DW groups; winner8 at both lengths. Register/shared stores were vectorized with stmatrix. This is **a CTA-role partition search**, not exhaustive autotuning of row/column tiles, pipeline stages and register budgets. Do not call it fully tuned.

## Verification and remaining accuracy limitation

- Initial output is bit-exact; all11 gradients meet relative-L2 <=5e-4 against the independent existing backward reference at both lengths. Region-level tests retain their stricter per-output bounds.
- After changing input, weights, upstream gradient, dropout scale and mask, captured graphs equal freshly executed new kernels bit-for-bit for output and all11 gradients. New-vs-saved maximum relative-L2 is0.00871% at384 and0.02685% at768.
- **The mutated L768 independent-reference check does not fully pass**: dWL relative-L2 is0.05485% for the new path and0.05413% for the saved baseline, exceeding the unchanged0.05% target. FP32 reduction segments2/4/8/16 did not resolve it. The failure is recorded in JSON; it was not converted to a pass by raising tolerance. It remains a limitation before production adoption.
- Memcheck: B1 and B7 independently at384, combined forward/backward at768:0 memory errors, numerical checks pass. Racecheck: both regions at384:0 errors,0 warnings. Initial B1 shared-buffer reuse had a synchronization bug that was fixed by a CTA barrier after both warp-groups finish TMA dtri writes; these are post-fix results.
- Trace: zero activation-restoration kernels,6 total cuBLAS contractions. No register spills in accepted cubins. The profiler's report-export helper prints an environment utf-8-sig warning after saving; .ncu-rep and CSV export both succeeded.

## Reproduction

Use node02 inside a two-GPU Slurm allocation. Existing training and Claude allocations were not stopped. This allocation was released after measurement.

```bash
export MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_fused_recompute_20260920/bench.py --length 384
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/trimul_fused_recompute_20260920/bench.py --length 768
```

`plans.py` builds cubins with SHA256 source/header/config keys and rejects spills. `b1_fused.cu`, `b7_fused.cu`, `common_recompute.cuh`, and `b1_saved_math.inc` contain the implementation. JSON includes source hashes, numerical results, graph replay checks, timing distributions, selected configs and kernel traces. `derive.py` records derivation from prior math. No engine default/dispatch was changed.

Attribution: this work follows and adapts Anthropic's Apache-2.0 TMA/WGMMA, LayerNorm fragment and matrix-layout primitives plus Miniworld's existing backward math. We are extending that inference work into training; these are not independent original inference kernels.
'''
(R/'README.md').write_text(report)
assets=SITE/'assets'
for n in (384,768):shutil.copyfile(R/('results-L%d.json'%n),assets/('trimul-onchip-recompute-L%d.json'%n))
shutil.copyfile(R/'ncu-summary.json',assets/'trimul-onchip-recompute-ncu.json')
shutil.copyfile(R/'README.md',assets/'trimul-onchip-recompute.md')
table=lambda head,rs:'<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr></thead><tbody>'+''.join(rs)+'</tbody></table></div>'
section='''<!-- ONCHIP_RECOMPUTE_BEGIN --><section class="panel" id="onchip-recompute">
<span class="pill amber">2026-09-20 · B1–B4 / B7–B12 재구현 · 실험 경로</span>
<h2>커널 내부 재계산은 구현했다. 이번 구현은 전체 학습이53–54% 느리다.</h2>
<p>기준은 <b>직전 Anthropic 파생 저장형 학습 forward + 최적화 CUDA backward</b>다. H100·BF16·L384/768·dropout25%·mask/residual·양방향·11개 gradient 조건이다. 각 경로600회 교대 측정했다.</p>
'''+table(['L','경로','Forward ms','Backward ms','전체 실측 ms'],hrows)+'''
<p>Forward는 약35–36% 짧아졌지만, backward는 약2.1배 걸린다. <b>기본 경로는 저장형을 유지한다.</b> 아래는 최적화가 끝난 결과가 아니라 재구현 후보의 실측이다.</p>
<h3>어떤 값을 어디서 다시 계산하는가</h3>
<div class="flow"><div class="node"><b>Forward · Anthropic 파생</b>K1 → cuBLAS2회 → K3<br>dropout·residual 포함<br><strong>left / right / tri만 중간값으로 유지</strong></div><div class="arrow">→</div><div class="node purple"><b>B1–B4 · CUDA1개</b>입력 LN·출력 LN / gate / 필요한 projection 재계산<br>즉시 dW·dtri·LN 미분에 사용<br>재계산 activation은 shared / register에만 존재</div><div class="arrow">→</div><div class="node gray"><b>B5–B6 · cuBLAS4회</b>저장된 left / right로 contraction 미분<br>forward contraction 반복 없음</div><div class="arrow">→</div><div class="node purple"><b>B7–B12 · CUDA1개</b>입력 LN·PL / GL / PR / GR 재계산<br>GLU 미분 → dW·dx → LN 미분·residual<br>HBM activation 복원 없음</div></div>
<p>입력·가중치·mask 참조와 gradient / dW reduction scratch는 별도로 필요하다. 중간 activation 유지량은 L384 <b>719.6→226.5MB</b>, L768 <b>2878.3→906.0MB</b>다. 전체 학습 peak 메모리 측정값은 아니다.</p>
<h3>느려진 위치 · profiler 구간 시간</h3>
'''+table(['L','구간','저장형 μs','새 재계산형 μs','비교'],hregion)+'''
<h3>NCU: 상한 근처가 아니다</h3>
'''+table(['L','구간','DRAM 사용률','SM throughput','Tensor pipe 활성도'],hmetrics)+'''
<p>B1은 재계산 때문에 단일 버퍼에 load→LN→GEMM→미분 GEMM이 이어진다. B7은 shared-memory 증가로 기존2 CTA/SM 대신1 CTA/SM을 사용한다. 재계산 중복과 실행 중첩을 먼저 개선해야 한다. <b>이 결과로 재계산 방식 자체가 느리다고 결론낼 수 없다.</b></p>
<h3>검증 범위와 남은 한계</h3>
<ul><li>일반 입력: forward 비트 일치,11개 gradient 상대 L2≤0.05%. 변경 입력에서 새 graph와 새 eager 결과도 비트 일치.</li>
<li>memcheck: L384 각 구간·L768 전체 오류0. racecheck: L384 두 구간 오류·경고0. B1 공유 버퍼 동기화 문제를 발견해 수정했다. 채택 후보 register spill0.</li>
<li><b>L768 변경 입력의 독립 기준 dWL 오차는 아직 초과</b>: 신규0.05485%, 기존0.05413% / 허용0.05%. 허용치를 올리지 않았으며 JSON에 실패로 기록했다.</li>
<li>현재 튜닝은 CTA 역할 배분이다. 타일·파이프라인 전체 탐색 완료나 SoL90 달성을 주장하지 않는다.</li></ul>
<p><a href="assets/trimul-onchip-recompute.md">구현·검증 보고서</a> · <a href="assets/trimul-onchip-recompute-L384.json">L384 원시 측정</a> · <a href="assets/trimul-onchip-recompute-L768.json">L768 원시 측정</a> · <a href="assets/trimul-onchip-recompute-ncu.json">NCU 수치</a></p>
</section><!-- ONCHIP_RECOMPUTE_END -->'''
p=SITE/'trimul.html';s=p.read_text()
s=re.sub(r'<!-- ONCHIP_RECOMPUTE_BEGIN -->.*?<!-- ONCHIP_RECOMPUTE_END -->','',s,flags=re.S)
s=s.replace('<main>','<main>'+section,1)
if '<a href="#onchip-recompute">' not in s:s=s.replace('<nav>','<nav><a href="#onchip-recompute">새 BWD 내부 재계산</a>',1)
if 'CHECKPOINT_HISTORY_BEGIN' not in s:
 s=s.replace('<!-- ALL_RECOMPUTE_BEGIN -->','<!-- CHECKPOINT_HISTORY_BEGIN --><details><summary>이전 실험: 모듈 전체 checkpoint (현재 커널 내부 재계산과 다름)</summary><!-- ALL_RECOMPUTE_BEGIN -->',1)
 s=s.replace('<!-- ALL_RECOMPUTE_END -->','<!-- ALL_RECOMPUTE_END --></details><!-- CHECKPOINT_HISTORY_END -->',1)
p.write_text(s)
print('Updated README and existing trimul.html with measured results')
