"""Publish measured LN-prefetch results; keep prior audits and training policy."""
from pathlib import Path
import csv, hashlib, html, json, re, shutil, tarfile
import xml.etree.ElementTree as ET
R=Path(__file__).resolve().parent
A=R.parent/'anthropic_adoption_20260919'; SITE=A/'site/dist'
read=lambda f:json.loads((R/f).read_text())
rows=read('sol-final-paired-results.json')
checks=read('dual_ln_prefetch-strict-validation.json')
assert len(checks)==24 and all(c['counters_zero'] for c in checks)
for c in checks:
 for errors in c['errors']:
  for k,e in errors.items():
   assert e['relative_l2']<=e['tolerance'],(k,e)
   if k=='dg':assert e['bit_exact']
maxerr={k:max(e[k]['relative_l2'] for c in checks for e in c['errors']) for k in checks[0]['errors'][0]}
saved=read('sol-saved-updates.json');assert len(saved)==4
for c in saved:
 for es in c['errors']:
  assert all(e['relative_l2']<=e['tolerance'] for e in es.values())
old=read('optimized-audit.json')
for f,sha in old['original_sha256'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==sha
san={}
for tool in ('memcheck','racecheck','synccheck','split-memcheck'):
 for n in (64,384):
  s=(R/f'sol-prefetch-{tool}-L{n}.log').read_text()
  assert 'ERROR SUMMARY: 0 errors' in s or 'RACECHECK SUMMARY: 0 hazards' in s
  san[f'{tool}/L{n}']='passed'
ptxas={}
for match in re.finditer(r'PTXAS dual_ln_prefetch (66|132) (1|2) (\S+\.cubin)',(R/'sol-prefetch-validation.log').read_text()):
 count,part,path=match.groups();p=Path(path);s=p.with_suffix('.ptxas.log').read_text()
 assert not re.search(r'(?<!\d)[1-9]\d* bytes spill (?:stores|loads)',s)
 ptxas[f'{count}/{part}']={'cubin':path,'command':json.loads(p.with_suffix('.json').read_text())['command'],'registers':re.findall(r'Used (\d+) registers',s),'spills':0}
assert len(ptxas)==4
sass=(R/'dual_ln_prefetch.sass').read_text()
sass_counts={op:len(re.findall(r'\b'+op+r'(?:\.|\s)',sass)) for op in ('HGMMA','UTMALDG','UTMASTG','LDL','STL')}
assert sass_counts['LDL']==sass_counts['STL']==0
profiles={};prof=[]
for n in (384,768):
 for label in ('balanced','prefetch'):
  data=list(csv.DictReader((R/f'sol-{label}-warm-L{n}.csv').open()));u,v=data[0],data[-1]
  keys=['gpu__time_duration.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_read.sum','dram__bytes_write.sum','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
  d={k:{'value':float(v[k].replace(',','')),'unit':u[k]} for k in keys};profiles[f'{label}/L{n}']=d
  val=lambda k:d[k]['value']
  byte=lambda k:val(k)*{'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9}[u[k]]
  prof.append([n,label,f'{val(keys[0]):.3f}',f'{val(keys[1]):.2f}%',f'{byte(keys[2])/1e6:.3f}',f'{byte(keys[3])/1e6:.3f}',f'{val(keys[4]):.2f}%'])
  assert val(keys[5])==val(keys[6])==0
def table(h,rs):return '| '+' | '.join(map(str,h))+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rs)
def htable(h,rs):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rs)+'</tbody></table></div>'
def lat(t,k):return f'{t[k]["median_us"]:.3f} / {t[k]["p90_us"]:.3f}'
perf=[];full=[]
for r in rows:
 t=r['times'];k='dual_ln_prefetch/part2';o='dual_balanced/part2'
 perf.append([r['L'],lat(t,'baseline'),lat(t,o),lat(t,k),f'{t["baseline"]["median_us"]/t[k]["median_us"]:.4f}x',f'{100*(1-t[k]["median_us"]/t[o]["median_us"]):.2f}%'])
 full.append([r['L'],lat(t,'full/baseline'),lat(t,'full/'+k),f'{t["full/baseline"]["median_us"]/t["full/"+k]["median_us"]:.4f}x'])
cueq_core=read('cueq-core-comparison.json');cueq=[]
for mode in ('default','tuned'):
 cs=read(f'cueq-{mode}-comparison.json');assert len(cs)==2
 for r in cs:
  t=r['times']
  cueq.append([r['L'],mode,lat(t,'cueq_eager'),lat(t,'cueq_compile'),lat(t,'cuda'),f'{t["cueq_compile"]["median_us"]/t["cuda"]["median_us"]:.3f}x'])
coretable=[]
for r in cueq_core:
 t=r['times'];coretable.append([r['L'],lat(t,'cueq_ln_hybrid'),lat(t,'cuda'),f'{t["cueq_ln_hybrid"]["median_us"]/t["cuda"]["median_us"]:.3f}x'])
phase=[]
for r in read('sol-prefetch-phases.json'):
 v=r['phases']['dual_ln_prefetch_timing/part2']['median_critical_frontier_us']
 phase.append([r['L'],*[f'{v[k]:.3f}' for k in ('body','partial_dump_and_publish','grid_wait','reduce_and_publish','reset')]])
experiments=[
 ['LN shared SoA; direct global statistics','sol-ln-candidates.json','No gain; direct global loses badly at L768.'],
 ['Persistent register gamma/beta accumulators','sol-lnacc.json','192 / 740 us; rejected.'],
 ['Per-warp SMEM accumulators; cached xhat','sol-epilogue.json','No reliable whole-path gain.'],
 ['TMA read-completion wait; streaming cache hints','sol-tma-read.json; sol-tma-stream.json','No consistent two-shape gain.'],
 ['Float mask decode / packed mask ballots','sol-mask-float.json; sol-mask-pack.json','Float loses; packed variation too small to adopt.'],
 ['Independent full-channel WG tiles','dual_wg_tiles build log','284B spill stores +260B loads: rejected BEFORE GPU execution.'],
 ['Independent WG tiles with shared reload','sol-wgreload.json','Zero spills but 197 / 711 us; rejected.'],
 ['Fewer WGMMA fence/commit groups','sol-mma.json','No gain; rejected.'],
 ['Vector mean/rstd before B1; gamma once per launch','sol-stats-prefetch.json; sol-final-paired-results.json','SELECTED dual_ln_prefetch.'],
 ['Two tiles ahead / cp.async statistics','sol-stats-prefetch.json; sol-async-stats.json','More overlap alone was slower.'],
 ['CTA placement, role ratios and combinations','sol-placement.json; sol-combinations.json','Keep40/92. place13 slight core win, full L768 regresses.'],
]
audit={'selected':'dual_ln_prefetch','status':'experimental; relative 1.7x met, SoL90 not established','strict_cases':24,'comparisons':72,'updated_saved_replays':8,'max_relative_l2':maxerr,'sanitizers':san,'ptxas':ptxas,'sass_static_counts':sass_counts,'profiles':profiles,'original_sha256':old['original_sha256'],'events':'sol-final-paired-results.json','cueq_full':['cueq-default-comparison.json','cueq-tuned-comparison.json'],'production_default_changed':False}
(R/'sol-audit.json').write_text(json.dumps(audit,indent=2))
report='''# B1–B4 CUDA: saved-LN prefetch and cuEquivariance comparison

2026-09-20. Selected experimental source: **dual_ln_prefetch.cu/.py**.
Triton/cuBLAS B1–B4: **1.738x /1.934x** at L384/768, dropout0.25.
This iteration saves1.68% /1.83% versus dual_balanced. **SoL90% is NOT established.**
Full backward integration is tested; production dispatch has not been promoted.

Anthropic's inference results were superior to our previous development.
This training extension builds on their native v5 CUDA primitives and ideas.
Training ratios below do not measure an improvement over Anthropic inference.

## 1. Operation, ownership, shared memory and assumptions

```
y = dy * ds[row % L]
dp = bf16(y * gate)
dg = bf16(((y * proj) * gate) * (1-gate))
dWg = bf16(sum_fp32(xn.T @ dg))
dWp = bf16(sum_fp32(dp.T @ norm))
dnorm = bf16(dp @ Wp)
xhat = (tri-mean) * rstd; h = dnorm * gamma
dtri = bf16(rstd * ((h-mean(h))-xhat*mean(h*xhat)))
dgamma = sum_fp32(dnorm*xhat); dbeta = sum_fp32(dnorm)
```

Fixed M=L²,C128,H256, six outputs (dg,dWg,dtri,dgamma,dbeta,dWp).
Forward saves and BF16 rounding boundaries are unchanged. B5+ stays Triton.
One cooperative launch:40 DW CTAs and92 DX/LN CTAs,256 threads each,
two128-thread WGs. Ratio10:23 gives20/46 at count66. Cyclic64-row tiles.
DW does B1+B2+B3b; DX does B1+B3a+B4. Both calculate dp and load dy/gate.
There is no global dp or dnorm, no cluster/multicast, and no extra dW launch.

DW shared slots0/98304 each96KiB: dy->dp16KiB,gate->dg16KiB,
proj16KiB,xn16KiB,norm32KiB. Each WG owns three64x128 dW tiles;
192 FP32 accumulators/thread; partials are stored after all owned rows.

DX shared slots0/98304 each64KiB: dy->dp16KiB,gate->row statistics16KiB,
tri->dtri32KiB. Wp resident163840..229376 (64KiB).
Warp gamma/beta partials65536..73728 (8KiB).
NEW saved mean/rstd:73728..74240 (slot0),74240..74752 (slot1).
NEW gamma resident74752..75776 (1KiB), loaded once PER LAUNCH.
Running gamma/beta sums229376..231424 (2KiB).
Dynamic231424B +static1024B;252 registers, zero spills, oneCTA/SM.

After TMA mbarrier wait,16 threads vector-load64 saved means and inverse
standard deviations, before B1. B1's existing CTA barrier publishes these
loads and gamma. B4 consumes them later, removing late global reads and
one CTA barrier per tile. Register tri/dnorm lifetime and arithmetic remain.
Two-stage slot=round&1,phase=(round/2)&1. Consumers and TMA stores finish
before reuse for tile+2*role_count. Existing WGMMA register/proxy fences,
commit/wait, cross-WG LN barriers and final ticket publication remain.
PART2 reduces in the same cooperative launch and resets counters;
PART1 uses the standalone reducer. No claim of eliminating all CTA barriers.

Contiguous aligned BF16 data, FP32 LN parameters/statistics, L>=64 and
L²%64=0, int32 indices, per-stream non-reentrant plan workspace. ds is zero
or a common positive BF16 scale. Mask cache falls back to ordinary loads
for periods above3. No L384/768-specific mask code. Packing, descriptors,
allocation and compilation are outside timing; saved pointers stay stable.

## 2. Correctness findings

All required cases pass, including changing saved mean/rstd/gamma between
graph replays. Gamma is refreshed each launch, not cached across launches.
Original SHA-256 values still match. Baseline and prior selected files stay
available. No new source is called qualified merely because it compiles.
The independent-WG variant with ptxas spills never ran on GPU.

Historical extra L72 seed20261151 has dtri relativeL2=2.1726e-5 in both
original and balanced, bit-identically; the larger generic shape matrix
was22/24, not all-pass. New selected path is strictly qualified for the
requested24 cases; no universal all-shape/all-seed claim.

## 3. Time split and roofline

Completion-frontier medians from20 instrumented diagnostic launches, us:

'''+table(['L','Body','Dump+publish','Grid wait','Reduce+publish','Reset'],phase)+'''
These are diagnostic frontiers, not the headline CUDA-event time breakdown.
Earlier partial writes overlap later body work. Timed instrumented/control
were170.464/171.360us and611.008/610.640us in that separate process.
Body dominates; exposed grid waiting is below1us, reduction around2.6–2.8us.

Useful traffic=(2056 read+768 write)*L² =416.416/1665.663MB.
At3.35TB/s, bandwidth-only minima124.30/497.21us;90% of that throughput
requires138.11/552.45us. Current172.64/611.23us does not reach either.
An independent3-read/1-write streaming probe measured2.992/3.062TB/s;
useful-byte throughput is about80.6%/89.0% of that probe, which is still
not proof of90% SoL. Actual algorithm has instructions, barriers and extra
requests. Keep useful traffic distinct from measured total HBM traffic.

Both roles load dy/gate: extra512B/row logical requests can partly hit L2.
dW/LN partials touch8,052,736B one-way,16.105MB read+write. Host capacity
is still26.223MB; no allocation-saving claim. No52MB HBM saving inferred.

## 4. Ranked experiments and decisions

Priorities were (1) source-level LN store conflicts and scheduling,
(2) overlap statistics loads with existing B1 work, (3) independent WG
row ownership and TMA/WGMMA scheduling. Each keeps the mathematical boundary.

'''+table(['Experiment','Evidence','Decision'],experiments)+'''
Source counters traced the old36,864 excessive shared wavefronts to the
two row-statistic STS64 stores, not the8KiB gamma/beta partial buffer.
Changing to SoA did not improve runtime. The selected prefetch improves
long-scoreboard stalls, but aggregate shared bank conflicts increased;
do not describe the improvement as bank-conflict elimination.

## 5. Code and integration

dual_ln_prefetch.cu/.py is the selected full device source and Plan wrapper.
dual_experiment.py hashes dependencies/flags and rejects spills before launch.
integrate.backward_cuda replaces B1–B4; B5+ uses the previous engine code.
dual_ln_prefetch_timing.cu adds diagnostic timestamp probes only.
Production default/autotune dispatch are unchanged; the code is an
experimental integration, not a completed production rollout.
Anthropic native v5 headers remain external repository dependencies.

## 6. Validation

L64/384/768 × dropout0/.25 × count66/132 × PART1/2 =24 cases.
Ordinary launch +two graph replays changing dy/ds =72 six-output comparisons;
counters zero. Maximum relativeL2:

'''+table(['Output','Maximum'],[[k,f'{v:.9g}'] for k,v in maxerr.items()])+'''
Limits: dg bit-exact including signed zero; dtri2e-5; gamma/beta5e-6;
dWg/dWp5e-4. Full backward11 gradients pass5e-4 at both target shapes.
Eight further replays change gamma,mean,rstd,dy,ds at both shapes/PART1/2.
memcheck/racecheck/synccheck at L64/384 and PART1 unfiltered memcheck all pass.
Four compiled count/PART variants have zero spills. SASS confirms explicit
HGMMA/TMA and no local loads/stores; NCU local sectors are also zero.

## 7. Reproduction and measured times

node02 H10080GB, CUDA12.9 sm90a, BF16, dropout0.25.
Same-process explicit one-call CUDA Graphs, alternating order,20 warmups+
200 samples per block,3 blocks. Pooled600-sample median/p90, us.
No profiler attached for headline timing; clocks were not locked.

'''+table(['L','Triton/cuBLAS','Previous balanced','Selected prefetch','Speedup vs Triton','Time saved vs balanced'],perf)+'''
Full backward:

'''+table(['L','Triton/cuBLAS','Selected CUDA +Triton B5+','Speedup'],full)+'''
```
P=runs/anthropic_b1b4_pipeline_20260919
ENV=runs/anthropic_adoption_20260919/env.sh
bash "$ENV" python -u -B "$P/check_experiment.py" --source dual_ln_prefetch
bash "$P/sol_prefetch_audit.sh"
bash "$ENV" python -u -B "$P/check_sol_saved_updates.py"
bash "$ENV" python -u -B "$P/measure_sol_final.py"
bash "$ENV" python -u -B "$P/compare_cueq.py" --output cueq-default-comparison.json
CUEQ_TRITON_TUNING=ONDEMAND CUEQ_TRITON_CACHE_DIR="$PWD/$P/cueq-cache" bash "$ENV" python -u -B "$P/compare_cueq.py" --output cueq-tuned-comparison.json
```

Exact compiler command (other configurations change count/PART):

```
'''+ ' '.join(ptxas['132/2']['command'])+'''
```

### cuEquivariance comparison: two distinct scopes

Installed cuequivariance-torch0.9.1, ops-cu12 0.10.0; this is not a claim
about all newer releases. Public TMU is single-direction. Calling it twice
normalizes128-channel halves separately; our bidirectional operation has
one shared256-channel LN. Thus compare the engine's equivalent composition:
vendor input LN +gated dual GEMM +two contractions +shared output LN,
Torch gate/projection, same supplied pair mask/dropout/residual, vendor autograd.
[Public API](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html).

Full backward only; forward and saves are prepared before timing. Static
fullgraph torch.compile, Inductor cudagraphs off; explicit CUDA Graph on
ALL timed paths. Default and ONDEMAND tuning are separate same-process runs.
Independent leaves are created on each forward/capture stream to avoid an
AccumulateGrad legacy-stream capture dependency. Numeric BF16 0/1 pair mask
avoids the vendor autotuner's randn_like(bool) error without changing its math.

'''+table(['L','Tuning','cuEq eager us','cuEq compile us','Current full bwd us','vs compiled cuEq'],cueq)+'''
Vendor preserves its own internal rounding/saves. All11 gradients are finite
and differ by roughly0.1–0.6% relativeL2; forward compile is about4–5e-5.
This is equivalent mathematical computation, not identical BF16 arithmetic
or the strict same-saves B1–B4 comparison. The entire full-backward advantage
must not be attributed to the newly optimized B1–B4 alone. B5+ also differs.
The Triton full baseline emits a heuristic-cache warning for input dual
backward; it is the exact current harness, not a claim of exhaustive tuning.

Controlled B1–B4 primitive experiment below uses the same engine B1 and
three cuBLAS GEMMs, replacing only B4 with vendor LN backward +parameter
reductions. Six outputs satisfy the strict comparison limits. It is a
cuEq-LN HYBRID, not the public cuEq TMU or a complete vendor B1–B4 kernel.

'''+table(['L','cuEq LN hybrid us','Current fused CUDA us','Ratio'],coretable)+'''
## 8. NCU observed signals

Warm application replay after20 warmups, cache-control none, clock-control
none. This aligns cache conditions with event benchmarks; default full
profiles also exist but are a separate cold/replay measurement.
[NCU profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

'''+table(['L','Version','NCU us','DRAM %peak','Read MB','Write MB','Tensor elapsed %peak'],prof)+'''
Warm DRAM74.14% /81.98% does not meet90%. Default full profiler duration
204.8/738.112us and DRAM62.16%/76.85% must not be mixed with warm events.
Long-scoreboard ratios drop0.467->0.370 and0.349->0.239. Barrier ratios
remain1.529/1.385; these are per-issue-active ratios, not elapsed percentages.
NCU zero local traffic agrees with ptxas/SASS. Static SASS counts:

'''+table(['Instruction','Count'],list(sass_counts.items()))+'''
## 9. Gap, limitations and next experiment

Relative1.7x reached at both target shapes; L384172.640us also barely
passes the absolute173us goal in this run. Current gain versus balanced
is only1.7–1.8%; do not present the full1.74/1.93x as this iteration's gain.
SoL90% remains open. No end-to-end model training-step speedup is established.
NCU, useful-byte bounds and the bandwidth probe are complementary evidence,
not interchangeable definitions chosen to make90% pass.

Next candidate: cooperative tail work sharing between DW and DX roles,
keeping six outputs and rounding fixed. Measured body frontiers leave only
~2.6us DW-to-DX slack at L384, so setup/extra resident-Wp loads may erase
the benefit. Prototype only if zero-spill static resource accounting leaves
room; require lower body-frontier time and repeatable whole-backward gains.
Not implemented and no estimated speedup is claimed as a result.
'''
(R/'SOL_AUDIT_20260920.md').write_text(report)

# Actual operation wiring, including the new saved-statistics dependency.
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1450" role="img" aria-labelledby="title desc"><title id="title">B1-B4 saved LN prefetch CUDA wiring</title><desc id="desc">One launch, 40 DW and92 DX CTAs. Mean/rstd prefetch before B1; its barrier publishes stats. Gamma resident per launch; B5 remains Triton.</desc><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#58717b"/></marker></defs><style>text{font-family:Arial,sans-serif;fill:#18343f}.edge{fill:none;stroke:#58717b;stroke-width:2;marker-end:url(#arr)}</style><rect width="1200" height="1450" fill="#f4f8fa"/>']
def rect(x,y,w,h,fill='#fff',stroke='#a7bac6'):svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}"/>')
def txt(x,y,s,size=16,bold=False):svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}">{html.escape(s)}</text>')
def edge(d):svg.append(f'<path d="{d}" class="edge"/>')
txt(35,45,'TriMul 학습 B1–B4 · dual_ln_prefetch',27,True)
txt(35,77,'Anthropic v5 CUDA primitives 계승 · 132 CTAs × 256 threads · 1 cooperative launch')
rect(35,102,1130,83,'#e9eff4');txt(55,132,'입력 / forward saves · 수식과 BF16 반올림 유지',19,True)
txt(55,162,'dy, ds, xn, gate, proj, norm, tri, mean, rstd, gamma, Wp · M=L² / C128 / H256')
rect(25,210,1150,850,'#fff','#367d70');txt(46,243,'하나의 CUDA 커널 · 내부 역할별 CTA 배정',21,True)
edge('M317 185V273');edge('M881 185V273')
for x,color,title in [(48,'#edf8f3','DW · 40 CTAs · B1 + B2 + B3b'),(615,'#eef3fc','DX / LN · 92 CTAs · B1 + B3a + B4')]:
 rect(x,273,535,655,color);txt(x+20,307,title,19,True)
for x,lines in [(70,['TMA: dy, gate, proj, xn, norm','96KiB slot0 ↔ slot1; mbarrier wait']), (637,['TMA: dy, gate, tri + resident Wp','64KiB slot0 ↔ slot1 + Wp64KiB; wait'])]:
 txt(x,341,lines[0]);txt(x,367,lines[1],15)
rect(637,390,491,97,'#fff7d9','#d7b869')
txt(654,416,'NEW · B1 전에 mean/rstd를 vector preload',17,True)
txt(654,443,'shared73728..74752 · 두 tile의 통계 버퍼',15)
txt(654,469,'gamma: launch마다1회 → shared74752..75776',15)
edge('M881 377V390');edge('M881 487V515');edge('M315 377V515')
for x in (70,637):rect(x,515,491,93);edge(f'M{x+245} 608V635')
txt(87,543,'B1 · dp=bf16(dy×ds×gate)',17,True)
txt(87,570,'dg=bf16(((dy×ds)×proj×gate)×(1−gate))',15)
txt(87,596,'dp/dg → shared; dg → global',15)
txt(654,543,'B1 · dp를 DX 역할에서 재계산 → shared',17,True)
txt(654,570,'기존 B1 CTA barrier가 mean/rstd/γ도 publish',15)
txt(654,596,'통계용 CTA barrier1회/tile 제거',15,True)
for x in (70,637):rect(x,635,491,149);edge(f'M{x+245} 784V810')
txt(87,664,'WGMMA · B2 dWg += xnᵀ @ dg',17,True)
txt(87,693,'WGMMA · B3b dWp += dpᵀ @ norm',17,True)
txt(87,723,'두 WG가 weight tile6개 분담',15)
txt(87,750,'FP32 accum192/thread · split+40×round',15)
txt(654,664,'B3a · WGMMA: dnorm=bf16(dp @ Wp)',17,True)
txt(654,693,'B4 · register dnorm/tri + prefetched stats',16,True)
txt(654,723,'LN backward → dtri → stmatrix → TMA store',15)
txt(654,750,'split+92×round · global dp/dnorm 없음',15)
txt(70,840,'종료: dWg/dWp FP32 partials → global',17,True)
txt(70,872,'40×49152 floats =7.864MB',15)
txt(637,840,'종료: dgamma/dbeta partials → global',17,True)
txt(637,872,'92×512 floats =0.188MB · shared sums8+2KiB',15)
edge('M315 928V955H600V977');edge('M881 928V955H600')
rect(70,977,1058,61,'#f8f0df','#d7b869')
txt(88,1002,'threadfence → grid ticket barrier → FP32 final reduction → counters reset',17,True)
txt(88,1027,'PART2: 같은 launch에서 완료 · PART1: standalone reducer fallback',15)
edge('M600 1060V1090');rect(35,1090,1130,80,'#e9eff4')
txt(55,1121,'6개 출력: dg, dWg, dtri, dgamma, dbeta, dWp',19,True)
txt(55,1150,'dg / dtri → 기존 Triton B5+ · 실험 경로; production 기본 dispatch는 미변경')
rect(35,1195,1130,222)
txt(55,1228,'Dropout25% · 3×200 CUDA-event samples · Triton/cuBLAS 기준',18,True)
txt(55,1260,'L384: 300.00 → 172.64 µs (1.738×)     L768:1182.26 → 611.23 µs (1.934×)',20,True)
txt(55,1293,'이번 개선: 이전 balanced 대비1.68% /1.83% · 전체 backward1.129× /1.146×')
txt(55,1325,'252 registers /spill0 · 24 strict cases +saved-value replay8회 +sanitizer 통과')
txt(55,1357,'Warm NCU DRAM74.14% /81.98% · SoL90% 도달로 판단하지 않음',18,True)
txt(55,1388,'cuEquivariance의 전체 backward와 B1–B4 hybrid 비교는 HTML의 별도 표 참조',15)
svg=''.join(svg)+'</svg>';ET.fromstring(svg)
history=R/'b1b4-shared-balanced-history.svg'
if not history.exists():shutil.copyfile(R/'b1b4-shared.svg',history)
(R/'b1b4-shared.svg').write_text(svg)

section='''<!-- B1B4_AUDIT_BEGIN --><section class="panel" id="b1b4-audit"><span class="pill green">2026-09-20 · LN prefetch · B1–B4 1.738× /1.934×</span><h2>B1–B4 학습 CUDA · 통계 prefetch 추가</h2>
<p>Anthropic의 inference 결과가 우리의 이전 개발보다 우수했다. <b>Anthropic native v5 CUDA primitives와 아이디어를 계승하여 학습 경로를 확장</b>한다. Forward 저장값·수식·BF16 반올림·6개 출력·Triton B5+를 유지한다.</p>
<p>40 DW +92 DX/LN CTA를 유지하면서, mean/rstd를 B1 앞에서 읽고 gamma를 launch마다1회 shared에 저장한다. B1의 기존 barrier를 공유하여 late load와 통계용 barrier를 줄였다. 이번 개선폭은 이전 balanced 대비<b>1.68% /1.83%</b>다.</p>'''+htable(['L','Triton/cuBLAS µs median/p90','이전 balanced','현재 CUDA','Triton 대비','이전 대비 시간 감소'],perf)+'''
<h3>전체 backward · B5+ 동일</h3>'''+htable(['L','Triton/cuBLAS µs','현재 CUDA +Triton B5+ µs','배속'],full)+'''
<p>node02 H10080GB · BF16 · dropout25% · CUDA Graph · 같은 process 순서 교대 ·20 warmup+200 samples를3회 반복. Pooled600-sample median/p90. Production 기본 dispatch는 미변경이다.</p>
<div class="notice amber"><b>SoL90% 도달은 아직 아니다.</b> Warm NCU DRAM74.14% /81.98%. Useful traffic과3.35TB/s로 계산한90% 목표시간138.1/552.5µs에 미달한다. 별도 실측 대역폭 probe 대비80.6%/89.0%도90% 도달 근거로 사용하지 않는다.</div>
<h3 id="cueq-bwd-comparison">cuEquivariance 대비 · 전체 backward</h3><p>cuEq ops0.10.0 /torch wrapper0.9.1. 공개 TMU는 단방향이다. 두 단방향 호출과 수식이 다른 <b>공유256채널 LN 양방향 연산</b>에 맞춰 cuEq 기본 연산을 구성했다. Default와ONDEMAND 자동 튜닝 모두 측정. Vendor 내부 저장/반올림 유지;11 gradients 상대L2 차이 약0.1–0.6%. 동일 BF16 연산순서 비교는 아니다.</p>'''+htable(['L','cuEq 튜닝','cuEq eager µs','cuEq compile µs','현재 전체 bwd µs','compiled cuEq 대비'],cueq)+'''
<p>두 경로 모두 explicit CUDA Graph. cuEq compile은 static fullgraph; forward는 타이밍 밖이다. 이 전체 배속에는 기존 B5+ 차이도 포함된다. B1–B4 변경만의 효과로 해석하지 않는다.</p>
<details><summary>B1–B4 통제 실험: cuEq LayerNorm을 넣은 hybrid</summary><p>공통 engine B1 +cuBLAS GEMM3개 +cuEq LN backward/parameter reductions. 공개 cuEq TriMul 전체가 아니다.</p>'''+htable(['L','cuEq LN hybrid µs','현재 CUDA µs','배속'],coretable)+'''</details>
<h3>실제 연산 연결 / 융합 / 저장</h3><p><a href="assets/b1b4-shared.svg">SVG 원본</a></p><img src="assets/b1b4-shared.svg" alt="40 DW CTA와92 DX CTA; B1 이전 mean/rstd prefetch, gamma resident, B1 barrier 공유, B5 Triton 연결" style="width:100%;height:auto"/>
<h3>검증 / 남은 병목</h3><p>24개 strict 조합·72회 비교, 저장 통계/γ를 바꾼 replay8회, sanitizer3종 및 PART1 memcheck 통과.252 registers, spill0. 본체가 지배적이며 마지막 reduction만 제거해서 큰 이득을 기대하기 어렵다. CTA 배치·비동기 통계 load·독립 WG·누적 방식 등은 직접 실험하고 더 느린 변형을 제외했다.</p>
<p>이전 추가L72 한 seed의 dtri 오차는 원본과 비트 단위로 동일하다. 모든 일반 shape를 완전히 검증했다고 표시하지 않는다. Triton 전체 비교에는 input-dual-backward heuristic cache 경고가 있어 완전 튜닝된 모든 baseline보다 빠르다는 주장도 하지 않는다.</p>
<p><a href="assets/b1b4-sol-audit-20260920.md">9항목 상세 보고서</a> · <a href="assets/b1b4-sol-audit.json">NCU /검증 요약</a> · <a href="assets/b1b4-sol-results.json">600 samples 원자료</a> · <a href="assets/b1b4-cueq-default.json">cuEq default</a> · <a href="assets/b1b4-cueq-tuned.json">cuEq tuned</a> · <a href="assets/b1b4-ln-prefetch.cu">CUDA 소스</a> · <a href="assets/b1b4-sol-source.tar.gz">재현 소스</a></p><details><summary>상세 보고서</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'''+html.escape(report)+'''</pre></details><details><summary>이전 balanced 기록</summary><a href="assets/b1b4-balanced-audit-20260920.md">이전 보고서</a> · <a href="assets/b1b4-shared-balanced-history.svg">이전 배선 SVG</a></details></section><!-- B1B4_AUDIT_END -->'''
s=(SITE/'trimul.html').read_text();before=s[s.index('id="training-shapes"'):]
s,n=re.subn(r'<!-- B1B4_AUDIT_BEGIN -->.*?<!-- B1B4_AUDIT_END -->',lambda _:section,s,flags=re.S);assert n==1
assert 'id="training-shapes"' in s
for path in (SITE/'trimul.html',A/'web/trimul.html',R.parent.parent/'ANTHROPIC_TRIMUL.html'):path.write_text(s)
assets={'b1b4-shared.svg':'b1b4-shared.svg','b1b4-shared-balanced-history.svg':'b1b4-shared-balanced-history.svg','SOL_AUDIT_20260920.md':'b1b4-sol-audit-20260920.md','sol-audit.json':'b1b4-sol-audit.json','sol-final-paired-results.json':'b1b4-sol-results.json','cueq-default-comparison.json':'b1b4-cueq-default.json','cueq-tuned-comparison.json':'b1b4-cueq-tuned.json','cueq-core-comparison.json':'b1b4-cueq-core.json','dual_ln_prefetch.cu':'b1b4-ln-prefetch.cu'}
for src,dst in assets.items():shutil.copyfile(R/src,SITE/'assets'/dst)
files=['dual_ln_prefetch.cu','dual_ln_prefetch.py','dual_ln_prefetch_timing.cu','dual_primitives.cuh','dual.py','core.py','dual_experiment.py','check_experiment.py','measure_experiment.py','measure_sol_final.py','profile_experiment.py','sanitize_experiment.py','integrate.py','sol_prefetch_audit.sh','check_sol_saved_updates.py','compare_cueq.py','compare_cueq_core.py',*assets.keys(),'dual_ln_prefetch-strict-validation.json','sol-saved-updates.json']
files=list(dict.fromkeys(files))
manifest={f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in files}
(R/'sol-manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(SITE/'assets/b1b4-sol-source.tar.gz','w:gz') as tar:
 for f in files+['sol-manifest.json']:tar.add(R/f,arcname=f)
print(json.dumps({'selected':audit['selected'],'max_errors':maxerr,'strict_cases':24,'sanitizers':len(san),'assets':len(assets),'report_bytes':len(report),'cueq':cueq},indent=2))
