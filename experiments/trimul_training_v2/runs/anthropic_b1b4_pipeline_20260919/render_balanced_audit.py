"""Render selected B1-B4 result from measured artifacts; preserve history."""
from pathlib import Path
import csv, hashlib, html, json, re, shutil, tarfile
import xml.etree.ElementTree as ET
R=Path(__file__).resolve().parent
A=R.parent/'anthropic_adoption_20260919'; SITE=A/'site/dist'
rows=json.loads((R/'balanced-paired-results.json').read_text())
checks=json.loads((R/'dual_balanced-strict-validation.json').read_text())
assert len(checks)==24 and all(c['counters_zero'] for c in checks)
maxerr={k:max(e[k]['relative_l2'] for c in checks for e in c['errors']) for k in checks[0]['errors'][0]}
old=json.loads((R/'optimized-audit.json').read_text())
for name,sha in old['original_sha256'].items():
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha,name
sanitizers={}
for tool in ('memcheck','racecheck','synccheck','split-memcheck'):
    for n in (64,384):
        log=(R/f'balanced-{tool}-L{n}.log').read_text()
        assert 'ERROR SUMMARY: 0 errors' in log or 'RACECHECK SUMMARY: 0 hazards' in log
        sanitizers[f'{tool}/L{n}']='passed'
keys=[k for k in old['profiles']['optimized/L384'] if k!='L2_bytes_derived']+['smsp__inst_executed.sum']
profiles={}
for n in (384,768):
    rr=list(csv.DictReader((R/f'balanced-L{n}.csv').open()));units,vals=rr[0],rr[-1]
    metrics={k:dict(value=float(vals[k].replace(',','')),unit=units[k]) if k in vals else dict(unavailable=True) for k in keys}
    metrics['L2_bytes_derived']=float(vals['lts__t_sectors.sum'].replace(',',''))*32
    profiles[f'balanced/L{n}']=metrics
sass=(R/'dual_balanced.sass').read_text()
sass_counts={op:len(re.findall(r'\b'+op+r'(?:\.|\s)',sass)) for op in ('HGMMA','UTMALDG','UTMASTG','LDG','STG','LDS','STS','LDL','STL')}
assert sass_counts['LDL']==sass_counts['STL']==0
ptxas={}
for m in re.finditer(r'PTXAS dual_balanced (66|132) (1|2) (\S+\.cubin)',(R/'balanced-strict.log').read_text()):
    count,part,cubin=m.groups();path=Path(cubin);log=path.with_suffix('.ptxas.log').read_text()
    assert not re.search(r'(?<!\d)[1-9]\d* bytes spill (?:stores|loads)',log)
    ptxas[f'{count}/{part}']=dict(cubin=str(path),command=json.loads(path.with_suffix('.json').read_text())['command'],registers=re.findall(r'Used (\d+) registers',log),spills=0)
assert len(ptxas)==4
extra=json.loads((R/'balanced-extra-results.json').read_text())
audit=dict(selected='dual_balanced',status='experimental; measured relative target met, L384 has little margin; absolute 173us not met',profiles=profiles,max_relative_l2=maxerr,strict_cases=24,comparisons=72,sanitizers=sanitizers,ptxas=ptxas,sass_static_module_counts=sass_counts,extra_cases=len(extra),extra_passed=sum(c['passed'] for c in extra),extra_caveat='L72 seed20261151 dtri2.1726077e-5, bit-identical to original dual',original_sha256=old['original_sha256'],events='balanced-paired-results.json',full_integration='integrate.backward_cuda; B5+ unchanged; production default unchanged')
(R/'balanced-audit.json').write_text(json.dumps(audit,indent=2))
def table(h,d):
    return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in d)
def htable(h,d):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in d)+'</tbody></table></div>'
def lat(r,k):
    t=r['times'][k];return f'{t["median_us"]:.2f} / {t["p90_us"]:.2f}'
perf=[];full=[];blocks=[]
for r in rows:
    t=r['times'];b=t['baseline']['median_us'];n=t['dual_balanced/part2']['median_us'];o=t['dual_optimized/part2']['median_us']
    perf.append([r['L'],lat(r,'baseline'),lat(r,'dual_optimized/part2'),lat(r,'dual_balanced/part2'),f'{b/n:.4f}x',f'{100*(1-n/o):.1f}%'])
    b=t['full/baseline']['median_us'];n=t['full/dual_balanced/part2']['median_us']
    full.append([r['L'],lat(r,'full/baseline'),lat(r,'full/dual_balanced/part2'),f'{b/n:.4f}x',f'{100*(1-n/b):.1f}%'])
    for i,block in enumerate(r['blocks']['core'],1):
        b=block['baseline']['median_us'];n=block['dual_balanced/part2']['median_us']
        blocks.append([r['L'],i,f'{b:.3f}',f'{n:.3f}',f'{b/n:.6f}x'])
smem=[['0–16384','dy → dp','dy → dp (slot0)'],['16384–32768','gate → dg','gate → row stats / mean / rstd / gamma'],['32768–49152','proj','tri → dtri (first half)'],['49152–65536','xn','tri → dtri (second half)'],['65536–73728','norm (part)','8KiB warp dgamma/dbeta partials'],['73728–98304','norm (remainder)','unused'],['98304–114688','dy → dp (slot1)','dy → dp (slot1)'],['114688–131072','gate → dg (slot1)','gate → statistics (slot1)'],['131072–147456','proj (slot1)','tri → dtri (first half of slot1)'],['147456–163840','xn (slot1)','tri → dtri (second half of slot1)'],['163840–196608','norm (slot1)','resident Wp (first half)'],['196608–229376','unused','resident Wp (second half)'],['229376–231424','initialized, unused','running FP32 dgamma/dbeta sums']]
ph=[]
for fn,key,label in [('diagnostic-results.json','dual_dspref_timing/part2','original dspref'),('optimized-phase-results.json','dual_optimized_timing/part2','previous selected'),('balanced-phase-results.json','dual_balanced_timing/part2','balanced selected')]:
    for r in json.loads((R/fn).read_text()):
        v=r['phases'][key]['median_critical_frontier_us'];ph.append([r['L'],label,*[f'{v[k]:.3f}' for k in ('body','partial_dump_and_publish','grid_wait','reduce_and_publish','reset')]])
prof=[];stalls=[]
for n in (384,768):
    for name,d in [('previous',old['profiles'][f'optimized/L{n}']),('balanced',profiles[f'balanced/L{n}'])]:
        v=lambda k:d[k]['value'];factor={'byte':1,'Mbyte':1e6,'Gbyte':1e9}
        read=v('dram__bytes_read.sum')*factor[d['dram__bytes_read.sum']['unit']];write=v('dram__bytes_write.sum')*factor[d['dram__bytes_write.sum']['unit']]
        prof.append([n,name,f'{v("gpu__time_duration.sum"):.2f}',f'{v("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"):.2f}%',f'{read/1e6:.2f}',f'{write/1e6:.2f}',f'{d["L2_bytes_derived"]/1e6:.2f}',f'{v("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"):.2f}%',f'{v("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")/1e6:.3f}M'])
        stalls.append([n,name,*[f'{v("smsp__average_warps_issue_stalled_"+k+"_per_issue_active.ratio"):.4f}' for k in ('barrier','long_scoreboard','short_scoreboard','membar','wait','sleeping')]])
report='''# B1–B4 training CUDA: asymmetric CTA roles, 2026-09-20

**Selected: dual_balanced.cu + dual_balanced.py.** Exact same-process
core.baseline() Triton/cuBLAS, dropout0.25: **L384 1.7018x; L768 1.9051x**.
L384 has little margin and misses the absolute173us target (176.00us).
This is an experimental plan validated through the full backward harness;
production defaults and autotune dispatch are unchanged.

Anthropic native v5 TriMul CUDA primitives are the foundation. Our previous
inference development was behind Anthropic's results; this work extends
their implementation and ideas to training. These speedups compare training
B1–B4 against Triton/cuBLAS, not against Anthropic inference.

## 1. Operation, shared memory, ownership and assumptions

```
y = dy * ds[row % L]
dp = bf16(y * gate)
dg = bf16(((y * proj) * gate) * (1 - gate))
dWg = bf16(sum_fp32(xn.T @ dg))
dWp = bf16(sum_fp32(dp.T @ norm))
dnorm = bf16(dp @ Wp)
xhat = (tri - mean) * rstd; h = dnorm * gamma
dtri = bf16(rstd * ((h - mean(h)) - xhat * mean(h*xhat)))
dgamma = sum_fp32(dnorm*xhat); dbeta = sum_fp32(dnorm)
```

Forward saves, six outputs and BF16 rounding points are unchanged. B5+
consumes dg/dtri on Triton. Internal scheduling changes: one cooperative
launch has40 weight CTAs and92 input/LN CTAs at UCOUNT132. Each role computes
dp independently; no global dp/dnorm. dy/gate are loaded by both roles.
No clusters/multicast and no separate dW kernel in the selected version.

Compile-time ratio10:23 gives20/46 at count66. Each role processes cyclic
64-row tiles: split+round*role_count. Empty CTAs publish zero partials.
L384: DW57/58 tiles, DX25/26. L768: DW230/231, DX100/101.
Each CTA has256 threads, two128-thread WGs. DW: three64x128 weight-gradient
tiles per WG,192 FP32 accumulators/thread. WG0 owns two dWg and one dWp
tiles; WG1 owns three dWp tiles. DX: each WG handles128 LN channels.
Raw tri and BF16 dnorm stay in registers through LN; no32KiB dnorm
shared-memory round trip.

Full dynamic SMEM map, half-open byte ranges; columns are alternative CTA roles:

'''+table(['Bytes','DW role','DX role'],smem)+'''
Dynamic231424B (226KiB) +static1024B; oneCTA/SM. DW has two96KiB stages.
DX has two64KiB stages,64KiB resident Wp,8KiB warp sums and2KiB running sums.
Selected count66/132 × PART1/2:252 registers, zero stack/spills (hardware
allocation rounds to256). No setmaxnreg rebalance in this selected layout.

Initialize/proxy-fence two mbarriers before issuing TMA. First DW96KiB;
first DX128KiB including Wp. Later DW96KiB/DX64KiB. Slot=round&1;
phase=(round/2)&1. Refill a slot for tile+2*role_count after every consumer
finishes. Generic B1 stores are proxy-fenced before WGMMA; register fences,
commit/wait protect accumulators and operands. CTA barriers publish both
channel halves' LN statistics and all warp partials. Each WG fences dtri,
commits/waits TMA stores before slot reuse. All partial writers threadfence
before ticket publication; volatile final reads follow the grid barrier.
Last completion ticket resets both counters. Same-stream replay begins
after completion. PART1 uses a separate194-block reducer.

Assumptions: contiguous TMA-aligned BF16, C128/H256, L>=64 and L²%64=0,
int32 index range, one stream/workspace per non-reentrant plan. ds is zero
or a common positive BF16 dropout scale. Mask cache applies only when
L/gcd(64*role_count,L)<=3; longer periods use normal ds loads. No L384/L768
or p0.25 special case. Wp/saves are fixed for a plan. Packing/allocation/
compilation/descriptors precede graph capture and event timing.

## 2. Correctness issues in the original and audit findings

- Original host wrapper rejected count>row_tiles. Experiment wrapper allows
  empty CTAs needed for L64 tests; original dual.py remains unchanged.
- No new arithmetic/race defect was established in dual_dspref for required
  shapes. Prior descriptions of52MB as unavoidable HBM traffic and dedicated
  simultaneous dW/dX WGs were inaccurate.
- New slot/fragment lifetimes have explicit synchronization; numerical
  graph replay and requested sanitizer tools pass.
- Extra L72 seed20261151 repeats dtri relativeL2=2.1726077e-5, above the
  large-shape2e-5 bound. Original dual and selected dtri are bit-identical
  across three diagnostic seeds. No universal all-L/all-seed accuracy claim.
- SHA-256 of original dual.cu, dual_dspref.cu, dual_primitives.cuh, dual.py
  and core.py still matches the pre-experiment audit. Previous selected
  kernel is preserved for A/B.

## 3. Time split, bottleneck and roofline

Instrumented completion-frontier medians, us,20 diagnostic launches:

'''+table(['L','Version','Body','Dump+publish','Grid wait','Reduce+publish','Reset'],ph)+'''
A frontier uses the latest CTA completion at each timestamp. Earlier DW
partial dumps overlap DX body;0.704us is exposed tail, not all DW store work.
Per-CTA waits can be much larger. These phases are not headline event timing.
32-bit timer probe rejects wrap-crossing launches, compiles without spills;
uninstrumented/instrumented174.032/173.872us and621.056/620.944us in its
process: no measurable probe penalty. Historical phase deltas are approximate.

Useful row bytes=(2056 read+768 write)*L²=416.416/1665.663MB. Supplied
3.35TB/s gives bandwidth-only minima124.30/497.21us;3.0TB/s gives
138.81/555.22us. GEMM work24.16/96.64GFLOP. This does not rule out1.7x,
but is not an achievable runtime promise. New roles request512B/row extra
dy/gate before L2 reuse: nominal3336B/row,491.913/1967.653MB. Measured HBM
is lower because requests may hit L2. Do not infer partial HBM by subtraction.

Partial bytes one direction=40*49152*4+92*512*4=8,052,736.
Read+write16.105MB versus52.445MB:69.3% less global traffic. Host capacity
remains the oversized26.223MB allocation; only touched regions shrink.
This is not a52MB HBM saving. Body dominates; final reduction about2.5us.

## 4. Ranked experiments and measured decisions

Conditional estimates, not measured savings or additive guarantees:

'''+table(['Priority','Hypothesis; estimated saving L384/L768 vs previous selected','Result'],[
 [1,'Asymmetric roles and fewer partials:20–40 /80–150us','Adopt40/92; final paired savings35.17 /121.68us.'],
 [2,'Two32-row stages in one CTA with register LN:0–25 /0–100us','Built zero-spill variants; best244 /878us, slower. More instructions.'],
 [3,'Three WGs, wider WGMMA or earlier TMA:0–10 /0–40us','Three-WG spills rejected before execution. N128, store widths, early prefetch/reduction gave no reliable gain.']])+'''
Controlled ratio sweep, local tuning (not final headline):66/66 gave
243.024/893.568us;44/88 gave186.032/694.080;42/90 gave175.456/668.736;
40/92 gave174.688/624.928. Rename40/92 to dual_balanced, then independently
validate and repeat. Longer mask periods can lose register-cache benefits;
shared mask-cache variants were also tested but not selected. The32-row
schedule reduced long-scoreboard stalls but raised executed instructions
59.0M→84.3M at L384, explaining its loss.

## 5. Complete code and host integration

- dual_balanced.cu: complete selected device source; existing
  dual_primitives.cuh and Anthropic v5 headers are reused.
- dual_balanced.py: Plan wrapper, count132/PART2 default; PART1 supported.
- dual_experiment.py: source/dependency/flag hashed cubins, zero-spill hard
  gate, workspace/maps and experiment launch support.
- integrate.backward_cuda: B1–B4 replaced in the full backward harness.
- Production dispatch and original sources unchanged. No cutlass-prefixed
  files or toolkit upgrade. Source archive includes local harness files;
  Anthropic headers and MiniWorld engine remain repository dependencies.

## 6. Correctness tests and results

L64/384/768 × p0/.25 × count66/132 × PART1/2 =24 cases. Each normal launch
plus two graph replays with changed dy/ds:72 six-output comparisons, counters
zero throughout. Maximum relativeL2:

'''+table(['Output','Maximum error','Limit'],[[k,f'{v:.9g}',{'dg':'bit-exact incl signed zero','dtri':'2e-5','dgamma':'5e-6','dbeta':'5e-6','dWg':'5e-4','dWp':'5e-4'}[k]] for k,v in maxerr.items()])+'''
Full backward11 gradients pass relativeL2<=5e-4 at both target shapes.
PART2 memcheck/racecheck/synccheck at L64/384 pass. Unfiltered PART1
memcheck also covers standalone reducer, both shapes pass.
Extra L72/80/136 × p.1/.25 × count66/132 × PART1/2:22/24 pass these strict
limits. Two L72 cases repeat the original bit-identical issue in section2.
Do not label this extra matrix all-pass.

## 7. Reproduction commands and measured results

All GPU commands run in an assigned node02 H100 allocation, CUDA12.9/sm90a.
Verification GPU was released; existing training job13228 remained running.
No node01 use. Exact compiler command for selected count132/PART2:

```bash
'''+ ' '.join(ptxas['132/2']['command'])+'''
```

Other validated variants change UCOUNT66/132 and PART_ONLY1/2 only.
The complete timed/sanitizer/NCU sequence is balanced_validation.sh:

```bash
P=runs/anthropic_b1b4_pipeline_20260919
E=runs/anthropic_adoption_20260919/env.sh
bash "$P/balanced_validation.sh"
bash "$E" python -u -B "$P/measure_balanced_final.py"
bash "$E" python -u -B "$P/check_balanced_extra.py"
bash "$E" python -u -B "$P/measure_experiment.py" --sources dual_balanced dual_balanced_timing --parts 2 --output balanced-phase-results.json
```

Sanitizer/profiler commands used for length64/384 (sanitizer) and384/768 (NCU):

```bash
for tool in memcheck racecheck synccheck; do
  for length in 64 384; do
    bash "$E" compute-sanitizer --tool "$tool" --error-exitcode 3 --kernel-name kns=dual_b1b4 python -u -B "$P/sanitize_experiment.py" --source dual_balanced --length "$length" --count 132 --part 2
  done
done
for length in 64 384; do
  bash "$E" compute-sanitizer --tool memcheck --error-exitcode 3 python -u -B "$P/sanitize_experiment.py" --source dual_balanced --length "$length" --count 132 --part 1
done
for length in 384 768; do
  bash "$E" ncu --set full --import-source yes -k regex:dual_b1b4 --profile-from-start off --force-overwrite -o "$P/balanced-L$length" python -u -B "$P/profile_experiment.py" --source dual_balanced --length "$length"
done
```

Primary protocol: same-process CUDA graphs, separate core/full timing
domains,3 blocks each20 warmups+200 events, alternating order. Report pooled
600-sample median/p90, not fastest block. Dropout0.25; no attached profiler.

'''+table(['L','Triton/cuBLAS median/p90 us','Previous CUDA median/p90 us','Balanced median/p90 us','Speedup','Time saved vs previous'],perf)+'''
Full backward, B5+ unchanged:

'''+table(['L','Baseline median/p90 us','Balanced median/p90 us','Speedup','Time reduction'],full)+'''
Every primary core block:

'''+table(['L','Block','Baseline median us','Balanced median us','Speedup'],blocks)+'''
Earlier independent20/200 validation mixed core/full timing domains:
L384298.240→175.008us (1.704x), L7681189.136→645.728us (1.842x).
PART1 gave175.024/646.064us, similar: keep simple split-reducer fallback.
Those results remain in balanced-final-results.json. Separate-domain
repeated results above are primary; warm cache state and system/clock
variation affect timing. No locked-clock claim.

## 8. Profiler signals: observed versus expected

NCU full-replay durations differ from warm event medians:

'''+table(['L','Version','NCU us','DRAM %','HBM read MB','HBM write MB','L2 MB derived','Tensor active %','Bank conflicts'],prof)+'''
DRAM>70% goal: L38460.51% misses, L76870.62% reaches the profiler threshold.
This does not prove roofline saturation. Instruction work, shared stores,
barriers and cache traffic remain.252 registers and local load/store
sectors0 on both shapes agree with ptxas. L2 bytes=sectors*32; direct
lts__t_bytes.sum unavailable. Independently replayed aggregate/submetric
bank counters need not add exactly; avoid inferring exact overlap from them.

Stall metrics are per-issue-active ratios, not elapsed-time percentages:

'''+table(['L','Version','Barrier','Long scoreboard','Short scoreboard','Membar','Wait','Sleeping'],stalls)+'''
Long-scoreboard and wait ratios improve. Exposed spin-barrier tail<1us;
removing only it cannot yield another large gain. Shared-store conflicts
and CTA barriers remain. Static module SASS, including standalone reducer:

'''+table(['Instruction','Static count'],list(sass_counts.items()))+'''
Explicit TMA/WGMMA and zero local spills confirmed. Static counts are not
dynamic traffic. Selected executed instructions59.46M/232.10M; rejected
32-row schedule84.34M at L384.

## 9. Gap, risks and the single next experiment

Relative1.7x passes all3 target-shape core blocks. L384 pooled1.7018x has
only0.19us margin against299.52/1.7=176.19us; this is not a robust guarantee
across machines/runs. Absolute173us is still missed by3.00us. L768620.48us
beats its693us target. Full backward1.122x/1.143x because B5+ is unchanged;
no end-to-end training-step speedup has been established.

Limits: fixed C128/H256, single-stream non-reentrant workspace, no production
dispatch/cache promotion, extra-L72 original numerical caveat, shape-dependent
mask periods/tails, cache/clock variability. Prior sources retained.

**Single next experiment:** change only DX warp-partial shared-memory layout
to reduce store-bank conflicts. Keep40/92 roles, fragment arithmetic and TMA
maps fixed. Conditional estimate2–5us at L384 /5–15us at L768, not a promise.
Require lower NCU bank/store stalls, zero spills, strict checks and repeated
same-process gains. Proposed only; not implemented or claimed faster.
'''
(R/'BALANCED_AUDIT_20260920.md').write_text(report)

# Code-native SVG of actual operation ownership and shared/global boundaries.
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1330" role="img" aria-labelledby="title desc"><title id="title">B1–B4 CUDA: 40 DW plus 92 DX CTAs in one launch</title><desc id="desc">Anthropic v5 based training extension; two input stages per role, register LN, no global dp or dnorm; B5+ remains Triton.</desc><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#58717b"/></marker></defs><style>text{font-family:Arial,sans-serif;fill:#18343f}.edge{fill:none;stroke:#58717b;stroke-width:2;marker-end:url(#arr)}</style><rect width="1200" height="1330" fill="#f4f8fa"/>']
def rect(x,y,w,h,fill='#fff',stroke='#a7bac6',sw=1.5):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def txt(x,y,s,size=16,bold=False):
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}">{html.escape(s)}</text>')
def edge(d):svg.append(f'<path d="{d}" class="edge"/>')
txt(35,45,'TriMul 학습 B1–B4 · dual_balanced · 현재 선택안',26,True)
txt(35,76,'Anthropic v5 CUDA primitives 계승 · 1 cooperative launch · 132 CTAs × 256 threads')
rect(35,100,1130,90,'#e9eff4');txt(55,129,'입력 / forward 저장값 · 변경 없음',18,True)
txt(55,156,'dy, ds, xn, gate, proj, norm, tri, mean, rstd, gamma, Wp')
txt(55,179,'M=L², C=128, H=256 · dp / dg / dnorm의 BF16 반올림 시점 유지',14)
rect(25,216,1150,754,'#fff','#367d70',3)
txt(48,247,'하나의 CUDA 커널 · 내부 CTA 역할을 분리 (cluster / multicast 없음)',20,True)
edge('M322 190V275');edge('M878 190V275')
rect(48,275,535,560,'#edf8f3','#73b099');rect(615,275,535,560,'#eef3fc','#8fa9d4')
for x,title,inp,buf in [(70,'DW · 40 CTAs · B1 + B2 + B3b','TMA: dy, gate, proj, xn, norm','96 KiB slot0 ↔ 96 KiB slot1'),(637,'DX / LN · 92 CTAs · B1 + B3a + B4','TMA: dy, gate, tri + resident Wp','64 KiB slot0 ↔ 64 KiB slot1 + 64 KiB Wp')]:
    txt(x,309,title,20,True);txt(x,340,inp);txt(x,365,buf)
    edge(f'M{x+245} 379V405');rect(x,405,491,91);edge(f'M{x+245} 496V526');rect(x,526,491,155);edge(f'M{x+245} 681V709')
txt(87,432,'B1 · y=dy×ds; dp=bf16(y×gate)',17,True)
txt(87,458,'dg=bf16(y×proj×gate×(1−gate))');txt(87,482,'dp / dg → shared; dg는 global 출력으로도 기록',14)
txt(654,432,'B1 · y=dy×ds; dp=bf16(y×gate)',17,True)
txt(654,458,'DX 역할에서 dp를 다시 계산 → shared');txt(654,482,'DW의 dp를 global로 주고받지 않음',14)
txt(87,555,'WGMMA · 두 WG가 6개 weight tile 분담',17,True)
txt(87,585,'B2: dWg += xnᵀ @ dg');txt(87,615,'B3b: dWp += dpᵀ @ norm')
txt(87,644,'FP32 192 accumulators/thread · tile 반복 누적',14);txt(87,666,'owned tile: split, split+40, split+80, …',14)
txt(654,555,'B3a · WGMMA: dnorm=bf16(dp @ Wp)',17,True)
txt(654,585,'dnorm + tri를 register fragment에 유지');txt(654,615,'B4 · LayerNorm backward',17,True)
txt(654,644,'dtri → stmatrix → TMA global store',14);txt(654,666,'owned tile: split, split+92, split+184, …',14)
txt(70,736,'반복 종료: dWg / dWp partial 기록',17,True);txt(70,764,'40 × 49152 FP32 = 7.864 MB');txt(70,797,'float2 vector store · 두 WG 모두 dW 담당',14)
txt(637,736,'반복 종료: dgamma / dbeta partial 기록',17,True);txt(637,764,'92 × 512 FP32 = 0.188 MB');txt(637,797,'warp partial 8 KiB + running sums 2 KiB shared',14)
edge('M315 835V864H599V890');edge('M883 835V864H599')
rect(70,890,1058,58,'#f8f0df','#d7b869');txt(89,915,'threadfence → grid ticket barrier → 최종 FP32 reduction → BF16 dW / FP32 LN grads',17,True)
txt(89,937,'PART2: 이 launch 안에서 완료 · counters reset / PART1: 별도 unified_reduce fallback 유지',14)
edge('M600 970V1004');rect(35,1004,1130,79,'#e9eff4')
txt(55,1035,'6개 출력: dg, dWg, dtri, dgamma, dbeta, dWp',17,True)
txt(55,1062,'dg / dtri → 기존 Triton B5+ · forward 저장값과 후속 연결 그대로')
rect(35,1110,1130,186)
txt(55,1140,'Dropout 25% · CUDA events · 3 × 200 samples · Triton/cuBLAS 기준',17,True)
txt(55,1170,'L384: 299.52 → 176.00 µs (1.702×)     L768: 1182.08 → 620.48 µs (1.905×)',20,True)
txt(55,1200,'전체 backward: 1.122× / 1.143× · 실험 경로이며 production 기본값은 아직 미변경')
txt(55,1228,'partial 읽기+쓰기 52.45 → 16.11 MB (global traffic; 전부 HBM은 아님)')
txt(55,1256,'252 registers, spill 0 · required 24 cases + sanitizer 통과 · NCU DRAM 60.5% / 70.6%')
txt(55,1280,'L384 절대 173 µs 목표 미달. 추가 L72 한 seed의 원본 수치오차는 상세 보고서에 명시.',14)
svg=''.join(svg)+'</svg>';ET.fromstring(svg)
hist=R/'b1b4-shared-optimized-history.svg'
if not hist.exists():shutil.copyfile(R/'b1b4-shared.svg',hist)
(R/'b1b4-shared.svg').write_text(svg)
section='''<!-- B1B4_AUDIT_BEGIN --><section class="panel" id="b1b4-audit"><span class="pill green">2026-09-20 · 최신 반복 실측 · B1–B4 1.702× / 1.905×</span><h2>B1–B4 학습 CUDA · 40 DW + 92 DX/LN CTA</h2>
<p>Anthropic v5 CUDA primitives를 계승한 학습 커널. 한 launch 안에서 <b>역할별 CTA 배정 + 입력 이중 버퍼 + register LN</b>을 적용했다. Forward 저장값·BF16 반올림·6개 출력·Triton B5+ 연결을 유지한다. 두 역할은 dp를 각각 재계산하며 dy/gate도 각각 읽는다.</p>
<p>우리의 기존 inference 개발보다 Anthropic의 결과가 우수했고, 이를 기반으로 학습 경로를 확장한다. 아래 배속은 정확히 <code>core.baseline()</code>의 <b>Triton/cuBLAS B1–B4 대비</b>다.</p>'''+htable(['L','Triton/cuBLAS µs median/p90','이전 CUDA µs median/p90','현재 CUDA µs median/p90','배속','이전 CUDA 대비 시간 감소'],perf)+'''
<div class="notice amber"><b>L384는 목표선에 가깝다.</b> 반복3개 block은1.7008–1.7027×. Pooled176.00µs로 상대1.7×는 넘었지만 절대173µs 목표에는3µs 부족하다. L768은 별도 검증 실행1.842×, 최신 반복 측정1.905×였다.</div><h3>전체 backward · B5+ 동일</h3>'''+htable(['L','기준 µs median/p90','현재 µs median/p90','배속','시간 감소'],full)+'''
<p>node02 H100 · dropout25% · 같은 process에서 순서 교대 · core/full 각각20 warmup+200 samples를3회 반복. 표는600개 pooled median/p90. <b>Production 기본 경로는 아직 바꾸지 않았다.</b></p><h3>연산 연결 / 융합 / 저장 위치</h3><p><a href="assets/b1b4-shared.svg">SVG 원본 크게 보기</a></p><img src="assets/b1b4-shared.svg" alt="하나의 B1부터 B4 CUDA 커널 안에서 40 DW CTA와 92 DX LayerNorm CTA의 연산 및 저장 위치" style="width:100%;height:auto"/>
<h3>검증과 남은 병목</h3><p>요구된24개 조합·입력 변경 graph replay72회 비교·sanitizer3종 통과.252 registers, spill0. 추가 L72 한 seed의 dtri 오차2.17e-5는 원본과 비트 단위로 동일하다. 추가 shape 검사는22/24 통과이며 전부 통과로 표시하지 않는다.</p>
<p>NCU DRAM60.5% /70.6%; local load/store0. 부분합 global 읽기·쓰기52.45→16.11MB. 전부 HBM은 아니며 workspace 할당 크기는 그대로다. 다음 후보는 DX shared partial store의 bank conflict 감소다.</p>
<p><a href="assets/b1b4-balanced-audit-20260920.md">9개 항목 상세 보고서</a> · <a href="assets/b1b4-balanced-audit.json">검증 / NCU 요약</a> · <a href="assets/b1b4-balanced-results.json">600 samples 원자료</a> · <a href="assets/b1b4-balanced.cu">CUDA 소스</a> · <a href="assets/b1b4-balanced-source.tar.gz">재현 소스 묶음</a></p><details><summary>상세 보고서: 수식·SMEM·시간 분해·실험·재현 명령·한계</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'''+html.escape(report)+'''</pre></details>
<details><summary>이전 선택안1.42× /1.58× 기록</summary><p>같은 CTA에서 dW→dX를 순차 실행했던 이전 버전이다.</p><a href="assets/b1b4-audit-20260920.md">이전 보고서</a> · <a href="assets/b1b4-shared-optimized-history.svg">이전 배선 SVG</a></details></section><!-- B1B4_AUDIT_END -->'''
s=(SITE/'trimul.html').read_text();s,n=re.subn(r'<!-- B1B4_AUDIT_BEGIN -->.*?<!-- B1B4_AUDIT_END -->',lambda _:section,s,flags=re.S);assert n==1
s=s.replace('<a href="#b1b4-shared">현재 B1–B4 CUDA</a>','<a href="#b1b4-shared">이전 B1–B4 기록</a>')
for path in (SITE/'trimul.html',A/'web/trimul.html',R.parent.parent/'ANTHROPIC_TRIMUL.html'):path.write_text(s)
assets={'b1b4-shared.svg':'b1b4-shared.svg','b1b4-shared-optimized-history.svg':'b1b4-shared-optimized-history.svg','BALANCED_AUDIT_20260920.md':'b1b4-balanced-audit-20260920.md','balanced-audit.json':'b1b4-balanced-audit.json','balanced-paired-results.json':'b1b4-balanced-results.json','dual_balanced.cu':'b1b4-balanced.cu'}
for src,dest in assets.items():shutil.copyfile(R/src,SITE/'assets'/dest)
files=['dual_balanced.cu','dual_balanced.py','dual_primitives.cuh','dual.py','core.py','dual_experiment.py','check_experiment.py','measure_experiment.py','measure_balanced_final.py','profile_experiment.py','sanitize_experiment.py','integrate.py','balanced_validation.sh','dual_balanced_timing.cu','check_balanced_extra.py','BALANCED_AUDIT_20260920.md','balanced-audit.json','dual_balanced-strict-validation.json','balanced-extra-results.json','balanced-extra72-results.json','balanced-final-results.json','balanced-paired-results.json']
(R/'balanced-manifest.json').write_text(json.dumps({f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in files},indent=2))
with tarfile.open(SITE/'assets/b1b4-balanced-source.tar.gz','w:gz') as tar:
    for f in files+['balanced-manifest.json']:tar.add(R/f,arcname='b1b4-balanced/'+f)
print(json.dumps(dict(report=str(R/'BALANCED_AUDIT_20260920.md'),strict_cases=24,extra_passed=sum(c['passed'] for c in extra),assets=7)))
