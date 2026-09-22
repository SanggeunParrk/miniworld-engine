"""Render measured B1-B4 status and retain historical result sections."""
from pathlib import Path
import json,re,hashlib,shutil,html,xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;a=r.parent/'anthropic_adoption_20260919';site=a/'site'/'dist';root=r.parent.parent
rows=json.loads((r/'optimized-final-results.json').read_text());audit=json.loads((r/'optimized-audit.json').read_text())
phase=json.loads((r/'optimized-phase-results.json').read_text());oldphase=json.loads((r/'diagnostic-results.json').read_text())
def table(headers,data):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in data)
def time(row,key):return row['times'][key]['median_us']
perf=[];full=[]
for row in rows:
 t=row['times'];base=time(row,'baseline');new=time(row,'dual_optimized/part2');old=time(row,'dual/part2')
 perf.append([row['L'],f'{base:.2f} / {t["baseline"]["p90_us"]:.2f}',f'{old:.2f}',f'{new:.2f} / {t["dual_optimized/part2"]["p90_us"]:.2f}',f'{base/new:.3f}x',f'{100*(1-new/old):.1f}%'])
 b=time(row,'full/baseline');n=time(row,'full/dual_optimized/part2')
 full.append([row['L'],f'{b:.2f} / {t["full/baseline"]["p90_us"]:.2f}',f'{n:.2f} / {t["full/dual_optimized/part2"]["p90_us"]:.2f}',f'{b/n:.3f}x',f'{100*(1-n/b):.1f}%'])
smem=[['0–16384','16 KiB','dy → dp'],['16384–32768','16 KiB','gate → dg'],['32768–49152','16 KiB','proj → LN row statistics / mean / rstd / gamma'],['49152–65536','16 KiB','xn'],['65536–98304','32 KiB','norm'],['98304–131072','32 KiB','tri → dtri'],['131072–196608','64 KiB','resident packed Wp'],['196608–229376','32 KiB','BF16 dnorm'],['229376–231424','2 KiB','FP32 dgamma / dbeta CTA sums']]
ph=[]
for old,new in zip(oldphase,phase):
 for name,data in [('old dspref',old['phases']['dual_dspref_timing/part2']),('selected',new['phases']['dual_optimized_timing/part2'])]:
  v=data['median_critical_frontier_us'];ph.append([new['L'],name,*[f'{v[k]:.3f}' for k in ['body','partial_dump_and_publish','grid_wait','reduce_and_publish','reset']]])
prof=[]
for n in (384,768):
 for name in ('dual-dspref','optimized'):
  d=audit['profiles'][f'{name}/L{n}'];v=lambda k:d[k]['value']
  unit={'Mbyte':1e6,'Gbyte':1e9,'byte':1};traffic=sum(d[k]['value']*unit[d[k]['unit']] for k in ['dram__bytes_read.sum','dram__bytes_write.sum'])/1e6
  prof.append([n,name,f'{v("gpu__time_duration.sum"):.2f}',f'{v("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"):.2f}%',f'{traffic:.2f}',f'{d["L2_bytes_derived"]/1e6:.2f}',f'{v("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"):.2f}%',f'{v("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")/1e6:.3f}M'])
text='''# B1–B4 CUDA training kernel — measured audit, 2026-09-20

**Selected: `dual_optimized.cu` + `dual_optimized.py`. 1.7x target NOT met.**
Anthropic's native v5 TriMul CUDA implementation supplies the TMA/WGMMA,
swizzle and matrix-load/store primitives. This experiment extends that work
to training. These results compare **B1–B4 against `core.baseline()`
(Triton/cuBLAS)**; they are not a claim to beat Anthropic inference.
All compilation, execution and profiling used one Slurm-assigned H100 on
node02. Existing training job 13228 was left running.

## 1. Operation, shared memory, warpgroup ownership and assumptions

`M=L², C=128, H=256`, L>=64 and M divisible by 64. Inputs/maps are contiguous
and TMA-aligned. Dimensions fit signed 32-bit indexing. `ds` is zero or a
common positive BF16 dropout scale. One plan owns one stream/workspace and
is not reentrant. The cooperative grid fits resident SM capacity.

```
y     = dy * ds[row % L]
dp    = bf16(y * gate)
dg    = bf16(((y * proj) * gate) * (1 - gate))
dWg   = bf16(sum_fp32(xn.T @ dg))
dWp   = bf16(sum_fp32(dp.T @ norm))
dnorm = bf16(dp @ Wp)
xh    = (tri - mean) * rstd
h     = dnorm * gamma
dtri  = bf16(rstd * ((h - mean(h)) - xh * mean(h*xh)))
dgamma = sum_fp32(dnorm * xh); dbeta = sum_fp32(dnorm)
```

Forward saves and all BF16 rounding points are unchanged. B5+ remains on
the existing Triton path, consuming dg/dtri. Host allocations, descriptors
and Wp packing happen before CUDA-event timing.

Each 256-thread CTA has two warpgroups. Both execute dW, then dX/LN;
there is no simultaneous dedicated dW and dX warpgroup in the selected
kernel. WG0 owns two 64x128 dWg tiles plus one dWp tile; WG1 owns three dWp
tiles. Each thread retains 192 FP32 dW accumulators. The two WGs then each
handle 128 LN channels. CTA b owns tiles `b, b+UCOUNT, ... < M/64`, including
ragged and empty tails. This cyclic order is the intentional scheduling
change from the original contiguous partition.

Dynamic shared-memory map, half-open byte intervals:

'''+table(['Offset','Size','Contents / reuse'],smem)+'''
Total dynamic SMEM 231,424 bytes (226 KiB), plus compiler-rounded static
storage 1,024 bytes. ptxas: **255 registers, 0 stack, 0 spill loads/stores**
for the selected PART1/PART2, counts66/132. This leaves no register headroom.

Synchronization comments are in the selected .cu. Barrier0 joins 256
threads; barriers1/2 join their respective WGs. One thread initializes the
mbarrier and publishes it before the first CTA barrier. First TMA phase
expects 196,608 bytes (128 KiB rows +64 KiB Wp); subsequent phases expect
131,072 bytes. Next-tile prefetch issues 80 KiB after dnorm/stat consumption;
late proj/tri loads supply 48 KiB. Phase toggles once per owned tile.
Empty CTAs issue/wait no TMA, but publish zero partials and join reduction.
Generic dp/dg writes are proxy-fenced before WGMMA. WGMMA register fences,
commit/wait protect accumulators and operands. CTA sync publishes both WGs'
LN row statistics. Each WG fences dtri stores before TMA output, then waits
before buffer reuse. Every partial writer threadfences before the leader
publishes its ticket; final loads remain volatile/coherent. The last
completion ticket resets both counters, and the next same-stream launch
waits for kernel completion. PART1 uses a separate reduction kernel.

## 2. Correctness issues found in the old implementation

- The original host wrapper rejected count>row_tiles, preventing requested
  L64/count66 or132 tests. New wrapper permits empty CTAs; their correctness
  and synchronization now pass numerical and sanitizer checks.
- Old tests used one loose 0.001 threshold. New checks enforce all six
  individual limits and signed-zero-sensitive BF16 bit equality for dg.
- No device race/OOB defect was demonstrated in the original by this audit.
- An extra L72 generic-shape probe found dtri relative L2 2.173e-5 for one
  seed. Original and optimized dtri were bit-identical. This exceeds 2e-5
  at that extra shape; it is not a regression and is not hidden as a pass.
  The requested L64/384/768 matrix passes. Universal tolerances for every
  arbitrary L/seed are not established.
- “52 MB partial traffic is all HBM”, “32% HBM/6% tensor is the latest
  profile”, and “moving Wp alone permits full double buffering” were
  inaccurate assumptions; measurements/accounting below correct them.

## 3. Time split, bottleneck and roofline

Diagnostic completion-frontier medians in us (20 launches). Each frontier
uses the latest CTA completion; separate per-CTA medians must not be added.

'''+table(['L','Version','Body','Dump + publish','Grid wait','Reduce + publish','Reset'],ph)+'''
These are instrumented diagnostics, not the headline benchmark. The selected
32-bit globaltimer probe costs about 1.8 /1.7 us at L384/768 and has zero
spills; launches crossing its 4.29s wrap are rejected. A first 64-bit probe
spilled and was rejected before execution. Original diagnostic data predates
optimization, selected diagnostics are fresh; phase deltas are approximate.

Useful row traffic: `(2056 read +768 write)*M = 416.416 MB /1665.663 MB`.
At 3.35 TB/s the bandwidth-only lower bounds are **124.30 /497.21 us**;
at the supplied 3.0 TB/s assumption, **138.81 /555.22 us**. Wp/gamma/ds
cache behavior and output weight epilogue are not included in this row bound.
GEMM work is 24.16 /96.64 GFLOP. The bound does not rule out 1.7x.

Partial stores+reads are `2*132*(49152+512)*4 = 52.445 MB` in global address
space, but mostly L2-served. Measured DRAM is close to row traffic; removing
all partial HBM traffic cannot be credited as a 52 MB bandwidth saving.
Grid waiting is below 1 us on the critical completion frontier. The body,
not the final spin barrier, now dominates.

Full double buffering needs `2*128+64+32+2=354 KiB`; removing Wp still leaves
290 KiB. A two-CTA role split with multicast and two stages was implemented
and tested; it is slower, including after register-resident LN optimization.

## 4. Ranked optimization plan, estimates and measured decisions

'''+table(['Priority','Experiment / conditional estimate L384, L768','Measured decision'],[
[1,'Vector partial stores/reduction: save 3–15 /3–20 us','Adopted float2 stores. float4 layout/tree reduction and fewer CTAs did not improve selected path.'],
[2,'TMA/row scheduling: save 10–35 /40–140 us','Adopted cyclic tiles and lossless mask reuse. Earlier issue/no-wait variants failed spill or speed gate.'],
[3,'Producer/consumer/two-stage split: save 0–25 /0–100 us','Implemented CTA roles, multicast, double buffering and register LN; all slower. Rejected.']])+'''
Estimates were hypotheses, not additive promises. Main controlled steps:
original ~251/866 us → float2 ~239/850 → cyclic ~229/825 → mask ~209/736.
Each step was measured against its predecessor in the same process.
Mask reuse applies only when `(64*UCOUNT)%L==0`; other lengths/counts retain
generic mask loads. The zero/common-scale representation is lossless and
reinitialized on every launch, including graph replay after mask changes.

Additional measured failures:

'''+table(['Family','Outcome / evidence'],[
['FP32 float4 partial layout, four-lane reduction','No material gain; vec4-results.json, tree4-results.json.'],
['UCOUNT 120/108/96','Fewer partials but less SM parallelism; count132 fastest (counts-results.json).'],
['LN unroll / retaining parameter sums','Several variants spilled; rejected by loader before GPU launch (epilogue-results.json).'],
['dtri TMA boxes 32/64/128 channels','No material gain over16; store-width experiments.'],
['Two CTA roles / two stages / multicast','~295/1081 us before fragment LN; fragments ~243/901 us, slower than209/742 controls (fragments-results.json).'],
['Shared-bank permutation + planar row stats','~220/779 us, slower than209/737 (banks-results.json).'],
['Redundant barrier removals','~209/730–733 us; no convincing broad gain (barriers-results.json).']])+'''
The selected source retains the fully validated original synchronization.

## 5. Complete optimized code

- `dual_optimized.cu`: selected complete CUDA entrypoints, annotated fences,
  barriers and workspace ownership; new file, original sources unchanged.
- `dual_optimized.py`: Plan wrapper, `part=1` explicit split fallback.
- `dual_experiment.py`: common host maps/workspace/cache, exact nvcc commands
  recorded by content hash; refuses any compiler-reported spills.
- Existing `dual_primitives.cuh` imports the Anthropic v5 primitives.
- `check_experiment.py`, `measure_experiment.py`, `profile_experiment.py`,
  `sanitize_experiment.py`, `prepare_timing.py`: reproducible harnesses.
- `optimized-audit.json` verifies SHA-256 of the five original files.

This is an experimental plan exercised through the engine backward in
`integrate.py`. No production default or cache dispatch was changed.

## 6. Correctness and sanitizer results

24 combinations = L64/384/768 × dropout0/.25 × count66/132 × PART1/2.
Each has one ordinary launch and two graph replays with changed dy and mask:
**72 six-output comparisons passed**; both counters zero after every call.
Fixed forward seed341 plus recorded gradient seeds. Maximum relative L2:

'''+table(['Output','Observed max','Required'],[[k,f'{v:.8g}',str({'dg':0,'dWg':5e-4,'dtri':2e-5,'dgamma':5e-6,'dbeta':5e-6,'dWp':5e-4}[k])] for k,v in audit['max_relative_l2'].items()])+'''
`dg` is bit-exact in all tests. Full backward's eleven returned gradients
also pass relative L2<=5e-4. L64 and L384 each pass memcheck, racecheck and
synccheck for PART2/count132, with p=.25 and two changed-input graph replays.
The PART1 fallback additionally passes unfiltered memcheck, including its
separate reducer, at both sizes. Sanitizer logs and exit statuses are saved.
No all-shape mathematical accuracy proof or multi-stream reentrancy claim.

## 7. Reproduction commands and measured results

Run inside the already assigned node02 GPU allocation. CUDA12.9/sm_90a,
20 warmups, 200 paired CUDA-event samples with alternating order, p=.25.
Graphs contain one invocation; host overhead is excluded. No toolkit upgrade.

B1–B4 latency, us (median /p90 where shown):

'''+table(['L','Triton/cuBLAS','Old dual median','Selected median /p90','Speedup vs baseline','Time saved vs old dual'],perf)+'''
Whole backward, same process, B5+ unchanged:

'''+table(['L','Baseline median /p90 us','Selected median /p90 us','Speedup','Time saved'],full)+'''
PART1 medians are 210.70 /745.94 us for B1–B4; PART2 210.61 /748.13.
A separate paired diagnostic run gave selected 209.06 /736.16 us against
299.07 /1178.22 us (1.43x /1.60x). Both runs are retained; headline numbers
above use the full comparison run, rather than the most favorable result.
B5+ uses the same pre-existing capped Triton tuning cache on both sides;
these absolute whole-backward numbers do not establish exhaustive B5+ tuning.

```bash
cd /home/psk6950/MiniWorld
P=runs/anthropic_b1b4_pipeline_20260919
E=runs/anthropic_adoption_20260919/env.sh
I=runs/trimul_sm90_parity_20260917/engine/third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc
bash "$E" nvcc -std=c++17 -O3 -arch=sm_90a --cubin -lineinfo \\
  -Xptxas=-v -I"$I" -DROLE=0 -DUCOUNT=132 -DPART_ONLY=2 \\
  "$P/dual_optimized.cu" -o "$P/build/dual_optimized-explicit.cubin"
bash "$E" python -u -B "$P/check_experiment.py" --source dual_optimized
bash "$E" python -u -B "$P/measure_experiment.py" \\
  --sources dual dual_dspref dual_optimized --parts 1 2 --full \\
  --output optimized-final-results.json
for tool in memcheck racecheck synccheck; do
  for length in 64 384; do
    bash "$E" compute-sanitizer --tool "$tool" --error-exitcode 3 \\
      --kernel-name kns=dual_b1b4 python -u -B "$P/sanitize_experiment.py" \\
      --source dual_optimized --length "$length" --count 132 --part 2
  done
done
for length in 384 768; do
  bash "$E" ncu --set full --import-source yes -k regex:dual_b1b4 \\
    --profile-from-start off --force-overwrite -o "$P/optimized-L$length" \\
    python -u -B "$P/profile_experiment.py" --source dual_optimized --length "$length"
done
python3 "$P/prepare_timing.py" --sources dual_optimized --timer-bits 32
bash "$E" python -u -B "$P/measure_experiment.py" \\
  --sources dual_optimized dual_optimized_timing --parts 2 \\
  --output optimized-phase-results.json
```

`final_validation.sh` records the full verification sequence. Every cached
build stores exact commands/ptxas evidence in `build/<sha>.json/.ptxas.log`.
Selected PART2/count132 cubin SHA-key:
`a1759201404153b87bb5b1e5ac0055b84b604215ccc714b4097b3f0a5f1a51ca`.

## 8. Profiler signals: expectation versus observation

Fresh NCU full reports, 38 replay passes per kernel. NCU times have profiler
and cache-state effects; never substitute them for event medians.

'''+table(['L','Source','NCU us','DRAM peak %','DRAM MB','L2 MB derived','Tensor active %','Shared bank conflicts'],prof)+'''
Expected DRAM>70% was **not reached**. Registers remain255; local load/store
sectors are zero. L2 traffic falls by about13%, while DRAM volume barely
changes. Long-scoreboard stall ratios drop ~2.01→1.00 and1.90→0.86. Barrier
ratios drop1.29→1.15 and1.10→0.94. Sleeping/spin ratios stay tiny. Short
scoreboard and wait remain near1.0. Bank conflicts increase ~9% despite
faster execution; blindly eliminating one access pattern made the measured
kernel slower. These ratios are per issued instruction, not time fractions.

`lts__t_bytes.sum` was not present in the full-report export; L2 bytes here
are explicitly derived from `32*lts__t_sectors.sum`. The report includes
requested metric units and missing-metric handling. NCU emitted an unrelated
post-report Python encoding warning, but .ncu-rep export and metrics succeeded.

SASS for the measured main symbol, excluding the separate reducer:
192 scalar partial STG become96 STG.64; remaining scalar STG198→6.
WGMMA count remains28 and TMA load/store count31/4. Static instruction count
3689→3873 because mask reuse adds a generic branch, but repeated mask loads
are avoided dynamically. Both have zero LDL/STL. `sass-audit.json` and the
full disassemblies preserve the evidence.

## 9. Remaining gap, risks and single next experiment

'''
for row in rows:
 b=time(row,'baseline');n=time(row,'dual_optimized/part2');target=b/1.7
 text+=f'- L{row["L"]}: {n:.2f} us now; target {target:.2f} us; must remove {n-target:.2f} us ({100*(1-target/n):.1f}% of current latency).\n'
text+='''
The roofline does not establish impossibility. Current register saturation
and body dependencies are the practical limit of this schedule, not a proved
hardware upper bound. Fewer partials or removing the spin barrier alone
cannot close the gap. The selected path is experimental, fixed C/H,
single-stream, and has the extra-L72 accuracy caveat described above.

**Single next experiment:** prototype a 32-row staged input schedule with
an explicit padding/lifetime proof for WGMMA's 64-row dnorm tile. Halving the
row working set could permit two input stages in one CTA, unlike the failed
two-CTA role split. Budget before padding is128 KiB staged rows +64 KiB Wp
+32 KiB dnorm +2 KiB LN =226 KiB; therefore zero-padding must reuse dead
buffers, with no additional SMEM. Validate TMA swizzle/tail layout first;
reject on spills, races or no event-timed gain. This is proposed, not built
or claimed to meet1.7x.
'''
(r/'AUDIT_20260920.md').write_text(text)
# Keep the source-native vector wiring and annotate actual scheduling/changes.
svg=(r/'b1b4-shared-before-audit.svg').read_text()
svg=svg.replace('0 0 1160 1130','0 0 1160 1240').replace('height="1130"','height="1240"')
svg=svg.replace('Persistent CTA · shared memory에서 타일을 재사용','Persistent CTA · tile b, b+132, … / 같은 WG가 dW → dX 순서로 실행')
svg=svg.replace('128-bit mask 읽기와 dg 쓰기. dp는 HBM에 저장하지 않음.','주기 일치 시 mask 32-bit 재사용; 그 외 원래 읽기. dp는 shared에 유지.')
svg=svg.replace('<path d="M690 373V410" class="edge"/>','<path d="M560 475H590" class="edge"/>')
svg=svg.replace('타일 반복 종료 → FP32 부분 dW와 dgamma/dbeta를 global workspace에 기록','타일 반복 종료 → dW partial을 FP32 × 2 vector store로 기록 (96 × STG.64/thread)')
svg=svg.replace('HBM · FP32 partial workspace','Global · FP32 partial workspace').replace('입력 반복 읽기는 줄였지만 부분합 HBM은 존재','global 왕복 52.45 MB · 대부분 L2에서 처리')
svg=svg.replace('</svg>','''<rect x="35" y="1120" width="1090" height="98" rx="12" fill="#fff3d8" stroke="#d9bb6f"/>
<text x="53" y="1149" class="b">2026-09-20 · B1–B4: L384 1.42× / L768 1.58× · 1.7× 목표 미달</text>
<text x="53" y="1175" class="small">p=0.25 · 같은 process의 Triton/cuBLAS 기준 · 20 warmup + 200 CUDA-event samples</text>
<text x="53" y="1199" class="small">24 조합·graph replay 통과 / sanitizer 3종 통과 / 255 registers, zero spills / 실험 경로</text></svg>''')
ET.fromstring(svg);(r/'b1b4-shared.svg').write_text(svg)
# Current summary first; older measured sections remain labeled by their dates.
tr=''.join(f'<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[3]}</td><td>{row[4]}</td></tr>' for row in perf)
fr=''.join(f'<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td></tr>' for row in full)
section='''<!-- B1B4_AUDIT_BEGIN --><section class="panel" id="b1b4-audit"><span class="pill amber">2026-09-20 · 최신 실측 / 1.7× 목표 미달</span><h2>B1–B4 학습 CUDA: 현재 선택안</h2>
<p>Anthropic v5 CUDA primitives를 계승한 학습 커널. <b>partial vector store + cyclic tile + dropout mask reuse</b>를 선택했다. 기존 forward 저장값·BF16 반올림·B5+ 연결을 유지한다. 아래 배속은 Triton/cuBLAS B1–B4 기준이며 Anthropic 추론 대비 배속이 아니다.</p>
<div class="table-wrap"><table><thead><tr><th>L</th><th>Triton/cuBLAS µs<br>median /p90</th><th>현재 CUDA µs<br>median /p90</th><th>배속</th></tr></thead><tbody>'''+tr+'''</tbody></table></div>
<h3>전체 backward · B5+ 동일</h3><div class="table-wrap"><table><thead><tr><th>L</th><th>기준 µs median /p90</th><th>현재 µs median /p90</th><th>배속</th></tr></thead><tbody>'''+fr+'''</tbody></table></div>
<p>node02 H100 · dropout 25% · 20 warmup + 200 CUDA-event samples. 24 검증 조합과 입력 변경 graph replay 통과. memcheck/racecheck/synccheck L64·384 통과. 255 registers, spill 0. 실험 plan이며 production 기본 경로는 아직 바꾸지 않았다.</p>
<p><b>남은 병목:</b> 본체의 메모리 접근·연산 의존성. NCU DRAM 54.5% /60.4%로 70% 목표 미달. 두 CTA 분리·double buffer·register LN·bank permutation은 더 느려 제외했다. L72 추가 검사 한 seed의 dtri 허용치 초과는 원본에서도 동일하게 나타났다.</p>
<img src="assets/b1b4-shared.svg" alt="현재 B1부터 B4까지의 CUDA 커널 연산 연결과 partial reduction" style="width:100%;height:auto"/>
<p><a href="assets/b1b4-audit-20260920.md">9개 항목 상세 보고서</a> · <a href="assets/b1b4-optimized-audit.json">NCU·검증 요약</a> · <a href="assets/b1b4-optimized-results.json">전체 실측 JSON</a> · <a href="assets/b1b4-optimized.cu">CUDA 소스</a></p>
<details><summary>수식·SMEM 배치·실험·재현 명령·남은 과제 전체</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'''+html.escape(text)+'''</pre></details></section><!-- B1B4_AUDIT_END -->'''
s=(site/'trimul.html').read_text();s=re.sub(r'<!-- B1B4_AUDIT_BEGIN -->.*?<!-- B1B4_AUDIT_END -->','',s,flags=re.S);s=s.replace('<main>','<main>'+section,1).replace('현재 분석·측정 준비','현재 B1–B4 실측')
old_block=re.search(r'<!-- B1B4_SHARED_BEGIN -->.*?<!-- B1B4_SHARED_END -->',s,re.S)
if old_block:
 old=old_block.group(0)
 updated=old.replace('최신 실측 · 1.7× 목표 미달','이전 실측 · 최적화 전 기준',1)
 updated=updated.replace('<h2>B1–B4 CUDA · 입력 공유 + 단일 커널 reduction</h2>', '<h2>이전 B1–B4 CUDA · 개선 전 기록</h2><p><a href="#b1b4-audit">현재 결과와 배선은 맨 위 최신 실측 참조</a></p>',1)
 updated=updated.replace('src="assets/b1b4-shared.svg"','src="assets/b1b4-shared-previous.svg"')
 s=s.replace(old,updated,1)
shutil.copyfile(r/'b1b4-shared-before-audit.svg',site/'assets'/'b1b4-shared-previous.svg')
for path in (site/'trimul.html',a/'web'/'trimul.html',root/'ANTHROPIC_TRIMUL.html'):path.write_text(s)
files=['dual_optimized.cu','dual_optimized.py','dual_experiment.py','check_experiment.py','measure_experiment.py','profile_experiment.py','sanitize_experiment.py','final_validation.sh','prepare_timing.py','dual_optimized_timing.cu','optimized-audit.json','optimized-final-results.json','optimized-phase-results.json','dual_optimized-strict-validation.json','AUDIT_20260920.md']
manifest=dict(status=audit['status'],sha256={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in files})
(r/'audit-manifest.json').write_text(json.dumps(manifest,indent=2))
for src,dst in [('b1b4-shared.svg','b1b4-shared.svg'),('AUDIT_20260920.md','b1b4-audit-20260920.md'),('audit-manifest.json','b1b4-audit-manifest.json'),('optimized-audit.json','b1b4-optimized-audit.json'),('optimized-final-results.json','b1b4-optimized-results.json'),('dual_optimized.cu','b1b4-optimized.cu')]:shutil.copyfile(r/src,site/'assets'/dst)
print(json.dumps(dict(report=str(r/'AUDIT_20260920.md'),html=str(site/'trimul.html'),assets=6)))
