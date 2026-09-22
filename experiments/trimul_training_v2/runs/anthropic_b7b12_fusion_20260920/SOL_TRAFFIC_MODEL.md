# SoL interpretation and traffic audit

`traffic_model.py` enumerates the selected ring source's TMA payload sizes and parameter partials. AtL768 it accounts for8.800GB of logical transfers and328.565GFLOP of tensor work per invocation, before mask/statistics/gamma/flag traffic, polling, cache-sector overfetch and write allocation. Of that, repeated saved x_n reads are1.208GB and projection/gate weight reads2.718GB. These are costs of this tiling, not mandatory costs of every possible algorithm.

NCU cache3:1.444704ms,328288189 L2 sectors (~10.505GB),2.648GB DRAM,89.257% reported L2 throughput. The raw throughput percentage alone does not establish an algorithmic SoL90. Observed L2 sectors and logical tensor payloads have different accounting, and optimizing reuse can change their denominator. The 512-thread two-tile prototype deliberately reduces repeated reads, but its greater synchronization latency made it slower; this is evidence of a tradeoff, not proof of a global optimum.

Source sampling attributes2359296 generic L2 sectors to mask loads (~75.5MB); producer/consumer flag loads in the two largest instructions total570586sectors (~18.3MB). TMA is accounted separately. Thus generic polling traffic is not the main explanation for the gap between logical payload bytes and total L2 sectors. Poll delay can still affect latency and scheduler behavior, and is being measured rather than assumed.

NCU command line retained cache-control none and clock-control none. A meaningful final SoL claim needs the selected source, reproducible frequencies/workload, a justified resource-specific ceiling and correct traffic/issue accounting. The current result is a bottleneck profile, with1.7× and independent SoL90 still unachieved.

## L2 throughput breakdown, 2026-09-20

The aggregate L2 throughput is the maximum of its constituent percentages, not a
single byte-bandwidth measurement. NVIDIA defines `ltcfabric` as the communication
fabric between L2 partitions. See the [NCU profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

Separate warm profiles collected `breakdown:lts__throughput.avg.pct_of_peak_sustained_elapsed`
after20 warm calls, with cache/clock controls disabled. The selected source hashes,
units and raw CSV hashes are in `l2-breakdown-summary.json`.

| Metric | L384 storepipe | L768 ring96 |
|---|---:|---:|
| Profile time |365.728µs|1426.688µs|
| LTC fabric activity / peak |78.63%|90.27%|
| L2 data sectors / peak |53.23%|57.04%|
| L2 tag sectors / peak |45.30%|56.65%|

The dominant reported L2 component is inter-partition fabric activity. It would
be incorrect to interpret90.27% as90.27% of peak total L2 data bandwidth, or as a
proof that no algorithm can be faster. The measured denominator includes repeated
operand reads and the chosen ring layout. These breakdown measurements are
separate from the earlier full-profile1.444704ms result.

Three600-sample timing controls narrow the next experiments:

- Shared workspace addresses confirm dW split13 atL384 and split20 atL768 for the
  original schedules. Retuning role counts alone is insufficient.
- L2 promotion settings none/64/128/256B show no useful gain; all256B is slower.
- Identical sources with different allocation addresses differ by about1%, with
  additional drift across measurement blocks. CTA-order gains seen with separate
  buffers almost disappear with shared addresses. Small isolated wins are not
  selected improvements.

A new one-producer/one-consumer warpgroup dX pipeline initially spilled in the
unchanged dW branch. Reducing GLU unrolling removed all stack/spills. A register
restore race on idle CTAs was fixed with a CTA rendezvous before restoring quotas.
The resulting candidates passed initial numerical checks but measured slower,
including after role-count retuning; they are not selected or production-qualified.
