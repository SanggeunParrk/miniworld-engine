# SoL interpretation and traffic audit

`traffic_model.py` enumerates the selected ring source's TMA payload sizes and parameter partials. AtL768 it accounts for8.800GB of logical transfers and328.565GFLOP of tensor work per invocation, before mask/statistics/gamma/flag traffic, polling, cache-sector overfetch and write allocation. Of that, repeated saved x_n reads are1.208GB and projection/gate weight reads2.718GB. These are costs of this tiling, not mandatory costs of every possible algorithm.

NCU cache3:1.444704ms,328288189 L2 sectors (~10.505GB),2.648GB DRAM,89.257% reported L2 throughput. The raw throughput percentage alone does not establish an algorithmic SoL90. Observed L2 sectors and logical tensor payloads have different accounting, and optimizing reuse can change their denominator. The 512-thread two-tile prototype deliberately reduces repeated reads, but its greater synchronization latency made it slower; this is evidence of a tradeoff, not proof of a global optimum.

Source sampling attributes2359296 generic L2 sectors to mask loads (~75.5MB); producer/consumer flag loads in the two largest instructions total570586sectors (~18.3MB). TMA is accounted separately. Thus generic polling traffic is not the main explanation for the gap between logical payload bytes and total L2 sectors. Poll delay can still affect latency and scheduler behavior, and is being measured rather than assumed.

NCU command line retained cache-control none and clock-control none. A meaningful final SoL claim needs the selected source, reproducible frequencies/workload, a justified resource-specific ceiling and correct traffic/issue accounting. The current result is a bottleneck profile, with1.7× and independent SoL90 still unachieved.
