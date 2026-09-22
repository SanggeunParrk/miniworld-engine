# B7 hardware roofline measurement

NCU job14187, H100 node01, selected nextrow candidate, L384/L768, C128/H256,
BF16, dropout25%, pair mask and residual. B1 is outside this measurement.

| L | dW roofline | dX roofline | B7 combined | Lower bound / measured us |
| --- | ---: | ---: | ---: | ---: |
| 384 | 48.5% | 54.0% | **51.6%** | 207.393 / 401.760 |
| 768 | 53.2% | 65.9% | **60.4%** | 855.682 / 1416.480 |

This is a **current-work hardware roofline**, not a calibrated attainable
algorithm ceiling. Compute the maximum of dense BF16 HGMMA, HBM, and L2
minimum times per kernel; sum these minimum times across the two sequential
launches and divide by summed profiled durations. dW is limited by the tensor
roof; dX by the L2 roof in this model.

Use measured SM/DRAM/L2 clocks, not nominal marketing peaks. Check the
sparsity_off HGMMA peak: the generic BF16 peak includes sparse support and
would incorrectly halve the dense efficiency reported here. Unit conversions
are asserted by report.py. The GEMM work counters also match the source-derived
counts: dW = M*524288 and dX = M*557056 floating-point operations.

Scalar GLU/LN work, synchronization and dependencies are not modeled. Memory
traffic is the observed traffic; removing redundant transfers can lower this
percentage while improving latency. Therefore **90% has not been reached**,
and this estimate is not evidence that the fused algorithm can attain the
entire gap. Do not substitute unrelated NCU utilization percentages for it.

Reference: [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
Local definitions: /opt/nvidia/nsight-compute/2025.2.0/sections/SpeedOfLight_HierarchicalTensorRooflineChart.section.
