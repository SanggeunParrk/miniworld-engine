from pathlib import Path
import csv,json,hashlib
R=Path(__file__).resolve().parent
out={'definition':'Current-work hardware roofline: max(dense BF16 HGMMA work / measured-clock peak, observed HBM traffic / measured-clock peak, observed L2 writeback / measured-clock peak), divided by kernel time. Sequential launches aggregate lower bounds and durations.', 'limitations':['Not a calibrated dependency-aware attainable limit','Scalar GLU/LN instruction throughput and mandatory synchronization are not in this roofline','Observed memory traffic includes removable work; reducing traffic can lower this percentage while making the kernel faster','NCU durations are separate from alternating CUDA-event measurements','Dense HGMMA BF16 sparsity_off peak is required; aggregate BF16 peak also permits sparsity and is twice as high'], 'source':'https://docs.nvidia.com/nsight-compute/ProfilingGuide/', 'job':14187,'candidate':'trimul_b7_nextrow_20260921','SoL90_achieved':False,'lengths':{},'source_sha256':{}}
for n in (384,768):
 p=R/('ncu-L%d-mode0.csv'%n);out['source_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest();rows=list(csv.DictReader(p.open()));units=rows.pop(0);results=[]
 for r in rows:
  work='sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum'
  assert units['gpu__time_duration.sum']=='us' and units[work+'.peak_sustained_elapsed.per_second']=='1/ns'
  assert units['dram__bytes.sum.per_second']=='Tbyte/s' and units['dram__bytes.sum.peak_sustained']=='Kbyte/cycle' and units['dram__cycles_elapsed.avg.per_second']=='Ghz'
  assert units['derived__lts__lts2xbar_bytes.sum.per_second']=='Tbyte' and units['derived__lts__lts2xbar_bytes.sum.peak_sustained']=='Kbyte' and units['lts__cycles_elapsed.avg.per_second']=='Ghz'
  t=float(r['gpu__time_duration.sum']);flops=float(r[work]);tcpeak=float(r[work+'.peak_sustained_elapsed.per_second'])*1e9
  hbm_rate=float(r['dram__bytes.sum.per_second'])*1e12;hbm_peak=float(r['dram__bytes.sum.peak_sustained'])*1e3*float(r['dram__cycles_elapsed.avg.per_second'])*1e9
  l2rate=float(r['derived__lts__lts2xbar_bytes.sum.per_second'])*1e12;l2peak=float(r['derived__lts__lts2xbar_bytes.sum.peak_sustained'])*1e3*float(r['lts__cycles_elapsed.avg.per_second'])*1e9
  bounds=dict(dense_tensor=flops/tcpeak*1e6,hbm=hbm_rate/hbm_peak*t,l2=l2rate/l2peak*t)
  results.append(dict(kernel=r['Kernel Name'],ncu_us=t,gemm_flops=flops,dense_tensor_peak_TFLOPs=tcpeak/1e12,hbm_peak_TBps=hbm_peak/1e12,l2_peak_TBps=l2peak/1e12,bounds_us=bounds,limiting_roof=max(bounds,key=bounds.get),roofline_pct=max(bounds.values())/t*100))
 total=sum(x['ncu_us'] for x in results);lower=sum(max(x['bounds_us'].values()) for x in results)
 out['lengths'][str(n)]=dict(kernels=results,ncu_total_us=total,hardware_lower_bound_us=lower,roofline_pct=100*lower/total)
(R/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
rows=[]
for n,d in out['lengths'].items():
 rows.append('| %s | %.1f%% | %.1f%% | **%.1f%%** | %.3f / %.3f |'%(n,d['kernels'][0]['roofline_pct'],d['kernels'][1]['roofline_pct'],d['roofline_pct'],d['hardware_lower_bound_us'],d['ncu_total_us']))
(R/'README.md').write_text('''# B7 hardware roofline measurement

NCU job14187, H100 node01, selected nextrow candidate, L384/L768, C128/H256,
BF16, dropout25%, pair mask and residual. B1 is outside this measurement.

| L | dW roofline | dX roofline | B7 combined | Lower bound / measured us |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+'''

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
''')
print({n:round(d['roofline_pct'],3) for n,d in out['lengths'].items()})
