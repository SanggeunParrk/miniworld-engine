import csv,json
from pathlib import Path
r=Path(__file__).resolve().parent;rows=list(csv.DictReader((r/'flat-profile.csv').open()));keys=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum','launch__registers_per_thread']
for z in rows[1:]:
 print(z['Kernel Name']);d={k:(z.get(k),rows[0].get(k)) for k in keys};print(d);print({k:v for k,v in z.items() if 'stalled' in k and ('avg.pct' in k or 'per_warp_active.pct' in k)});print({k:v for k,v in z.items() if 'local' in k})
