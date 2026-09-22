from pathlib import Path
import json
R=Path(__file__).resolve().parent;P=R.parent
out={'selected_candidate':'trimul_b7_nextrow_20260921','selected_launches':2,'production_ready':False,'SoL90_achieved':False,'candidates':{}}
for name in ('trimul_b7_joint_cluster_20260921','trimul_b7_joint8_20260922','trimul_b7_joint8_bulk_20260922'):
 d=P/name;items=[]
 for n in (384,768):
  files=sorted(d.glob('timing-check-L%d-C*.json'%n)) or sorted(d.glob('check-L%d-C*.json'%n))
  j=json.loads(files[0].read_text());items.append(j)
 out['candidates'][name]=items
out['hardware_roofline']=json.loads((P/'trimul_b7_roofline_20260921/summary.json').read_text())
(R/'report.json').write_text(json.dumps(out,indent=2))
import csv
out['profile_diagnostic_only']={}
for n in (384,768):
 f=R/('ncu-L%d-mode0.csv'%n)
 if not f.exists():continue
 rows=list(csv.DictReader(f.open()));unit,row=rows[0],rows[1]
 keys=['gpu__time_duration.sum','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active','smsp__warp_issue_stalled_barrier_per_warp_active.pct','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warps_eligible.avg.per_cycle_active']
 out['profile_diagnostic_only'][n]={k:{'value':float(row[k]),'unit':unit[k]} for k in keys}
(R/'report.json').write_text(json.dumps(out,indent=2))
