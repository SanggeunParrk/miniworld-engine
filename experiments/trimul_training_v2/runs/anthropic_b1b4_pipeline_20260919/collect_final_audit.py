from pathlib import Path
import csv,json,hashlib
r=Path(__file__).resolve().parent
keys=['gpu__time_duration.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_read.sum','dram__bytes_write.sum','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum','launch__registers_per_thread','lts__t_sectors.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
keys += ['smsp__average_warps_issue_stalled_'+v+'_per_issue_active.ratio' for v in ['barrier','long_scoreboard','short_scoreboard','membar','wait','sleeping']]
profiles={}
for name in ['dual-dspref','optimized']:
 for length in [384,768]:
  path=r/f'{name}-L{length}.csv'
  rows=list(csv.DictReader(path.open()));units,vals=rows[0],rows[-1]
  metrics={k:{'value':float(vals[k].replace(',','')),'unit':units[k]} if k in vals else {'unavailable':True} for k in keys}
  metrics['L2_bytes_derived']=float(vals['lts__t_sectors.sum'].replace(',',''))*32
  profiles[f'{name}/L{length}']=metrics
checks=json.loads((r/'dual_optimized-strict-validation.json').read_text())
assert len(checks)==24
maxerr={k:max(e[k]['relative_l2'] for c in checks for e in c['errors']) for k in checks[0]['errors'][0]}
originals={
 'dual.cu':'3026e8b5722106852571dc245a9693fe317b61537598f99cb7d3aa6fc6e468c8',
 'dual_primitives.cuh':'4968268aec02402b1eb0136ca5d95336409f453c4885cc04147c72a02f93234f',
 'dual.py':'0a08f370cc5640d3e429c783bc931bb8c4fb5382bedbb50e13cffec83c0ab355',
 'core.py':'b7fc3efd0f378ef2aa55e72e88997b9a83f02090ef6f705bfbd5bae5521ae058',
 'dual_dspref.cu':'7f7614f0d76309f8368df2aa0303c804960f72fd3f999f7464b6718a0d4de517'}
for name,sha in originals.items(): assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha,name
sanitizers={}
for name in ['memcheck-L64','memcheck-L384','racecheck-L64','racecheck-L384','synccheck-L64','synccheck-L384','split-memcheck-L64','split-memcheck-L384']:
 text=(r/f'optimized-{name}.log').read_text()
 ok='ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text
 assert ok,name
 sanitizers[name]='passed'
result=dict(status='measured; 1.7x target not met; experimental plan, no production default change',selected='dual_optimized',profiles=profiles,max_relative_l2=maxerr,strict_cases=len(checks),comparisons=len(checks)*3,sanitizers=sanitizers,original_sha256=originals,full_results='optimized-final-results.json',note='NCU replay/cold-cache durations differ from CUDA-event medians; L2 bytes are sectors times 32, not direct lts__t_bytes.sum.')
(r/'optimized-audit.json').write_text(json.dumps(result,indent=2))
for k,v in profiles.items(): print(k, json.dumps(v))
