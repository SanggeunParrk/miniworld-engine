from pathlib import Path
import subprocess,sys,os
R=Path(__file__).resolve().parent
stem=R/('ncu-D'+os.environ['NCU_WIDTH']+'-L384')
metrics='gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__inst_executed.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum'
env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'};env['PYTHONNOUSERSITE']='1'
cmd=['ncu','--profile-from-start','off','--section','SpeedOfLight','--section','MemoryWorkloadAnalysis','--section','SourceCounters','--section','SchedulerStats','--metrics',metrics,'--cache-control','none','--clock-control','none','--target-processes','all','--kernel-name','regex:^width_b7$','--launch-count','1','--force-overwrite','-o',str(stem),'env','PYTHONPATH='+os.environ.get('PYTHONPATH',''),sys.executable,'-B',str(R/'ncu_probe.py')]
with stem.with_suffix('.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=420)
with stem.with_suffix('.csv').open('w') as f:subprocess.run(['ncu','--import',str(stem)+'.ncu-rep','--page','raw','--csv'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
print('PROFILE COMPLETE',flush=True)
