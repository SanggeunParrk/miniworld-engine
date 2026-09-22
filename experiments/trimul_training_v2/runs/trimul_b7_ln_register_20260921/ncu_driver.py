from pathlib import Path
import argparse,subprocess,sys,os
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
metrics='gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed.sum,lts__t_sectors.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__warps_eligible.avg.per_cycle_active,'+','.join('smsp__warp_issue_stalled_'+s+'_per_warp_active.pct' for s in ('no_instruction','branch_resolving','wait','long_scoreboard','short_scoreboard','mio_throttle','barrier','math_pipe_throttle'))
env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'};env['PYTHONNOUSERSITE']='1'
for mode in (0,):
 stem=R/('ncu-L%d-mode%d'%(n,mode))
 cmd=['ncu','--profile-from-start','off','--section','SourceCounters','--section','SpeedOfLight','--import-source','yes','--metrics',metrics,'--cache-control','none','--clock-control','none','--target-processes','all','--kernel-name','regex:^front_b7b12$','--launch-count','2','--force-overwrite','-o',str(stem),'env','PYTHONPATH='+os.environ.get('PYTHONPATH',''),sys.executable,'-B',str(R/'profile_one.py'),'--length',str(n),'--profile']
 with stem.with_suffix('.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=360)
 with stem.with_suffix('.csv').open('w') as f:subprocess.run(['ncu','--import',str(stem)+'.ncu-rep','--page','raw','--csv'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
 with (R/('source-L%d.csv'%n)).open('w') as f:subprocess.run(['ncu','--import',str(stem)+'.ncu-rep','--page','source','--print-source','cuda,sass','--csv'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
 print('COMPLETE',n,mode,flush=True)
