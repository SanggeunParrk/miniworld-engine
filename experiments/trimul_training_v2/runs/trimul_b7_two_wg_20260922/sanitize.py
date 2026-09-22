from pathlib import Path
import subprocess,sys,json,os
R=Path(__file__).resolve().parent
os.environ['B7_CONSUMERS']='10'
os.environ.setdefault('B7_PRODUCER_REGS','48')
records=[]
for tool in ('memcheck','racecheck'):
 log=R/(tool+'-L384.log')
 cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','kns=b7_joint',sys.executable,'-B',str(R/'probe.py')]
 with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=480)
 records.append(dict(tool=tool,L=384,returncode=p.returncode,log=str(log)));(R/'sanitizer-L384.json').write_text(json.dumps(records,indent=2));print(tool,p.returncode,flush=True)
 assert p.returncode==0
