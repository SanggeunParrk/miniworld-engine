from pathlib import Path
import argparse,subprocess,sys,json
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);n=ap.parse_args().length
records=[]
for tool in ('memcheck','racecheck'):
 stem='%s-L%d'%(tool,n)
 cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','kns=b1_fused',sys.executable,'-B',str(R/'probe.py'),'--length',str(n),'--output',stem+'.json']
 with (R/(stem+'.log')).open('w') as f:
  p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 records.append(dict(tool=tool,returncode=p.returncode,log=stem+'.log'))
 (R/('sanitizers-L%d.json'%n)).write_text(json.dumps(records,indent=2))
 print(tool,n,p.returncode,flush=True);assert p.returncode==0
