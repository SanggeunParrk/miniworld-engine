from pathlib import Path
import argparse,subprocess,sys,json
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--role',choices=['dw','dx'],required=True);args=p.parse_args();n=args.length;role=args.role;records=[]
for tool in ('memcheck','racecheck'):
 log=R/('%s-%s-L%d.log'%(tool,role,n));cmd=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--kernel-name','kns=front_b7b12_'+role,sys.executable,'-B',str(R/('stress_'+role+'.py')),'--length',str(n)]
 with log.open('w') as f:q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1000)
 records.append(dict(tool=tool,role=role,returncode=q.returncode,log=str(log)));print('SANITIZER',n,role,tool,q.returncode,flush=True);(R/('sanitizer-%s-L%d.json'%(role,n))).write_text(json.dumps(records,indent=2));assert q.returncode==0
