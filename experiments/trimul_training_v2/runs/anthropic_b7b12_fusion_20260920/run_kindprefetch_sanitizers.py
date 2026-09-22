from pathlib import Path
import subprocess,json,sys
r=Path(__file__).resolve().parent;records={}
for n in [64,384]:
 for tool in ['memcheck','racecheck','synccheck']:
  key=f'{tool}-L{n}';cmd=['compute-sanitizer','--tool',tool,'--kernel-name','kns=front_b7b12','--error-exitcode','86',sys.executable,'-u','-B',str(r/'sanitize_kindprefetch.py'),'--length',str(n)]
  with (r/f'kindprefetch-{key}.log').open('w') as log:
   try:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120);rc=result.returncode
   except subprocess.TimeoutExpired:rc='timeout'
  text=(r/f'kindprefetch-{key}.log').read_text();ok=rc==0 and 'SANITIZER_CHECK_PASS' in text and ('ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text)
  records[key]=dict(result='passed' if ok else 'failed_or_incomplete',returncode=rc,log=f'kindprefetch-{key}.log');(r/'kindprefetch-sanitizers.json').write_text(json.dumps(records,indent=2));print(key,records[key],flush=True)
  if not ok:break
