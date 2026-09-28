import subprocess
import sys
import argparse
from pathlib import Path

out=Path('.bench/transition-wide-local/sanitize')
out.mkdir(parents=True,exist_ok=True)
tool='/usr/local/cuda-12.9/bin/compute-sanitizer'
subprocess.run([tool,'--version'],check=True)
p=argparse.ArgumentParser()
p.add_argument('--widths',type=int,nargs='+',default=[384,512])
args=p.parse_args()
failures=[]
for width in args.widths:
    for mode in ('memcheck','racecheck','synccheck'):
        full=mode=='memcheck'
        stem=f'D{width}-L768-{mode}'
        cmd=[tool,'--tool',mode,'--error-exitcode','99',sys.executable,
             str(Path(__file__).with_name('sanitize.py')),'--width',str(width),
             '--length','768','--out',str(out/(stem+'.json'))]
        if full:
            cmd.append('--full')
        print('START',stem,'full module' if full else 'isolated changed kernels',flush=True)
        with (out/(stem+'.txt')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        print('EXIT',stem,result.returncode,flush=True)
        if result.returncode:
            print((out/(stem+'.txt')).read_text()[-6000:],flush=True)
            failures.append(stem)
if failures:
    raise SystemExit('Failed checks: '+', '.join(failures))
