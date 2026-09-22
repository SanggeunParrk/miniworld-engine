"""Disassemble exact measured binaries on the allocated compute node."""
from pathlib import Path
import json, subprocess, re, collections
r=Path(__file__).resolve().parent
sources={
 'dual':'01f1f050ea559c609d3b7b00e38fdc91034f87d0a5c971179d0ed6ac8cc61daf',
 'dual_optimized':'a1759201404153b87bb5b1e5ac0055b84b604215ccc714b4097b3f0a5f1a51ca'}
result={}
for name,key in sources.items():
 cubin=r/'build'/(key+'.cubin')
 sass=subprocess.check_output(['cuobjdump','--dump-sass',str(cubin)],text=True)
 (r/(name+'-audit.sass')).write_text(sass)
 section=sass.split('Function : dual_b1b4',1)[1]
 if 'Function :' in section: section=section.split('Function :',1)[0]
 counts=collections.Counter()
 for line in section.splitlines():
  m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?P\w+\s+)?([A-Z][A-Z0-9]*(?:\.[A-Za-z0-9_]+)*)\b',line)
  if m: counts[m.group(1)]+=1
 result[name]=dict(cubin=str(cubin),instruction_count=sum(counts.values()),opcodes=dict(sorted(counts.items())))
 print(name,json.dumps(result[name]),flush=True)
(r/'sass-audit.json').write_text(json.dumps(result,indent=2))
