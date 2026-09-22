from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'measure_balanced_final.py').read_text()
s=s.replace("['dual_optimized','dual_balanced']", "['dual_balanced','dual_ln_prefetch','dual_pref_place13']")
s=s.replace("balanced-paired-results.json", "sol-final-paired-results.json")
(r/'measure_sol_final.py').write_text(s)

# Apply exactly the already qualified timing probes to the new prefetch body.
import difflib
base=(r/'dual_balanced.cu').read_text().splitlines(keepends=True)
timed=(r/'dual_balanced_timing.cu').read_text().splitlines(keepends=True)
new=(r/'dual_ln_prefetch.cu').read_text()
ops=list(difflib.SequenceMatcher(None,base,timed).get_opcodes())
for tag,i,j,a,b in ops:
 if tag=='equal':continue
 old=''.join(base[i:j]);rep=''.join(timed[a:b])
 if old:
  assert new.count(old)==1,(tag,old)
  new=new.replace(old,rep)
 else:
  # Locate insertions using the following original line as an anchor.
  anchor=''.join(base[i:i+3]);assert new.count(anchor)==1,anchor
  new=new.replace(anchor,rep+anchor)
(r/'dual_ln_prefetch_timing.cu').write_text(new)
