from pathlib import Path
import json
p=Path(__file__).resolve().parent
for base in ['front_prefetch_lnpair_storepipe','front_ring96_cache3_storepipe_acq32','front_ring96_queue']:
 s=(p/(base+'.cu')).read_text()
 for old,new in [('#define WGRAD_SLICES 2','#define WGRAD_SLICES 1'),('segmentRounds=(rounds+1)/2','segmentRounds=rounds'),('((group*DW_SPLITS+split)*2+r/segmentRounds)','((group*DW_SPLITS+split)+r/segmentRounds)'),('b<DW_SPLITS*2','b<DW_SPLITS'),('(group*DW_SPLITS*2+b)','(group*DW_SPLITS+b)')]:
  assert old in s,(base,old);s=s.replace(old,new)
 name=base+'_onesegment';(p/(name+'.cu')).write_text(s)
 cfg=json.loads((p/(base+'.launch.json')).read_text());cfg['wgrad_slices']=1
 (p/(name+'.launch.json')).write_text(json.dumps(cfg))
