from pathlib import Path
import json
p=Path(__file__).resolve().parent
for src in ['front_warp','front_hybrid']:
 s=(p/(src+'.cu')).read_text();a=s.index('TMN_DEVI void dw_consumer');b=s.index('\n}\n',a)+3;f=s[a:b]
 f=f.replace('float acc[2][64]={};int r=0;','float acc[2][64]={};int r=0,rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+1)/2;')
 f=f.replace('r>0||k>0','r%segmentRounds>0||k>0')
 needle='release(b,slot);\n }\n static_for<2>'
 assert needle in f
 f=f.replace(needle,'release(b,slot);\n if((r+1)%segmentRounds==0 || tile+DW_SPLITS>=p.tiles){\n static_for<2>')
 f=f.replace('(group*DW_SPLITS+split)*16384+n*8192','((group*DW_SPLITS+split)*2+r/segmentRounds)*16384+n*8192')
 assert f.endswith(' });\n}\n')
 f=f[:-3]+' }\n }\n}\n';s=s[:a]+f+s[b:]
 s=s.replace('for(int b=0;b<DW_SPLITS;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS+b)*16384+j];','for(int b=0;b<DW_SPLITS*2;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(group*DW_SPLITS*2+b)*16384+j];')
 name=src+'_seg';(p/(name+'.cu')).write_text('#define WGRAD_SLICES 2\n'+s);(p/(name+'.launch.json')).write_text(json.dumps({'direct_weights':True,'wgrad_slices':2}))
s=(p/'warp_plan.py').read_text().replace("self.config={'direct_weights':True};self.debug=0","self.config={'direct_weights':True};self.debug=0\n  cf=R/(source+'.launch.json')\n  if cf.exists():self.config.update(json.loads(cf.read_text()))")
s=s.replace('torch.empty((8,splits,16384),','torch.empty((8,splits*self.config.get("wgrad_slices",1),16384),')
(p/'warp_plan.py').write_text(s)
