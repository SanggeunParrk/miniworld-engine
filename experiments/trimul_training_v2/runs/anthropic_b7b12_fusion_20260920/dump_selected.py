from front_plan import *
with torch.no_grad():
 a=setup(64);p=Plan(a)
 # load metadata points to exact compiled source and flags.
 candidates=[]
 for f in (R/'build').glob('*.json'):
  d=json.loads(f.read_text())
  if all(d.get(k)==v for k,v in dict(source='front_roundonly',count=132,splits=15,part=2,debug=0).items()):candidates.append(f)
 assert len(candidates)==1,candidates
 f=candidates[0];src=f.with_suffix('.cubin')
 z=subprocess.run(['cuobjdump','--dump-sass',str(src)],capture_output=True,text=True,check=True);(R/'selected.sass').write_text(z.stdout)
 print('SASS',str(src),'TMA',z.stdout.count('UTMALDG'),'WGMMA',z.stdout.count('HGMMA'),'LDL',z.stdout.count(' LDL'),'STL',z.stdout.count(' STL'),flush=True)
 (R/'selected-build.json').write_text(json.dumps(dict(metadata=json.loads(f.read_text()),cubin=str(src),cubin_sha256=hashlib.sha256(src.read_bytes()).hexdigest()),indent=2))
try:
 import cairosvg
 cairosvg.svg2png(url=str(R/'front-b7b12.svg'),write_to=str(R/'front-b7b12.png'))
 print('SVG_RENDERED',flush=True)
except ImportError:print('CAIROSVG_UNAVAILABLE',flush=True)
