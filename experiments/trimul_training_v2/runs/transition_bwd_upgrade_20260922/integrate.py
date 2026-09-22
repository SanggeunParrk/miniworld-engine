from pathlib import Path
import shutil,json,sys,hashlib
R=Path(__file__).resolve().parent;name=sys.argv[1];rep=int(sys.argv[2]) if len(sys.argv)>2 else 8;T=R/'selected';T.mkdir(exist_ok=True)
for f in ('transition_fused_sm90a.cu','transition_fused_fwd_sm90a_kernel.cu'):shutil.copy2(R/f,T/f)
shutil.copytree(R/'anthropic_v5',T/'anthropic_v5',dirs_exist_ok=True)
s=(R/'transition_fused_bwd_sm90a_kernel.cu').read_text();host=s[s.index('// ================================================================================== host launcher'):]
host=host.replace('const int work = 3 * 8 * HS * D_ + 256;','const int work = 3 * 8 * HS * D_ + 4 * 256;')
(T/'transition_fused_bwd_sm90a_kernel.cu').write_text((R/(name+'.cu')).read_text()+host)
s=(R/'fused_sm90a.py').read_text().replace('DW_REPL = 8','DW_REPL = '+str(rep)).replace('transition_fused_sm90a_c{ctas}','transition_bwd_upgrade_sm90a_c{ctas}').replace('name="transition_fused_fwd_sm90a"','name="transition_upgrade_fwd_sm90a"').replace('name="transition_fused_bwd_sm90a"','name="transition_upgrade_bwd_sm90a"')
(T/'fused_sm90a.py').write_text(s)
(R/'selection.json').write_text(json.dumps(dict(variant=name,rep=rep,source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in T.glob('*') if p.is_file()}),indent=2))
