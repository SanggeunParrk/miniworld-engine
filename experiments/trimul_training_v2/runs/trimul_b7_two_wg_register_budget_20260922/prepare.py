from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b7_two_wg_lowreg_20260922'
for name in ('single_wg.inc','pair_producer.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm'):
 (R/name).write_text((S/name).read_text().replace(S.name,R.name))
s=(S/'joint.cu').read_text().replace('__launch_bounds__(384,2)','__maxnreg__(B7_REG_CAP)').replace('setmaxnreg_inc<104>()','setmaxnreg_inc<B7_COMPUTE_REGS>()')
(R/'joint.cu').write_text(s)
p=(S/'plan.py').read_text().replace("flags=['-DB7_HALF_GATE=", "flags=['-DB7_REG_CAP='+os.environ.get('B7_REG_CAP','96'),'-DB7_COMPUTE_REGS='+os.environ.get('B7_COMPUTE_REGS','128'),'-DB7_HALF_GATE=")
(R/'plan.py').write_text(p)
s=(S/'sweep.py').read_text()
s=s.replace("tag='configs-'", "tag='reg'+os.environ.get('B7_REG_CAP','96')+'g'+os.environ.get('B7_GROUPS','6')+'u'+os.environ.get('B7_C','10')+'-configs-'")
start=s.index(' for spec in args.configs.split');end=s.index(' limits=',start);t=s[start:end]
t=t.replace("variant,regs=map", "os.environ['B7_CONSUMERS']=os.environ.get('B7_C','10')\n  variant,regs=map").replace('clusters=10','clusters=int(os.environ.get("B7_GROUPS","6"))')
s=s[:start]+t+s[end:];(R/'sweep.py').write_text(s)
p=(R/'probe.py').read_text().replace("os.environ['B7_CONSUMERS']='10'", "os.environ['B7_CONSUMERS']=os.environ.get('B7_C','10')").replace('clusters=10','clusters=int(os.environ.get("B7_GROUPS","6"))');(R/'probe.py').write_text(p)
