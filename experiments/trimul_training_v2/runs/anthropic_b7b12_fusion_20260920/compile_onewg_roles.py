from front_plan import *
for src in ['front_onewg_dwonly','front_onewg_dxonly']:
 try:load(396,20,2,0,src)
 except RuntimeError as e:print('COMPILE_REJECT',src,str(e),flush=True)
