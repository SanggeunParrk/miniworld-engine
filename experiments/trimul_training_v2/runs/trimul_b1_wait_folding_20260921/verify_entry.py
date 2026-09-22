from pathlib import Path
import sys,json,torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));import trimul_training_current as C
with torch.no_grad():
 P=C._policy;Q=P.BASE.OLD.Q;a=Q.setup(384);old=P.B.Training(a);new=C.Training(a)
 ref=old();ref=(ref[0].clone(),tuple(t.clone() for t in ref[1]));out=new();errors=P.BASE.OLD.H.errors(out,ref)
 assert all(v['bit_exact'] for v in errors.values()),errors
 _,k=new.forward();assert k[1].dtype==torch.bfloat16 and k[3].numel()==2*384*384;new.backward(k);assert new.p1.xhat.data_ptr()==k[1].data_ptr()
 (R/'current-entry-check.json').write_text(json.dumps(dict(entry='runs/trimul_training_current.py:Training',all_outputs_bit_exact=True,errors=errors),indent=2));print('CURRENT_ENTRY_PASS',flush=True)
