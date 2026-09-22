from check_experiment import *
a=argparse.ArgumentParser();a.add_argument('sources',nargs='+');args=a.parse_args()
with torch.no_grad():
 for s in args.sources:
  try:run_case(s,64,66,1,.25)
  except RuntimeError as e:
   if 'Spill regression:' not in str(e):raise
   print('REJECT_SPILL',s,str(e),flush=True)
