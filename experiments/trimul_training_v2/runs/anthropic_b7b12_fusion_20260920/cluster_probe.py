from front_plan import *
with torch.no_grad():
 a=setup(64);p=Plan(a,count=128,splits=1,source='front_cluster_probe');p.partln=torch.empty((128,256),device='cuda');p.bind(a['dl'],a['dr'],a['dg'],a['dy']);p();torch.cuda.synchronize();expected=torch.arange(128*256,device='cuda').reshape(128,256).reshape(64,2,256).flip(1).reshape(128,256).float();print('CLUSTER_PROBE',torch.equal(expected,p.partln),flush=True)
