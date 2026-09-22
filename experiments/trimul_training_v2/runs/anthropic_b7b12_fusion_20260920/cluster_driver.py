from front_plan import *
class Config(ctypes.Structure):
 _fields_=[(k,ctypes.c_uint) for k in ('gridDimX','gridDimY','gridDimZ','blockDimX','blockDimY','blockDimZ','sharedMemBytes')]+[('hStream',ctypes.c_void_p),('attrs',ctypes.c_void_p),('numAttrs',ctypes.c_uint)]
def active_clusters(k,count=128,shared=229376,threads=256):
 lib=ctypes.CDLL('libcuda.so.1');fn=lib.cuOccupancyMaxActiveClusters;fn.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.POINTER(Config)];fn.restype=ctypes.c_int
 cfg=Config(count,1,1,threads,1,1,shared,int(torch.cuda.current_stream().cuda_stream),None,0);v=ctypes.c_int();rc=fn(ctypes.byref(v),int(k.handle),ctypes.byref(cfg));assert rc==0,rc;return v.value
if __name__=='__main__':
 with torch.no_grad():
  a=setup(64);k,_=load(128,1,2,0,'front_cluster_probe');cap=active_clusters(k);print('ACTIVE_CLUSTERS',cap,flush=True)
  count=cap*8;p=Plan(a,count=count,splits=1,source='front_cluster_probe');p.partln=torch.empty((count,256),device='cuda');p.bind(a['dl'],a['dr'],a['dg'],a['dy']);p();torch.cuda.synchronize();expected=torch.arange(count*256,device='cuda').reshape(count//2,2,256).flip(1).reshape(count,256).float();print('CLUSTER_PROBE',torch.equal(expected,p.partln),flush=True)
