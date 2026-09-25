import torch, drv
from cuda.bindings import driver as cu
k = drv.Kernel("build/tbwd.cubin", "transition_bwd_sm100", 230656 + 256, cluster=4)
for cl in (1, 2, 4, 8):
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = 148 - 148 % cl, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 512, 1, 1
    cfg.sharedMemBytes = k.smem
    at = cu.CUlaunchAttribute(); at.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cl, 1, 1
    cfg.attrs = [at]; cfg.numAttrs = 1
    n = drv._chk(cu.cuOccupancyMaxActiveClusters(k.func, cfg), "occ")
    print(f"cluster {cl}: max active clusters {n} -> {n * cl} CTAs resident")
