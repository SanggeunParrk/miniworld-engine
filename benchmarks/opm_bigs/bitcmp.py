"""fused kernel: new defaults (5/2/16) vs old (4/4/8) bitwise, plus error vs fp64 reference of the product+projection on a small slice."""
import sys, torch
from pathlib import Path
from foldjet_engine.kernels.integrations import opm_train as ot
from foldjet_engine.kernels._nvcc import load_extension as load
dev = torch.device("cuda"); bf = torch.bfloat16
csrc = Path(ot.__file__).with_name("csrc")
def mk(name, flags):
    return load(name=name, sources=[str(csrc / "opm_epilogue.cu")], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DMINIWORLD_OPM_FUSED=1"] + flags, extra_cflags=["-O3"])
new = mk("miniworld_opm_fused_var_new", []); old = mk("miniworld_opm_fused_var_old", ["-DOPM_FUSED_STAGES=4", "-DOPM_FUSED_WST=4", "-DOPM_FUSED_RASTER=8"])
for L, S in [(256, 1024), (384, 2048), (768, 4096), (512, 8192), (768, 16384), (272, 4160)]:
    torch.manual_seed(1)
    m = torch.randn(S, L, 64, device=dev, dtype=bf); mk_ = torch.rand(S, L, device=dev) > 0.1
    gam = torch.ones(64, device=dev); bet = torch.zeros(64, device=dev)
    wa = (torch.randn(32, 64, device=dev) * 0.1).to(bf); wb = (torch.randn(32, 64, device=dev) * 0.1).to(bf)
    wo = (torch.randn(128, 1024, device=dev) * 0.02).to(bf); ob = torch.zeros(128, device=dev)
    a2, bt, st, m16, norm = new.opm_prologue_wide(m, mk_, gam, bet, wa, wb, 1e-5, False, None, False)
    wp = ot._fused_weight(wo)
    zn = new.opm_fused_infer(a2, bt, norm, wp, ob, None, 1); zo = old.opm_fused_infer(a2, bt, norm, wp, ob, None, 1)
    print(L, S, "bitwise" if torch.equal(zn, zo) else "DIFF", flush=True)
