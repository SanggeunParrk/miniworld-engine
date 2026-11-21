# dwmtest.py : DwModTf32 (dw_tf32.cu DWM=1) against the cuBLAS TF32 GEMM and fp64, random non-zero data, determinism
import torch
from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_bwd
dev = torch.device("cuda"); torch.manual_seed(3)
K = tf32_bwd.kernels_bwd(0)
ok = True
for M in (256, 1024, 2048, 3072, 4096, 4104 // 8 * 8, 6144, 8192, 16384):
    dmod = torch.randn(M, 768, device=dev) * 0.05
    a = torch.randn(M, 128, device=dev)
    if not K.dwmod.ok(dmod, a): print(M, "not ok"); continue
    ref = dmod.double().t() @ a.double()
    o1, _ = K.dwmod.run(dmod, a); o2, _ = K.dwmod.run(dmod, a); torch.cuda.synchronize()
    cb = tf32_bwd.mm_tf32(dmod.t(), a)
    e_new = (o1.double() - ref).abs().max().item() / ref.abs().max().item()
    e_cb = (cb.double() - ref).abs().max().item() / ref.abs().max().item()
    det = torch.equal(o1, o2)
    print(f"M {M:6d}: max rel err vs fp64: kernel {e_new:.2e}  cuBLAS-TF32 {e_cb:.2e}  deterministic {det}", flush=True)
    ok &= det and e_new < 3 * e_cb + 1e-6
print("PASS" if ok else "FAIL")
