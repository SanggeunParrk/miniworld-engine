"""Check the QK^T mechanics against torch."""
import sys
from pathlib import Path
import torch
_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_dir))
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension  # noqa: E402

ensure_cuda_home()
_v5 = "/home/psk6950/miniworld-engine-dit2/src/miniworld_engine/kernels/transition/cuda/anthropic_v5"
ext = load_extension(name="tdit_qk_probe", sources=[str(_dir / "qk_probe.cu")],
                     extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), f"-I{_v5}",
                                        "--expt-relaxed-constexpr", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
                     extra_cflags=["-std=c++17"], verbose=False)
torch.manual_seed(0)
M, DS, H = 64, 768, 16
qkvg = (torch.randn(M, 4 * DS, device="cuda") * 0.3).to(torch.bfloat16)
for head in (0, 3, 15):
    qc, kc = head * 48, DS + head * 48
    got = ext.qk(qkvg, qc, kc, 48)
    q = qkvg[:, qc:qc + 48].float()
    k = qkvg[:, kc:kc + 48].float()
    ref = q @ k.t()
    err = float((got - ref).norm() / ref.norm())
    print(f"head {head:2d}: rel {err:.2e}  {'ok' if err < 5e-3 else 'FAIL'}", flush=True)

p = (torch.randn(64, 64, device="cuda") * 0.3).to(torch.bfloat16)
v = (torch.randn(64, 64, device="cuda") * 0.3).to(torch.bfloat16)
ref_pv = p.float() @ v.float()[:, :48]
for mode in (0, 1):
    for lbo in (16, 128, 1024, 2048):
        try:
            got = ext.pv(p, v, mode, lbo)
            err = float((got - ref_pv).norm() / ref_pv.norm())
            print(f"pv mode {mode} lbo {lbo:5d}: rel {err:.2e}  {'ok' if err < 5e-3 else ''}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"pv mode {mode} lbo {lbo}: {type(e).__name__}", flush=True)
