"""TMA read bandwidth from an L2-resident buffer, at the core's tile shape: the denominator for any SoL claim here."""
import sys
from pathlib import Path
import torch
_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_dir.parent))
from bench import us  # noqa: E402
from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension  # noqa: E402

ensure_cuda_home()
_v5 = "/home/psk6950/miniworld-engine-dit2/src/miniworld_engine/kernels/transition/cuda/anthropic_v5"
ext = load_extension(name="tdit_l2_roof", sources=[str(_dir / "l2_roof.cu")],
                     extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), f"-I{_v5}",
                                        "--expt-relaxed-constexpr", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
                     extra_cflags=["-std=c++17"], verbose=False)
TILE = 64 * 64 * 2
for mb in (16, 32):                                                  # both fit the 50 MB L2
    rows = mb * 2**20 // (64 * 2)
    buf = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)
    for ctas, iters in ((264, 200), (528, 100)):
        t = us(lambda: ext.roof(buf, iters, ctas))
        gb = ctas * iters * TILE / 1e9
        print(f"buffer {mb} MB, {ctas} CTAs x {iters} tiles: {t:7.1f} us  {gb / t * 1e6:6.2f} TB/s", flush=True)
