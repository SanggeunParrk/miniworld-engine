"""Feasibility probe: benchmark CUTLASS tuned Blackwell persistent collective
on the tm1 skinny shape mnkl=(1048576,128,128,1). One-side plain GEMM.
Answers: does the tuned collective beat our naive 0.63ms (two-GEMM) / ~0.315ms per side?
"""
import sys, warnings, importlib.util
warnings.filterwarnings("ignore")
import cutlass, torch

spec = importlib.util.spec_from_file_location(
    "refp", "/home/snu_hwle/psk/ncu/ref_dense_gemm_persistent.py")
refp = importlib.util.module_from_spec(spec); spec.loader.exec_module(refp)

M, N, K, L = 1048576, 128, 128, 1
mnkl = (M, N, K, L)

configs = [
    # (mma_tiler_mn, cluster_shape_mn, use_2cta_instrs)
    ((128, 128), (1, 1), False),
    ((256, 128), (2, 1), True),
    ((256, 128), (1, 1), False),
    ((128, 256), (1, 1), False),
    ((256, 256), (2, 1), True),
    ((128, 128), (2, 1), False),
]

for mma_tiler_mn, cluster, use_2cta in configs:
    try:
        t_us = refp.run(
            mnkl,
            ab_dtype=cutlass.BFloat16, c_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32,
            a_major="k", b_major="k", c_major="n",
            mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster,
            use_2cta_instrs=use_2cta, use_tma_store=True,
            tolerance=2.0, warmup_iterations=20, iterations=50,
            skip_ref_check=False, benchmark=True,
        )
        print(f"RESULT tiler={mma_tiler_mn} cluster={cluster} 2cta={use_2cta}: {t_us/1000:.4f} ms/side", flush=True)
    except Exception as e:
        print(f"FAIL   tiler={mma_tiler_mn} cluster={cluster} 2cta={use_2cta}: {type(e).__name__}: {str(e)[:120]}", flush=True)
