import warnings, importlib.util
warnings.filterwarnings("ignore")
import cutlass
spec = importlib.util.spec_from_file_location("refp","/home/snu_hwle/psk/ncu/ref_dense_gemm_persistent.py")
refp = importlib.util.module_from_spec(spec); spec.loader.exec_module(refp)
M,N,K,L = 1048576,128,128,1
for cmaj in ["n","m"]:
    try:
        t = refp.run((M,N,K,L), ab_dtype=cutlass.BFloat16, c_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32,
            a_major="k", b_major="k", c_major=cmaj, mma_tiler_mn=(128,128), cluster_shape_mn=(1,1),
            use_2cta_instrs=False, use_tma_store=True, tolerance=2.0, warmup_iterations=10, iterations=30,
            skip_ref_check=False, benchmark=True)
        print(f"RESULT c_major={cmaj}: {t/1000:.4f} ms/side (ref-check passed)", flush=True)
    except Exception as e:
        print(f"FAIL c_major={cmaj}: {type(e).__name__}: {str(e)[:160]}", flush=True)
