# ffnv.py "FLAG,FLAG" CFG FILTER : append -D flags to every ffn_fwd_tf32* cubin entry, then ifeab.py CFG (single value) FILTER
import runpy, sys
from foldjet_engine.kernels.swa_dit.cuda.sm100 import tf32_fwd
fl = tuple(f"-D{x}" for x in sys.argv[1].split(",") if x); cfg, flt = sys.argv[2], sys.argv[3]
for k in list(tf32_fwd.CUBINS):
    if k.startswith("ffn_fwd_tf32"):
        stem, ex = tf32_fwd.CUBINS[k]; tf32_fwd.CUBINS[k] = (stem, tuple(ex) + fl)
print("ffn builds:", {k: v[1] for k, v in tf32_fwd.CUBINS.items() if k.startswith("ffn_fwd_tf32")}, flush=True)
sys.argv = ["ifeab.py", cfg, "foldjet_engine.modules.embedder._SCATTER8=True", flt]
runpy.run_path("n1cfg/ifeab.py", run_name="__main__")
