"""Run upstream's flash builder with a source-matched recipe across torch ABIs.

Upstream build_flash_prebuilt.py requires an already-shipped manifest for the
target ABI, preventing a new ABI from being built. Only recipe lookup changes;
source/flags/CUTLASS checks and the original build/load check remain intact.
"""
from pathlib import Path
import os
import sys

root=Path(os.environ['MINIWORLD_ANTHROPIC_ROOT'])/'common/opt_core/opt_core/kernels/transition'
script=root/'build_flash_prebuilt.py'
src=script.read_text()
old='sealed_man = json.load(open(os.path.join(SEALED, "prebuilt", sealed_key, "manifest.json"), encoding="utf-8"))'
new='''recipe_path = os.path.join(SEALED, "prebuilt", sealed_key, "manifest.json")
    if not os.path.isfile(recipe_path):
        import glob
        candidates = sorted(glob.glob(os.path.join(SEALED, "prebuilt", "*", "manifest.json")))
        candidates = [p for p in candidates if json.load(open(p))["source_sha256"].get(B.SRC) == _sha256(os.path.join(SEALED, "csrc", B.SRC))]
        if not candidates:
            raise RuntimeError("No source-matched upstream build recipe")
        recipe_path = candidates[0]
    print("[recipe]", recipe_path, flush=True)
    sealed_man = json.load(open(recipe_path, encoding="utf-8"))'''
assert src.count(old)==1
src=src.replace(old,new)
exec(compile(src,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
