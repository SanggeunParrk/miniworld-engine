"""Token DiT FP8 error against fp64 (judge form of docs/fp8.md: max|y - y64| / max|y64|, rms_rel = ||y - y64|| / ||y64||) + the e4m3 weight pack
of ``quant_weight`` against its torch-op reference (``quant_weight_ref``, when the tree has it).

    python td_err.py [--lengths 384,768,1536] [--module TokenDiTProtenix]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="384,768,1536")
    ap.add_argument("--modules", default="TokenDiTProtenix,TokenDiTChai1,TokenDiTBoltz2,TokenDiTRoseTTAFold3")
    a = ap.parse_args()
    import torch
    from paper_common import cfg as cfgmod
    from foldjet_engine import fp8
    from foldjet_engine.kernels.conditioned_transition.cuda import token_dit_fp8 as td
    from foldjet_engine.measure import config
    from foldjet_engine.modules.primitives import unpadded_masks

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(td, "quant_weight_ref"):
        for shape in ((4096, 768), (768, 768), (3072, 768), (768, 1024)):
            w = (torch.randn(shape, device=dev) * 0.05).bfloat16()
            a8, sa = td.quant_weight(w)
            r8, sr = td.quant_weight_ref(w)
            diff = (a8.float() != r8.float()).float().mean().item()
            print(json.dumps({"weight_pack": list(shape), "e4m3 values differing": diff, "scale max rel diff": ((sa - sr).abs() / sr).max().item()}))
    for mod in a.modules.split(","):
        cfg = cfgmod.load(str(Path(__file__).resolve().parent / "cfg_tokendit.yaml"), argparse.Namespace(module=[mod], lengths=a.lengths, dtypes=["master"], modes=["eval"]))
        seen = set()
        for case in cfgmod.limit(cfg, 0):
            if case.samples != 5:
                continue
            key = (case.module, tuple(sorted(case.kwargs.items())) if hasattr(case, "kwargs") else case.row, case.length)
            if key in seen:
                continue
            seen.add(key)
            base = config.build(case, device=dev)
            with unpadded_masks():
                inputs = config.inputs(case, base, device=dev)
            ac = config.autocast(case)

            def run(m, inp, autocast):
                with torch.no_grad(), torch.autocast("cuda", dtype=autocast) if autocast is not None else contextlib.nullcontext():
                    return m._forward_cuda(**inp) if hasattr(m, "_forward_cuda") else m(**inp)

            ref_m = copy.deepcopy(base).double().set_backend("pytorch").eval()
            inp64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inputs.items()}
            with torch.no_grad():
                y64 = ref_m(**inp64)
            y64 = (y64[0] if isinstance(y64, tuple) else y64).double()
            res = {"case": config.describe(case, cfg)["label"]}
            for name, backend in (("default", "cuda"), ("fp8", "engine-fp8")):
                m = copy.deepcopy(base).set_backend(backend).eval()
                fp8.reset_route_stats()
                run(m, inputs, ac)
                y = run(m, inputs, ac)
                y = (y[0] if isinstance(y, tuple) else y).double()
                d = (y - y64).abs()
                res[name] = {"judge": (d.max() / y64.abs().max()).item(), "rms_rel": (d.norm() / y64.norm()).item()}
                if name == "fp8":
                    res["fired"] = fp8.route_stats()["token_dit"]["calls"] > 0
            print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
