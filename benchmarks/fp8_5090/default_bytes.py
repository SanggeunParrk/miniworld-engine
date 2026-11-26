"""The default (FP8 off) outputs of the token / atom DiT blocks as sha256 of their bytes, for a byte comparison of two trees (run once per tree with
PYTHONPATH pointing at it, every FOLDJET_ENGINE_*FP8* variable unset): eval / master, shared conditioning, eager and a CUDA-graph replay.

    python default_bytes.py Module:A:L[:pick] ...
"""

import hashlib
import os
import sys

import torch
from dit_common import Rig, case, graph

assert not any("FP8" in k for k in os.environ if k.startswith("MINIWORLD")), "unset every FOLDJET_ENGINE_*FP8* variable"
for spec in sys.argv[1:]:
    mod, A, L, *opt = spec.split(":")
    rig = Rig(case(mod, int(L), int(A), pick=opt[0] if opt else None))
    y = rig.run()
    g, _gf, _h = graph(rig.step, n=1, flush=False)
    g.replay()
    torch.cuda.synchronize()
    with torch.no_grad():
        o = rig.step()  # eager again after the capture
    torch.cuda.synchronize()
    yo = (o if torch.is_tensor(o) else o[0]).float()
    print(spec, hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest()[:16], hashlib.sha256(yo.cpu().numpy().tobytes()).hexdigest()[:16], flush=True)
