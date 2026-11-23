"""Default-route bytes of OuterProductMean / MSAPairWeightedAveraging on the RTX 5090 (FP8 off): sha1 of each case's output, one JSON line per case.
Run once per tree (PYTHONPATH = that tree's src) and compare the files: the opt-in e4m3 routes must leave the default bytes unchanged.

    python msa_bit.py OUT.jsonl"""
import hashlib
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _msa_common import opm_case, pwa_case  # noqa: E402

CASES = [
    "opm:64:128:256:1024", "opm:64:128:384:2048:a", "opm:128:256:256:1024:a", "opm:256:512:128:1024", "opm:64:128:208:296",
    "pwa:64:32:128:1024:384", "pwa:64:32:128:512:768", "pwa:64:8:128:1024:400", "pwa:128:8:256:256:768", "pwa:128:16:256:256:512",
    "pwa:256:32:512:256:384",
]  # fmt: skip
with open(sys.argv[1], "w") as fh:
    for c in CASES:
        kind, spec = c.split(":", 1)
        m, inp = (opm_case if kind == "opm" else pwa_case)(spec)
        with torch.no_grad():
            y = m(*inp)
        torch.cuda.synchronize()
        h = hashlib.sha1(y.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        fh.write(json.dumps({"case": c, "sha1": h}) + "\n")
        print(c, h, flush=True)
        del m, inp, y
