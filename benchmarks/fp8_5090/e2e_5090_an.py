"""End-level table of ``e2e_5090.py`` runs (the H100 rule of docs/fp8.md): per model and target, the bf16 trunk-seed pair RMSD max / mean, the same-seed
FP8 vs bf16 RMSD max / mean, max |d ipTM|, inside = max same-seed RMSD <= max(1.1 x max bf16 pair RMSD, 0.5 A). All-atom coordinates, Kabsch superposition.

    python e2e_5090_an.py DIR [DIR ...]          # DIR = one model's output directory (bf16_t* and fp8_t* cases)"""

import glob
import itertools
import json
import os
import statistics as st
import sys

import torch


def kabsch(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _s, vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.linalg.det(u @ vt))
    D = torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=a.dtype))
    return ((a @ (u @ D @ vt) - b) ** 2).sum(-1).mean().sqrt().item()


def ld(p):
    d = torch.load(p)
    c = d["coords"].reshape(-1, 3).double()
    return c, (None if d["iptm"] is None else float(d["iptm"].float().mean()))


MODE = os.environ.get("E2E_MODE", "fp8")
for dr in sys.argv[1:]:
    files: dict = {}
    for p in glob.glob(f"{dr}/*.pt"):
        t, c = os.path.basename(p)[:-3].split(".")
        files.setdefault(t, {})[c] = p
    n_in = n_all = 0
    sat = 0
    print(f"== {dr}")
    for t in sorted(files):
        C = files[t]
        bf = sorted(c for c in C if c.startswith("bf16_t"))
        if len(bf) < 2:
            continue
        D = {n: ld(C[n]) for n in bf}
        sp = [kabsch(D[a][0], D[b][0]) for a, b in itertools.combinations(bf, 2)]
        thr = max(1.1 * max(sp), 0.5)
        fs = sorted(c for c in C if c.startswith(MODE + "_t") and "bf16_t" + c.split("_t")[-1] in D)
        if not fs:
            print(f"{t[:4]} bf16 n={len(bf)} {max(sp):.2f}/{st.mean(sp):.2f} | fp8: -")
            continue
        same, dip = [], []
        for n in fs:
            b = "bf16_t" + n.split("_t")[-1]
            c, i = ld(C[n])
            same.append(kabsch(c, D[b][0]))
            if i is not None and D[b][1] is not None:
                dip.append(abs(i - D[b][1]))
            j = C[n][:-3] + ".json"
            if os.path.exists(j) and any(json.load(open(j)).get("saturated", [])):
                sat += 1
        ok = max(same) <= thr
        n_in += ok
        n_all += 1
        print(f"{t[:4]} bf16 n={len(bf)} pair max/mean {max(sp):.2f}/{st.mean(sp):.2f} | fp8 n={len(fs)} same-seed max/mean {max(same):.2f}/{st.mean(same):.2f} "
              f"[{' '.join(f'{x:.2f}' for x in same)}] thr {thr:.2f} max|dipTM| {max(dip) if dip else float('nan'):.3f} {'inside' if ok else 'OUT'}", flush=True)
    print(f"inside {n_in}/{n_all}, fp8 cases with a saturation flag {sat}")
