"""repro: same-named case in two dirs: bytes equal? RMSD, ipTM.  usage: rep.py DIRA DIRB CASEPREFIX (e.g. bf16)"""
import sys, glob, os, torch
a, b, pre = sys.argv[1:4]
def kabsch(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    u, s, vt = torch.linalg.svd(x.T @ y); dd = torch.sign(torch.linalg.det(u @ vt)); D = torch.diag(torch.tensor([1, 1, dd]))
    return ((x @ (u @ D @ vt) - y) ** 2).sum(-1).mean().sqrt().item()
for p in sorted(glob.glob(f"{a}/*.{pre}_t*.pt")):
    q = f"{b}/{os.path.basename(p)}"
    if not os.path.exists(q): continue
    x, y = torch.load(p), torch.load(q)
    print(os.path.basename(p), "bytes-equal" if torch.equal(x["coords"], y["coords"]) else "DIFF", f"rmsd {kabsch(x['coords'].reshape(-1,3), y['coords'].reshape(-1,3)):.3f}", f"ipTM {float(x['iptm'].mean()):.3f} vs {float(y['iptm'].mean()):.3f}")
