import sys, torch
a = torch.load(sys.argv[1]).float(); b = torch.load(sys.argv[2]).float()
d = (a - b).abs()
print(sys.argv[1].split("/")[-1], "vs", sys.argv[2].split("/")[-1], "maxdiff", float(d.max()), "mean", float(d.mean()), "ref absmax", float(a.abs().max()), "equal", bool((a == b).all()))
