# cmpscan.py A.pt B.pt : bitwise comparison of two bitscan outputs
import sys, torch
a = torch.load(sys.argv[1]); b = torch.load(sys.argv[2])
n = eq = 0; worst = 0.0; bad = []
for k in a:
    if k.endswith("|repeat"):
        if not bool(a[k]) or (k in b and not bool(b[k])): bad.append(("NONDETERMINISTIC", k))
        continue
    if k not in b: continue
    n += 1
    d = (a[k] - b[k]).abs().max().item()
    if d == 0.0: eq += 1
    else:
        worst = max(worst, d); bad.append((k, d, float(a[k].abs().max())))
print(f"cases compared {n}, bitwise equal {eq}, worst max|diff| {worst}")
for x in bad[:20]: print(x)
