import torch
from common import graph_time
n = 4 * 2 ** 30 // 4
s = torch.empty(n, device="cuda", dtype=torch.float32).normal_(); d = torch.empty_like(s)
r = torch.empty(1, device="cuda")
print("read  (sum) TB/s", n * 4 / graph_time(lambda: torch.sum(s, dim=(0,), out=r.view(())), reps=5) / 1e6)
print("write (fill) TB/s", n * 4 / graph_time(lambda: d.fill_(1.0), reps=5) / 1e6)
print("copy TB/s", 2 * n * 4 / graph_time(lambda: d.copy_(s), reps=5) / 1e6)
print("add TB/s", 3 * n * 4 / graph_time(lambda: torch.add(s, s, out=d), reps=5) / 1e6)
