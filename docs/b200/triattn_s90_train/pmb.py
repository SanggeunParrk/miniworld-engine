# pmb.py: schedule C (save P in bf16) -- the HBM time of the P tensor alone: written once in the forward core, read by each of the two backward kernels.
# Measured with L2 flushed: a streaming write (fill) and a streaming read (sum into a scalar, no write), 4 heads, bf16 P = 2 L^3 H bytes.
import torch
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
def t(fn, n=10):
    ts = []
    for _ in range(n):
        flush.zero_(); torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
for L in (128, 256, 384, 512, 640, 768):
    n = L ** 3 * 4  # scores, 4 heads
    p = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    w = t(lambda: p.fill_(1.0))
    x = p.view(-1, 4096)
    rd = t(lambda: x.amax(dim=1))   # reads P, writes n/4096 values
    cp = t(lambda: p.clone())
    print(f"L{L} H4: P {n*2/1e6:8.1f} MB | write {w:8.1f} us ({n*2/w/1e6:.2f} TB/s) | read {rd:8.1f} us ({n*2/rd/1e6:.2f} TB/s) | clone r+w {cp:8.1f} us | C traffic = 1 write + 2 reads = {w+2*rd:8.1f} us", flush=True)
    del p, x
