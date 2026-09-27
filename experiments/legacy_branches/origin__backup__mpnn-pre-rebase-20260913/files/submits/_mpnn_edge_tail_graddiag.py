"""Per-gradient breakdown for mpnn_edge_tail at the real workload point.

The bench scores all ten gradients as one concatenated vector, which catches a broken
gradient but does not say WHICH. This splits them, and adds an FP32 arbiter so BF16
rounding in the reference can be told apart from error in the kernel: if the kernel sits
CLOSER to FP32 than the BF16 reference does, the gap is the reference's rounding.

At B=8 T=8192 every intermediate is 805 MiB in BF16 and 1.6 GiB in FP32, so the three
gradient sets are produced one at a time, evacuated to host memory, and the graph freed
before the next one starts. Holding two of them at once does not fit on a 24 GiB card.

  python submits/_mpnn_edge_tail_graddiag.py [seq_len] [batch]
"""

import sys

import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.mpnn_edge_tail.triton.compute import edge_tail_compute

DEVICE = "cuda"
BF16 = torch.bfloat16
NAMES = ["edge", "query", "table", "w1", "w2", "b2", "w3", "b3", "gamma", "beta"]
CHUNK = 1 << 24


def rel(a, b):
    """Relative Frobenius with FP64 accumulation, chunked, operands left on host."""
    a, b = a.reshape(-1), b.reshape(-1)
    num = torch.zeros((), dtype=torch.float64)
    den = torch.zeros((), dtype=torch.float64)
    for i in range(0, a.numel(), CHUNK):
        x = a[i : i + CHUNK].double()
        y = b[i : i + CHUNK].double()
        num += ((x - y) ** 2).sum()
        den += (y**2).sum()
    return (num.sqrt() / den.sqrt().clamp_min(1e-30)).item()


def free():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run(seq_len: int, batch: int, d_pair: int = 128, k: int = 48):
    D = d_pair
    neighbors = min(k, seq_len)
    nodes = batch * seq_len
    rows = nodes * neighbors
    eps = 1e-5
    torch.manual_seed(0)

    def _w():
        return torch.randn(D, D, device=DEVICE, dtype=torch.float32) * (D**-0.5)

    params = (
        _w(), _w(), torch.zeros(D, device=DEVICE), _w(),
        torch.zeros(D, device=DEVICE), torch.ones(D, device=DEVICE),
        torch.zeros(D, device=DEVICE),
    )
    torch.manual_seed(1)
    edge = torch.randn(rows, D, device=DEVICE, dtype=BF16)
    query = torch.randn(nodes, D, device=DEVICE, dtype=BF16)
    table = torch.randn(nodes, D, device=DEVICE, dtype=BF16)
    index = torch.arange(rows, device=DEVICE) % nodes
    groups = torch.arange(rows, device=DEVICE) // neighbors
    dy = torch.randn(rows, D, device=DEVICE, dtype=BF16)

    print(f"\nseq_len={seq_len} batch={batch} rows={rows:,} "
          f"({rows * D * 2 / 2**30:.2f} GiB per BF16 intermediate)")

    def leaves():
        return [t.clone().requires_grad_(True) for t in (edge, query, table, *params)]

    def harvest(ls):
        """Grads to host, then drop every device reference this graph is holding."""
        out = [t.grad.detach().to("cpu", copy=True) for t in ls]
        for t in ls:
            t.grad = None
        return out

    def bf16_reference(e, q, t, w1, w2, b2, w3, b3, gamma, beta):
        """Exactly the bench's inline oracle: FP32 accumulate, BF16 between stages."""
        pre = (
            F.linear(e, w1.to(BF16)).float() + q[groups].float() + t[index].float()
        ).to(BF16)
        hidden = (
            F.linear(F.gelu(pre.float()).to(BF16), w2.to(BF16)).float() + b2
        ).to(BF16)
        update = F.linear(F.gelu(hidden.float()).to(BF16), w3.to(BF16)).float() + b3
        values = (e.float() + update).to(BF16)
        return F.layer_norm(values.float(), (D,), gamma, beta, eps).to(BF16)

    def fp32_reference(e, q, t, w1, w2, b2, w3, b3, gamma, beta):
        """Same maths with no BF16 rounding between stages -- the arbiter."""
        pre = F.linear(e.float(), w1) + q.float()[groups] + t.float()[index]
        hidden = F.linear(F.gelu(pre), w2) + b2
        update = F.linear(F.gelu(hidden), w3) + b3
        return F.layer_norm(e.float() + update, (D,), gamma, beta, eps)

    ls = leaves()
    bf16_reference(*ls).backward(dy)
    g_ref = harvest(ls)
    del ls
    free()

    ls = leaves()
    edge_tail_compute(*ls[:3], index, *ls[3:], eps, 0.0).backward(dy)
    g_kern = harvest(ls)
    del ls
    free()

    g_exact = None
    try:
        ls = leaves()
        fp32_reference(*ls).backward(dy.float())
        g_exact = harvest(ls)
        del ls
        free()
    except torch.cuda.OutOfMemoryError:
        free()
        print("  (FP32 arbiter OOMed at this size -- BF16 comparison only)")

    header = f"  {'grad':<8}{'triton vs bf16ref':>20}"
    if g_exact is not None:
        header += f"{'triton vs fp32':>17}{'bf16ref vs fp32':>18}   verdict"
    print(header)
    for i, name in enumerate(NAMES):
        line = f"  {name:<8}{rel(g_kern[i], g_ref[i]):>20.3e}"
        if g_exact is not None:
            tk, tr = rel(g_kern[i], g_exact[i]), rel(g_ref[i], g_exact[i])
            verdict = (
                "kernel MORE accurate" if tk < tr
                else "ok" if tk < 3 * tr
                else "SUSPECT"
            )
            line += f"{tk:>17.3e}{tr:>18.3e}   {verdict}"
        print(line)


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), torch.__version__,
          f"{torch.cuda.mem_get_info()[1] / 2**30:.1f} GiB")
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(L, B)
