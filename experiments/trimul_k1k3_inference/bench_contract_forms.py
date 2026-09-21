"""Upper bound for fusing the bidirectional contraction into one GEMM.

The bidirectional update contracts the two channel halves differently: outgoing is NT (a . b^T), incoming is TN (a^T . b),
so the current composition is two strided-batched GEMMs of c_hidden/2 planes each.  If K1 stored the incoming half already
transposed (upstream does exactly that for the unidirectional incoming direction, INCOMING_MODE="kt"), both halves would be
NT and one GEMM of c_hidden planes would do.  This measures what that would buy, on the same buffers:

  split   = bmm(a[:h], b[:h]^T) + bmm(a[h:]^T, b[h:])     what the composition runs today
  one_nt  = bmm(a, b^T)                                   the same shapes in one call (numerically the wrong thing for the
                                                          second half -- a timing bound, not an implementation)
  two_nt  = bmm(a[:h], b[:h]^T) + bmm(a[h:], b[h:]^T)     isolates "TN vs NT" from "two calls vs one"
"""
import argparse, json, statistics
import torch

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--hidden", type=int, default=128, help="per direction; the planes tensor is 2 x (2h)")
p.add_argument("--output", required=True)
a = p.parse_args()

torch.manual_seed(4103)
torch.backends.cuda.matmul.allow_tf32 = False
N = (a.length + 15) // 16 * 16
h, ch = a.hidden, 2 * a.hidden
A = torch.randn(ch, N, N, device="cuda", dtype=torch.bfloat16)
B = torch.randn(ch, N, N, device="cuda", dtype=torch.bfloat16)
T = torch.empty(ch, N, N, device="cuda", dtype=torch.bfloat16)

forms = {
    "split_nt_tn": lambda: (torch.bmm(A[:h], B[:h].transpose(1, 2), out=T[:h]), torch.bmm(A[h:].transpose(1, 2), B[h:], out=T[h:])),
    "one_nt": lambda: torch.bmm(A, B.transpose(1, 2), out=T),
    "two_nt": lambda: (torch.bmm(A[:h], B[:h].transpose(1, 2), out=T[:h]), torch.bmm(A[h:], B[h:].transpose(1, 2), out=T[h:])),
}
res = {}
for name, fn in forms.items():
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    rounds = []
    for _ in range(3):
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(50):
            g.replay()
        en.record(); torch.cuda.synchronize()
        rounds.append(st.elapsed_time(en) * 1000 / 50)
    res[name] = dict(rounds=rounds, median=statistics.median(rounds))
    print("RESULT", name, json.dumps(res[name]), flush=True)
json.dump(dict(length=a.length, Np=N, hidden_per_direction=h, forms=res), open(a.output, "w"), indent=1)
