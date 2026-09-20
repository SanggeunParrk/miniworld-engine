"""Standalone benchmark of the TriMul contraction (no payload needed) X[c,i,j] = sum_k a[c,i,k] b[c,j,k] (bf16, fp32 accumulate) as the engine issues it:
torch.bmm(a, b.transpose(1,2), out=x) with planes [2ch, Np, Np].  Variants: default cuBLAS, cuBLASLt preferred, fp32 out, layouts."""
import argparse, json, statistics, torch
p = argparse.ArgumentParser(); p.add_argument('--length', type=int, default=768); p.add_argument('--ch', type=int, default=128); p.add_argument('--output', required=True); a = p.parse_args()
N, ch = a.length, a.ch
torch.manual_seed(1)
ab = torch.randn(2 * ch, N, N, device='cuda', dtype=torch.bfloat16)
A_, B_ = ab[:ch], ab[ch:]
x = torch.empty(ch, N, N, device='cuda', dtype=torch.bfloat16)
Bt = B_.transpose(1, 2).contiguous()           # [c, k, j] (NN form) — only possible if K1 wrote b transposed
x32 = torch.empty(ch, N, N, device='cuda', dtype=torch.float32)

def timeit(fn, name, iters=100):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    ev = [(e.name, e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total) for e in prof.events() if e.device_type.name == 'CUDA']
    tot = sum(t for _, t in ev) / iters
    names = sorted({n[:60] for n, _ in ev})
    # also CUDA-graph timing
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): fn()
    rounds = []
    for _ in range(3):
        for _ in range(10): g.replay()
        torch.cuda.synchronize(); st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(50): g.replay()
        en.record(); torch.cuda.synchronize(); rounds.append(st.elapsed_time(en) * 1000 / 50)
    r = dict(name=name, cupti_us=tot, graph_us=statistics.median(rounds), kernels=names)
    print('RESULT', json.dumps(r), flush=True); return r

res = []
with torch.no_grad():
    res.append(timeit(lambda: torch.bmm(A_, B_.transpose(1, 2), out=x), 'bmm TN (engine form)'))
    res.append(timeit(lambda: torch.bmm(A_, Bt, out=x), 'bmm NN (b pre-transposed)'))
    res.append(timeit(lambda: torch.bmm(B_, A_.transpose(1, 2), out=x), 'bmm TN swapped operands (X^T)'))
    res.append(timeit(lambda: torch.matmul(A_, B_.transpose(1, 2), out=x), 'matmul TN'))
    res.append(timeit(lambda: torch.bmm(A_.float(), B_.transpose(1, 2).float(), out=x32), 'bmm fp32 (reference cost)'))
    try:
        torch.backends.cuda.preferred_blas_library('cublaslt')
        res.append(timeit(lambda: torch.bmm(A_, B_.transpose(1, 2), out=x), 'bmm TN cublaslt-preferred'))
        res.append(timeit(lambda: torch.bmm(A_, Bt, out=x), 'bmm NN cublaslt-preferred'))
        torch.backends.cuda.preferred_blas_library('cublas')
    except Exception as e:
        print('cublaslt pref failed', repr(e)[:200])
    # batch splitting: two half-batches on two streams (overlap tail waves)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    def two_streams():
        cur = torch.cuda.current_stream()
        s1.wait_stream(cur); s2.wait_stream(cur)
        with torch.cuda.stream(s1): torch.bmm(A_[:ch // 2], B_[:ch // 2].transpose(1, 2), out=x[:ch // 2])
        with torch.cuda.stream(s2): torch.bmm(A_[ch // 2:], B_[ch // 2:].transpose(1, 2), out=x[ch // 2:])
        cur.wait_stream(s1); cur.wait_stream(s2)
    res.append(timeit(two_streams, 'bmm TN two half-batches on two streams'))
    # flops
    flops = 2.0 * ch * N * N * N
    for r in res: r['tflops_graph'] = flops / (r['graph_us'] * 1e-6) / 1e12
    print('FLOPS', flops / 1e9, 'GFLOP', flush=True)
json.dump(dict(N=N, ch=ch, results=res), open(a.output, 'w'), indent=1)
