"""q|k|v|g GEMM in fp8 at the step's shapes: is any available kernel fast enough to matter?

  bf16 quack      what the step runs today (its picked config)
  scaled_mm rw    torch._scaled_mm, e4m3, per-row A scale x per-column B scale, bias, bf16 out (the accurate recipe)
  quack fp8 pt    quack.gemm.gemm on e4m3 with one per-tensor alpha (quack sm90 has no row/column scale; gemm_act
                  does not expose alpha)
Timed both ways: do_bench (bench.us) and a 20-call graph replay (how the picker times, the step's regime)."""
import statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import us                                                  # noqa: E402
from tdit.runner import _quack_gemm_act, PLAIN_CFGS                   # noqa: E402

DS, dev = 768, "cuda"
f8 = torch.float8_e4m3fn


def graph_us(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        for _ in range(n): fn()
    best = 1e9
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / n)
    return best


gemm_act = _quack_gemm_act()
from quack.gemm import gemm as qgemm                                  # noqa: E402  (takes alpha; gemm_act does not)
import cutlass                                                        # noqa: E402
from quack import cute_dsl_utils as _cdu                              # noqa: E402
# quack's torch -> CuTe dtype map has no fp8 entry although its sm90 GEMM accepts Float8E4M3FN: add it here only.
_cdu.torch2cute_dtype_map.setdefault(torch.float8_e4m3fn, cutlass.Float8E4M3FN)
for M in (3840, 1920):
    N, K = 4 * DS, DS
    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev).to(torch.bfloat16)
    W = (torch.randn(N, K, device=dev) * K ** -0.5).to(torch.bfloat16)
    bias = torch.zeros(N, device=dev, dtype=torch.bfloat16); bias[:DS] = 0.1
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ref = torch.addmm(bias.float(), A.float(), W.float().t())
    rows = []

    def quack_bf16(c):
        return lambda: gemm_act(A[None], W[None], None, None, out[None], None, None, c[0], c[1], c[2], c[3],
                                pingpong=c[4], rowvec_bias=bias[None])
    best = None
    for c in PLAIN_CFGS:
        try:
            t = graph_us(quack_bf16(c))
        except Exception:
            continue
        if best is None or t < best[0]: best = (t, c)
    fn = quack_bf16(best[1]); fn()
    rows.append((f"bf16 quack {best[1]}", us(fn), best[0], float((out.float() - ref).norm() / ref.norm())))

    sa = A.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448
    sb = W.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448
    Aq = (A.float() / sa).to(f8); Wq = (W.float() / sb).to(f8)
    sbt = sb.t().contiguous()
    try:
        fn = lambda: torch._scaled_mm(Aq, Wq.t(), scale_a=sa, scale_b=sbt, bias=bias, out_dtype=torch.bfloat16)
        o = fn()
        rows.append(("scaled_mm rowwise", us(fn), graph_us(fn), float((o.float() - ref).norm() / ref.norm())))
    except Exception as e:  # noqa: BLE001
        print(f"  scaled_mm rowwise failed: {type(e).__name__}: {str(e)[:120]}", flush=True)

    ta = float(A.float().abs().max()) / 448; tb = float(W.float().abs().max()) / 448
    Ap = (A.float() / ta).to(f8); Wp = (W.float() / tb).to(f8)
    best8 = None
    for c in PLAIN_CFGS + ((128, 256, 1, 1, True), (128, 256, 2, 1, True), (128, 128, 2, 1, True)):
        def q8(c=c):
            return qgemm(Ap[None], Wp[None], out[None], None, None, c[0], c[1], c[2], c[3],
                         pingpong=c[4], rowvec_bias=bias[None], alpha=ta * tb)
        try:
            t = graph_us(q8)
        except Exception as e:  # noqa: BLE001
            err = f"{c}: {type(e).__name__}: {str(e)[:160]}"
            continue
        if best8 is None or t < best8[0]: best8 = (t, c, q8)
    if best8:
        best8[2]()
        rows.append((f"quack fp8 per-tensor {best8[1]}", us(best8[2]), best8[0], float((out.float() - ref).norm() / ref.norm())))
    else:
        print(f"  quack fp8: no config ran ({err})", flush=True)

    print(f"M={M} N={N} K={K}  ({2 * M * N * K / 1e9:.1f} GFLOP)", flush=True)
    for n, t1, t2, e in rows:
        print(f"  {n:<44} do_bench {t1:6.2f} us   graph {t2:6.2f} us   rel err vs fp32 {e:.2e}", flush=True)
