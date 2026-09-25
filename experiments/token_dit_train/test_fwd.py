"""Forward kernel against the fp64 reference (and the bf16-rounded-input reference, to separate kernel error from
input rounding)."""
import math
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_dit_train.ref import make, reference, rel, LOG2E  # noqa: E402
from token_dit_train.fwd import prep, attn_fwd  # noqa: E402

ok = True
for L, A, bs, mf in ((384, 4, 1.0, 0.0), (768, 8, 1.0, 0.0), (384, 8, 4.0, 0.0), (384, 8, 1.0, 0.2), (768, 48, 1.0, 0.0)):
    q, k, v, bias, mask = make(A, L, bias_scale=bs, mask_frac=mf, seed=L + A)
    o_ref, lse_ref = reference(q, k, v, bias, mask)
    # the same math on the bf16-rounded inputs: what a perfect bf16-operand kernel would give
    bias_r = (bias * LOG2E).bfloat16().double() / LOG2E                 # the kernels' rounding: bf16 of log2-unit bias
    o_bref, lse_bref = reference(q.bfloat16().float(), k.bfloat16().float(), v.bfloat16().float(), bias_r, mask)
    o, lse = attn_fwd(*prep(q, k, v, bias, mask), A, L)
    torch.cuda.synchronize()
    eo, el, eb, elb = rel(o, o_ref), rel(lse / LOG2E, lse_ref), rel(o_bref, o_ref), rel(lse_bref, lse_ref)
    good = eo < 1.2 * eb + 1e-4 and el < 1.5 * elb + 1e-5 and math.isfinite(eo)
    ok &= good
    print(f"fwd: L{L} A{A} bias*{bs} mask{mf}: O rel {eo:.2e} (bf16-input floor {eb:.2e})  LSE rel {el:.2e} ({elb:.2e})  "
          f"{'ok' if good else 'FAIL'}", flush=True)
print("fwd: ALL OK" if ok else "fwd: FAILED", flush=True)
