# ConditionedTransition forward — the 1+2 | 3+4+5 two-kernel grouping — H100, fp32 / TF32

_Forward only. The post-AdaLN ops numbered: 1=expand(a=x@Wa^T,b=x@Wb^T), 2=SwiGLU(h=silu(a)·b),
3=squeeze(out=h@Ws^T), 4=to_scale(scale=cond@Wsc^T+b_sc), 5=gate(y=sigmoid(scale)·out)._

**Grouping = 1+2 | 3+4+5 — exactly TWO triton kernels, UNIFORM for atom (d=128) and token
(d=768)** (entry: `cond_transition_fwd_12_345`):
- **Kernel 1 (1+2)** `_expand_swiglu`: K-tiled `a=x@Wa^T`, `b=x@Wb^T` (tl.dot tf32), `h=silu(a)·b`
  → writes `h:(M,ND)` to HBM.
- **Kernel 2 (3+4+5)** `_squeeze_gate`: `out=h@Ws^T` (ND-tiled), `scale=cond@Wsc^T+b_sc` (DC-tiled),
  `y=sigmoid(scale)·out` — all fused → writes `y:(M,d)`.

`h` round-trips HBM between the two kernels — no register-resident squeeze, no b2b, no spill, so
the SAME path works for any d (atom and token). fp32 io, TF32 tensor cores.

## Forward bench (CUDA graph) — 1+2|3+4+5 vs b2b single-kernel vs eager vs torch.compile

_Baselines: b2b = the atom-only fused single kernel (`cond_transition_inference`, won't compile at
d=768 → n/a for token); torch.compile = `compile(reference, mode="reduce-overhead")`. Source:
`forward_12_345.out`._

| stream | M | d | cos | 1+2\|3+4+5 us | b2b us | eager us | compile us | vs b2b | vs eager | vs compile |
|---|---|---|---|---|---|---|---|---|---|---|
| atom | 2048 | 128 | 0.999999 | 13.3 | 25.1 | 30.8 | 45.4 | **1.88×** | 2.32× | 3.41× |
| atom | 4096 | 128 | 0.999999 | 16.5 | 25.3 | 39.6 | 50.2 | **1.53×** | 2.40× | 3.04× |
| atom | 8192 | 128 | 0.999999 | 29.0 | 27.0 | 57.8 | 70.5 | 0.93× | 2.00× | 2.43× |
| token | 384 | 768 | 0.999999 | 29.8 | n/a | 49.9 | 66.9 | n/a | 1.67× | 2.25× |
| token | 512 | 768 | 0.999999 | 32.6 | n/a | 49.8 | 73.5 | n/a | 1.53× | 2.25× |
| token | 768 | 768 | 0.999999 | 57.0 | n/a | 60.8 | 79.1 | n/a | 1.07× | 1.39× |
| token | 1024 | 768 | 0.999999 | 60.7 | n/a | 68.0 | 87.2 | n/a | 1.12× | 1.44× |

## Findings

- **Correct** (cos 0.999999) for both atom and token — the uniform two-kernel grouping handles
  d=768 cleanly (where the b2b single-kernel won't even compile).
- **Beats torch.compile everywhere: 1.39–3.41×.** Beats eager everywhere: 1.07–2.40×.
- **vs the b2b single-kernel (atom):** the simpler 1+2|3+4+5 is FASTER at small/mid atom
  (M=2048 1.88×, M=4096 1.53×) and only marginally behind at M=8192 (0.93×). The b2b's
  register-resident squeeze (avoiding the `h` HBM round-trip) only pays off at the largest M;
  the two-kernel grouping's lower per-kernel pressure wins below that.

## Verdict

The 1+2 | 3+4+5 two-kernel forward is the clean, uniform structure to ship: one code path for
both streams, correct, simpler than the b2b/CUTLASS paths, and it beats torch.compile at every
shape (and the b2b kernel at all but the largest atom M). Entry: `cond_transition_fwd_12_345`.
