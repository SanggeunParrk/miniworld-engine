"""Anthropic's fused SWAAtomBlock (uplifting-biomolecular-modeling @f4f62fa, esmfold2/.../driver/ef2_atom.py, the fast tier with gemm="bf16":
K1 rms-adaLN + q|k|v|gate + qk-norm + RoPE, K2 window attention, K3 out_proj + residual + rms-adaLN + SwiGLU + residual) on the same
block, inference only (no backward). Its residual stream is fp32 and its modulation / RoPE tables are per row."""
import sys, types, torch
from common import make, hoist_mod, block_ref, rel, graph_time, HW, C
import anthropic_ef2_atom as E

E._CFG["gemm"] = "bf16"
E._sm_count()


def fake_block(w):
    at = types.SimpleNamespace(Wqkv=types.SimpleNamespace(weight=w["wqkv"]), gate_proj=types.SimpleNamespace(weight=w["wg"]),
                               out_proj=types.SimpleNamespace(weight=w["wo"]), n_heads=4, head_dim=32, scale=32 ** -0.5, half_window=HW)
    ffn = types.SimpleNamespace(w_up=types.SimpleNamespace(weight=w["wu"]), w_down=types.SimpleNamespace(weight=w["wd"]))
    return types.SimpleNamespace(attn=at, ffn=ffn)


def bind(A, S, seed=0):
    q, cb, cos, sin, su, w = make(A, S, seed=seed)
    mod = hoist_mod(cb, w["wmod"])                                        # [S, 6C] -> per row [A S, 6C]
    M = A * S
    fs = {"M": M, "N": S, "Bp": A, "cos": cos.repeat(A, 1).contiguous(), "sin": sin.repeat(A, 1).contiguous(), "seqlen": su}
    modr = mod.repeat(A, 1).contiguous()
    x = q.reshape(M, C).float().contiguous()
    blk = fake_block(w)
    run = lambda: E._fused_block(blk, x, modr, fs)[0]
    return run, (q, mod, cos, sin, su, w)


run, (q, mod, cos, sin, su, w) = bind(4, 1024)
out = run().view(q.shape)
ref = block_ref(q, mod, cos, sin, su, *(w[k] for k in ("wqkv", "wg", "wo", "wu", "wd")))
print(f"anthropic vs fp64 (A4, S1024): out rel {rel(out, ref):.2e}", flush=True)
for L in [int(x) for x in (sys.argv[1:] or ["384", "768"])]:
    run, _ = bind(5, 8 * L)
    print(f"L{L} (S={8 * L}) anthropic  inference {graph_time(run):10.1f} us", flush=True)
