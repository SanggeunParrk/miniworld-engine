"""Backward kernel latency (do_bench, L2 evicted) against the unit floors at A=48."""
import os
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_overlap"))
from bench import us  # noqa: E402
from token_dit_train.ref import make  # noqa: E402
from token_dit_train.fwd import prep, attn_fwd  # noqa: E402
from token_dit_train.bwd import bwd_prep, attn_dq, attn_dkv, attn_dkv2, attn_dqb, attn_dkv_nobias  # noqa: E402

TC, CLK, SMS = 757e12, 1.755e9, 132
A, H = 48, 16
for L in (384, 768):
    q, k, v, bias, _ = make(A, L)
    args = prep(q, k, v, bias, None)
    o, lse = attn_fwd(*args, A, L)
    dob, dd = bwd_prep(torch.randn_like(q), o, A, L)
    qs, kb, vb, bb, km = args
    t = us(lambda: attn_dq(qs, kb, vb, dob, bb, km, lse, dd, A, L))
    pairs = A * H * L * L
    f_tc = pairs * 288 / TC * 1e6                                   # 3 GEMMs of 48
    f_mufu = pairs / (16 * SMS * CLK) * 1e6
    print(f"bench: dq{os.environ.get('ATTN_DQ_DEFS', '')} L{L} A{A}: {t:8.1f} us  {pairs * 288 / t / 1e6:6.1f} TF/s | "
          f"floors: tensor {f_tc:6.1f}  mufu {f_mufu:6.1f}  -> {max(f_tc, f_mufu) / t * 100:5.1f} %", flush=True)
    if os.environ.get("TDT_DQB"):
        t1 = us(lambda: attn_dqb(qs, kb, vb, dob, bb, km, lse, dd, A, L))
        t2 = us(lambda: attn_dkv_nobias(qs, kb, vb, dob, bb, km, lse, dd, A, L))
        print(f"bench: dqb L{L} A{A}: {t1:8.1f} us   dkv(no dbias) {t2:8.1f} us   bwd total {t1 + t2:8.1f} us", flush=True)
        continue
    if os.environ.get("TDT_ONLY1"):
        t = us(lambda: attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, A, L))
        f_tc = pairs * 384 / TC * 1e6
        print(f"bench: dkv{os.environ.get('ATTN_DKV_DEFS', '')} L{L} A{A}: {t:8.1f} us  {pairs * 384 / t / 1e6:6.1f} TF/s | "
              f"floors: tensor {f_tc:6.1f}  mufu {f_mufu:6.1f}  -> {max(f_tc, f_mufu) / t * 100:5.1f} %", flush=True)
        continue
    t = us(lambda: attn_dkv2(qs, kb, vb, dob, bb, km, lse, dd, A, L))
    f_tc = pairs * 384 / TC * 1e6
    print(f"bench: dkv2 sch{os.environ.get('TDT_DKV_SCH', 'auto')} L{L} A{A}: {t:8.1f} us  {pairs * 384 / t / 1e6:6.1f} TF/s | "
          f"floors: tensor {f_tc:6.1f}  mufu {f_mufu:6.1f}  -> {max(f_tc, f_mufu) / t * 100:5.1f} %", flush=True)
    if os.environ.get("TDT_ONLY2"):
        continue
    t = us(lambda: attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, A, L))
    f_tc = pairs * 384 / TC * 1e6                                   # 4 GEMMs of 48
    print(f"bench: dkv{os.environ.get('ATTN_DKV_DEFS', '')} L{L} A{A}: {t:8.1f} us  {pairs * 384 / t / 1e6:6.1f} TF/s | "
          f"floors: tensor {f_tc:6.1f}  mufu {f_mufu:6.1f}  -> {max(f_tc, f_mufu) / t * 100:5.1f} %", flush=True)
